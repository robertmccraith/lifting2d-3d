import numpy as np
from numpy.core.fromnumeric import compress
import torch
import glob
from scipy.spatial import ConvexHull
from copy import deepcopy
from scipy import interpolate
from tqdm import tqdm
import os 
from collections import defaultdict

from shapely.geometry import Polygon, Point
import cv2
import torch.nn.functional as F
from torchvision import transforms
cv2.setNumThreads(0)

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

def project_image_to_rect(P, uv_depth):
	''' Input: nx3 first two channels are uv, 3rd channel
				is depth in rect camera coord.
		Output: nx3 points in rect camera coord.
	'''
	c_u = P[0,2]
	c_v = P[1,2]
	f_u = P[0,0]
	f_v = P[1,1]
	b_x = P[0,3]/(-f_u) # relative 
	b_y = P[1,3]/(-f_v)

	n = uv_depth.shape[0]
	x = ((uv_depth[:,0]-c_u)*uv_depth[:,2])/f_u + b_x
	y = ((uv_depth[:,1]-c_v)*uv_depth[:,2])/f_v + b_y
	pts_3d_rect = np.zeros((n,3))
	pts_3d_rect[:,0] = x
	pts_3d_rect[:,1] = y
	pts_3d_rect[:,2] = uv_depth[:,2]
	return pts_3d_rect

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def get_centre_view_rot_angle(intrinsics, bb2d):
	''' Get the frustum rotation angle, it isshifted by pi/2 so that it
	can be directly used to adjust GT heading angle '''
	uv = np.array([[ bb2d[[0,2]].mean(), bb2d[[1,3]].mean(), 20]])
	
	box2d_centre_rect = project_image_to_rect(intrinsics, uv)

	frustum_angle = -1 * np.arctan2(box2d_centre_rect[0,2], box2d_centre_rect[0,0])
	
	return np.pi / 2.0 + frustum_angle


def get_centre_view_box3d_centre(corners, angle):
	''' Frustum rotation of 3D bounding box centre. '''
	box3d_centre = (corners[0, :] + corners[6, :]) / 2.0
	# box3d_centre = get_box3d_centre(corners, angle)

	return rotate_pc_along_y(np.expand_dims(box3d_centre, 0), angle).squeeze()



def get_centre_view_point_set(point_set, angle):
	''' Frustum rotation of point clouds.
	NxC points with first 3 channels as XYZ
	z is facing forward, x is left ward, y is downward
	'''
	return rotate_pc_along_y(point_set, angle)



class KittiObjectsDataset(torch.utils.data.Dataset):

	def __init__(self,
				data_path,
				kitti_path,
				n_pts=1024,
				reflectance=True,
				augs=False,
				inliers=False,
				mask_2d=False,
				frames=range(1),
				width_min=25,
				lidar_min=0,
				mask_min=0,
				truncation=0,
				ret_all=False,
				img_scale=1.,
				verbose=False,
				min_conf=0.5,
				yaw="",
				render=False):
		self.ret_all = ret_all
		
		files = list(map(lambda a: data_path+str(a)+".npz",range(len(glob.glob(data_path+"*[0-9].npz")))))

		self.objects = {}
		

		self.data_path = data_path
		self.kitti_path = kitti_path

		self.augs = augs
		self.n_pts = n_pts
		self.reflectance = reflectance

		self.inliers = inliers
		self.mask_2d = mask_2d
		self.frames = frames
		self.img_scale = img_scale
		self.verbose=verbose
		self.yaw = yaw

		self.render = render

		self.files = []
		self.indexing = defaultdict(list)
		self.filters = defaultdict(list)

		split = "train" if "train" in data_path else "val"

		folder = data_path[:-1].split("-")[-1]
		compressed_dir = f"{folder}-{split}.npy"

		if os.path.exists(compressed_dir):# and False:
			self.files = np.load(compressed_dir,allow_pickle=True)
		else:
			for f in tqdm(files[:]):
				# print(f)
				inputs = {}
				lo = np.load(f, allow_pickle=True)

				for k,v in lo.items():
					inputs[k] = v


				if len(inputs["box"]) == 0 or inputs["box"][0] is None:
					continue
				

				inputs["box"] = np.array([b for b in inputs["box"] if b is not None])

				wmin = inputs["box"][0,3]-inputs["box"][0,1]>=width_min
				


				if inputs["pts"].shape[0]<len(self.frames) or inputs["Rt"].shape[0]<len(self.frames):
					continue
				
				if truncation>=0:
					trunc = np.all([box[0]>0+truncation and box[1]>0+truncation and box[2]<inputs["img_size"][0]-truncation and box[3]<inputs["img_size"][1]-truncation for box in inputs["box"][:1]])
				else:
					trunc = True
				

				min_pts = np.all([len(inputs["pts"][frame])>lidar_min for frame in frames])
				
				l_min = np.all([inputs["pts"][frame].shape[0]>mask_min for frame in frames])

				m_min = np.all([inputs["mask_2d"][frame].sum()>mask_min for frame in frames]) if self.mask_2d else True


				occ = inputs["occlusion"]<1 if "occlusion" in inputs else True

				if wmin and m_min and l_min and trunc and min_pts and inputs["scores"][0]>min_conf and 2<=inputs["box"].shape[0]:
					

					self.files.append(inputs)

					if "img_id" in inputs:
						self.indexing[int(inputs["img_id"])].append(len(self.files)-1)
						self.filters[len(self.files)-1] = [wmin, m_min, l_min, trunc, min_pts]

					if "raw_id" in inputs:
						self.indexing["/".join(inputs["raw_id"])].append(len(self.files)-1)
			np.save(compressed_dir, self.files)

		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

		self.img_size = (int(1242//self.img_scale),int(375//self.img_scale))

	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, ind):
		inputs = deepcopy(self.files[ind])


		folder, drive, f = inputs["raw_id"]



		inputs["K"][:2] /= self.img_scale
		color = [0.485*255, 0.456*255, 0.406*255]

		for ind, frame in enumerate(self.frames):
			if self.render:
				d=self.kitti_path.replace("object", "raw")
				masks = np.load(f"{d}/{folder}/{drive}/mask_rcnn_02_big/data/{(int(f)+frame):0>10}.npz", allow_pickle=True)['instances']
				
				current = masks[inputs["mask2d_img"][frame][-1]]

				inputs[("current",ind)] = cv2.resize(current.astype(np.uint8), self.img_size, cv2.INTER_NEAREST)
				
				occluding = 1-masks[inputs["mask2d_img"][frame][:-1]].sum(0)
				inputs[("occluding",ind)] = cv2.resize(occluding.astype(np.uint8), self.img_size, cv2.INTER_NEAREST)

			if len(inputs["pts"])<=frame:continue
			if len(inputs["pts"][frame])==0:continue

			pts = inputs["pts"][frame]

			inputs[("npts",ind)] = pts.shape[0]

			if self.mask_2d:
				fmask = inputs["mask_2d"][frame]
				pts = pts[fmask]

			if self.n_pts > 0:

				inds = np.random.choice(pts.shape[0], self.n_pts, pts.shape[0] < self.n_pts)
				pts = pts[inds]

			inputs[("pts",ind)] = pts.transpose()

			inputs[("Rt",ind)] = inputs["Rt"][frame]


			continue
			if "frames" in inputs:
				f_id = int(f)+inputs["frames"][frame]
			else:
				f_id = int(f)+frame
			img_path = self.kitti_path.replace("object", "raw")+f"{folder}/{drive}/image_02/data/{(f_id):0>10}.jpg"
			box = inputs["box"].astype(np.int32)[frame]

			img = cv2.imread(img_path)

			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			
			crop = img.copy()

			# crop[np.where(inputs["masks"][frame]!=2)] = color
			crop = crop[box[1]:box[3],box[0]:box[2]].copy()

			cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color=(0,255,0), thickness=2)

			inputs[("img",ind)] = cv2.resize(img, self.img_size, cv2.INTER_NEAREST)/255.
			
			#TODO: colour augs
			desired_size = max(crop.shape)
			delta_w = desired_size - crop.shape[1]
			delta_h = desired_size - crop.shape[0]
			top, bottom = delta_h//2, delta_h-(delta_h//2)
			left, right = delta_w//2, delta_w-(delta_w//2)

			
			crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

			crop = cv2.resize(crop, (64,64), cv2.INTER_NEAREST)/255.
			
			inputs[("crop_raw",ind)] = crop.copy()

			crop = torch.from_numpy(np.transpose(crop, (2,0,1))).float()
			crop = self.normalize(crop)

			inputs[("crop",ind)] = crop


			
		
		inputs["box"] = np.array([b for b in inputs["box"] if b is not None])
		inputs["box"] = inputs["box"][self.frames]/self.img_scale

		if "velocities" in inputs:
			if np.linalg.norm(np.sum(inputs["velocities"],0))>2:

				inputs["velocities"] = np.cumsum(inputs["velocities"],0)[self.frames[:-1]]
				

			else:
				inputs["velocities"] = np.zeros_like(inputs["velocities"][self.frames[:-1]])
		
		inputs["scores"] = inputs["scores"][0]
		



		if self.augs and np.random.rand()>0.5:
			ran = self.frames if len(self.frames)<20 else [0]
			for frame in ran:
				inputs[("pts",frame)][0] *= -1
				inputs["box"][:,[0,2]] = inputs[("img",0)].shape[1] - inputs["box"][:,[0,2]]
				inputs["box"][:,[1,3]] = inputs[("img",0)].shape[1] - inputs["box"][:,[1,3]]
				
				inputs[("img",frame)] = cv2.flip(inputs[("img",frame)],1)

			if "corners" in inputs:
				inputs["corners"][:,0] *= -1
				inputs["yaw"] = (3.14 - inputs["yaw"])
				if inputs["yaw"]>3.14:
					inputs["yaw"] -= 2*np.pi
			
			inputs["flipped"] = -1

		else:
			inputs["flipped"] = 1
		
		

		delete = ["pts","mask","Rt", "mask_2d","mask2d_img","mask_file","frames","masks","raw_id","yaw"]

		if not self.verbose:
			delete.extend(["centre", "corners", "bound_lines","box_seq"])

		else:
			if "corners" in inputs:
				inputs["centre"] = (inputs["corners"][0,:3]+inputs["corners"][6,:3])/2.

		if "val" in self.data_path:
			delete.append("velocities")			
		for k in delete:
			if k in inputs:
				del inputs[k]

		return inputs