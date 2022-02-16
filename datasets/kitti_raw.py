# Based on F. K. Gustafsson code https://github.com/fregu856/3DOD_thesis

# from datasets.kittiloader import LabelLoader2D3D
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
import os
import skimage.transform
import pickle
import numpy as np
import math
import random
random.seed(1)
import cv2
from PIL import Image, ImageDraw
import pykitti
from torchvision import transforms
from utils import *
from kitti_utils import *
np.set_printoptions(suppress=True)
import glob
from shapely.geometry import box as Sbox
from tqdm import tqdm
import pandas

def _sided_distance(p1, p2):
	p1 = p1.reshape(-1, 1, 3)
	p2 = p2.reshape(1, -1, 3)
	dists = np.abs(p1 - p2)
	vecs = dists.copy()
	dists = np.sum(dists, axis=-1)
	amin = np.argmin(dists,axis=-1)
	vecs = np.stack([vecs[i,j] for i,j in enumerate(amin)])
	dist = np.min(dists, axis=-1)

	return vecs, dist

def read_gt(img_id):
	names = "type,truncated,occluded,alpha,left,top,bottom,right,h,w,l,X,Y,Z,yaw".split(",")
	types = [str,float,float,float,float,float,float,float,float,float,float,float,float,float]
	dtypes = {n:t for n,t in  zip(names, types)}

	l = pandas.read_csv(img_id, sep=" ", header=None, dtype=dtypes,names=names)

	return l[l["type"]=="Car"]


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def cv2_loader(path):
	return cv2.imread(path)
	return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
	# return cv2.cvtColor(np.array(pil_loader(path)), cv2.COLOR_BGR2RGB)



class KittiRawDataset(torch.utils.data.Dataset):
	def __init__(self, data_path, type,opt, prev=[], is_train=False, process=True, sequence=False):
		self.data_path = data_path
		self.raw_path = self.data_path.replace("object","raw")
		self.opt = opt
		self.prev = prev
		self.is_train = is_train
		self.img_dir = data_path + "/training/image_2/"

		self.calib_dir = data_path + "/training/calib/"
		self.image_loader = pil_loader
		self.height = opt.height
		self.width = opt.width
		self.seg_net = opt.seg_net

		if sequence:
			type = "val"
		
		img_ids = np.genfromtxt(f"kitti_eval/cpp/ImageSets/{type}.txt", dtype=str)

		self.mapping = np.array(open(data_path+"/devkit_object/mapping/train_mapping.txt", "r").readlines())
		rand = open(data_path+"/devkit_object/mapping/train_rand.txt", "r").readlines()[0].rstrip("\n").split(",")
		
		rand = np.array(list(map(int,rand)))-1

		self.mapping = self.mapping[rand]

		self.mapping = {i:a.rstrip("\n") for i,a in enumerate(self.mapping) if "{:06}".format(int(i)) in img_ids}


		if not sequence:
			self.examples = []
			

			for img_id in tqdm(img_ids):
				folder, drive, frame = self.mapping[int(img_id)].split()
				frame = int(frame)
				skip = False

				preds = np.load(f"{self.raw_path}/{folder}/{drive}/{self.seg_net}/data/{(int(frame)):0>10}.npz", allow_pickle=True)
				
				if preds["scores"].shape[0] > 0:
					self.examples.append((folder, drive, frame, img_id))
		
		else:
			bad_sequences = [v[1] for v in self.mapping.values()]
			
			files = glob.glob(f"{self.raw_path}/*/*/{self.seg_net}/data/*.npz")
			files_ids = list(map(lambda a: a.split("/"),files))
			file_ids = list(map(lambda a: (a[-5],a[-4],a[-1].split(".")[0]),files_ids))
			file_ids = [f for f in file_ids if f not in bad_sequences]

			file_ids = sorted(file_ids, key=lambda a: (a[1],a[2]) )


			inverse_mapping = {v:k for k,v in self.mapping.items()}


			self.examples = []


			for img_id in tqdm(file_ids):
				folder, drive, frame = img_id
				frame = int(frame)
				skip = False


				if (folder, drive, frame) in file_ids:
					img_id = inverse_mapping[(folder, drive, frame)]
				else:
					img_id=-1
				self.examples.append((folder, drive, frame, img_id))

		
		scene_names = set(map(lambda a: a[1] ,self.examples))
		self.poses = {}
		self.p_data = {}

		self.imu2cams = {}

		for scene in scene_names:

			name_contents = scene.split('_')
			date = name_contents[0] + '_' + name_contents[1] + '_' + name_contents[2]
			drive = name_contents[4]

			p_data_full = pykitti.raw(self.data_path.replace("object","raw"), date, drive, imtype="jpg")

			nimg = len(p_data_full)

			p_data =  pykitti.raw(self.data_path.replace("object","raw"), date, drive, imtype="jpg")
			
			M_imu2cam = p_data.calib.T_cam2_imu

			self.imu2cams[scene] = M_imu2cam

			self.p_data[scene] = p_data

			poses = []

			for i_img in range(nimg): 
				imgname = p_data.cam2_files[i_img].split('/')[-1]
				poses.append(np.matmul(M_imu2cam, np.linalg.inv( p_data.oxts[i_img].T_w_imu) ))

			self.poses[scene] = poses
		self.kitti_dir=self.data_path

	def __getitem__(self, index):
		folder, drive, frame, img_id = self.examples[index]

		inputs = {}
		inputs["raw_id"] = (folder, drive, frame)
		inputs["img_id"] = img_id

		if img_id!=-1:
			gts = read_gt(f"{self.data_path}/training/label_2/{img_id}.txt").reset_index()
			gt_centres = np.stack([gts["X"],gts["Y"], gts["Z"]],-1).reshape(-1,3)
		else:
			gts=[]
		
		pose = self.poses[drive][frame]

		inputs["imu2cam"] = torch.from_numpy(self.imu2cams[drive]).unsqueeze(0)

		segmentation = np.load(f"{self.raw_path}/{folder}/{drive}/{self.seg_net}/data/{(int(frame)):0>10}.npz", allow_pickle=True)


		inputs["3d_pts"] = [[] for k in range(len(segmentation["boxes"]))]
		inputs["3d_mask"] = [[] for k in range(len(segmentation["boxes"]))]
		inputs["2d_mask"] = [[] for k in range(len(segmentation["boxes"]))]
		inputs["2d_mask_img"] = [[] for k in range(len(segmentation["boxes"]))]
		inputs["bound_lines"] = [[] for k in range(len(segmentation["boxes"]))]
		inputs["box"] = [[] for k in range(len(segmentation["boxes"]))]
		inputs["scores"] = [[] for k in range(len(segmentation["boxes"]))]

		inputs["lidar_median"] = [[] for k in range(len(segmentation["boxes"]))]

		inputs["velocities"] = [[] for k in range(len(segmentation["boxes"]))]

		inputs["yaw"] = [-100 for k in range(len(segmentation["boxes"]))]

		done = []

		inputs["Rts"] = []


		for i in self.prev:
			if not os.path.exists(f"{self.raw_path}/{folder}/{drive}/image_02/data/{(int(frame)+i):0>10}.jpg") or not os.path.exists(f"{self.raw_path}/{folder}/{drive}/velodyne_points/data/{(int(frame)+i):0>10}.bin"):
				return None

			velo, velo_pts_im, P_rect = generate_pcds(self.raw_path, folder, drive, int(frame)+i)
			inputs[("K",-1)] = P_rect


			segmentation = np.load(f"{self.raw_path}/{folder}/{drive}/{self.seg_net}/data/{(frame+i):0>10}.npz", allow_pickle=True)

			used_predictions = []

			inputs["Rts"].append(pose.dot(np.linalg.inv(self.poses[drive][frame+i])))


			areas2d = np.array([(a[2]-a[0])*(a[3]-a[1]) for ind, a in enumerate(segmentation["boxes"])]).argsort()[::-1]
			
			rel_rt = self.poses[drive][frame].dot(np.linalg.inv(self.poses[drive][frame+i]))

			instances = {}
			cham_inliers = np.zeros((len(inputs["lidar_median"]), len(segmentation["boxes"])))
			distances = np.zeros((len(inputs["lidar_median"]), len(segmentation["boxes"]))) + np.inf

			velocities = np.zeros((len(inputs["lidar_median"]), len(segmentation["boxes"]),3))

			bigger_cars = []

			for k in areas2d:
				xmin,ymin,xmax,ymax = segmentation["boxes"][k]

				box = np.array([xmin,ymin,xmax,ymax])
				scores = segmentation["scores"][k]

				box_fov_inds = (velo_pts_im[:,0]<xmax) & \
								(velo_pts_im[:,0]>=xmin) & \
								(velo_pts_im[:,1]<ymax) & \
								(velo_pts_im[:,1]>=ymin)
				
				pc_in_box_fov = velo[box_fov_inds]

				inst = segmentation["instances"][k]
				
				
				inst_mask = inst[velo_pts_im[box_fov_inds,1].astype(np.int), velo_pts_im[box_fov_inds,0].astype(np.int)].copy()
				
				inst = inst.astype(np.int8)
				inst *= 2
				inst[np.where(bigger_cars==1)] = 1

				bigger_cars.append(k)
				
				if inst_mask.sum()==0 and i==0:
					inputs["lidar_median"][k].append(np.zeros((3))-1000)
					inputs["box"][k].append(np.zeros(4))
					inputs["2d_mask_img"][k].append(bigger_cars.copy())
					instances[k] =  () 
					
					continue

				if i==0:
					inputs["box"][k].append(np.array([xmin,ymin,xmax,ymax]))
					inputs["scores"][k].append(segmentation["scores"][k])
					inputs["2d_mask"][k].append(inst_mask)
					inputs["2d_mask_img"][k].append(bigger_cars.copy())
					lidar_median = np.median(pc_in_box_fov[inst_mask],0)[:3]
					inputs["lidar_median"][k].append(lidar_median)

					inputs["3d_pts"][k].append(pc_in_box_fov)

					if len(gts)>0:
						dist = np.linalg.norm(gt_centres - lidar_median, axis=1)
						amin = np.argmin(dist)
						if np.min(dist)<5:
							inputs["yaw"][k] = gts["yaw"][amin]

				else:
					if inst_mask.sum()>10:
						lidar_median = np.median(pc_in_box_fov[inst_mask], 0)
					else:
						instances[k] = ()
						continue

					lidar_median[-1] = 1.0

					locs = np.expand_dims(np.linalg.pinv(inputs["Rts"][-1]).dot(lidar_median),0)

					locs = np.expand_dims(rel_rt.dot(lidar_median),0)
					lm0 = np.stack([a[-1] for a in inputs["lidar_median"]])
					

					d = np.linalg.norm(lm0[:,:3]-locs[:,:3], axis=1)
					velocities[:,k] = lm0[:,:3]-locs[:,:3]
					distances[:,k] = d

					instances[k] =  (np.array([xmin,ymin,xmax,ymax]), segmentation["scores"][k], inst_mask, bigger_cars.copy(), pc_in_box_fov.copy(), locs[0,:3])



			if i!=0:
				for _ in range(min(distances.shape)):
					if np.all(distances==np.inf):
						break

					l,m = np.unravel_index(distances.argmin(), distances.shape)

					
					if  distances.min()<2 and l not in done:
						box, scores, inst_mask, seg2d, pc_in_box_fov, lm = instances[m]

						if inst_mask.sum()<inputs["2d_mask"][l][0].sum()*0.5:
							distances[:,m] = np.inf
							continue

						inputs["box"][l].append(box)
						inputs["scores"][l].append(scores)
						inputs["2d_mask"][l].append(inst_mask)
						inputs["2d_mask_img"][l].append(seg2d)
						inputs["3d_pts"][l].append(pc_in_box_fov)
						inputs["lidar_median"][l].append(lm)
						inputs["velocities"][l].append(velocities[l,m])
					else:
						done.append(l)
					

					distances[l] = np.inf
					distances[:,m] = np.inf
					cham_inliers[l] = 0.
					cham_inliers[:,m] = 0.
			
			for k in range(len(inputs["box"])):
				if len(inputs["box"][k])<self.prev.index(i)+1:
					done.append(k)
		return inputs





	def __len__(self):
		return len(self.examples)



