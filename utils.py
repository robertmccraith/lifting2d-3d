import torch
import torch.nn as nn
import numpy as np
import socket
hn = socket.gethostname()
# print(hn)

from kaolin.metrics.pointcloud import sided_distance

from tqdm import tqdm
import shutil
from shutil import copyfile
import pathlib
import glob, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas
import subprocess
from collections import defaultdict

from datasets.kitti_utils import compute_box_3d

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def calc_AP(vals):
	s = np.mean(vals[1:],0)
	# s = np.mean(vals[0,:-1:4])
	return s


def write2kittiObjFile(pred_dir, img_ids, box2d, XYZ, yaw, size, scores, npts):
	
	pathlib.Path(f"{pred_dir}").mkdir(parents=True, exist_ok=True)
	
	g = glob.glob(f"{pred_dir}/*.txt")+glob.glob(f"{pred_dir}/plot/*.txt")

	for f in g:
		os.remove(f)
	

	for i in range(len(img_ids)):


		f = f"{pred_dir}/{int(img_ids[i]):06}.txt"
		
		with open(f, "a+") as output:

			line = ["Car", "-1", "-1", "-1"]
			line.extend([f"{a:.2f}" for a in box2d[i]])
			line.extend([f"{a:.2f}" for a in size[i]])
			line.extend([f"{a:.2f}" for a in XYZ[i]])
			line.append(f"{yaw[i]:.2f}")
			line.append(f"{scores[i]:.5f}")
			
			print(" ".join(line), file=output)



def evaluate(kitti_path, pred_dir, split="val", threshold=0.7):

	string= f"cd kitti_eval/cpp/; ./evaluate_{split}{threshold} "+kitti_path+"/training/ "+pred_dir
	# print(string)
	result=subprocess.getoutput(string)
	# print(result)
	my_data = np.genfromtxt(pred_dir+"/plot/car_detection.txt", delimiter=' ')
	bb = calc_AP(my_data)[1:]

	my_data = np.genfromtxt(pred_dir+"/plot/car_detection_ground.txt", delimiter=' ')
	bev = calc_AP(my_data)[1:]


	my_data = np.genfromtxt(pred_dir+"/plot/car_detection_3d.txt", delimiter=' ')
	bb3d = calc_AP(my_data)[1:]

	return bb, bev, bb3d



def plot_pcds(pcds, models, logits, centre, yaw,front,next, size,gt, writer,i,frame):
	centre = centre.detach().cpu().numpy()
	yaw = yaw.detach().cpu().numpy()
	front = front.detach().cpu().numpy()
	if next is not None: 
		next = next.detach().cpu().numpy()

	gt = gt.detach().cpu().numpy() if gt is not None else None


	for k in range(min(4, pcds.shape[0])):
		pts = pcds[k][...,:3]
		mesh = models[k]
		
		colours = torch.ones((pcds.shape[1] + mesh.shape[0], 3), dtype=torch.int) * 255

		colours[:, :2] = 0

		mask = logits.detach()[k].cpu()
		
		ma = mask.max()
		mi = mask.min()

		per = ((mask-mi) / (ma-mi)).unsqueeze(-1).repeat(1,3)

		red = torch.as_tensor([255, 0, 0], dtype=torch.int).unsqueeze(0).repeat(pts.shape[0],1)
		green = torch.as_tensor([0, 255,0], dtype=torch.int).unsqueeze(0).repeat(pts.shape[0],1)
		

		colours[:pts.shape[0]] = red

		colours[mask+pts.shape[0]] = green


		verts = torch.cat((pts, mesh))

		point_size_config = {
			'material': {
				'cls': 'PointsMaterial',
				'size': 0.03
			}
		}

		verts[:, 1] *= -1

		writer.add_mesh(f"3D/{frame}/{k}",
						config_dict=point_size_config,
						vertices=verts.unsqueeze(0),
						colors=colours.unsqueeze(0),
						global_step=i)




		fig = plt.figure()
		
		plt.scatter(*pcds.transpose(-1,-2)[k,[0,2]].detach().cpu().numpy())


		if gt is not None:
			plt.plot(gt[k,:,0],gt[k,:,2], c="green")
		


		pred = convert2corners_np(*centre[k,[0,2]], yaw[k], *size[k,[0,2]])

		plt.plot(pred[:,0], pred[:,1], c="red")


		# X,Y = gt[k,:,[0,2]].mean(1)
		X,Y = centre[k,[0,2]]
		plt.axis((X-10,X+10,Y-10,Y+10))
		
		if next is not None:
			plt.plot([centre[k,0], next[k,0]], [centre[k,2], next[k,2]], c="green")
		
		plt.plot([centre[k,0], front[k,0]], [centre[k,2], front[k,2]], c="red")
		

		writer.add_figure(f"BEV Plot/{frame}/{k}", fig, i)

	writer.flush()


def make_Rt(yaw, centre):
	Rt = torch.eye(4, device=yaw.device).unsqueeze(0).repeat(len(yaw), 1, 1)
	Rt[:, :3, 3] = centre


	angle = yaw + 1.57
	ca = torch.cos(angle)
	sa = torch.sin(angle)
	Rt[:, 0, 0] = ca
	Rt[:, 2, 2] = ca
	Rt[:, 0, 2] = sa
	Rt[:, 2, 0] = -sa

	return Rt

def get_car_model(tensor=False, mean_size=True):
	faces = np.load("mean-car/mesh.faces.npy")

	vertices = np.load("mean-car/mesh.vertices.npy")[:, [2,1,0]]

	vertices[:,2] -= 2.1739-(2.1739+2.0873)/2.
	vertices[:,0] -= 0.8541-(0.8541+0.8519)/2.
	vertices[:,1] -= 1.0505-(1.0505+0.4982)/2.

	height = vertices[:, 1].max() - vertices[:, 1].min()
	vertices[:, 1] += height / 2.

	vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)


	l = vertices[:, 2].max() - vertices[:, 2].min()
	w = vertices[:, 0].max() - vertices[:, 0].min()
	h = vertices[:, 1].max() - vertices[:, 1].min()

	size = [1.52986348, 1.61876715, 3.89206519][::-1]

	vertices[:, 2] = vertices[:, 2] * (size[0] / l)
	vertices[:, 0] = vertices[:, 0] * (size[1] / w)
	vertices[:, 1] = vertices[:, 1] * (size[2] / h)



	if tensor:
		return torch.from_numpy(vertices).float(), torch.from_numpy(faces).long()
	
	return vertices, faces


def extra_pts_model(faces, mpts, vertices):
	extra_pts_faces = faces[np.random.choice(faces.shape[0], mpts)]

	rs = torch.rand((mpts,2)).to(faces.device)


	
	a = 1-torch.sqrt(rs[:,0])
	b = torch.sqrt(rs[:,0])*(1-rs[:,1])
	c = torch.sqrt(rs[:,0])*rs[:,1]
	
	A = vertices[extra_pts_faces[:,0]] * a.unsqueeze(-1).repeat(1,4)
	B = vertices[extra_pts_faces[:,1]] * b.unsqueeze(-1).repeat(1,4)
	C = vertices[extra_pts_faces[:,2]] * c.unsqueeze(-1).repeat(1,4)

	extra_vertices = A + B + C

	vertices = torch.cat((vertices,extra_vertices),0)

	return vertices


def transform_model(vertices, size, centre=None, yaw=None):
	#size = l,w,h

	model = vertices.clone().float().unsqueeze(0).repeat(len(centre), 1, 1)
	model[:, :, -1] = 1.

	if size is not None:
		l = model[0, :, 2].max() - model[0, :, 2].min()
		w = model[0, :, 0].max() - model[0, :, 0].min()
		h = model[0, :, 1].max() - model[0, :, 1].min()


		models = torch.ones_like(model)

		for i in range(len(model)):

			models[i, :, 2] = model[i, :, 2] * (size[i, 0] / l)
			models[i, :, 0] = model[i, :, 0] * (size[i, 1] / w)
			models[i, :, 1] = model[i, :, 1] * (size[i, 2] / h)
	

	Rt = make_Rt(yaw, centre)

	model = Rt.bmm(model.transpose(2,1))

	model = model.transpose(2,1)[:,:,:3]

	return model




def erase_copy(output_dir):
	try:
		shutil.rmtree(f"{output_dir}")
	except OSError as e:
		print("Failed with:", e.strerror)

	pyfiles = glob.glob("**/*.py", recursive=True)
	pathlib.Path(f"{output_dir}/weights/").mkdir(parents=True, exist_ok=True)
	for p in pyfiles:
		if f"{output_dir}/boards" in p: continue

		pathlib.Path(f"{output_dir}/code/" + "/".join(p.split("/")[:-1])).mkdir(parents=True,exist_ok=True)

		copyfile(p, f"{output_dir}/code/" + p)


def sided_chamfer(pts, models, loss_fn="Seuclidean"):
	if loss_fn in ["euclidean","Seuclidean"]:
		distances,inds = sided_distance(pts[...,:3], models[...,:3])

		if loss_fn=="Seuclidean":
			distances = torch.sqrt(distances)
	
	else:
		pts = pts.unsqueeze(-2)[...,:3]
		models = models.unsqueeze(1).repeat(1,pts.shape[1],1,1)[...,:3]

		if loss_fn == "l1":
			l = F.l1_loss(pts, models, reduction="none").sum(-1)
		
		elif loss_fn == "sl1":
			l = F.smooth_l1_loss(pts, models, reduction="none").sum(-1)
		
		elif loss_fn == "l2":
			l = F.mse_loss(pts, models, reduction="none").sum(-1)

		distances, inds = torch.min(l,-1)

	return distances, inds



def convert2corners(centre, yaw,l=3.89,w=1.61):

	corners = torch.cuda.FloatTensor([
			[l/2.0, 0, w/2.0],
			[-l/2.0, 0, w/2.0],
			[-l/2.0, 0, -w/2.0],
			[l/2.0, 0, -w/2.0]
		])



	Rmat = torch.eye(3).repeat(yaw.shape[0], 1,1).to(centre.device)
	Rmat[:,0,0] = torch.cos(yaw)
	Rmat[:,-1,-1] = torch.cos(yaw)

	Rmat[:,0,-1] = torch.sin(yaw)
	Rmat[:,-1,0] = -torch.sin(yaw)



	cars = []

	for i in range(Rmat.shape[0]):

		car = []
		for c in corners:
			car.append(torch.matmul(Rmat[i], c))
		
		car = torch.stack(car)[:,[0,2]]

		car+=centre[i]

		cars.append(car)

	
	return torch.stack(cars)


def convert2corners_np(x,z,yaw,l,w):

	centre = np.array([x,z])


	
	Rmat = np.eye(3)
	Rmat[0,0] = np.cos(yaw)
	Rmat[-1,-1] = np.cos(yaw)

	Rmat[0,-1] = np.sin(yaw)
	Rmat[-1,0] = -np.sin(yaw)

	corners = np.array([
			[l/2.0, 0, w/2.0],
			[-l/2.0, 0, w/2.0],
			[-l/2.0, 0, -w/2.0],
			[l/2.0, 0, -w/2.0]
	])

	cars = []


	car = []
	for c in corners:
		car.append(np.matmul(Rmat, c))


	car = np.stack(car)[:,[0,2]] + centre

	return car



def get_cat(gt,v=False):
	cat,truncated,occluded,alpha,left,top,right,bottom,h,w,l,X,Y,Z,ry = gt
	diff=-1
	
	if cat=="Car" and truncated<0.15 and occluded==0 and abs(top-bottom)>40:
		diff = 0
	elif cat=="Car" and truncated<0.3 and occluded==1 and abs(top-bottom)>25:
		diff=1
	elif cat=="Car" and truncated<0.5 and occluded==2 and abs(top-bottom)>25:
		diff=2
	
	return diff



def read_labels(dir, dtypes, names, files=None):

	if files is None:
		print(dir+"*.txt")
		gs = glob.glob(dir+"*.txt")
		gs = sorted(gs,key=lambda a: int(a.split("/")[-1].split(".")[0]))
	else:
		gs = [dir+g+".txt" for g in files]

	labels = defaultdict(list)

	for g in tqdm(gs):
		if not os.path.exists(g):continue

		l = pandas.read_csv(g, sep=" ", header=None, dtype=dtypes,names=names)

		
		if "label_2" in g:
			labels[g.split("/")[-1].split(".")[0]] = np.array(l)
		else:
			g = g.split("/")
			g = (g[-3],g[-2],g[-1].split(".")[0])
			labels[g] = np.array(l)
	
	if files is None:
		return list(labels.keys()), labels
	
	return labels