import torch
import torch.nn as nn
import numpy as np
from models.frusum_pointnet import FrustumPointNetv1, PointNetInstanceSeg, PointNetEstimation, point_cloud_masking, STNxyz
from kaolin.render.mesh import dibr_rasterization
from kaolin.render.mesh.utils import *
from kaolin.render.camera import generate_perspective_projection
from torchvision import models
from torch.cuda.amp import autocast
from utils import *
import cv2

torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
import random
random.seed(0)

import numpy as np
np.random.seed(0)


class Model(nn.Module):
	def __init__(self,
				frame_ids=range(1),
				yaw="bins",
				yaw_bins=16,
				threshold=0.9,
				mpts=202,
				sizeGt=False,
				cham=False,
				stn=False,
				nomaskcut=False,
				loss_fn="l1",
				box_loss=False,
				render_mask_loss=False,
				offsets=0):
		
		super(Model, self).__init__()

		if not cham:
			self.seg = PointNetInstanceSeg(sigma=not cham)
		self.yaw_bins = yaw_bins
		if yaw in ["img","img_app", "vec","bins"]:
			self.est = [PointNetEstimation(ret_feats = True)]
			if yaw=="img_app":
				self.yaw_net = Img_Yaw(apparent=True,yaw_bins=yaw_bins)


			elif yaw=="img":
				self.yaw_net = Img_Yaw(yaw_bins=yaw_bins)

			elif yaw=="vec":
				self.yaw_net = Vec_Yaw(yaw_bins=2)
			elif yaw=="bins":
				self.yaw_net = Vec_Yaw(yaw_bins=yaw_bins)
		else:
			self.est = []
			for b in yaw:
				self.est.append(PointNetEstimation(False))
		
		self.est = nn.ModuleList(self.est)

		self.cham = cham
		self.sigma = not self.cham
		self.threshold = threshold
		self.box_loss=box_loss
		self.render_mask_loss = render_mask_loss

		self.nomaskcut = nomaskcut

		self.frame_ids = frame_ids if len(frame_ids) > 0 else [0]
		self.yaw = yaw
		self.sizeGT = sizeGt
		self.mpts = mpts
		self.loss_fn = loss_fn
		self.offsets = offsets

		vertices, self.faces = get_car_model(True)

		if self.mpts > 202:
			self.vertices = extra_pts_model(self.faces, self.mpts-202, vertices)
		else:
			self.vertices = vertices[np.random.choice(vertices.shape[0], self.mpts)]
		
		self.stn = stn
		if self.stn:
			self.STN = STNxyz(False)
	

	def forward(self, inputs, val=False):

		losses = {}
		outputs = {}

		single = False

		for frame in self.frame_ids:
			if ("pts",frame) not in inputs:
				single = True
				break
			pts = inputs[("pts",frame)]

			if not self.cham:
				outputs[("sigmas",frame)] = self.seg(inputs[("pts", frame)])
			
			if not self.nomaskcut and self.sigma:
				object_pts_xyz, mask_xyz_mean, mask = point_cloud_masking(pts, outputs[("sigmas",frame)])
		
			else:
				mask_xyz_mean = torch.median(pts[:,:3], dim=-1, keepdim=True).values
				object_pts_xyz = pts[:,:3] - mask_xyz_mean.repeat(1,1,pts.shape[-1])
				mask_xyz_mean = mask_xyz_mean.squeeze(-1)


			if self.stn:
				delta = self.STN(object_pts_xyz).reshape(-1,3)
				mask_xyz_mean += delta
				object_pts_xyz_new = object_pts_xyz - delta.view(delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])
			else:
				object_pts_xyz_new = object_pts_xyz


			pts = pts.transpose(2, 1)
			bins = []

			if self.yaw in ["img","img_app"]:
				centre,feats = self.est[0](object_pts_xyz_new)
				centre += mask_xyz_mean
				outputs[("centre",frame,0)] = centre

				yaw = self.yaw_net(inputs[("crop",frame)], feats)
				if self.yaw_bins!=-1:
					bin_size = 2*3.14/self.yaw_bins
					outputs[("yaw",frame,0)] = yaw.argmax(-1)*bin_size-3.14+bin_size/2

				else:
					outputs[("yaw",frame,0)] = yaw

				if self.yaw=="img_app":
					outputs[("yaw",frame,0)] += torch.arctan(centre[:,0]/(centre[:,2]+1e-7))
					outputs[("yaw",frame,0)][outputs[("yaw",frame,0)]>3.14] -= 2*3.14
					outputs[("yaw",frame,0)][outputs[("yaw",frame,0)]<-3.14] += 2*3.14
				
				bins.append(0)
			


			elif self.yaw == "vec":
				centre,feats = self.est[0](object_pts_xyz_new)
				centre += mask_xyz_mean
				outputs[("centre",frame,0)] = centre

				yaw = self.yaw_net(feats)
				yaw = yaw[:,:2]/torch.norm(yaw[:,:2], dim=1, keepdim=True)
				outputs[("yaw",frame)] = yaw
			
				if not val:
					bin_size = 2*np.pi/self.yaw_bins

					for b,offset in enumerate(np.arange(-np.pi,np.pi,bin_size)):
						outputs[("yaw",frame,b)] = torch.zeros_like(yaw[:,0])+offset + bin_size/2

						outputs[("centre",frame,b)] = centre
						bins.append(b)
				
				else:
					outputs[("yaw", frame,0)] = torch.atan2(yaw[:,0], yaw[:,1])
					bins.append(0)
			
			elif self.yaw=="bins":
				centre,feats = self.est[0](object_pts_xyz_new)
				centre += mask_xyz_mean
				outputs[("centre",frame,0)] = centre

				yaw = self.yaw_net(feats)

				bin_size = 2*3.14/self.yaw_bins
				outputs[("yaw",frame)] = yaw.argmax(-1)*bin_size+bin_size/2 - 3.14
			
				if not val:
					bin_size = 2*np.pi/self.yaw_bins

					for b,offset in enumerate(np.arange(-np.pi,np.pi,bin_size)):
						outputs[("yaw",frame,b)] = torch.zeros_like(yaw[:,0])+offset + bin_size/2

						outputs[("centre",frame,b)] = centre
						bins.append(b)
				else:
					outputs[("yaw",frame,0)] = outputs[("yaw",frame)]

					outputs[("centre",frame,0)] = centre
					bins.append(0)

			else:
				for rt_net in range(len(self.est)):
					centre, yaw = self.est[rt_net](object_pts_xyz_new)
					centre += mask_xyz_mean
					outputs[("centre",frame,rt_net)] = centre
					
					bin_size = 2*3.14 / len(self.yaw)

					outputs[("yaw",frame,rt_net)] = rt_net*bin_size + bin_size*F.sigmoid(yaw[:,0])-bin_size/2

					bins.append(rt_net)
				


			outputs[("pts",frame)] = pts.clone()


			bin_losses = []
			bin_model_losses = []


			for b in bins:
				vertices = self.vertices.clone().to(pts.device)
			
				if self.sizeGT:
					size = inputs["size"]
				else:
					size = None


				models = transform_model(vertices, size, outputs[("centre", frame, b)], outputs[("yaw", frame, b)])


				outputs[("models", frame, b)] = models.clone()

				distances, inds = sided_chamfer(pts, models, self.loss_fn)
					
				outputs[("inds",frame,b)] = inds

				if self.cham:
					d = distances
					bin_losses.append(d)
				
				else:
					sigma = (outputs[("sigmas",frame)].squeeze(-1) + 1e-7)
					d = (distances / sigma) + torch.log(sigma)
					if self.threshold>0.0:
						d = d + distances*(distances<self.threshold)

					bin_losses.append(d)
				

			bin_losses = torch.stack(bin_losses,1)


			amin = bin_losses.mean(-1).argmin(1)

			outputs[("models",frame)] = torch.stack([outputs[("models", frame, j.item())][i]for i,j in enumerate(amin)])
			
			outputs[("centre", frame)] = torch.stack([outputs[("centre", frame, j.item())][i] for i,j in enumerate(amin)])
			

			
			outputs[("inds", frame)] = torch.stack([outputs[("inds", frame, j.item())][i] for i,j in enumerate(amin)])

			losses[("chamfer", frame)] = torch.stack([bin_losses[i,j] for i,j in enumerate(amin)])
			


			if self.yaw == "vec":
				picked_yaw = torch.stack([outputs[("yaw", frame, j.item())][i] for i,j in enumerate(amin)])
				picked_yaw = torch.stack([torch.cos(picked_yaw), torch.sin(picked_yaw)],1)
				losses[("yaw",frame)] = F.mse_loss(yaw,picked_yaw)


				outputs[("yaw", frame)] = torch.atan2(yaw[:,0], yaw[:,1])

			elif self.yaw == "bins":
				losses[("yaw",frame)] = nn.CrossEntropyLoss()(yaw, amin)




			centre = torch.cat((outputs[("centre",frame)],torch.ones_like(outputs[("centre", frame)][:,[0]])), 1)

			Rts = inputs[("Rt",frame)].clone()

			outputs[("trans_centres",frame)] = Rts.bmm(centre.unsqueeze(-1))

			frontRT = make_Rt(outputs[("yaw", frame)], centre[:,:3])


			front_pt = torch.zeros_like(centre)
			front_pt[:,2] = 3.89/2
			front_pt[:,3] = 1

			front_pt = frontRT.bmm(front_pt.unsqueeze(-1))
			
			outputs[("front",frame)] = front_pt.clone()
			outputs[("trans_front",frame)] = Rts.bmm(front_pt)
		


		if len(self.frame_ids)>1 and not single:
			warped_centre = torch.stack([outputs[("trans_centres", frame)] for frame in self.frame_ids],1)

			warped_centre = (warped_centre[:,[0]] - warped_centre)[:,1:,:3,0]
			if "velocities" in inputs:
				motion = inputs["velocities"]

				warped_centre = warped_centre + motion


			losses["centre"] = torch.norm(warped_centre, dim=-1).mean(-1).mean(-1)

			warped_front = torch.stack([outputs[("front", frame)] for frame in self.frame_ids],1)

			warped_front = (warped_front[:,[0]] - warped_front)[:,1:,:3,0]
			if "velocities" in inputs:
				warped_front = warped_front + motion
			
			losses["front"] = torch.norm(warped_front, dim=-1).mean(-1).mean(-1)

		
		return outputs, losses





class Img_Yaw(nn.Module):
	def __init__(self, apparent=False, yaw_bins=-1):
		super(Img_Yaw, self).__init__()
		self.apparent = apparent
		self.features = models.vgg16_bn(pretrained=True).features
		self.yaw_bins = yaw_bins
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		
		f = 1024 if not apparent else 512
		self.linear = nn.Sequential(
			nn.Linear(f, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.Dropout(),
		
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.Dropout(),
		
			nn.Linear(512, self.yaw_bins if self.yaw_bins!=-1 else 2),
		)

	def forward(self, x, feats):
		x = self.features(x)
		
		x = self.avgpool(x)
		
		x = torch.flatten(x, 1)
		if not self.apparent:
			x = torch.cat((x, feats),1)
		
		x = self.linear(x)

		
		return x

class Vec_Yaw(nn.Module):
	def __init__(self, yaw_bins=-1):
		super(Vec_Yaw, self).__init__()
		self.yaw_bins = yaw_bins
		self.linear = nn.Sequential(
			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(512, self.yaw_bins if self.yaw_bins!=-1 else 2),
		)

	def forward(self,feats):
		x = self.linear(feats)

		if self.yaw_bins==-1:
			v = x[:,:2]/torch.norm(x[:,:2], dim=1, keepdim=True)
			yaw =  torch.atan2(v[:,0], v[:,1])
			return yaw
		
		return x