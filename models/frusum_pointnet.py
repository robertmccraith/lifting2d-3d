import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init


class PointNetInstanceSeg(nn.Module):
	def __init__(self,reflectance=False, sigma=False):
		super(PointNetInstanceSeg, self).__init__()
		self.conv1 = nn.Conv1d(4, 64, 1)
		self.conv2 = nn.Conv1d(64, 64, 1)
		self.conv3 = nn.Conv1d(64, 64, 1)
		self.conv4 = nn.Conv1d(64, 128, 1)
		self.conv5 = nn.Conv1d(128, 1024, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)


		self.dconv1 = nn.Conv1d(1024+64, 512, 1)
		self.dconv2 = nn.Conv1d(512, 256, 1)
		self.dconv3 = nn.Conv1d(256, 128, 1)
		self.dconv4 = nn.Conv1d(128, 128, 1)
		# self.dropout = nn.Dropout(0)

		if sigma:
			self.dconv5 = nn.Conv1d(128,1,1)
			self.out = nn.Softplus()
		else:
			self.dconv5 = nn.Conv1d(128, 2, 1)
			self.out = lambda a: a

		self.dbn1 = nn.BatchNorm1d(512)
		self.dbn2 = nn.BatchNorm1d(256)
		self.dbn3 = nn.BatchNorm1d(128)
		self.dbn4 = nn.BatchNorm1d(128)

	# @autocast()
	def forward(self, pts): # bs,4,n
		'''
		:param pts: [bs,4,n]: x,y,z,intensity
		:return: logits: [bs,n,2],scores for bkg/clutter and object
		'''

		out1 = F.relu(self.bn1(self.conv1(pts))) 
		out2 = F.relu(self.bn2(self.conv2(out1))) 
		out3 = F.relu(self.bn3(self.conv3(out2))) 
		out4 = F.relu(self.bn4(self.conv4(out3)))
		out5 = F.relu(self.bn5(self.conv5(out4)))

		out5 = torch.max(out5, 2, keepdim=True)[0].view(pts.shape[0], -1,1).repeat(1,1,pts.shape[-1])

		out = torch.cat((out2, out5),1)


		x = F.relu(self.dbn1(self.dconv1(out)))
		x = F.relu(self.dbn2(self.dconv2(x)))
		x = F.relu(self.dbn3(self.dconv3(x)))
		x = F.relu(self.dbn4(self.dconv4(x)))
		# x = self.dropout(x)
		x = self.dconv5(x)

		seg_pred = x.transpose(2,1).contiguous()
		return self.out(seg_pred)


class PointNetEstimation(nn.Module):
	def __init__(self, ret_feats=False):
		super(PointNetEstimation, self).__init__()
		self.conv1 = nn.Conv1d(3, 128, 1)
		self.conv2 = nn.Conv1d(128, 128, 1)
		self.conv3 = nn.Conv1d(128, 256, 1)
		self.conv4 = nn.Conv1d(256, 512, 1)
		self.bn1 = nn.BatchNorm1d(128)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(256)
		self.bn4 = nn.BatchNorm1d(512)

		self.centre_fc1 = nn.Linear(512, 512)
		self.centre_fc2 = nn.Linear(512, 256)
		self.centre_fc3 = nn.Linear(256,3)
		self.centre_fcbn1 = nn.BatchNorm1d(512)
		self.centre_fcbn2 = nn.BatchNorm1d(256)


		self.yaw_fc1 = nn.Linear(512, 512)
		self.yaw_fc2 = nn.Linear(512, 256)
		self.yaw_fc3 = nn.Linear(256,2)
		self.yaw_fcbn1 = nn.BatchNorm1d(512)
		self.yaw_fcbn2 = nn.BatchNorm1d(256)

		self.ret_feats = ret_feats

	# @autocast()
	def forward(self, pts): # bs,3,m
		'''
		:param pts: [bs,3,m]: x,y,z after InstanceSeg
		:return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
			including box centers, heading bin class scores and residual,
			and size cluster scores and residual
		'''
		bs = pts.size()[0]
		n_pts = pts.size()[2]

		out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
		out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
		out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
		out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
		global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

		x = F.relu(self.centre_fcbn1(self.centre_fc1(global_feat)))#bs,512
		x = F.relu(self.centre_fcbn2(self.centre_fc2(x)))  # bs,256
		xyz = self.centre_fc3(x)

		yaw = F.relu(self.yaw_fcbn1(self.yaw_fc1(global_feat)))#bs,512
		yaw = F.relu(self.yaw_fcbn2(self.yaw_fc2(yaw)))  # bs,256
		yaw = self.yaw_fc3(yaw)

		if self.ret_feats:
			return xyz, global_feat
		return xyz, yaw





class STNxyz(nn.Module):
	def __init__(self,n_classes=3):
		super(STNxyz, self).__init__()
		self.conv1 = torch.nn.Conv1d(3, 128, 1)
		self.conv2 = torch.nn.Conv1d(128, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 256, 1)
		#self.conv4 = torch.nn.Conv1d(256, 512, 1)
		self.fc1 = nn.Linear(256, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 3)

		init.zeros_(self.fc3.weight)
		init.zeros_(self.fc3.bias)

		self.bn1 = nn.BatchNorm1d(128)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(256)
		self.fcbn1 = nn.BatchNorm1d(256)
		self.fcbn2 = nn.BatchNorm1d(128)
	
	# @autocast()
	def forward(self, pts):
		bs = pts.shape[0]
		x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
		x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
		x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
		x = torch.max(x, 2)[0]# bs,256


		x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
		x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
		x = self.fc3(x)# bs,
		return x





class FrustumPointNetv1(nn.Module):
	def __init__(self, reflectance=False, sigma=False, nostn=False, nomaskcut=False):
		super(FrustumPointNetv1, self).__init__()
		self.sigma = sigma
		self.nostn = nostn
		self.nomaskcut = nomaskcut
		
		self.est = PointNetEstimation(False)
	

	def forward(self, point_cloud, segmentation):
		outputs = {}

		# 3D Instance Segmentation PointNet
		# if self.sigma:
			# sigmas = self.InsSeg(point_cloud)
			# outputs["sigmas"] = sigmas

		if not self.nomaskcut and self.sigma:
			object_pts_xyz, mask_xyz_mean, mask = point_cloud_masking(point_cloud, segmentation)
		
		else:
			mask_xyz_mean = torch.median(point_cloud[:,:3], dim=-1, keepdim=True).values
			object_pts_xyz = point_cloud[:,:3] - mask_xyz_mean.repeat(1,1,point_cloud.shape[-1])
			mask_xyz_mean = mask_xyz_mean.squeeze(-1)

		
		stage1_centre = mask_xyz_mean
		object_pts_xyz_new = object_pts_xyz

		outputs["stage1_centre"] = stage1_centre

		centre, yaw = self.est(object_pts_xyz_new)

		centre += stage1_centre
		outputs["centre"] = centre
		outputs["yaw"] = yaw

		return outputs
	
	def update_bn_momentum(self, i, epochs):
		for subnetwork in self.children():
			for layer in subnetwork.children():
				if type(layer) == torch.nn.BatchNorm1d:
					layer.momentum = max(0.5 - 0.49* (float(i)/float(epochs)), 0.01)
	

def point_cloud_masking(pts, logits, xyz_only=True):
	'''
	:param pts: bs,c,n in frustum
	:param logits: bs,n,2
	:param xyz_only: bool
	:return:
	'''

	bs = pts.shape[0]
	n_pts = pts.shape[2]
	# Binary Classification for each point
	if logits.shape[-1] == 1:
		mask = []
		for l in logits:
			l = l.squeeze()
			# med = torch.median(l)
			med = torch.sort(l)[0][int(l.shape[0]*0.5)]
			mask.append(l<med)
		mask = torch.stack(mask)

	else:
		mask = logits[:, :, 0] < logits[:, :, 1]

	mask = mask.unsqueeze(1).float()

	mask_count = mask.sum(2,keepdim=True).repeat(1, 3, 1) 
	pts_xyz = pts[:, :3, :] 


	mask_xyz_mean = (mask.repeat(1, 3, 1) * pts_xyz).sum(2,keepdim=True)
	mask_xyz_mean = mask_xyz_mean / torch.clamp(mask_count,min=1)
	mask = mask.squeeze(1)
	pts_xyz_stage1 = pts_xyz - mask_xyz_mean.repeat(1, 1, n_pts)

	if xyz_only:
		pts_stage1 = pts_xyz_stage1
	else:
		pts_features = pts[:, 3:, :]
		pts_stage1 = torch.cat([pts_xyz_stage1, pts_features], dim=-1)
	
	object_pts, _ = gather_object_pts(pts_stage1, mask)

	object_pts = object_pts.reshape(bs, mask.shape[-1]//2, -1)
	object_pts = object_pts.float().view(bs,3,-1)
	return object_pts, mask_xyz_mean.squeeze(), mask


def gather_object_pts(pts, mask):
	'''
	:param pts: (bs,c,1024)
	:param mask: (bs,1024)
	:param n_pts: max number of points of an object
	:return:
		object_pts:(bs,c,n_pts)
		indices:(bs,n_pts)
	'''
	n_pts=pts.shape[-1]//2
	
	bs = pts.shape[0]
	indices = torch.zeros((bs, n_pts), dtype=torch.int64)
	object_pts = torch.zeros((bs, pts.shape[1], n_pts), device=pts.device)
	
	for i in range(bs):
		pos_indices = torch.where(mask[i, :] > 0.5)[0]

		if len(pos_indices) > 0:
			if len(pos_indices) > n_pts:
				choice = torch.randperm(len(pos_indices))[:n_pts]
			
			else:
				c1 = torch.randperm(len(pos_indices))
				choice = torch.randint(len(pos_indices), size=(n_pts-len(pos_indices),))
				
				choice = torch.cat((c1, choice))

			
			indices[i, :] = pos_indices[choice]
			object_pts[i,:,:] = pts[i,:,indices[i,:]]
		###else?

	return object_pts, indices