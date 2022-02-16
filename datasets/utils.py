from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import random
from kitti_utils import generate_depth_map
import os 
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

def in_hull(p, hull):
	if not isinstance(hull,Delaunay):
		hull = Delaunay(hull)

	return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
	''' pc: (N,3), box3d: (8,3) '''
	box3d_roi_inds = in_hull(pc, box3d)
	return pc[box3d_roi_inds,:], box3d_roi_inds


def dm2XYZ(img, K):
	if isinstance(img, np.ndarray):
		u,v = np.where(img[...,0]>0)
	else:
		u,v = torch.nonzero(img, as_tuple=True)

	img = img[u,v]

	if len(K.shape)>2:
		K=K[0]
	
	fx = K[0,0]
	fy = K[1,1]
	cx = K[0,2]
	cy = K[1,2]

	x = img[...,0] * (v-cx)/fx
	y = img[...,0] * (u-cy)/fy
	return x,y,img



def make_depthmap(data_path, folder, frame_id, index=-1):
	
	npy_filename = os.path.join(
		data_path.replace("object","raw"),
		folder,
		"velodyne_points/",
		"lidar4",
		"image_2",
		"{:010d}.npz".format(int(frame_id)))
	

	calib_dir = os.path.join(data_path, folder.split("/")[0])
	velo_filename = os.path.join(data_path, folder,
					"velodyne_points/data", "{:010d}.bin".format(frame_id))

	try:
		d = np.load(npy_filename, allow_pickle=True)
		
		return d["depth"], d["interp"]

	except:
		gt_depth, inter = generate_depth_map(calib_dir, velo_filename, 2)

		Path("/".join(npy_filename.split("/")[:-1]) ).mkdir(parents=True, exist_ok=True)

		np.savez(npy_filename, depth=gt_depth, interp=inter)


		return gt_depth, inter



		


from collections import namedtuple

OxtsPacket = namedtuple('OxtsPacket',
						'lat, lon, alt, ' +
						'roll, pitch, yaw, ' +
						'vn, ve, vf, vl, vu, ' +
						'ax, ay, az, af, al, au, ' +
						'wx, wy, wz, wf, wl, wu, ' +
						'pos_accuracy, vel_accuracy, ' +
						'navstat, numsats, ' +
						'posmode, velmode, orimode')
def read_calib_file(filepath):
	"""
	Read in a calibration file and parse into a dictionary

	Parameters
	----------
	filepath : str
		File path to read from

	Returns
	-------
	calib : dict
		Dictionary with calibration values
	"""
	data = {}
	print("H")

	with open(filepath, 'r') as f:
		for line in f.readlines():
			key, value = line.split(':', 1)
			# The only non-float values in these files are dates, which
			# we don't care about anyway
			try:
				data[key] = np.array([float(x) for x in value.split()])
			except ValueError:
				pass

	return data

def rotx(t):
	"""
	Rotation about the x-axis

	Parameters
	----------
	t : float
		Theta angle

	Returns
	-------
	matrix : np.array [3,3]
		Rotation matrix
	"""
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[1,  0,  0],
					 [0,  c, -s],
					 [0,  s,  c]])
def roty(t):
	"""
	Rotation about the y-axis

	Parameters
	----------
	t : float
		Theta angle

	Returns
	-------
	matrix : np.array [3,3]
		Rotation matrix
	"""
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[c,  0,  s],
					 [0,  1,  0],
					 [-s, 0,  c]])


def rotz(t):
	"""
	Rotation about the z-axis

	Parameters
	----------
	t : float
		Theta angle

	Returns
	-------
	matrix : np.array [3,3]
		Rotation matrix
	"""
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[c, -s,  0],
					 [s,  c,  0],
					 [0,  0,  1]])

def pose_from_oxts_packet(raw_data, scale):
	"""
	Helper method to compute a SE(3) pose matrix from an OXTS packet

	Parameters
	----------
	raw_data : dict
		Oxts data to read from
	scale : float
		Oxts scale

	Returns
	-------
	R : np.array [3,3]
		Rotation matrix
	t : np.array [3]
		Translation vector
	"""
	packet = OxtsPacket(*raw_data)
	er = 6378137.  # earth radius (approx.) in meters

	# Use a Mercator projection to get the translation vector
	tx = scale * packet.lon * np.pi * er / 180.
	ty = scale * er * \
		np.log(np.tan((90. + packet.lat) * np.pi / 360.))
	tz = packet.alt
	t = np.array([tx, ty, tz])

	# Use the Euler angles to get the rotation matrix
	Rx = rotx(packet.roll)
	Ry = roty(packet.pitch)
	Rz = rotz(packet.yaw)
	R = Rz.dot(Ry.dot(Rx))

	# Combine the translation and rotation into a homogeneous transform
	return R, t

def transform_from_rot_trans(R, t):
	"""
	Transformation matrix from rotation matrix and translation vector.

	Parameters
	----------
	R : np.array [3,3]
		Rotation matrix
	t : np.array [3]
		translation vector

	Returns
	-------
	matrix : np.array [4,4]
		Transformation matrix
	"""
	R = R.reshape(3, 3)
	t = t.reshape(3, 1)
	return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def invert_pose_numpy(T):
	"""Inverts a [4,4] np.array pose"""
	Tinv = np.copy(T)
	R, t = Tinv[:3, :3], Tinv[:3, 3]
	Tinv[:3, :3], Tinv[:3, 3] = R.T, - np.matmul(R.T, t)
	return Tinv