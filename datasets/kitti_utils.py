from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter
import scipy.interpolate as interpolate

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

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data
###3d->2d

def project_rect_to_image(pts_3d_rect, P):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def project_velo_to_image(pts_3d_velo, v2c, R0, P):
    ''' Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = project_velo_to_rect(pts_3d_velo, v2c, R0)
    return project_rect_to_image(pts_3d_rect, P)



### 3D
def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

def project_ref_to_rect(pts_3d_ref, R0):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))

def project_velo_to_ref(pts_3d_velo, v2c):
    pts_3d_velo = cart2hom(pts_3d_velo) # nx4
    return np.dot(pts_3d_velo, np.transpose(v2c))


def project_velo_to_rect(pts_3d_velo, v2c, R0):
    pts_3d_ref = project_velo_to_ref(pts_3d_velo, v2c)
    return project_ref_to_rect(pts_3d_ref, R0)

def get_lidar_in_image_fov(pc_velo, calibs, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = project_velo_to_image(pc_velo, calibs['Tr_velo_to_cam'].reshape(3,4),
                                            calibs['R0_rect'].reshape(3,3),
                                            calibs['P2'].reshape(3,4))

    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]

    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo



def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def lin_interp(shape, xyd):
	# taken from https://github.com/hunse/kitti
	m, n = shape
	ij, d = xyd[:, 1::-1], xyd[:, 2]
	f = interpolate.LinearNDInterpolator(ij, d, fill_value=0)
	J, I = np.meshgrid(np.arange(n), np.arange(m))
	IJ = np.vstack([I.flatten(), J.flatten()]).T
	disparity = f(IJ).reshape(shape)
	return disparity


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]
    # print(velo.shape)

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((*im_shape[:2], 2))

    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2:]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape[:-1], velo_pts_im[:, 1], velo_pts_im[:, 0])
    
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    # if interp:
    return depth, lin_interp(im_shape, velo_pts_im)
    # return depth



def generate_pcds(data_path,folder, drive, frame):
    data_path = data_path.replace("object", "raw")

    cam2cam = read_calib_file(f"{data_path}/{folder}/calib_cam_to_cam.txt")
    velo2cam = read_calib_file(f"{data_path}/{folder}/calib_velo_to_cam.txt")
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    velo = np.fromfile(f"{data_path}/{folder}/{drive}/velodyne_points/data/{(frame):0>10}.bin", dtype=np.float32).reshape(-1,4)

    velo = velo[velo[:, 0] >= 0, :]

    ref = velo[:,3].copy()
    velo[:,3]=1.0


    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]


    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    # velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    # velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]


    velo = np.dot(velo, velo2cam.T)
    velo = np.dot(R_cam2rect, velo.T).T

    # velo = velo[:,[1,2,0,3]]

    velo[:,3]=ref
    
    velo = velo[val_inds, :]

    return velo, velo_pts_im, P_rect


def compute_box_3d(ry, l, w, h, X,Y,Z):
	''' Takes an object and a projection matrix (P) and projects the 3d
		bounding box into the image plane.
		Returns:
			corners_2d: (8,2) array in left image coord.
			corners_3d: (8,3) array in in rect camera coord.
	'''
	# compute rotational matrix around yaw axis
	R = roty(ry)

	# 3d bounding box corners
	x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
	y_corners = [0,0,0,0,-h,-h,-h,-h]
	z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

	# rotate and translate 3d bounding box
	corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
	#print corners_3d.shape
	corners_3d[0,:] = corners_3d[0,:] + X
	corners_3d[1,:] = corners_3d[1,:] + Y
	corners_3d[2,:] = corners_3d[2,:] + Z
	#print 'cornsers_3d: ', corners_3d 
	# only draw 3d bounding box for objs in front of the camera

	if np.any(corners_3d[2,:]<0.1):
		return np.transpose(corners_3d)


	return np.transpose(corners_3d)