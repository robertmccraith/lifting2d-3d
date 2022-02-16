import numpy as np
import cv2
from pathlib import Path


import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.ndimage

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from kitti_raw import KittiRawDataset

from pathlib import Path

class Opts(object):pass	
opt = Opts()
opt.frame_ids = range(5)
opt.height = 376
opt.width = 1242

opt.seg_net = "cityscapes_02"


base_dir = "/datasets/"

opt.data_path = f"/{base_dir}kitti/object/"



raw = True
draw = False

def process(split):

	train = KittiRawDataset(opt.data_path, split, opt, prev=opt.frame_ids, is_train=False,sequence=False)

	
	train_ = DataLoader(train, None, False, None, None, 16, lambda a: a)
	ind = 0

	for j, inputs in enumerate(tqdm(train_)):
		# if j>=10:break
		if inputs is None:
			continue
		if not raw:
			ran = range(len(inputs["bb2d"]))
		else:
			ran = range(len(inputs["box"]))



		for i in ran:
			if not raw:
				if inputs["type"][i] != "Car" or inputs["ignore"][i]:
					continue

				if inputs["bb2d"][i][0] == -1:
					continue
			
				center = inputs["centers"][i]
				size = inputs["size"][i]
				corners = inputs["corners"][i]
				yaw = inputs["yaw"][i]

				mask = inputs["3d_mask"][i]

			box = inputs["box"][i]
			
			pts = inputs["3d_pts"][i]

			skip = False
			for p in pts:
				if p.shape[0]==0:
					skip=True
			
			mask_2d = inputs["2d_mask"][i]
			if len(mask_2d) == 0 or mask_2d[0] is None or mask_2d[0].sum()==0:
				continue

			
			folder = f"/{base_dir}/pcd/{split}-unsup{max(opt.frame_ids)+1}-cityscapes1"

			Path(folder).mkdir(exist_ok=True)

			f = f"{folder}/{ind}"

			if skip:
				continue
			
			np.savez(f,
				pts=pts, 
				mask_2d=mask_2d,
				box=box,
				K=inputs[("K",-1)],
				mask2d_img=inputs["2d_mask_img"][i],
				# img_size=inputs[("color",  0, -1)].size,
				scores=inputs["scores"][i],
				Rt=inputs["Rts"],
				img_id=int(inputs["img_id"]),
				raw_id=inputs["raw_id"],
				velocities=inputs["velocities"][i],
			)


			ind +=1

		if draw:
			plt.show()
			



process("train")
process("val")