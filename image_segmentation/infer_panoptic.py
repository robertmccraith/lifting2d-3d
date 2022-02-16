import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
import glob, random
from run_detectron import run_detectron
import matplotlib.pyplot as plt



root = "/datasets/kitti/raw/"

g = glob.glob(root+"*/*/image_02/data/*.jpg")

# fs = open("splits/object/train_files.txt", "r").readlines()
# fs = list(map(lambda a: a.strip().split(" "), fs))

# g = list(map(lambda a: root+"/"+a[0]+"/image_02/data/"+a[1]+".jpg", fs))

# shuffle(g)
# root = "/datasets/kitti/raw/"



for rgb_file in tqdm(g):
	out_filename = rgb_file.replace("image_02", "cityscapes_02").replace("jpg", "npz")
	
	rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
	inst, cl, bs, sc = run_detectron(rgb)


	instances = []


	boxes = []
	scores = []
	idx = 0
	for x,i in enumerate(inst):
		
		if cl[x] == 2:# in both MS-COCO and Cityscapes 2 is the thing index for cars
			instances.append(i)

			boxes.append(bs[x])
			scores.append(sc[x])

	Path("/".join(out_filename.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

	np.savez_compressed(out_filename, instances=instances, boxes=boxes, scores=scores)