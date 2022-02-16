from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
import glob 
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model

import socket
hn = socket.gethostname()


cfg.merge_from_file("~/detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml")	
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl"

# cfg.merge_from_file("~/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"


predictor = DefaultPredictor(cfg)

def run_detectron(im):
	outputs = predictor(im)

	# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

	# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


	instances = outputs['instances'].pred_masks.cpu().numpy()
	classes = outputs['instances'].pred_classes.cpu().numpy()

	
	scores = outputs["instances"].scores.cpu().numpy()
	boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
	# print(boxes)

	# print( MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)

	# For COCO and Cityscapes thing_class[2] == "Car"

	return instances, classes, boxes, scores 
