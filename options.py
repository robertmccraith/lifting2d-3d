from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Options:
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="options")

		self.parser.add_argument("--data_path",
						type=str,
						default="/datasets/pcd/",
						help="path to the training data")
		
		self.parser.add_argument("--kitti_path",
						type=str,
						help="path to the training data",
						default="/datasets/kitti/object/")

		self.parser.add_argument("--exp_name",
						type=str,
						default="test",
						help="experiment name")

		self.parser.add_argument("--out_dir",
						type=str,
						help="experiment name",
						 default="boards/")

		self.parser.add_argument("--folder",
						type=str,
						help="data folder",
						default="unsup5-motion")


		# OPTIMIZATION options
		self.parser.add_argument("--batch_size",
						type=int,
						help="batch size",
						default=64)

		self.parser.add_argument("--lr",
						type=float,
						help="learning rate",
						default=3e-3)

		self.parser.add_argument("--num_epochs",
						type=int,
						help="number of epochs",
						default=100)
		self.parser.add_argument("--seed",
						type=int,
						help="seed",
						default=0)

		self.parser.add_argument("--npts",
						type=int,
						help="number of epochs",
						default=1024)
		
		self.parser.add_argument("--mpts",
						type=int,
						help="number of epochs",
						default=1000)


		self.parser.add_argument("--step",
						type=int,
						help="step size of the scheduler",
						default=30)
		
		self.parser.add_argument("--inference",
						help="inference mode",
						action="store_true")


		self.parser.add_argument("--weights",
						type=str,
						help="path to pretrained weights")


		self.parser.add_argument("--seg",
						help="use segmentation network",
						action="store_true")
		

		self.parser.add_argument("--reflectance",
						help="us reflectance value in ",
						action="store_true")

		

		self.parser.add_argument("--loss_fn",
						type=str,
						default="Seuclidean")

		self.parser.add_argument("--rotate_to_centre",
						help="rotate so image box centred",
						action="store_true")
		
		self.parser.add_argument("--frame_ids",
								 nargs="+",
								 type=int,
								 help="frames to load",
								 default=[1])
		
					
		self.parser.add_argument("--sizeGT",
						help="sigmoid segmentation output",
						action="store_true")
		
		self.parser.add_argument("--inliers",
						help="sigmoid segmentation output",
						action="store_true")
		
		self.parser.add_argument("--mask_2d",
						help="sigmoid segmentation output",
						action="store_true")


		self.parser.add_argument("--threshold",
						type=float,
						help="remove chamfer error above this percent",
						default=0.0)
		
		self.parser.add_argument("--cham",
						help="chamfer rather than lfit",
						action="store_true")
		
		self.parser.add_argument("--stn",
						help="use stn network",
						action="store_true")

		self.parser.add_argument("--nomaskcut",
						help="chamfer rather than lfit",
						action="store_true")

		self.parser.add_argument("--yaw_bins",
						type=int,
						help="remove chamfer error above this percent",
						default=-1)
		
		self.parser.add_argument("--offsets",
						type=int,
						help="remove chamfer error above this percent",
						default=0)
		
		self.parser.add_argument("--yaw",
						type=str,
						help="remove chamfer error above this percent",
						default="bins")
		
		self.parser.add_argument("--wmin",
						type=int,
						help="smallest width car in image",
						default=25)
		
		self.parser.add_argument("--lidar_min",
						type=int,
						help="min 3d inliers",
						default=0)
		
		self.parser.add_argument("--mask_min",
						type=int,
						help="min 3d inliers",
						default=0)


		self.parser.add_argument("--trunc",
						type=int,
						help="examples with no 2d box coord this close to edge of image",
						default=-1)
		
		self.parser.add_argument("--f_inter",
						help="chamfer rather than lfit",
						action="store_true")

		self.parser.add_argument("--box_loss",
						help="chamfer rather than lfit",
						action="store_true")

		self.parser.add_argument("--render_mask_loss",
						help="render model in image and compare to mask-rcnn in pixels",
						action="store_true")
		
		self.parser.add_argument("--img_scale",
						type=float,
						help="examples with no 2d box coord this close to edge of image",
						default=2)

		self.parser.add_argument("--verbose",
						help="chamfer rather than lfit",
						action="store_true")
		
		self.parser.add_argument("--min_confidence",
						type=float,
						help="min confidence masks to train with",
						default=0.0)


	def parse(self):
		self.options = self.parser.parse_args()

		self.options.frame_ids = range(*self.options.frame_ids)

		return self.options



	






