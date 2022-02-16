import numpy as np
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from datasets.kitti_seq import KittiObjectsDataset
from models.model import Model
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from options import Options
from pathlib import Path
from collections import defaultdict

from torch.cuda.amp import GradScaler, autocast



opts = Options()
opt = opts.parse()

torch.set_printoptions(precision=10)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
import random
random.seed(opt.seed)
np.random.seed(opt.seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

output_dir = f"{opt.out_dir}/boards/{opt.exp_name}"
print(output_dir)


results = {}

if not opt.inference:
	kind = "train"+"-"+opt.folder
	print(opt.data_path+f"{kind}/")
	train = KittiObjectsDataset(opt.data_path+f"{kind}/", 
								opt.kitti_path,
								n_pts=opt.npts,
								reflectance=opt.reflectance,
								inliers=opt.inliers, 
								mask_2d=opt.mask_2d,
								frames=opt.frame_ids,
								width_min=opt.wmin,
								lidar_min=opt.lidar_min,
								mask_min=opt.mask_min,
								truncation=opt.trunc,
								img_scale=opt.img_scale,
								verbose=opt.verbose,
								min_conf=opt.min_confidence,
								yaw=opt.yaw,
								render=opt.render_mask_loss)

	train_loader = DataLoader(train,
							opt.batch_size,
							True,
							pin_memory=False,
							drop_last=True,
							num_workers=12,
							worker_init_fn=seed_worker,
							persistent_workers=True)
	print(f"Train examples:{len(train)}")

	erase_copy(output_dir)


	writer_train = SummaryWriter(f"{output_dir}/Train/")


	writer_val = SummaryWriter(f"{output_dir}/Val/")

	for k, v in vars(opt).items():
		writer_train.add_text(k, str(v), 0)



kind = "val"+"-"+opt.folder


val = KittiObjectsDataset(opt.data_path+f"{kind}/",
							opt.kitti_path,
							n_pts=opt.npts,
							reflectance=opt.reflectance,
							inliers=opt.inliers, 
							mask_2d=opt.mask_2d,
							frames=range(1),
							width_min=25,
							mask_min=0,
							lidar_min=opt.lidar_min,
							truncation=-1,
							img_scale=opt.img_scale,
							verbose=True,
							min_conf=0.,
							yaw=opt.yaw,
							render=opt.render_mask_loss)

print(f"Val examples:{len(val)}")

val_loader = DataLoader(val,
						opt.batch_size,
						False,
						pin_memory=False,
						drop_last=False,
						num_workers=12,
						worker_init_fn=seed_worker,
						persistent_workers=True)








model = nn.DataParallel(Model(frame_ids=opt.frame_ids, 
				yaw=opt.yaw,
				yaw_bins=opt.yaw_bins,
				threshold=opt.threshold, 
				mpts=opt.mpts,
				sizeGt= opt.sizeGT,
				cham=opt.cham,
				stn=opt.stn,
				nomaskcut=opt.nomaskcut, 
				loss_fn=opt.loss_fn,
				box_loss=opt.box_loss,
				render_mask_loss=opt.render_mask_loss,
				offsets=opt.offsets)).cuda()


optimizers = optim.Adam(model.parameters(), opt.lr)

schedulers = optim.lr_scheduler.StepLR(optimizers, opt.step,0.3)






def run_epoch(i, loader, writer, label, train=False, inference=False):
	log = i%10==0 and label!="Train" and not inference
	
	centres = []
	yaws = []
	sizes = []
	img_ids = []
	box2d = []
	npts = []
	scores = []
	total_losses = defaultdict(list)

	total_loss = []		
	
	for j, inputs in enumerate(tqdm(loader,position=1,leave=False)):

		for key, ipt in inputs.items():
			if not isinstance(ipt, list):
				inputs[key] = ipt.to("cuda", non_blocking=True).float()
		
		if train:
			optimizers.zero_grad()


		outputs, losses = model(inputs,label=="Val")
		
		loss = 0.
		for k,v in losses.items():
			total_losses[k].append(v.mean().item())
			loss += v.mean()
		
		if train:
			loss.mean().backward()
			optimizers.step()


		size = inputs["size"].detach().cpu().numpy() if opt.sizeGT else np.array([[1.52986348, 1.61876715, 3.89206519][::-1]]).repeat(inputs[("pts",0)].shape[0],0)

		if "img_id" in inputs:
			img_ids.extend(inputs["img_id"].tolist())
			box2d.extend((inputs["box"]*opt.img_scale)[:,0].tolist())
			scores.extend(inputs["scores"].tolist())
			npts.extend(inputs[("npts",0)].tolist())
		
		
		centres.extend(outputs[("centre", 0)].tolist())
		yaws.extend(outputs[("yaw", 0)].tolist())
		sizes.extend(size[:,::-1].tolist())


		total_loss.append(loss.item())


		if j == 0 and not inference:
			for frame in opt.frame_ids:
				if ("img",frame) not in inputs:
					break

				for k in range(4):
					writer.add_image(f"image/{frame}/{k}", inputs[("img",frame)][k], i, dataformats="HWC")	

					writer.add_image(f"crop/{frame}/{k}", inputs[("crop_raw",frame)][k], i, dataformats="HWC")

					if opt.render_mask_loss:
						 writer.add_image(f"mask2d_img/{0}/{k}", inputs[("current",0)][k], i, dataformats="HW")

					plot_pcds(inputs[("pts",frame)].transpose(-1,-2),
							outputs[("models",frame)],
							outputs[("inds",frame)],
							outputs[("centre",frame)],
							outputs[("yaw",frame)],
							outputs[("front",frame)],
							None,
							size,
							inputs["corners"] if "corners" in inputs else None,
							writer,
							i,
							frame)
			

			if opt.render_mask_loss:
				for frame in opt.frame_ids:
					if ("current",frame) not in inputs:
						break

					masks = torch.stack([inputs[("current",frame)], outputs[("render",frame)], inputs[("occluding",frame)]],1)

					for k in range(4):
						writer.add_image(f"masks/{frame}/{k}", masks[k], i)

						writer.add_image(f"difference/{frame}/{k}", outputs[("difference",frame)][k], i, dataformats="HW")

						writer.add_image(f"difference_mask/{frame}/{k}", outputs[("difference_mask",frame)][k], i, dataformats="HW")

						writer.add_image(f"mask2d_img_occlusion/{frame}/{k}", inputs[("occluding",frame)][k], i, dataformats="HW")
		


	if not opt.inference:
		writer.add_scalar(f"Loss", np.mean(total_loss), i)

		for k,v in total_losses.items():
			if type(k) is tuple:
				writer.add_scalar(" ".join([str(a) for a in k]), np.mean(v), i)	
			else:
				writer.add_scalar(k, np.mean(v), i)

	


	if (i%10==0 and "img_id" in inputs and label=="Val") or inference:
		centres = np.array(centres)
		sizes = np.array(sizes)
		yaws = np.array(yaws)

		# Y centre is at bottom of car
		centres[:,1] += sizes[:,1]/2

		predictions = output_dir+"/predictions/"
		if img_ids[0]==-1:
			img_ids = [a["raw_id"].tolist() for a in val.files]

		img_ids = np.array(img_ids)
		box2d = np.array(box2d)
		npts=np.array(npts)
		scores=np.array(scores)

		write2kittiObjFile(predictions, img_ids, box2d, centres, yaws, sizes, scores, npts)

		for threshold in [0.7, 0.5]:
			bb, bev, bb3d = evaluate(opt.kitti_path, predictions, label.lower(), threshold)

			print(f"{opt.exp_name} T{threshold}:", bb, bev, bb3d)
			
			results[("bev", threshold)] = [str(a) for a in bev]
			results[("bb3d", threshold)] = [str(a) for a in bb3d]
			results[("bb2d", threshold)] = [str(a) for a in bb]

			if inference:continue

			writer.add_scalar(f"Kitti {threshold} Image/Easy", bb[0], i)
			writer.add_scalar(f"Kitti {threshold} Image/Medium", bb[1], i)
			writer.add_scalar(f"Kitti {threshold} Image/Hard", bb[2], i)


			writer.add_scalar(f"Kitti {threshold} iou2d/Easy", bev[0], i)
			writer.add_scalar(f"Kitti {threshold} iou2d/Medium", bev[1], i)
			writer.add_scalar(f"Kitti {threshold} iou2d/Hard", bev[2], i)

			writer.add_scalar(f"Kitti {threshold} iou3d/Easy", bb3d[0], i)
			writer.add_scalar(f"Kitti {threshold} iou3d/Medium", bb3d[1], i)
			writer.add_scalar(f"Kitti {threshold} iou3d/Hard", bb3d[2], i)

				


if __name__ == '__main__':

	if opt.inference:
		model.load_state_dict(torch.load(opt.weights))
		model.eval()

		with torch.no_grad():
			run_epoch(0,val_loader, None, "Val", inference=True)
	
	else:
		for i in tqdm(range(opt.num_epochs + 1)):
			model.train()

			run_epoch(i,train_loader, writer_train, "Train", train=True)

			schedulers.step()


			if i % 10 == 0:
				model.eval()
				with torch.no_grad():
					run_epoch(i,val_loader, writer_val, "Val")

				torch.save(model.state_dict(), output_dir+f"/weights/{i}.pth")

		with open("results.txt", "a+") as f:
			print(opt.exp_name, ",", ",".join(results[("bev", 0.5)]), ",",",".join(results[("bb3d", 0.5)]),",",",".join(results[("bev", 0.7)]),",", ",".join(results[("bb3d", 0.7)]), ",",",".join(results[("bb2d", 0.5)]),",",",".join(results[("bb2d", 0.7)]), file=f)

	print(opt.exp_name, ",", ",".join(results[("bev", 0.5)]), ",",",".join(results[("bb3d", 0.5)]),",",",".join(results[("bev", 0.7)]),",", ",".join(results[("bb3d", 0.7)]), ",",",".join(results[("bb2d", 0.5)]),",",",".join(results[("bb2d", 0.7)]))

