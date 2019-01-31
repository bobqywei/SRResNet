import os
import torch
import glob
import torchvision.transforms as transforms

from PIL import Image

MEAN = [0.35671806, 0.3557934, 0.33836347]
STD = [0.20286339, 0.20125221, 0.20376566]

class KITTIDataset(torch.utils.data.Dataset):
	def __init__(self, gt_path, lr_path, mask_path):
		super().__init__()
		self.gt_imgs = sorted(glob.glob(os.path.join(gt_path, '*.png')))
		self.lr_imgs = sorted(glob.glob(os.path.join(lr_path, '*.png')))
		self.masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
		self.mask_transform = transforms.ToTensor()

	def __len__(self):
		return len(self.gt_imgs)

	def __getitem__(self, index):
		gt = Image.open(self.gt_imgs[index]).convert('RGB')
		w, h = gt.size
		gt = self.img_transform(gt)
		
		lr = Image.open(self.lr_imgs[index]).convert('RGB')
		lr = self.img_transform(lr.resize((w, h), resample=Image.BICUBIC))

		mask = Image.open(self.masks[index]).convert('RGB')
		mask = self.img_transform(mask)

		return lr, gt, mask


