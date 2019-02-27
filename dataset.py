import os
import torch
import glob
import torchvision.transforms as transforms

from PIL import Image

# MEAN = [0.37159888, 0.38509135, 0.3721678]
# STD = [0.1006065,  0.10916227, 0.11465776]

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# reverses the earlier normalization applied to the image to prepare output
def unnormalize(x):
	x = (x * 0.5) + 0.5
	return x

class KITTIDataset(torch.utils.data.Dataset):
	def __init__(self, gt_path, lr_path, mask_path=None):
		super().__init__()
		self.gt_imgs = sorted(glob.glob(os.path.join(gt_path, '*.png')))
		self.lr_imgs = sorted(glob.glob(os.path.join(lr_path, '*.png')))
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

		if not mask_path is None:
			self.masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))
			self.mask_transform = transforms.ToTensor()

	def __len__(self):
		return len(self.gt_imgs)

	def __getitem__(self, index):
		gt = Image.open(self.gt_imgs[index]).convert('RGB')
		gt = self.img_transform(gt)

		lr = Image.open(self.lr_imgs[index]).convert('RGB')
		lr = self.img_transform(lr)

		if hasattr(self, 'masks'):
			mask = Image.open(self.masks[index])
			mask = self.mask_transform(mask)
			return lr, gt, mask
		else:
			return lr, gt
