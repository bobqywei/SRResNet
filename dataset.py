import os
import torch
import glob
import torchvision.transforms as transforms

from PIL import Image

MEAN = [0.37159888, 0.38509135, 0.3721678]
STD = [0.1006065,  0.10916227, 0.11465776]

# reverses the earlier normalization applied to the image to prepare output
def unnormalize(x):
	x.transpose_(1, 3)
	x = x * torch.Tensor(STD).cuda() + torch.Tensor(MEAN).cuda()
	x.transpose_(1, 3)
	return x

class KITTIDataset(torch.utils.data.Dataset):
	def __init__(self, gt_path, lr_path, mask_path):
		super().__init__()
		self.gt_imgs = sorted(glob.glob(os.path.join(gt_path, '*.png')))
		self.lr_imgs = sorted(glob.glob(os.path.join(lr_path, '*.png')))
		# self.masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
		# self.mask_transform = transforms.ToTensor()

	def __len__(self):
		return len(self.gt_imgs)

	def __getitem__(self, index):
		gt = Image.open(self.gt_imgs[index]).convert('RGB')
		# mask = Image.open(self.masks[index]).convert('RGB')
		w, h = gt.size

		if w % 2 != 0:
			gt = gt.resize((w-1, h), resample=Image.BICUBIC)
			# mask = mask.resize((w-1, h), resample=Image.BICUBIC)
		if h % 2 != 0:
			gt = gt.resize((w, h-1), resample=Image.BICUBIC)
			# mask = mask.resize((w, h-1), resample=Image.BICUBIC)

		gt = self.img_transform(gt)
		# mask = self.mask_transform(mask)

		lr = Image.open(self.lr_imgs[index]).convert('RGB')
		lr = self.img_transform(lr)

		return lr, gt#, mask
