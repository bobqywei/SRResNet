import os
import numpy as np
import argparse
import torch

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def gauss2d(shape, sigma=1.0):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='source directory')
parser.add_argument('--factor', type=int, default=2, help='factor to downsample by')
args = parser.parse_args()

curr_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(curr_dir, args.src)
img_files = os.listdir(src_dir)
target_dir = os.path.join(curr_dir, 'lr_x{0}/'.format(args.factor))

if not os.path.exists(target_dir):
	os.makedirs(target_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tr = transforms.ToTensor();

conv = torch.nn.Conv2d(1, 1, 5, padding=2, bias=False)
with torch.no_grad():
	conv.weight.data[0][0] = torch.from_numpy(gauss2d((5,5)))
conv.to(device)

for name in img_files[0:1]:
	img = Image.open(os.path.join(src_dir, name)).convert('RGB')
	w, h = img.size
	img = tr(img.resize((w//args.factor, h//args.factor), resample=Image.BICUBIC)).to(device)

	# batch dimension
	img = torch.unsqueeze(img, 0)
	img_R = conv(torch.unsqueeze(img[:,0,:,:], 0))
	img_G = conv(torch.unsqueeze(img[:,1,:,:], 0))
	img_B = conv(torch.unsqueeze(img[:,2,:,:], 0))
	img = torch.cat((img_R, img_G, img_B), 1)
	save_image(img, os.path.join(target_dir, name))
	# img.save(os.path.join(target_dir, name))
