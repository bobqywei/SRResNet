import os
import sys
import glob
import argparse
import numpy as np
import scipy.misc as m

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='path to images')
parser.add_argument('--depth', type=int, help='depth value to split')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--name', type=str)
args = parser.parse_args()

dest = [os.path.join(args.mode, 'masks', args.name, 'level_1'), os.path.join(args.mode, 'masks', args.name, 'level_2')]
for pth in dest:
	if not os.path.exists(pth):
		os.makedirs(pth)

img_paths = glob.glob(os.path.join(args.src, '*.png'))

for i in tqdm(range(len(img_paths))):
	img_path = img_paths[i]
	f_name = img_path.split('/')[-1]
	img = np.array(Image.open(img_path))
	mask1 = np.logical_and(img > 0, img <= args.depth) * 255
	mask2 = (img > args.depth) * 255

	mask1 = Image.fromarray(mask1.astype('uint8'))
	mask2 = Image.fromarray(mask2.astype('uint8'))

	mask1.save(os.path.join(dest[0], f_name))
	mask2.save(os.path.join(dest[1], f_name))
