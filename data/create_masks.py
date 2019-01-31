import os
import sys
import glob
import argparse
import numpy as np
import scipy.misc as m

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='path to images')
parser.add_argument('--depth', type=int, help='depth value to split')
args = parser.parse_args()

curr = os.path.dirname(os.path.realpath(sys.argv[0]))
dest = [os.path.join(curr, 'masks', 'd_level_1'), os.path.join(curr, 'masks', 'd_level_2')]
for pth in dest:
	if not os.path.exists(pth):
		os.makedirs(pth)

img_paths = glob.glob(os.path.join(args.src, '*.png'))

for img_path in img_paths:
	f_name = img_path.split(args.src+'/')[1]
	img = np.array(Image.open(img_path))
	mask1 = np.logical_and(img > 0, img <= args.depth) * 255
	mask2 = (img > args.depth) * 255

	mask1 = Image.fromarray(mask1.astype('uint8'))
	mask2 = Image.fromarray(mask2.astype('uint8'))

	mask1.save(os.path.join(dest[0], f_name))
	mask2.save(os.path.join(dest[1], f_name))
