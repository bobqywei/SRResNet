import os
import numpy as np
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='source directory')
parser.add_argument('--factor', type=int, help='factor to downsample by')
args = parser.parse_args()

curr_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(curr_dir, args.src)
img_files = os.listdir(src_dir)
target_dir = os.path.join(curr_dir, 'lr_down_x{0}/'.format(args.factor))

if not os.path.exists(target_dir):
	os.makedirs(target_dir)

for name in img_files:
	img = Image.open(os.path.join(src_dir, name)).convert('RGB')
	w, h = img.size
	img = img.resize((w//args.factor, h//args.factor), resample=Image.BICUBIC)
	img.save(os.path.join(target_dir, name))
