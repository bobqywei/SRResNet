import argparse
import requests
import zipfile
import os
import random
import shutil
import glob

from io import BytesIO

zipfiles = {
	# Train
	# "2011_09_26": [1,9,11,14,15,17,18,19,22,27,28,29,32,35,39,46,48,51,52,56,57,59,60,61,64,70,84,86,87,91,93,96,101,104,106,117],
	# "2011_09_28": [1,2,16,21,34,35,38,39,43,45,47],
	# "2011_09_29": [4]

	# Train: Large Size
	# "2011_09_29": [71],
	# "2011_09_30": [18,20,27,28,33,34],
	# "2011_10_03": [27,34,42]

	# Val
	"2011_09_26": [2,5,13,20,23,36,79,95,113],
	"2011_09_28": [37],
	"2011_09_29": [26],
	"2011_09_30": [16],
	"2011_10_03": [47]
}

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/", type=str)
parser.add_argument("--dest", type=str, default="/home/bobw/val/rgb/")
parser.add_argument("--d_dest", type=str, default="/home/bobw/val/depth_16bit/")
parser.add_argument("--d_src", type=str, default="/home/bobw/kitti_depth/")
args = parser.parse_args()

if not os.path.exists(args.dest):
	os.makedirs(args.dest)

if not os.path.exists(args.d_dest):
	os.makedirs(args.d_dest)

img_index = 1

for date, drivelist in zipfiles.items():
	for drive in drivelist:
		name = date + "_drive_{:04d}_sync".format(drive)
		print(name)
		request = requests.get("https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{0}_drive_{1}/{2}.zip".format(date, "{:04d}".format(drive), name))
		print('...download complete')

		file = zipfile.ZipFile(BytesIO(request.content))
		depth_path = os.path.join(args.d_src, name, "proj_depth", "groundtruth", "image_02/")
		imgs = glob.glob(depth_path + "*.png")

		for d_img in imgs:
			img = os.path.join(date, name, "image_02", "data", d_img.split(depth_path)[1])
			file.extract(img)
			out_name = "{0}_{1}_{2}.png".format(img_index, date, drive)
			os.rename(img, os.path.join(args.dest, out_name))
			shutil.copyfile(d_img, os.path.join(args.d_dest, out_name))
			img_index += 1

		shutil.rmtree(date)
