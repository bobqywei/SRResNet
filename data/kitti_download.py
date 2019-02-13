import argparse
import requests
import zipfile
import os
import random
import shutil
import glob

from io import BytesIO

zipfiles = {
	# "2011_09_26": [1,9,11,14,15,17,18,19,22,27,28,29,32,35,39,46,48,51,52,56,57,59,60,61,64,70,84,86,87,91,93,96,101,104,106,117],
	# "2011_09_28": [1,2,16,21,34,35,38,39,43,45,47,71],
	# "2011_09_29": [4],
	# "2011_09_30": [18,20,27,28,33,34],
	# "2011_09_30": [18, 28],
	"2011_09_30": [20,27,34],
	# "2011_10_03": [27,34,42]
	"2011_10_03": [42]
}

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/", type=str)
parser.add_argument("--img_per_scene", type=int, default=10)
# parser.add_argument("--date", type=str, default="2011_09_26")
# parser.add_argument("--drive", type=int, default=1)
parser.add_argument("--dest", type=str, default="/home/bobw/rgb/")
parser.add_argument("--depth", type=str, default="/home/bobw/depth_16bit/")
parser.add_argument("--d_src", type=str, default="/home/bobw/kitti_depth/")
args = parser.parse_args()

if not os.path.exists(args.dest):
	os.makedirs(args.dest)

if not os.path.exists(args.depth):
	os.makedirs(args.depth)

img_index = 501

for date, drivelist in zipfiles.items():
	for drive in drivelist:
		name = date + "_drive_{:04d}_sync".format(drive)
		print(name)
		request = requests.get("https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{0}_drive_{1}/{2}.zip".format(date, "{:04d}".format(drive), name))#args.url + name + "/" + name + "_sync.zip")
		file = zipfile.ZipFile(BytesIO(request.content))
		depth_path = os.path.join(args.d_src, "train", name, "proj_depth", "groundtruth", "image_02/")
		imgs = glob.glob(depth_path + "*.png")
		imgs = random.sample(imgs, args.img_per_scene)
		# imgs = [x for x in file.namelist() if x.startswith(os.path.join(date, name, "image_02", "data/"))]

		for d_img in imgs:
			img = os.path.join(date, name, "image_02", "data", d_img.split(depth_path)[1])
			file.extract(img)
			os.rename(img, args.dest + "{0}.png".format(img_index))
			shutil.copyfile(d_img, args.depth + "{0}.png".format(img_index))
			img_index += 1

		shutil.rmtree(date)
