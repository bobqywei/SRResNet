import os
import glob
import numpy as np

from PIL import Image

src_path = '/home/bobw/SRResNet/data/depth_16bit/'

paths = glob.glob(os.path.join(src_path, '*.png'))

for path in paths:
	img = Image.open(path)
	w, h = img.size
	if w != 1244 or h != 376:
		img = img.resize((1244, 376))
		img.save(path)

# test = Image.open(paths[0])
# w, h = test.size
# total = np.zeros([w * h, 3])
#
# for path in paths:
# 	img = np.array(Image.open(path).convert('RGB'))
# 	img = img.astype('float') / 255.0
# 	total = np.add(total, img.reshape(w * h, 3))
#
# total /= len(paths)
# print(np.mean(total, axis=0))
# print(np.std(total, axis=0))
