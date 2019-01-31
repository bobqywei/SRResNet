import os
import glob
import numpy as np

from PIL import Image

src_path = '/home/bobw/debug_kitti/pytorch-SRResNet/data/rgb/'

paths = glob.glob(os.path.join(src_path, '*.png'))
test = Image.open(paths[0])
w, h = test.size
total = np.zeros([w * h, 3])

for path in paths:
	img = np.array(Image.open(path).convert('RGB'))
	img = img.astype('float') / 255.0
	total = np.add(total, img.reshape(w * h, 3))

total /= len(paths)
print(np.mean(total, axis=0))
print(np.std(total, axis=0))
