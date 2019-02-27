import os
import numpy as np
import argparse
import torch

from PIL import Image
from tqdm import tqdm
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
parser.add_argument('--src', type=str)
parser.add_argument('--dest', type=str, default='dest')
parser.add_argument('--factor', type=int, default=2)
parser.add_argument('--gauss', action='store_true')
parser.add_argument('--fix_size_only', action='store_true')
args = parser.parse_args()

if not args.fix_size_only and not os.path.exists(args.dest):
	os.makedirs(args.dest)

if args.gauss:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tr = transforms.ToTensor();
    conv = torch.nn.Conv2d(1, 1, 5, padding=2, bias=False)
    with torch.no_grad():
        conv.weight.data[0][0] = torch.from_numpy(gauss2d((5,5), sigma=1.0))
    conv.to(device)

img_names = os.listdir(args.src)
for i in tqdm(range(len(img_names))):
    img = Image.open(os.path.join(args.src, img_names[i]))
    w, h = img.size
    if w != 1244 or h != 376:
        img = img.resize((1244, 376))
        img.save(os.path.join(args.src, img_names[i]))
        w, h = (1244, 376)

    if not args.fix_size_only:
        img = img.resize((w//args.factor, h//args.factor), resample=Image.BICUBIC)

        if args.gauss:
            img = tr(img).to(device)
            # batch dimension
            img = torch.unsqueeze(img, 0)
            img_R = conv(torch.unsqueeze(img[:,0,:,:], 0))
            img_G = conv(torch.unsqueeze(img[:,1,:,:], 0))
            img_B = conv(torch.unsqueeze(img[:,2,:,:], 0))
            img = torch.cat((img_R, img_G, img_B), 1)
            save_image(img, os.path.join(args.dest, img_names[i]))
        else:
            img.save(os.path.join(args.dest, img_names[i]))
