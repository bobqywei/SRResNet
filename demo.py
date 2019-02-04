import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_194.pth", type=str, help="model path")
parser.add_argument("--data", default="testsets/", type=str, help="dataset name")
parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 2")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location='cpu')["model"]
im_gt = Image.open(opt.data + "gt/" + "0000000005.png").convert('RGB')
im_l = Image.open(opt.data + "lr/" + "0000000005.png").convert('RGB')

# w,h = im_gt.size
# if w % 2 != 0:
#     im_gt = im_gt.resize((w-1, h), resample=Image.BICUBIC)
# if h % 2 != 0:
#     im_gt = im_gt.resize((w, h-1), resample=Image.BICUBIC)

img_tf = transforms.ToTensor()
im_l = img_tf(im_l)
im_l.unsqueeze_(0)

if cuda:
    model = model.cuda()
    im_l = im_l.cuda()
else:
    model = model.cpu()
    
# start_time = time.time()
model.eval()
out = model(im_l)
# elapsed_time = time.time() - start_time

out = out.cpu()
im_h = out.data[0].numpy().astype(np.float32)

im_h = im_h*255.
im_h[im_h<0] = 0
im_h[im_h>255.] = 255.            
im_h = im_h.transpose(1,2,0)

im_gt = np.array(im_gt)

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_h.astype(np.uint8))
ax.set_title("Output(SRResNet)")
plt.show()
