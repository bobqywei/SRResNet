import argparse, os, glob
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import time, math
import matplotlib.pyplot as plt

from torch.autograd import Variable
from PIL import Image
from dataset import unnormalize, MEAN, STD
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--name", type=str, help="training run name")
parser.add_argument("--gt", default="testsets/gt", type=str)
parser.add_argument("--lr", default="testsets/lr", type=str)
parser.add_argument("--mask", default="testsets/masks", type=str)
# parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 2")
parser.add_argument("--epoch", default=500, type=int, help="training epoch")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--dest", default="inference_on_train", type=str)

def PSNR(pred, gt):
    diff = pred - gt
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

model = torch.load(os.path.join("checkpoints", opt.name, "model_epoch_{0}.pth").format(opt.epoch), map_location='cpu')["model"]
img_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
mask_tf = transforms.ToTensor()

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = model.cuda()

imgs = glob.glob(os.path.join(opt.lr, "*.png"))
output_dir = os.path.join(opt.dest, opt.name, str(opt.epoch))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

avg_psnr_bicubic = 0.0
avg_psnr_sr = 0.0
skipped = 0

for i in tqdm(range(len(imgs))):
    img_name = imgs[i].split('/')[-1]

    im_gt = Image.open(os.path.join(opt.gt, img_name)).convert('RGB')
    mask = np.array(Image.open(os.path.join(opt.mask, img_name)))
    im_lr = Image.open(imgs[i]).convert('RGB')
    im_b = np.array(im_lr.resize(im_gt.size, resample=Image.BICUBIC)).astype(float)
    im_gt = np.array(im_gt).astype(float)

    mask = mask != 0
    mask = np.transpose(np.tile(mask, (3,1,1)), (1,2,0))
    if np.sum(mask) == 0:
        skipped += 1
        continue

    im_b = im_b[mask]
    im_gt = im_gt[mask]
    avg_psnr_bicubic += PSNR(im_b, im_gt)

    im_lr = img_tf(im_lr).unsqueeze_(0)
    if cuda:
        im_lr = im_lr.cuda()

    model.eval()
    out = unnormalize(model(im_lr)).cpu()
    utils.save_image(out, os.path.join(output_dir, img_name))

    out_np = out.data[0].numpy().transpose(1,2,0).astype(float) * 255.0
    out_np = out_np[mask]
    avg_psnr_sr += PSNR(out_np, im_gt)

with open(os.path.join(output_dir, "PSNR.txt"), 'a+') as file:
    n = len(imgs) - skipped
    file.write("PSNR Bicubic: {0}\nPSNR SRResNet: {1}".format(avg_psnr_bicubic / n, avg_psnr_sr / n))
