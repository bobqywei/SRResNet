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
parser.add_argument("--data", default="testsets/", type=str, help="dataset name")
parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 2")
parser.add_argument("--epoch", default=500, type=int, help="training epoch")
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

model = torch.load(os.path.join("checkpoints", opt.name, "model_epoch_{0}.pth").format(opt.epoch), map_location='cpu')["model"]
imgs = glob.glob(os.path.join(opt.data, "lr", "*.png"))
output_dir = os.path.join("inference_test", opt.name, str(opt.epoch))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

avg_psnr_bicubic = 0.0
avg_psnr_sr = 0.0

for img_path in imgs:
    img_name = img_path.split("/lr/")[1]

    im_gt = Image.open(opt.data + "gt/" + img_name).convert('RGB')
    w,h = im_gt.size
    if w % 2 != 0:
        im_gt = im_gt.resize((w-1, h), resample=Image.BICUBIC)
    if h % 2 != 0:
        im_gt = im_gt.resize((w, h-1), resample=Image.BICUBIC)
    im_l = Image.open(img_path).convert('RGB')

    im_b_np = np.array(im_l.resize(im_gt.size, resample=Image.BICUBIC)).astype(float)
    im_gt_np = np.array(im_gt).astype(float)
    avg_psnr_bicubic += PSNR(im_b_np, im_gt_np)

    img_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    im_l = img_tf(im_l).unsqueeze_(0)

    if cuda:
        model = model.cuda()
        im_l = im_l.cuda()
    else:
        model = model.cpu()

    model.eval()
    out = unnormalize(model(im_l).cpu())
    utils.save_image(out, os.path.join(output_dir, img_name))

    out_np = out.data[0].numpy().transpose(1,2,0).astype(np.float32) * 255
    avg_psnr_sr += PSNR(out_np, im_gt_np)

with open(os.path.join(output_dir, "PSNR.txt"), 'a+') as file:
    file.write("PSNR Bicubic: {0}\nPSNR SRResNet: {1}".format(avg_psnr_bicubic/len(imgs), avg_psnr_sr/len(imgs)))
