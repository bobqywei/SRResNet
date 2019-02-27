import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import glob

from srresnet import _NetG
from dataset import KITTIDataset, unnormalize, MEAN, STD

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tensorboardX import SummaryWriter
from PIL import Image

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--name", type=str, help="name of current run")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
# parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--val", action="store_true")
parser.add_argument("--use_mask", action="store_true")

parser.add_argument("--save_epoch", default=50, type=int, help="number of epochs for each save")
parser.add_argument("--gt", type=str, default="data/lr_x2", help="ground truth images directory")
parser.add_argument("--mask", type=str, default="data/masks/lr_x2/level_1", help="mask images directory")
parser.add_argument("--input", type=str, default="data/lr_x4", help="input images directory")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="directory to store saved models")
parser.add_argument("--log_dir", type=str, default="tensorboard", help="tensorboard directory")
parser.add_argument("--val_in", type=str, default="data/lr_x2", help="validation input images directory")
parser.add_argument("--val_gt", type=str, default="data/rgb", help="validation ground truth images directory")
parser.add_argument("--val_mask", type=str, default="data/masks/full/level_2", help="validation mask directory")

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    tensorboard_dir = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    print("===> Loading datasets")
    train_set = KITTIDataset(opt.gt, opt.input, mask_path=opt.mask)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    print("Loaded {0} training images".format(len(train_set)))

    print("===> Building model")
    model = _NetG()
    criterion = nn.MSELoss(size_average=False)

    if opt.cuda:
        print("===> Setting GPU")
        model = model.cuda()
        criterion = criterion.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in (range(opt.start_epoch, opt.epochs + 1)):
        train(training_data_loader, optimizer, model, criterion, epoch, writer)
        if epoch % opt.save_epoch == 0:
            save_checkpoint(model, epoch, os.path.join(opt.ckpt_dir, opt.name))
            if opt.val:
                validation(os.path.join(opt.ckpt_dir, opt.name), criterion, epoch, writer)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, writer):
    lr = adjust_learning_rate(optimizer, epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        inp, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            inp = inp.cuda()
            target = target.cuda()

        output = model(inp)

        if opt.use_mask:
            mask = Variable(batch[2], requires_grad=False).cuda()
            output = output * mask
            target = target * mask

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        overall_iter = iteration * opt.batch_size + (epoch - 1) * len(training_data_loader) * opt.batch_size

        if overall_iter % (50 * opt.batch_size * len(training_data_loader)) == 0:
            out_image = unnormalize(output[0].data.cpu())
            writer.add_image("train/output", out_image, epoch)

        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(training_data_loader), loss.data.item()))
            writer.add_scalar("train/MSE", loss.data.item(), overall_iter)

def save_checkpoint(model, epoch, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_out_path = os.path.join(dir, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def validation(model_path, criterion, epoch, writer, max_len=50):
    in_paths = sorted(glob.glob(os.path.join(opt.val_in, '*.png')))
    gt_paths = sorted(glob.glob(os.path.join(opt.val_gt, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(opt.val_mask, '*.png')))

    model = torch.load(os.path.join(model_path, "model_epoch_{0}.pth").format(epoch))["model"]
    model = model.cuda()

    in_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    mask_tf = transforms.ToTensor()

    avg_mse = 0.0
    for i in range(max_len):#len(in_paths)):
        img_in = in_tf(Image.open(in_paths[i]).convert('RGB')).unsqueeze(0).cuda()
        img_gt = in_tf(Image.open(gt_paths[i]).convert('RGB')).unsqueeze(0).cuda()

        model.eval()
        out = model(img_in)
        if opt.use_mask:
            mask = mask_tf(Image.open(mask_paths[i])).unsqueeze(0).cuda()
            out = out * mask
            img_gt = img_gt * mask

        avg_mse += criterion(out, img_gt).data.cpu()

        if i == max_len-1:
            writer.add_image("validation/output_{0}".format(epoch), unnormalize(out).squeeze().data.cpu(), epoch)

    avg_mse /= max_len
    writer.add_scalar("validation/MSE", avg_mse, epoch)

if __name__ == "__main__":
    main()
