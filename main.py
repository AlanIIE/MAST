import argparse
import os
import time
from datetime import datetime
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter

import functional.feeder.dataset.YouTubeVOSTrain as Y
import functional.feeder.dataset.YTVOSTrainLoader as YL

import matplotlib.pyplot as plt
import logger

from models.mast import MAST

parser = argparse.ArgumentParser(description='MAST')

# Data options
parser.add_argument('--datapath', default='/home/kuang/DataSets/YouTube_VOS/train_all_frames/JPEGImages/',
                    help='Data path for Kinetics')
parser.add_argument('--validpath',
                    help='Data path for Davis')
parser.add_argument('--csvpath', default='functional/feeder/dataset/ytvos.csv',
                    help='Path for csv file')
parser.add_argument('--savepath', type=str, default='results/debug',
                    help='Path for checkpoints and logs')
parser.add_argument('--resume', type=str, default=None,
                    help='Checkpoint file to resume')

# Training options
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--bsize', type=int, default=12,
                    help='batch size for training (default: 12)')
parser.add_argument('--worker', type=int, default=12,
                    help='number of dataloader threads')
parser.add_argument('--mode', type=str, default='faster',
                    help='faster for cuda tensor multiply, slower for frame by frame, cpu for cpu tensor multiply')

# Debug options
parser.add_argument('--multi_scale', type=str, default=None,
                    help='None for origin setting;\n \
                        (a) for residual 1\&5 plus;\n \
                        (b) for residual 1\&5 cat;\n')
parser.add_argument('--num_long', type=int, default=1,
                    help='Long term memory')
parser.add_argument('--dil_int', type=int, default=15,
                    help='For frame interval < dil_int, no need for deformable resampling, default 15.')
parser.add_argument('--num_short', dest='ref_num', type=int, default=3,
                    help='Short term memory.')
parser.add_argument('--w_s', type=float, default=0.1,
                    help='Weight for smoothness loss.')
parser.add_argument('--ksargmax', action='store_true', dest='ksargmax', default=False,
                    help='Use kernel soft argmax.')

args = parser.parse_args()

def main():
    args.training = True

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
        
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    log = logger.setup_logger(args.savepath + '/training'+TIMESTAMP+'.log')
    writer = SummaryWriter(args.savepath + '/runs_'+TIMESTAMP)

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    dirates = np.random.choice([1,2,3,4])
    TrainData, frame_indices = Y.dataloader(args.csvpath, args.num_long, args.ref_num, args.dil_int, dirates)
    TrainImgLoader = torch.utils.data.DataLoader(
        YL.myImageFloder(args.datapath, TrainData, frame_indices, True),
        batch_size=args.bsize, shuffle=True, num_workers=args.worker,drop_last=True
    )

    model = MAST(args).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    epoch_start = -1
    if args.resume:
        if os.path.isdir(args.resume):
            checkpoint_path = args.resume
            num_file_last = -1
            for file in os.listdir(checkpoint_path):
                if not file[-2:] == 'pt': continue
                num_file = int(file[:-3].split('_')[2])
                if num_file > num_file_last:
                    args.resume = os.path.join(checkpoint_path,file)
                    num_file_last = num_file
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
            epoch_start = checkpoint['epoch']
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()
    model = nn.DataParallel(model).cuda()

    for epoch in range(epoch_start+1, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainImgLoader, model, optimizer, log, writer, epoch, [dirates for i in range(args.num_long)])

        dirates = np.random.choice([1,2,3,4])
        TrainData, frame_indices = Y.dataloader(args.csvpath, args.num_long, args.ref_num, args.dil_int, dirates)
        TrainImgLoader = torch.utils.data.DataLoader(
            YL.myImageFloder(args.datapath, TrainData, frame_indices, True),
            batch_size=args.bsize, shuffle=True, num_workers=args.worker, drop_last=True
        )

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

iteration = 0
def train(dataloader, model, optimizer, log, writer, epoch, dirates):
    global iteration
    _loss = AverageMeter()
    rec_loss = AverageMeter()
    smo_loss = AverageMeter()
    n_b = len(dataloader)
    b_s = time.perf_counter()

    for b_i, (images_lab, images_rgb_, images_quantized, ref_index) in enumerate(dataloader):
        model.train()

        adjust_lr(optimizer, epoch, b_i, n_b)

        images_lab_gt = [lab.clone().cuda() for lab in images_lab]
        images_lab = [r.cuda() for r in images_lab]
        images_rgb_ = [r.cuda() for r in images_rgb_]

        _, ch = model.module.dropout2d_lab(images_lab)
        loss, err_maps = compute_lphoto(model, images_lab, images_lab_gt, ch, ref_index, dirates)
        
        sum_loss = loss[0]+loss[1]

        sum_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        rec_loss.update(loss[0].item())
        smo_loss.update(loss[1].item())
        _loss.update(sum_loss.item())

        iteration = iteration + 1
        writer.add_scalar("Training loss", sum_loss.item(), iteration)

        info = 'Loss = {:.3f}({:.3f}), '.format(_loss.val, _loss.avg)+\
               'reconstruction loss: {:.3f}({:.3f}), '.format(rec_loss.val, rec_loss.avg) + \
               'smoothness loss: {:.3f}({:.3f}). '.format(smo_loss.val, smo_loss.avg)
        b_t = time.perf_counter() - b_s
        b_s = time.perf_counter()

        for param_group in optimizer.param_groups:
            lr_now = param_group['lr']
        # if b_i % 1 == 0:
        log.info('Epoch{} [{}/{}] {} T={:.2f}  LR={:.6f}'.format(
            epoch, b_i, n_b, info, b_t, lr_now))


        if (b_i * args.bsize) % 2000 < args.bsize:
            b = 0
            fig = plt.figure(figsize=(16,3))
            # Input
            plt.subplot(151)
            image_rgb0 = images_rgb_[0][b].cpu().permute(1, 2, 0)
            plt.imshow(image_rgb0)
            plt.title('Frame t')

            plt.subplot(152)
            image_rgb1 = images_rgb_[1][b].cpu().permute(1, 2, 0)
            plt.imshow(image_rgb1)
            plt.title('Frame t+1')

            plt.subplot(153)
            plt.imshow(torch.abs(image_rgb1 - image_rgb0))
            plt.title('Frame difference ')

            # Error map
            plt.subplot(154)
            err_map = err_maps[b]
            plt.imshow(err_map.cpu(), cmap='jet')
            plt.colorbar()
            plt.title('Error map')

            writer.add_figure('ErrorMap', fig, iteration)

        n_iter = b_i + n_b * epoch

    log.info("Saving checkpoint.")
    savefilename = args.savepath + f'/checkpoint_epoch_{epoch}.pt'
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, savefilename)

def compute_lphoto(model, image_lab, images_rgb_, ch, index, dirates):
    b, c, h, w = image_lab[0].size()

    ref_x = [lab for lab in image_lab[:-1]]   # [im1, im2, im3]
    ref_y = [rgb[:,ch] for rgb in images_rgb_[:-1]]  # [y1, y2, y3]
    tar_x = image_lab[-1]  # im4
    tar_y = images_rgb_[-1][:,ch]  # y4
    ref_index = [index[:,ind] for ind in range(index.shape[1]-1)] #list(5),each with 12 elements
    tar_index = index[:,-1]

    outputs, smoothness_loss = model(ref_x, ref_y, tar_x, ref_index, tar_index, dirates)   # only train with pairwise data

    outputs = F.interpolate(outputs, (h, w), mode='bilinear')
    loss = [F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean'), args.w_s*torch.mean(smoothness_loss)]

    err_maps = torch.abs(outputs - tar_y).sum(1).detach()

    return loss, err_maps


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_lr(optimizer, epoch, batch, n_b):
    iteration = (batch + epoch * n_b) * args.bsize

    if iteration <= 400000:
        lr = args.lr
    elif iteration <= 600000:
        lr = args.lr * 0.5
    elif iteration <= 800000:
        lr = args.lr * 0.25
    elif iteration <= 1000000:
        lr = args.lr * 0.125
    else:
        lr = args.lr * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
