import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np

import functional.feeder.dataset.Davis2017 as D
import functional.feeder.dataset.DavisLoaderLab as DL
from functional.utils.f_boundary import db_eval_boundary
from functional.utils.jaccard import db_eval_iou
from models.mast import MAST
from functional.utils.io import imwrite_indexed

import logger

def main():
    args.training = False

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData = D.dataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        DL.myImageFloder(TrainData[0], TrainData[1], False),
        batch_size=1, shuffle=False,num_workers=0,drop_last=False
    )

    model = MAST(args)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

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
            checkpoint = torch.load(args.resume) # , map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')
    model = nn.DataParallel(model).cuda()

    start_full_time = time.time()

    test(TrainImgLoader, model, log)

    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def test(dataloader, model, log):
    model.eval()

    torch.backends.cudnn.benchmark = True

    Fs = AverageMeter()
    Js = AverageMeter()

    n_b = len(dataloader)

    log.info("Start testing.")
    for b_i, (images_rgb, annotations) in enumerate(dataloader):
        # if b_i<=27: continue
        fb = AverageMeter(); jb = AverageMeter()

        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]

        N = len(images_rgb)
        outputs = [annotations[0].contiguous()]

        for i in range(N-1):
            mem_gap = 2
            # ref_index = [i]
            if args.ref == 0:
                ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(filter(lambda x:x>0,range(i,i-mem_gap*3,-mem_gap)))[::-1]
                ref_index = sorted(list(set(ref_index)))
            elif args.ref == 1:
                ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
            elif args.ref == 2:
                ref_index = [i]
            else:
                raise NotImplementedError

            rgb_0 = [images_rgb[ind] for ind in ref_index]
            rgb_1 = images_rgb[i+1]

            anno_0 = [outputs[ind] for ind in ref_index]
            anno_1 = annotations[i+1]

            _, _, h, w = anno_0[0].size()

            max_class = anno_1.max()

            with torch.no_grad():
                _output, _ = model(rgb_0, anno_0, rgb_1, ref_index, i+1)
                # _output, _ = model(rgb_0, anno_0, rgb_0[0], ref_index, i+1)
                # rgb_0:    list(5),[1,3,480,910]
                # anno_0:   list(5),[1,1,480,910]
                # rgb_1:    tensor, [1,3,480,910]


                # warp_grid = F.interpolate(_.permute(0,3,1,2), (h,w), mode='bilinear')
                # _output = F.grid_sample(torch.cat(anno_0,0).float(), warp_grid.permute(0,2,3,1), mode='bilinear')
                # output = F.interpolate(torch.mean(_output,0,keepdim=True), (h,w), mode='bilinear')
                # outputs.append(torch.round(output).long())

                _output = F.interpolate(_output, (h,w), mode='bilinear')
                output = torch.argmax(_output, 1, keepdim=True).float()
                outputs.append(output)

                if i > 15:
                    # draw flow from grid with cv2.line
                    from functional.data import dataset
                    from functional.utils import util
                    import cv2
                    warp_grid = F.interpolate(_, (h,w), mode='bilinear')
                    # warp_grid = F.interpolate(_.permute(0,3,1,2), (h,w), mode='bilinear')

                    # draw gate
                    # util.save_image_tensor2cv2(warp_grid[0,...].unsqueeze(0), 'test.png', cv2.COLOR_GRAY2RGB)

                    # draw grid or flow
                    grid_x = warp_grid.permute(0,2,3,1)[...,[0]]
                    grid_y = warp_grid.permute(0,2,3,1)[...,[1]]
                    b=warp_grid.shape[0]
                    # regular grid / [-1,1] normalized
                    grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w),
                                                            np.linspace(-1, 1, h))  # grid_X & grid_Y : feature_H x feature_W
                    grid_X = torch.tensor(grid_X, dtype=torch.float, requires_grad=False, device=warp_grid.device)
                    grid_Y = torch.tensor(grid_Y, dtype=torch.float, requires_grad=False, device=warp_grid.device)

                    grid_X = grid_X.expand(b, h, w)  # x coordinates of a regular grid
                    grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
                    grid_Y = grid_Y.expand(b, h, w)  # y coordinates of a regular grid
                    grid_Y = grid_Y.unsqueeze(1)

                    from functional.utils.track_vis import draw_track_trace, draw_trace_from_flow
                    # draw pic from grid
                    # source = torch.cat(((grid_Y.permute(0,2,3,1)+1)/2*h,(grid_X.permute(0,2,3,1)+1)/2*w),3).cpu()
                    # target = torch.cat(((grid_y+1)/2*h,(grid_x+1)/2*w),3).cpu()
                    # frame = (torch.cat(rgb_0,0)+1)/2*255
                    # imgs_dbg = draw_track_trace(np.array(source),np.array(target),np.array(frame.permute(0,2,3,1).cpu()))
                    # util.save_image_tensor2cv2(torch.tensor(imgs_dbg[0]/255).permute(2,0,1).unsqueeze(0), 'test1.png', cv2.COLOR_Lab2BGR)
                    # util.save_image_tensor2cv2((rgb_1+1)/2, 'test2.png', cv2.COLOR_Lab2BGR)

                    # draw pic from flow
                    source = torch.cat(((grid_Y+1)/2*h,(grid_X+1)/2*w),1).permute(0,2,3,1).cpu()
                    target = torch.cat(((warp_grid[:,1,...].unsqueeze(1)+grid_Y+1)/2*h,
                                        (warp_grid[:,0,...].unsqueeze(1)+grid_X+1)/2*w),1).cpu()
                    target = target.permute(0,2,3,1)
                    # frame = torch.zeros((b,3,h,w))
                    frame = (torch.cat(rgb_0,0)+1)/2*255
                    imgs_dbg = draw_track_trace(np.array(source),np.array(target),np.array(frame.permute(0,2,3,1).cpu()))
                    util.save_image_tensor2cv2(torch.tensor(imgs_dbg[0]/255).permute(2,0,1).unsqueeze(0), 'test.png')


                    # draw dense flow map
                    from functional.utils.flow_vis import flow_to_color
                    tmp = flow_to_color(np.array((source[0,...]-target[0,...]).cpu()))
                    util.save_image_tensor2cv2(torch.tensor(tmp).permute(2,0,1).unsqueeze(0), 'test.png')

                # warp_grid = F.interpolate(_.permute(0,3,1,2), (h,w), mode='bilinear')
                # pred_anno_1 = F.grid_sample(torch.cat(rgb_0,0), warp_grid.permute(0,2,3,1), mode='bilinear')

                # util.save_image_tensor2cv2((rgb_0[0]+1)/2, 'test.png', cv2.COLOR_LAB2BGR)
                # util.save_image_tensor2cv2((_output+1)/2, 'test.png', cv2.COLOR_LAB2BGR)
                # util.save_image_tensor2cv2(torch.sum(anno_0[0], 1,keepdim=True), 'test.png', cv2.COLOR_GRAY2RGB)
                
                # util.save_image_tensor2cv2(anno_1[:,1,...].unsqueeze(1), 'test.png', cv2.COLOR_GRAY2RGB)
                # util.save_image_tensor2cv2(_[:,0,...].unsqueeze(1), 'test.png', cv2.COLOR_GRAY2RGB)
                # util.save_image_tensor2cv2(torch.tensor(_).permute(2,0,1).unsqueeze(0), 'test.png', cv2.COLOR_BGR2RGB)
                # util.save_image_tensor2cv2(torch.tensor((frame[0]/frame[0].max()).transpose(0,3,1,2)), 'test.png', cv2.COLOR_GRAY2RGB)
                # util.save_image_tensor2cv2(torch.tensor((frame/frame[0].max()).transpose(0,3,1,2)), 'test.png', cv2.COLOR_GRAY2RGB)
                # util.save_image_tensor2cv2(torch.tensor((mask/mask[0].max()).transpose(0,3,1,2)), 'test.png', cv2.COLOR_GRAY2RGB)
                # util.save_image_tensor2cv2(torch.tensor(tmp).permute(2,0,1).unsqueeze(0), 'test.png')
                # util.save_image_tensor2cv2(_[1,...].unsqueeze(0), 'test.png', cv2.COLOR_GRAY2RGB)
                # util.save_image_tensor2cv2((pred_anno_1[0,...].unsqueeze(0)+1)/2, 'test.png', cv2.COLOR_Lab2BGR)
                # util.save_image_tensor2cv2(dataset.UnNormalize()(rgb_0[0]), 'test.png', cv2.COLOR_Lab2BGR)
                # util.save_image_tensor2cv2((rgb_1+1)/2, 'test.png', cv2.COLOR_Lab2BGR)

            js, fs = [], []

            for classid in range(1, max_class + 1):
                obj_true = (anno_1 == classid).cpu().numpy()[0, 0]
                obj_pred = (output == classid).cpu().numpy()[0, 0]

                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)

                fs.append(f); js.append(j)
                Fs.update(f); Js.update(j)

            ###
            folder = os.path.join(args.savepath,'benchmark')
            if not os.path.exists(folder): os.mkdir(folder)

            output_folder = os.path.join(folder, D.catnames[b_i].strip())

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad =  ((0,0), (0,0))
            if i == 0:
                # output first mask
                output_file = os.path.join(output_folder, '%s.png' % str(0).zfill(5))
                out_img = anno_0[0][0, 0].cpu().numpy().astype(np.uint8)

                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                imwrite_indexed(output_file, out_img )

            output_file = os.path.join(output_folder, '%s.png' % str(i + 1).zfill(5))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img)

        info = '\t'.join(['Js: ({:.3f}). Fs: ({:.3f}).'
                          .format(Js.avg, Fs.avg)])

        log.info('[{}/{}] {}'.format( b_i, n_b, info ))

    return Js.avg


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAST')

    # Data options
    parser.add_argument('--ref', type=int, default=0)
    parser.add_argument('--mode', type=str, default='faster',
                        help='faster for cuda tensor multiply, slower for frame by frame, cpu for cpu tensor multiply')

    parser.add_argument('--datapath', help='Data path for Davis')
    parser.add_argument('--savepath', type=str, default='results/test',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    
    # Debug option
    parser.add_argument('--multi_scale', type=str, default=None,
                        help='None for origin setting;\n \
                        (a) for residual 1\&5 plus;\n \
                        (b) for residual 1\&5 cat;\n')
    parser.add_argument('--ksargmax', action='store_true', dest='ksargmax', default=False,
                    help='Use kernel soft argmax.')

    args = parser.parse_args()

    main()
