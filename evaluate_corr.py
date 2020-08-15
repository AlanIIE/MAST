r"""Runs Hyperpixel Flow framework"""

import argparse
import datetime
import os

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import cv2


from models.mast import MAST
from functional.utils import evaluation, util
from functional.data import download

os.environ["CUDA_VISIBLE_DEVICES"]=''
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
MISSING_VALUE = 1 # label for outliers 

def run(datapath, benchmark, thres, alpha,
        logpath, beamsearch, model=None, dataloader=None, visualize=False):
    r"""Runs Hyperpixel Flow framework"""
    args.training = False

    # 1. Logging initialization
    if not os.path.isdir(logpath):
        os.makedirs(logpath)
    if not beamsearch:
        cur_datetime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logfile = os.path.join(logpath, 'run' + cur_datetime + '.log')
        util.init_logger(logfile)
        util.log_args(args)
        if visualize: os.mkdir(os.path.join(logpath, 'run' + cur_datetime + 'vis'))

    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataloader is None:
        download.download_dataset(os.path.abspath(datapath), benchmark)
        split = 'val' if beamsearch else 'test'
        dset = download.load_dataset(benchmark, datapath, thres, device, split)
        dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 3. Model initialization
    args.training = False
    model = MAST(args)
    
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
            util.log_info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))
            model.load_state_dict(checkpoint['state_dict'])
            util.log_info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            util.log_info("=> No checkpoint found at '{}'".format(args.resume))
            util.log_info("=> Will start from scratch.")
    else:
        util.log_info('=> No checkpoint file. Start from scratch.')
    model = nn.DataParallel(model).to(device)
    model.eval()
    torch.backends.cudnn.benchmark = True

    # 4. Evaluator initialization
    evaluator = evaluation.Evaluator(benchmark, device)

    for idx, data in enumerate(dataloader):

        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
        data['alpha'] = alpha

        ref_index = [0]

        c0, h0, w0 = data['src_img'].shape
        rgb_0 = [data['src_img'].unsqueeze(0)]
        c1, h1, w1 = data['trg_img'].shape
        rgb_1 = data['trg_img'].unsqueeze(0)
        # rgb_1 = F.interpolate(data['trg_img'].unsqueeze(0), (h0,w0), mode='bilinear')

        c0 = data['src_kps'].shape[1]
        anno_0 = cords_to_map(np.array(data['src_kps'][[1,0]].cpu()), [h0, w0]).transpose(2,0,1)
        anno_0 = [torch.tensor(anno_0.reshape(1, c0, h0, w0), dtype=torch.float, device=device)]
        c1 = data['trg_kps'].shape[1]
        anno_1 = cords_to_map(np.array(data['trg_kps'][[1,0]].cpu()), [h1, w1]).transpose(2,0,1)
        anno_1 = torch.tensor(anno_1.reshape(1, c1, h1, w1), dtype=torch.float, device=device)

        # b) Feed a pair of images to Hyperpixel Flow model
        with torch.no_grad():
            # pred_anno_1, warp_grid = model(rgb_0, anno_0, rgb_1, ref_index, 16) # try to change the parameter 16
            # pred_anno_1, warp_grid = model(rgb_0, anno_0, rgb_1, ref_index, 32) 
            # pred_anno_1, warp_grid = model(rgb_0, anno_0, rgb_1, ref_index, 48) 
            pred_anno_1, warp_grid = model(rgb_0, anno_0, 
                                            F.interpolate(data['trg_img'].unsqueeze(0), (h0,w0), mode='bilinear'), 
                                            ref_index, 1) 
            # pred_anno_1, warp_grid = model(rgb_0, anno_0, rgb_0[0], ref_index, 16)

            pred_anno_1 = F.interpolate(pred_anno_1, (h1,w1), mode='bilinear')

            # output = torch.argmax(anno_1, 1, keepdim=True).float()

        # from functional.data import dataset
        # util.save_image_tensor2cv2(dataset.UnNormalize()(rgb_0[0]), 'test.png', cv2.COLOR_LAB2RGB)
        # util.save_image_tensor2cv2(torch.sum(output, 1,keepdim=True), 'test.png', cv2.COLOR_GRAY2RGB)
        # util.save_image_tensor2cv2(torch.sum(anno_1, 1,keepdim=True), 'test.png', cv2.COLOR_GRAY2RGB)
        # util.save_image_tensor2cv2(torch.sum(pred_anno_1, 1,keepdim=True), 'test.png', cv2.COLOR_GRAY2RGB)
        # util.save_image_tensor2cv2(anno_1[:,1,...].unsqueeze(1), 'test.png', cv2.COLOR_GRAY2RGB)
        # util.save_image_tensor2cv2(dataset.UnNormalize()(data['trg_img'].unsqueeze(0)), 'test.png', cv2.COLOR_Lab2BGR)
        # util.save_image_tensor2cv2(dataset.UnNormalize()(rgb_1), 'test.png', cv2.COLOR_Lab2BGR)

        # c) Predict key-points & evaluate performance
            # warp_grid = F.interpolate(warp_grid.permute(0,3,1,2), (h1,w1), mode='bilinear')
            # pred_anno_1 = F.grid_sample(anno_0[0].float(), warp_grid.permute(0,2,3,1), mode='bilinear')
            prd_kps = map_to_cord(np.array(pred_anno_1.permute(0,2,3,1).squeeze().cpu()), lbl_set=data['trg_kps'].shape[-1], threshold=data['alpha'])
            prd_kps = torch.tensor(prd_kps.T,dtype=torch.float, device=device)

        evaluator.evaluate(prd_kps[[1,0]], data)

        # d) Log results
        if not beamsearch:
            evaluator.log_result(idx, data=data)
        if visualize:
            vispath = os.path.join(logpath, 'run' + cur_datetime + 'vis', '%03d_%s_%s' % (idx, data['src_imname'][0], data['trg_imname'][0]))
            util.visualize_prediction(data['src_kps'].transpose(0,1).cpu(), prd_kps.transpose(0,1).cpu(),
                                      data['src_img'], data['trg_img'], vispath)
    if beamsearch:
        return (sum(evaluator.eval_buf['pck']) / len(evaluator.eval_buf['pck'])) * 100.
    else:
        evaluator.log_result(len(dset), data=None, average=True)

# draw pose from https://github.com/AlanIIE/PCGAN/blob/master/pose_utils.py
def cords_to_map(cords, img_size, sigma=6):
    # cords: [kps,nch]
    # img_size: [h,w]
    cords = cords.T
    result = np.zeros(img_size + np.array(cords.shape[0:1]).tolist(), dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result
    # return np.transpose(result,[1, 0, 2])

def map_to_cord(pose_map, lbl_set=15, threshold=0.1):
    all_peaks = [[] for i in range(lbl_set)]
    pose_map = pose_map[..., :lbl_set]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)), pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(lbl_set):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename, cvt=cv2.COLOR_RGB2BGR):
    """
    save tensor as cv2
    :param input_tensor: tensor (b, c, h, w) to save 
    :param filename: target filename
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # copy
    tmp_tensor = input_tensor.clone().detach().to(torch.device('cpu'))
    # tmp_tensor = dataset.UnNormalize()(tmp_tensor)
    # b c h w -> h*b1 w*b2 c
    b,c,h,w = tmp_tensor.shape
    # b1 = int(np.sqrt(b))
    # b2 = b//b1 + 1 if b % b1 else b//b1
    # tmp_tensor = torch.cat([tmp_tensor,torch.zeros(b1*b2-b,c,h,w)], 0)
    # tmp_tensor = tmp_tensor.permute(1,0,2,3).contiguous().reshape(h*b1,w*b2,c)
    tmp_tensor = tmp_tensor.permute(0,2,3,1).contiguous().reshape(h,w,c)
    # [0,1] -> [0,255],-> cv2
    tmp_tensor = tmp_tensor.mul_(255).clamp_(0, 255).type(torch.uint8).numpy()
    # RGB -> BRG
    tmp_tensor = cv2.cvtColor(tmp_tensor, cvt)
    cv2.imwrite(filename, tmp_tensor)

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Hyperpixel Flow in pytorch')
    parser.add_argument('--datapath', type=str, default='../Datasets_HPF')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='./log')
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--mode', type=str, default='faster',
                        help='faster for cuda tensor multiply, slower for frame by frame, cpu for cpu tensor multiply')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    parser.add_argument('--multi_scale', type=str, default=None,
                        help='None for origin setting;\n \
                        (a) for residual 1\&5 plus;\n \
                        (b) for residual 1\&5 cat;\n')
    parser.add_argument('--ksargmax', action='store_true', dest='ksargmax', default=False,
                        help='Use kernel soft argmax.')
                        
    args = parser.parse_args()

    run(datapath=args.datapath, benchmark=args.dataset, thres=args.thres, alpha=args.alpha,
        logpath=args.logpath, beamsearch=False, visualize=args.visualize)
