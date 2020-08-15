r"""Helper functions"""

import logging
import re

import torchvision.transforms.functional as ff
import torch.nn.functional as F
import torch
from PIL import Image
import cv2

from functional.utils import geometry
from functional.data import dataset

unnorm = dataset.UnNormalize()


def init_logger(logfile):
    r"""Initialize logging settings"""
    logging.basicConfig(filemode='w',
                        filename=logfile,
                        level=logging.INFO,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M:%S')

    # Configuration on console logs
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_info(in_str):
    logging.info(in_str)

def log_args(args):
    r"""Log program arguments"""
    logging.info('\n+========== Hyperpixel Flow Arguments ===========+')
    for arg_key in args.__dict__:
        logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    logging.info('+================================================+\n')


def resize(img, kps, side_thres=300):
    r"""Resize given image with imsize: (1, 3, H, W)"""
    imsize = torch.tensor(img.size()).float()
    kps = kps.float()
    side_max = torch.max(imsize)
    inter_ratio = 1.0
    if side_max > side_thres:
        inter_ratio = side_thres / side_max
        img = F.interpolate(img,
                            size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
                            mode='bilinear',
                            align_corners=False)
        kps *= inter_ratio
    return img.squeeze(0), kps, inter_ratio


def where(predicate):
    r"""Returns indices which match given predicate"""
    matching_idx = predicate.nonzero()
    n_match = len(matching_idx)
    if n_match != 0:
        matching_idx = matching_idx.t().squeeze(0)
    return matching_idx


def intersect1d(tensor1, tensor2):
    r"""Takes two 1D tensor and returns tensor of common values"""
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def parse_hyperpixel(hyperpixel_ids):
    r"""Parse given hyperpixel list (string -> int)"""
    return list(map(int, re.findall(r'\d+', hyperpixel_ids)))


def visualize_prediction(src_kps, prd_kps, src_img, trg_img, vispath, relaxation=2000):
    r"""TPS transform source image using predicted correspondences"""
    src_imsize = src_img.size()[1:][::-1]
    trg_imsize = trg_img.size()[1:][::-1]

    img_tps = geometry.ImageTPS(src_kps, prd_kps, src_imsize, trg_imsize, relaxation)
    wrp_img = ff.to_pil_image(img_tps(unnorm(src_img.cpu())))
    trg_img = ff.to_pil_image(unnorm(trg_img.cpu()))

    new_im = Image.new('RGB', (trg_imsize[0] * 2, trg_imsize[1]))
    new_im.paste(wrp_img, (0, 0))
    new_im.paste(trg_img, (trg_imsize[0], 0))
    new_im.save(vispath)

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename, cvt=cv2.COLOR_RGB2BGR):
    r"""
    save tensor as cv2
    :param input_tensor: tensor (b, c, h, w) to save 
    :param filename: target filename
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # copy
    tmp_tensor = input_tensor.clone().detach().to(torch.device('cpu'))
    # b c h w -> h*b1 w*b2 c
    b,c,h,w = tmp_tensor.shape
    tmp_tensor = tmp_tensor.permute(0,2,3,1).contiguous().reshape(h,w,c)
    # [0,1] -> [0,255],-> cv2
    tmp_tensor = tmp_tensor.mul_(255).clamp_(0, 255).type(torch.uint8).numpy()
    # RGB -> BRG
    tmp_tensor = cv2.cvtColor(tmp_tensor, cvt)
    cv2.imwrite(filename, tmp_tensor)