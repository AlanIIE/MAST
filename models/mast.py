import torch
import torch.nn as nn

import pdb
from .submodule import ResNet18
from .colorizer import Colorizer

import numpy as np

class MAST(nn.Module):
    def __init__(self, args):
        super(MAST, self).__init__()

        # Model options
        self.p = 0.3
        self.C = 7
        self.args = args

        self.feature_extraction = ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        if self.args.multi_scale=='a' or self.args.multi_scale=='b':
            # self.post_convolution0 = nn.Conv2d(64, 64, 3, 2, 1) # new layer would cause OUTOFMEMORY in test
            self.post_convolution1 = nn.Conv2d(64, 64, 3, 2, 1)
            # self.post_convolution2 = nn.Conv2d(128, 64, 3, 1, 1)
            # self.post_convolution3 = nn.Conv2d(256, 64, 3, 1, 1)
        self.D = 4

        # Use smaller R for faster training
        if args.training:
            self.R = 6
        else:
            self.R = 12

        self.colorizer = Colorizer(self.D, self.R, self.C, args.mode, args.training, args.ksargmax)

    def forward(self, rgb_r, quantized_r, rgb_t, ref_index=None, current_ind=None, dirates=None):
        if self.args.multi_scale==None:
            feats_r = [self.post_convolution(self.feature_extraction(rgb)[4]) for rgb in rgb_r]
        elif self.args.multi_scale=='a':
            feats_r = []
            for rgb in rgb_r:
                feats_r_all = self.feature_extraction(rgb)
                feats_r.append(self.post_convolution1(feats_r_all[0])+self.post_convolution(feats_r_all[4]))
        elif self.args.multi_scale=='b':
            feats_r = []
            for rgb in rgb_r:
                feats_r_all = self.feature_extraction(rgb)
                feats_r.append(torch.cat((self.post_convolution1(feats_r_all[0]),self.post_convolution(feats_r_all[4])), 1))
        if self.args.multi_scale==None:
            feats_t = self.post_convolution(self.feature_extraction(rgb_t)[4])
        elif self.args.multi_scale=='a':
            feats_t_all = self.feature_extraction(rgb_t)
            feats_t = self.post_convolution1(feats_t_all[0])+self.post_convolution(feats_t_all[4])
        elif self.args.multi_scale=='b':
            feats_t_all = self.feature_extraction(rgb_t)
            feats_t = torch.cat((self.post_convolution1(feats_t_all[0]),self.post_convolution(feats_t_all[4])), 1)

        if self.args.training:
            quantized_t, smooth = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind, 
                                    dirates, self.args.num_long, self.args.dil_int)
            return quantized_t, smooth
        else:
            quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
            return quantized_t


    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked