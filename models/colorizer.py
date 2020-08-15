import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import one_hot
from spatial_correlation_sampler import SpatialCorrelationSampler
from .deform_im2col_util import deform_im2col
import pdb
import numpy as np

class Colorizer(nn.Module):
    def __init__(self, D=4, R=6, C=32, mode='faster', training=False, ksargmax=True):
        super(Colorizer, self).__init__()
        self.D = D
        self.R = R  # window size
        self.C = C

        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0

        self.training = training
        self.mode = mode
        self.beta = 50
        self.kernel_sigma = 1.0
        self.ksargmax=ksargmax
        self.memory_patch_R = 12
        self.memory_patch_P = self.memory_patch_R * 2 + 1
        self.memory_patch_N = self.memory_patch_P * self.memory_patch_P

        self.correlation_sampler_dilated = [
            SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.memory_patch_P,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=dirate) for dirate in range(2,6)]

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)

    def prep(self, image, HW):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.D,::self.D]

        if c == 1 and not self.training:
            x = one_hot(x.long(), self.C)

        return x

    def get_confidence_from_flow(self, flow, thre=0.9):
        b,c,h,w = flow.shape
        flow = flow.reshape(b,c,h*w)
        X_r = torch.ones(b, 1, h*w).to(flow.device)
        rad = torch.sqrt(torch.mul(flow[:,0,:],flow[:,0,:]) + torch.mul(flow[:,1,:],flow[:,1,:]))
        _a, idx_th = torch.topk(rad, int((1-thre) * h * w))
        X_r.scatter_(2, idx_th.unsqueeze(1), 0)
        return X_r.reshape(b,1,h,w)

    def get_confidence(self, corr, thre=0.9):
        b, r, n2, hw = corr.size()
        corr = corr.reshape(b*r,n2,hw)
        X_r = torch.zeros(b*r, 1, hw).to(corr.device)
        _a, idx_th = torch.topk(torch.max(corr, dim=1, keepdim=True)[0], int(thre * hw))
        # _a, _b = torch.sort(torch.max(corr, dim=1, keepdim=True)[0], dim=2) # sort decrease default
        # idx_th = _b[:, :, range(0,int(thre * hw))]
        X_r.scatter_(2, idx_th, 1)
        return X_r

    def get_flow(self, grid):
        grid_x = grid[...,[0]].permute(0,3,1,2)
        grid_y = grid[...,[1]].permute(0,3,1,2)
        b, _, h, w = grid_x.size()
        # regular grid / [-1,1] normalized
        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w),
                                                np.linspace(-1, 1, h))  # grid_X & grid_Y : feature_H x feature_W
        grid_X = torch.tensor(grid_X, dtype=torch.float, requires_grad=False, device=grid.device)
        grid_Y = torch.tensor(grid_Y, dtype=torch.float, requires_grad=False, device=grid.device)

        grid_X = grid_X.expand(b, h, w)  # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
        grid_Y = grid_Y.expand(b, h, w)  # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)

        flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),
                            1)  # 2-channels@1st-dim, first channel for x / second channel for y

        # from functional.data import dataset
        # from functional.utils import util
        # import cv2
        # from functional.utils.track_vis import draw_track_trace, draw_trace_from_flow
        # source = torch.cat(((grid_Y+1)/2*4*h,(grid_X+1)/2*4*w),1).cpu()
        # source = F.interpolate(source, (h*4,w*4), mode='bilinear').permute(0,2,3,1)
        # target = torch.cat(((flow[:,1,...].unsqueeze(1)+grid_Y+1)/2*4*h,(flow[:,0,...].unsqueeze(1)+grid_X+1)/2*4*w),1).cpu()
        # # target = torch.cat(((grid_y+1)/2*4*h,(grid_x+1)/2*4*w),1).cpu()
        # target = F.interpolate(target, (h*4,w*4), mode='bilinear').permute(0,2,3,1)
        # frame = torch.zeros((b,3,h*4,w*4))
        # imgs_dbg = draw_track_trace(np.array(source),np.array(target),np.array(frame.permute(0,2,3,1).cpu()))
        # util.save_image_tensor2cv2(torch.tensor(imgs_dbg[0]/255).permute(2,0,1).unsqueeze(0), 'test.png')

        return flow

    def get_flow_smoothness(self, flow):
        # kernels for computing gradients
        dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float,
                                 requires_grad=False, device=flow.device).view(1, 1, 1, 3).expand(1, 2, 1, 3)
        dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float,
                                 requires_grad=False, device=flow.device).view(1, 1, 3, 1).expand(1, 2, 3, 1)

        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx)
        flow_dy = torch.abs(flow_dy)

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness


    def get_grid(self, corr, kernel=False):
        d = 1
        b, hw, h, w = corr.size()
        
        x_normal = np.linspace(-1, 1, h)
        y_normal = np.linspace(-1, 1, w)
        x_normal = torch.tensor(x_normal, dtype=torch.float, requires_grad=False, device=corr.device)
        y_normal = torch.tensor(y_normal, dtype=torch.float, requires_grad=False, device=corr.device)
        y_normal,x_normal = torch.meshgrid(x_normal,y_normal)
        x_grid = F.unfold(x_normal.reshape(1,1,h,w), kernel_size=self.P, padding =self.R)
        y_grid = F.unfold(y_normal.reshape(1,1,h,w), kernel_size=self.P, padding =self.R)
        weight = corr.reshape(b,hw,h*w).clone()
        weight = weight/torch.sum(weight,1,keepdim=True)
        x_grid = (weight * x_grid).sum(1).reshape([b,h,w,1])
        y_grid = (weight * y_grid).sum(1).reshape([b,h,w,1])
        grid = torch.cat((x_grid, y_grid), 3)
        return grid

    def kernel_soft_argmax(self, corr, dim = 0):
        b, hw, h, w = corr.size()
        kernel_size = int(np.sqrt(corr.size()[dim]))

        # apply_gaussian_kernel
        idx_x=idx_y=torch.tensor((kernel_size-1)/2, dtype=torch.float, requires_grad=False, device=corr.device).reshape(1,1,1)

        # 1-d indices for generating Gaussian kernels
        x = np.linspace(0, kernel_size - 1, kernel_size)
        x = torch.tensor(x, dtype=torch.float, requires_grad=False, device=corr.device)
        y = np.linspace(0, kernel_size - 1, kernel_size)
        y = torch.tensor(y, dtype=torch.float, requires_grad=False, device=corr.device)

        x = x.view(1, 1, kernel_size).expand(1, 1, kernel_size)
        y = y.view(1, kernel_size, 1).expand(1, kernel_size, 1)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * self.kernel_sigma ** 2))
        gauss_kernel = gauss_kernel.reshape(1,kernel_size**2,1,1).expand(b,kernel_size**2,h,w)
        corr = gauss_kernel * corr

        # softmax_with_temperature
        M, _ = corr.max(dim=dim, keepdim=True)
        corr = corr - M  # subtract maximum value for stability
        # val = torch.empty(corr.size()).cuda()
        # val.copy_(corr)
        exp_x = torch.exp(self.beta * corr)
        exp_x_sum = exp_x.sum(dim=dim, keepdim=True)
        corr = exp_x / exp_x_sum


        corr = corr.view(-1, kernel_size, kernel_size, h, w)  # (target hxw) x (source hxw)
        

        # # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        # x_normal = np.linspace(-1, 1, self.P)
        # x_normal = torch.tensor([x_normal], dtype=torch.float, requires_grad=False).cuda()
        # y_normal = np.linspace(-1, 1, self.P)
        # y_normal = torch.tensor(y_normal, dtype=torch.float, requires_grad=False).cuda()

        # grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        # x_normal = x_normal.expand(b, self.P)
        # x_normal = x_normal.view(b, self.P, 1, 1)
        # grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        # grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        # y_normal = y_normal.expand(b, self.P)
        # y_normal = y_normal.view(b, self.P, 1, 1)
        # grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        # grid = torch.cat((grid_x.permute(0, 2, 3, 1), grid_y.permute(0, 2, 3, 1)), 3)
        return corr#val
    
    def calculate_corr(self, feats_t, feats_r, searching_index, offset0):
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        if self.mode == 'cpu':
            col_0 = deform_im2col(feats_r[searching_index].cpu(), offset0.cpu(), kernel_size=self.P, mode = 'faster')  # b,c*N,h*w
            col_0 = col_0.reshape(b,c,N,h,w)
            ##
            corr = (feats_t.cpu().unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)
            return corr.to(feats_t.device)
        else:
            col_0 = deform_im2col(feats_r[searching_index], offset0, kernel_size=self.P, mode = self.mode)  # b,c*N,h*w
            torch.cuda.empty_cache()
            col_0 = col_0.reshape(b,c,N,h,w)
            ##
            if self.mode == 'faster' or self.training:
                return (feats_t.unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)
            else:
                # col_0[:,0,] = feats_t.unsqueeze(2)[:,0,]*col_0[:,0,]
                # col_0[:,1:,] = feats_t.unsqueeze(2)[:,1:,]*col_0[:,1:,]
                # corr = torch.zeros(col_0.shape).cuda()
                for batch in range(c//16):
                    col_0[:,16*batch:16*(batch+1),] = feats_t.unsqueeze(2)[:,16*batch:16*(batch+1),]*col_0[:,16*batch:16*(batch+1),]
            return col_0.sum(1)

    def forward(self, feats_r, feats_t, quantized_r, ref_index, current_ind, dirates=None, nsearch=2, dil_int = 15):
        """
        Warp y_t to y_(t+n). Using similarity computed with im (t..t+n)
        :param feats_r: f([im1, im2, im3])
        :param quantized_r: [y1, y2, y3]
        :param feats_t: f(im4)
        :param mode:
        :return:
        """
        # For frame interval < dil_int, no need for deformable resampling
        nref = len(feats_r)
        if self.training == False:
            nsearch = len([x for x in ref_index if current_ind - x > dil_int])

        # The maximum dilation rate is 4
        if dirates == None:
            dirates = [ min(4, (current_ind - x) // dil_int +1) for x in ref_index if current_ind - x > dil_int]
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        corrs = []

        offsets = []
        for searching_index in range(nsearch):
            ##### GET OFFSET HERE.  (b,h,w,2)
            samplerindex = dirates[searching_index]-2
            coarse_search_correlation = self.correlation_sampler_dilated[samplerindex](feats_t, feats_r[searching_index])  # b, p, p, h, w
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_N, h*w)
            coarse_search_correlation = F.softmax(coarse_search_correlation, dim=1)
            coarse_search_correlation = coarse_search_correlation.reshape(b,self.memory_patch_P,self.memory_patch_P,h,w,1)
            _y, _x = torch.meshgrid(torch.arange(-self.memory_patch_R,self.memory_patch_R+1),torch.arange(-self.memory_patch_R,self.memory_patch_R+1))
            _grid = torch.stack([_x, _y], dim=-1).unsqueeze(-2).unsqueeze(-2)\
                .reshape(1,self.memory_patch_P,self.memory_patch_P,1,1,2).contiguous().float().to(coarse_search_correlation.device)
            offset0 = (coarse_search_correlation * _grid ).sum(1).sum(1) * dirates[searching_index]  # 1,h,w,2

            corr = self.calculate_corr(feats_t, feats_r, searching_index, offset0)

            corr = corr.reshape([b, self.P * self.P, h * w]) 
            corrs.append(corr)
            offsets.append(offset0)

        for ind in range(nsearch, nref):
            corrs.append(self.correlation_sampler(feats_t, feats_r[ind]))
            _, _, _, h1, w1 = corrs[-1].size()
            corrs[ind] = corrs[ind].reshape([b, self.P*self.P, h1*w1])

        if self.ksargmax:
            corr = torch.cat(corrs, 0)  # b*nref,N,HW
            corr = self.kernel_soft_argmax(corr.reshape(-1,self.P*self.P,h,w), dim = 1) # b, hw, h, w = corr.size()
            corr = corr.reshape(b,1,self.N*nref,h*w)
        else:        
            corr = torch.cat(corrs, 1)  # b,nref*N,HW
            corr = F.softmax(corr, dim=1)
            corr = corr.unsqueeze(1)

        grid = self.get_grid(corr.reshape(-1,self.P*self.P,h,w))

        qr = [self.prep(qr, (h,w)) for qr in quantized_r]

        im_col0 = [deform_im2col(qr[i], offsets[i], kernel_size=self.P, mode = 'cpu')  for i in range(nsearch)]# b,3*N,h*w
        im_col1 = [F.unfold(r, kernel_size=self.P, padding =self.R) for r in qr[nsearch:]]
        image_uf = im_col0 + im_col1

        image_uf = [uf.reshape([b,qr[0].size(1),self.P*self.P,h*w]) for uf in image_uf]
        image_uf = torch.cat(image_uf, 2)


        if  self.training:
            out = (corr * image_uf).sum(2).reshape([b,qr[0].size(1),h,w])
            tgt_gate = self.get_confidence(corr.reshape(b, nref, self.P*self.P, h*w),thre=0.5).reshape(b,nref,h,w) # target fg mask

            # _y, _x = torch.meshgrid(torch.arange(0,h),torch.arange(0,w))
            # _grid = torch.stack([_x, _y], dim=-1).unsqueeze(0).repeat(b*nref,1,1,1).float().to(grid.device)
            # grid_idx = F.grid_sample(_grid.permute(0,3,1,2), grid, mode='bilinear').type(torch.long)
            # warp_grid = torch.zeros(b*nref,h,w,2)
            # for h_t in range(h):
            #     for w_t in range(w):
            #         warp_grid[range(b*nref),grid_idx[:,0,h_t,w_t],grid_idx[:,1,h_t,w_t],:] = torch.tensor([h_t,w_t], dtype=torch.float)
            # warp_grid = warp_grid.to(tgt_gate.device)
            # src_gate = F.grid_sample(tgt_gate, warp_grid/warp_grid.max()*2-1, mode='bilinear')

            flow = self.get_flow(grid) * (1-tgt_gate) # src_gate # times source fg mask
            smoothness = self.get_flow_smoothness(flow)

            # cycle flow within pariwised data
            warp_flow = F.grid_sample(flow, grid, mode='bilinear') * (1-tgt_gate)# times target fg mask
            cycle = [warp_flow, flow]

            return [out, smoothness, cycle]
        elif self.mode == 'cpu' or self.mode == 'faster':
            out = (corr * image_uf).sum(2).reshape([b,qr[0].size(1),h,w])
 
            return out, grid
        else:
            if corr.shape[1] == 1:
                for batch in range(image_uf.shape[1]):
                    image_uf[:,batch,:,:] = image_uf[:,batch,:,:]*corr
            else:
                for batch in range(image_uf.shape[1]):
                    image_uf[:,batch,:,:] = image_uf[:,batch,:,:]*corr[:,batch,:,:]
            # return image_uf.sum(2).reshape([b,qr[0].size(1),h,w]), grid
            # tgt_gate = self.get_confidence(corr.reshape(b, nref, self.P*self.P, h*w),thre=0.9).reshape(b*nref,h,w).unsqueeze(1)
            flow = self.get_flow(grid)
            tgt_gate = self.get_confidence_from_flow(flow,0.9)
            flow = self.get_flow(grid) * (tgt_gate)
            return image_uf.sum(2).reshape([b,qr[0].size(1),h,w]), flow
            
            # from functional.utils.flow_vis import flow_to_color
            # tmp = flow_to_color(np.array(flow.permute(0,2,3,1).squeeze(0).cpu()))
            # from functional.utils.track_vis import draw_track_trace, draw_trace_from_flow
            # calc_flow = flow*100
            # frame = draw_trace_from_flow(np.array(calc_flow.permute(0,2,3,1).long().cpu()))
            # return image_uf.sum(2).reshape([b,qr[0].size(1),h,w]), tmp
            # _y, _x = torch.meshgrid(torch.arange(0,h),torch.arange(0,w))
            # _grid = torch.stack([_y, _x], dim=-1).unsqueeze(0).repeat(b*nref,1,1,1).float().to(grid.device)
            # grid_idx = F.grid_sample(_grid.permute(0,3,1,2), grid, mode='bilinear').type(torch.long)
            # warp_grid = torch.zeros(b*nref,h,w,2)
            # for h_t in range(h):
            #     for w_t in range(w):
            #         warp_grid[range(b*nref),grid_idx[:,0,h_t,w_t],grid_idx[:,1,h_t,w_t],:] = torch.tensor([h_t,w_t], dtype=torch.float)
            # warp_grid = warp_grid.to(tgt_gate.device)
            # src_gate = F.grid_sample(tgt_gate.reshape(b*nref,1,h,w), warp_grid/warp_grid.max()*2-1, mode='bilinear').reshape(b,nref,h,w)
            # return image_uf.sum(2).reshape([b,qr[0].size(1),h,w]), src_gate#, grid

def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)
