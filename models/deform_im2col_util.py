import os
import torch
import torch.nn.functional as F

def image_meshgrid_from(x):
    # input: b,c,h,w
    # output: b,c,h,2
    shape = x.shape  # assume b,c,h,w
    _y, _x = torch.meshgrid(torch.arange(shape[2]), torch.arange(shape[3]))
    grid = torch.stack([_x, _y], dim=-1)
    return torch.stack([grid] * shape[0], dim=0).type(x.type()).to(x.device)


def normalize_meshgrid(grid):
    # normalize wrt to image size
    # input: b,h,w,2
    # output: b,h,w,2 (range = [-1,1])
    grid_new = torch.zeros_like(grid)
    b, h, w, _ = grid.shape
    grid_new[..., 0] = grid[..., 0] / (w - 1) * 2 - 1
    grid_new[..., 1] = grid[..., 1] / (h - 1) * 2 - 1
    return grid_new

def deform_im2col(im, offset, kernel_size=3, mode='faster'):
    # Faster on gpu, slower on CPU
    # input: b,c,h,w
    # output: b,N*c,h*w
    if mode == 'faster' or mode == 'cpu':
        with torch.no_grad():
            grid = image_meshgrid_from(im)
            b, c, h, w = im.shape

        N = kernel_size * kernel_size

        grid_ = torch.zeros(b * N, h, w, 2,  device=im.device).contiguous()
        im_ = im.repeat(N, 1, 1, 1)

        for dy in range(kernel_size):
            for dx in range(kernel_size):
                grid_[(dy * kernel_size + dx) * b:(dy * kernel_size + dx + 1) * b] =\
                    grid + offset + torch.tensor([dx - kernel_size // 2, dy - kernel_size // 2])[None, None, None, :].float().to(im.device)

        out = F.grid_sample(im_.contiguous(), normalize_meshgrid(grid_).contiguous())
        out = out.reshape(N, b, c, h * w).permute(1,2,0,3)

        return out.reshape(b, kernel_size * kernel_size * c, h * w)
    else:
        with torch.no_grad():
            grid = image_meshgrid_from(im)
            b, c, h, w = im.shape

        N = kernel_size * kernel_size

        out = torch.zeros(b * N, c, h, w,  device=im.device).contiguous()

        for dy in range(kernel_size):
            for dx in range(kernel_size):
                grid_ = grid+offset + torch.tensor([dx - kernel_size // 2, dy - kernel_size // 2])[None, None, None, :].float().to(im.device)
                out[(dy * kernel_size + dx) * b:(dy * kernel_size + dx + 1) * b] = \
                    F.grid_sample(im.contiguous(), normalize_meshgrid(grid_).contiguous())

        out = out.reshape(b, N, c, h * w)
        for ind_b in range(b):
            # tmp = out[ind_b,].transpose(0,1).contiguous()
            # out[ind_b,] = tmp.reshape(N,c,h*w)

            # tmp = out[ind_b,:,:,0].transpose(0,1).contiguous()
            # out[ind_b,:,:,0] = tmp.reshape(N,c)
            
            # tmp = out[ind_b,:,:,1:].transpose(0,1).contiguous()
            # out[ind_b,:,:,1:] = tmp.reshape(N,c,h*w-1)


            for ind_r in range(0,w):
                batchsize = h
                tmp = out[ind_b,:,:,ind_r*batchsize:(ind_r+1)*batchsize].transpose(0,1).contiguous()
                out[ind_b,:,:,ind_r*batchsize:(ind_r+1)*batchsize] = tmp.reshape(N,c,batchsize)

        return out.reshape(b, kernel_size * kernel_size * c, h * w)
