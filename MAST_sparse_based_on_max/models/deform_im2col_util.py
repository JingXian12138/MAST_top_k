'''
Author: silentchord 3334228261@qq.com
Date: 2022-09-22 18:37:23
LastEditors: silentchord 3334228261@qq.com
LastEditTime: 2022-10-06 17:11:32
FilePath: \MAST_sparse_based_on_max\models\deform_im2col_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import torch
import torch.nn.functional as F
import pynvml

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

def deform_im2col(im, offset, kernel_size=25):
    with torch.no_grad():
        grid = image_meshgrid_from(im)
        b, c, h, w = im.shape
    
    # print(grid.shape) # 1,120,228,2
    # print(grid[0][119][227]) 227,119
    N = kernel_size * kernel_size

    grid_ = torch.zeros(b * N, h, w, 2,  device=im.device).contiguous() # 625,120,228,2
    im_ = im.repeat(N, 1, 1, 1) # 625,7,120,228

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            grid_[(dy * kernel_size + dx) * b:(dy * kernel_size + dx + 1) * b] =\
                grid + offset + torch.tensor([dx - kernel_size // 2, dy - kernel_size // 2])[None, None, None, :].float().to(im.device)

    # print("b, c, h, w: ", b, c, h, w)
    out = F.grid_sample(im_.contiguous(), normalize_meshgrid(grid_).contiguous())
    out = out.reshape(N, b, c, h * w).permute(1,2,0,3)
    out = out.reshape(b, kernel_size * kernel_size * c, h * w)
    return out


def deform_im2col_max(im, offset, kernel_size=1):
    # Faster on gpu, slower on CPU
    # input: b,c,h,w
    # output: b,N*c,h*w
    print("#######")
    with torch.no_grad():
        grid = image_meshgrid_from(im)
        b, c, h, w = im.shape
    
    # print(grid.shape) # 1,120,228,2
    # print(grid[0][119][227]) 227,119
    N = kernel_size * kernel_size

    grid_ = torch.zeros(b * N, h, w, 2,  device=im.device).contiguous() # 625,120,228,2
    im_ = im.repeat(N, 1, 1, 1) # 625,7,120,228

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            grid_[(dy * kernel_size + dx) * b:(dy * kernel_size + dx + 1) * b] =\
                grid + offset + torch.tensor([dx - kernel_size // 2, dy - kernel_size // 2])[None, None, None, :].float().to(im.device)

    out = F.grid_sample(im_.contiguous(), normalize_meshgrid(grid_).contiguous())
    out = out.reshape(N, b, c, h * w).permute(1,2,0,3)

    return out.reshape(b, kernel_size * kernel_size * c, h * w)