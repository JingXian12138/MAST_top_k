import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import one_hot
from spatial_correlation_sampler import SpatialCorrelationSampler
from .deform_im2col_util import deform_im2col
from .deform_im2col_util import deform_im2col_max
import numpy as np



class Colorizer(nn.Module):
    def __init__(self, D=4, R=6, C=32):
        super(Colorizer, self).__init__()
        self.D = D
        self.R = R  # window size
        self.C = C

        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0

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

    def forward(self, feats_r, feats_t, quantized_r, ref_index, current_ind, dil_int = 15):
        """
        Warp y_t to y_(t+n). Using similarity computed with im (t..t+n)
        :param feats_r: f([im1, im2, im3])
        :param quantized_r: [y1, y2, y3]
        :param feats_t: f(im4)
        :param mode:
        :return:
        """
        dil_int = 16
        # dil_int = 1
        # For frame interval < dil_int, no need for deformable resampling
        nref = len(feats_r)
        nsearch = len([x for x in ref_index if current_ind - x > dil_int])

        # The maximum dilation rate is 4
        dirates = [ min(4, (current_ind - x) // dil_int +1) for x in ref_index if current_ind - x > dil_int]
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        
        query_values = []
        query_images = []
        k = 20

        # topk_query_values = []
        # topk_query_images = []
        # print("quantized_r[0].shape", quantized_r[0].shape) # 1,1,480,910
        # 测试时标签改为独热编码(最多为7类因此维度为7,人为设定)
        qr = [self.prep(qr, (h,w)) for qr in quantized_r] # qr[0].shape 1,7,120,228 

        
        # # 长期记忆im_col0和短期记忆im_col1
        # im_col0 = [deform_im2col(qr[i], offset0, kernel_size=self.P)  for i in range(nsearch)] # b,3*N,h*w
        # im_col1 = [F.unfold(r, kernel_size=self.P, padding =self.R) for r in qr[nsearch:]]
        # # print("im_col1[0].shape: ", im_col1[0].shape) # 1,4375,27360 # 1,7*25*25,120*228
        # image_uf = im_col0 + im_col1
        # offset0 = []

        # print("nsearch: ",nsearch)
        # print("nref: ",nref)

        # out.shape torch.Size([1, 64, 625, 27360])
        for searching_index in range(nsearch):
            ##### GET OFFSET HERE.  (b,h,w,2)
            samplerindex = dirates[searching_index]-2
            coarse_search_correlation = self.correlation_sampler_dilated[samplerindex](feats_t, feats_r[searching_index])  # b, p, p, h, w
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_N, h*w)
            coarse_search_correlation = F.softmax(coarse_search_correlation, dim=1)
            coarse_search_correlation = coarse_search_correlation.reshape(b,self.memory_patch_P,self.memory_patch_P,h,w,1)
            _y, _x = torch.meshgrid(torch.arange(-self.memory_patch_R,self.memory_patch_R+1),torch.arange(-self.memory_patch_R,self.memory_patch_R+1))
            grid = torch.stack([_x, _y], dim=-1).unsqueeze(-2).unsqueeze(-2)\
                .reshape(1,self.memory_patch_P,self.memory_patch_P,1,1,2).contiguous().float().to(coarse_search_correlation.device)
            offset0 = (coarse_search_correlation * grid ).sum(1).sum(1) * dirates[searching_index]  # 1,h,w,2
            
            

            col_0 = deform_im2col(feats_r[searching_index], offset0, kernel_size=self.P)  # b,c*N,h*w
            # deform_im2col(qr[0], offset0, kernel_size=self.P)
            # exit(0)
            col_0 = col_0.reshape(b,c,N,h,w)
            ##
            corr = (feats_t.unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)
            col_0 = []
            corr = corr.reshape([b, self.P * self.P, h * w])
            # corrs.append(corr)
            # max_query = torch.max(corr, dim=1, keepdim=True)
            # max_query_values = max_query.values
            # # print("max_query_values.shape", max_query_values.shape) # torch.Size([1, 1, 27360])
            # max_query_indices = max_query.indices

            topk_query = torch.topk(corr,k,dim=1,sorted=True)
            topk_query_values = topk_query.values
            # # print("max_query_values.shape", max_query_values.shape) # torch.Size([1, 1, 27360])
            # max_query_indices = max_query.indices
            topk_query_indices = topk_query.indices

            # print("max_query_indices[0]", max_query_indices)
            # print("max_query_indices.shape", max_query_indices.shape) # torch.Size([1, 1, 27360])
            query_values.append(topk_query_values)


            # long_img_tmp = process_long_image_uf(qr[searching_index], offset0, max_query_indices) 
           
            long_img_tmp = process_long_image_uf_topk(qr[searching_index], offset0, topk_query_indices, k)    
            query_images.append(long_img_tmp)

            

            # deform_im2col(qr[i], offset0, kernel_size=self.P)

        
        # 用correlation_sampler计算i+1帧和i,i-2,i-4帧的corr（相关程度）
        
        corr_tmp = torch.zeros([b, self.P, self.P, h, w], device=qr[0].device).contiguous()
        corr_tmp_r = torch.zeros([b, self.P*self.P, h*w], device=qr[0].device).contiguous()
        for ind in range(nsearch, nref):
            # corrs.append(self.correlation_sampler(feats_t, feats_r[ind]))
            # _, _, _, h1, w1 = corrs[-1].size()
            # corrs[ind] = corrs[ind].reshape([b, self.P*self.P, h1*w1])
            corr_tmp = self.correlation_sampler(feats_t, feats_r[ind])
            _, _, _, h1, w1 = corr_tmp.size()
            corr_tmp_r = corr_tmp.reshape([b, self.P*self.P, h1*w1]) # 1,625,27360
            # max_query = torch.max(corr_tmp_r, dim=1, keepdim=True)
            topk_query = torch.topk(corr_tmp_r,k,dim=1,sorted=True)
            # max_query_values = max_query.values
            topk_query_values = topk_query.values
            # # print("max_query_values.shape", max_query_values.shape) # torch.Size([1, 1, 27360])
            # max_query_indices = max_query.indices
            topk_query_indices = topk_query.indices
            # query_values.append(max_query_values)
            query_values.append(topk_query_values)
            # # query_indexs.append(max_query_indices)
            # img_uf = process_short_image_uf(qr[ind], max_query_indices)
            # print("topk_query_indices.shape: ", topk_query_indices.shape) # 1,20,27360
            topk_img_uf = process_short_image_uf_topk(qr[ind], topk_query_indices, k)
            # query_images.append(img_uf)
            query_images.append(topk_img_uf)


        corr_max = torch.cat(query_values, dim=1) # b,1*N,HW
        corr_max = F.softmax(corr_max, dim=1)
        corr_max = corr_max.unsqueeze(1) # b,1,1*N,HW

        new_image_uf = torch.cat(query_images, 2)
        new_out = (corr_max * new_image_uf).sum(2).reshape([b,qr[0].size(1),h,w])
        return new_out

def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)



def process_short_image_uf(q, indices):
    # q.shape 1,7,120,228 
    # indices.shape 1,1,27360
    # return image_uf_max 1,7,1,27360
    b,c,h,w = q.size()
    tmp = torch.zeros([b,c,1,h*w], device=q.device).contiguous()
    for r in range(h):
        for c in range(w):
            ind = r*w + c
            q_ind = indices[0][0][ind]
            dr = int(q_ind/25) - 12
            dc = q_ind%25 - 12
            tmp[0,:,0,ind] = q[0,:,r+dr,c+dc]
    return tmp


# 后期利用矩阵运算进行优化 [1,7,20,27360]*[1,7,1,27360]=>[1,7,20,27360]
def process_short_image_uf_topk(q, indices, k=20):
    # q.shape 1,7,120,228 
    # indices.shape 1,20,27360
    # return image_uf_max 1,7,20,27360
    b,c,h,w = q.size()
    tmp = torch.zeros([b,c,k,h*w], device=q.device).contiguous()
    for r in range(h):
        for c in range(w):
            ind = r*w + c
            for kk in range(k):
                q_ind = indices[0][kk][ind]
                dr = int(q_ind/25) - 12
                dc = q_ind%25 - 12
                tmp[0,:,kk,ind] = q[0,:,r+dr,c+dc]
    return tmp


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

def process_long_image_uf(im, offset, indices):
    # im.shape 1,7,120,228 
    # indices.shape 1,1,27360
    # return image_uf_max 1,7,1,27360

    # b,c,h,w = im.size()
    # tmp = torch.zeros([b,c,1,h*w], device=q.device).contiguous()

    # print("offset.shape", offset.shape) # [1, 120, 228, 2]

    with torch.no_grad():
        # grid = image_meshgrid_from(im)
        b, c, h, w = im.shape
    
    for r in range(h):
        for c in range(w):
            ind = r*w + c
            q_ind = indices[0][0][ind]
            dr = int(q_ind/25) - 12 # y
            dc = q_ind%25 - 12 # x
            offset[0][r][c][0] += dc
            offset[0][r][c][1] += dr
    return deform_im2col(im, offset, 1).unsqueeze(2)

def process_long_image_uf_topk(im, offset, indices,k=20):
    # im.shape 1,7,120,228 
    # offset 1,120,228,2
    # indices.shape 1,20,27360
    # return image_uf_topk 1,7,20,27360
    
    # b,c,h,w = im.size()
    # tmp = torch.zeros([b,c,1,h*w], device=q.device).contiguous()

    # print("offset.shape", offset.shape) # [1, 120, 228, 2]
    # exit(0)

    with torch.no_grad():
        # grid = image_meshgrid_from(im)
        b, c, h, w = im.shape
    
    ret = []
    
    for kk in range(k):
        off_tmp = offset.clone()
        for r in range(h):
            for c in range(w):
                ind = r*w + c
                q_ind = indices[0][kk][ind]
                dr = int(q_ind/25) - 12 # y
                dc = q_ind%25 - 12 # x
                off_tmp[0][r][c][0] += dc
                off_tmp[0][r][c][1] += dr
        tt = deform_im2col(im, off_tmp, 1).unsqueeze(2)
        ret.append(tt)
        off_tmp = []

    res = torch.cat(ret,2)
    return res



