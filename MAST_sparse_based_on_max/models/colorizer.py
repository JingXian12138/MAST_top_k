import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import one_hot
from spatial_correlation_sampler import SpatialCorrelationSampler
from .deform_im2col_util import deform_im2col
import pdb
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
        # For frame interval < dil_int, no need for deformable resampling
        nref = len(feats_r)
        nsearch = len([x for x in ref_index if current_ind - x > dil_int])

        # The maximum dilation rate is 4
        dirates = [ min(4, (current_ind - x) // dil_int +1) for x in ref_index if current_ind - x > dil_int]
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        corrs = []

        query_values = []
        query_indexs = []

        # offset0 = []
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
            col_0 = col_0.reshape(b,c,N,h,w)
            ##
            corr = (feats_t.unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)

            corr = corr.reshape([b, self.P * self.P, h * w])
            corrs.append(corr)
            max_query = torch.max(corr, dim=1, keepdim=True)
            max_query_values = max_query.values
            # print("max_query_values.shape", max_query_values.shape) # torch.Size([1, 1, 27360])
            max_query_indices = max_query.indices
            # print("max_query_indices[0]", max_query_indices)
            # print("max_query_indices.shape", max_query_indices.shape) # torch.Size([1, 1, 27360])
            query_values.append(max_query_values)
            query_indexs.append(max_query_indices)

        # 用correlation_sampler计算i+1帧和i,i-2,i-4帧的corr（相关程度）
        for ind in range(nsearch, nref):
            # corrs.append(self.correlation_sampler(feats_t, feats_r[ind]))
            # _, _, _, h1, w1 = corrs[-1].size()
            # corrs[ind] = corrs[ind].reshape([b, self.P*self.P, h1*w1])
            corr_tmp = self.correlation_sampler(feats_t, feats_r[ind])
            corrs.append(corr_tmp)
            _, _, _, h1, w1 = corrs[-1].size()
            corrs[ind] = corrs[ind].reshape([b, self.P*self.P, h1*w1]) # 1,625,27360

            # print("corrs[ind].shape", corrs[ind].shape)

            max_query = torch.max(corrs[ind], dim=1, keepdim=True)
            max_query_values = max_query.values
            # print("max_query_values.shape", max_query_values.shape) # torch.Size([1, 1, 27360])
            max_query_indices = max_query.indices
            # print("max_query_indices[0]", max_query_indices)
            # print("max_query_indices.shape", max_query_indices.shape) # torch.Size([1, 1, 27360])
            query_values.append(max_query_values)
            query_indexs.append(max_query_indices)


        # # 对参考域中的像素点进行选择，只挑选较大的那部分
        # for c in corrs:
        #     # print("c.shape", c.shape) # 1,625,27360
        #     # print(type(c))
        #     # med = torch.median(c, axis=1).values
        #     max = torch.max(c, dim=1).values
        #     c[c<max] = 0

        corr = torch.cat(corrs, 1)  # b,nref*N,HW
        corr = F.softmax(corr, dim=1)
        corr = corr.unsqueeze(1) # b,1,25*25*N,HW

        corr_max = torch.cat(query_values, dim=1) # b,1*N,HW
        corr_max = F.softmax(corr_max, dim=1)
        corr_max = corr_max.unsqueeze(1) # b,1,1*N,HW

        # print("quantized_r[0].shape", quantized_r[0].shape) # 1,1,480,910
        # 测试时标签改为独热编码(最多为7类因此维度为7,人为设定)
        qr = [self.prep(qr, (h,w)) for qr in quantized_r] # qr[0].shape 1,7,120,228 

        # 长期记忆im_col0和短期记忆im_col1
        im_col0 = [deform_im2col(qr[i], offset0, kernel_size=self.P)  for i in range(nsearch)] # b,3*N,h*w
        im_col1 = [F.unfold(r, kernel_size=self.P, padding =self.R) for r in qr[nsearch:]]
        # print("im_col1[0].shape: ", im_col1[0].shape) # 1,4375,27360 # 1,7*25*25,120*228
        image_uf = im_col0 + im_col1

        image_uf = [uf.reshape([b,qr[0].size(1),self.P*self.P,h*w]) for uf in image_uf]
        # print("image_uf[0].shape", image_uf[0].shape) # 1,7,625,27360
        new_image_uf = process_image_uf(image_uf, query_indexs)
        
        image_uf = torch.cat(image_uf, 2)
        new_image_uf = torch.cat(new_image_uf, 2)

        new_out = (corr_max * new_image_uf).sum(2).reshape([b,qr[0].size(1),h,w])
        # print("new_out.shape", new_out.shape)

        


        # print("corr.shape", corr.shape) #1,1,625*N,HW
        # print("image_uf.shape", image_uf.shape) # 1,7,625*N,HW

        # out = (corr * image_uf).sum(2).reshape([b,qr[0].size(1),h,w])
        # print("out.shape", out.shape) # b,qr[0].size(1),h,w
        return new_out

def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)


def process_image_uf(image_uf, query_indexs):
    # print("process_image_uf")
    # print(query_indexs[0].shape) # 1,1,27360
    # print("image_uf[0].shape", image_uf[0].shape) # 1,7,625,27360
    new_image_uf = []
    for i in range(len(image_uf)):
        uf = image_uf[i]
        # print("uf.shape", uf.shape) #1,7,625,27360
        query = query_indexs[i]
        b,c,_,hw = uf.size()
        tmp = torch.zeros([b,c,1,hw], device=uf.device).contiguous()
        
        # exit(0)
        for j in range(hw):
            # print("####################uf[0,:,query[j],j].shape", uf[0,:,query[0][0][j],j].shape)
            tmp[0,:,0,j] = uf[0,:,query[0][0][j],j]
        new_image_uf.append(tmp)
    return new_image_uf





def process_short(qr, query_indexs, nsearch):
    # qr[0] 1,7,120,228 # 原图片的1/4

    print("process short image")
    # print(len(qr)) # 1
    # print(len(query_indexs)) # 1
    # print(nsearch) # 0  

    # print(qr[0].shape) # 1,7,120,228
    # print(query_indexs[0].shape) # 1,1,27360
    _,_,h,w = qr[0].size()
    for frame in range(nsearch, len(qr)):
        query = query_indexs[frame] # 1,1,27360
        f = torch.zeros(qr[frame].shape)
        for ind in range(h*w):
            r = int(ind/w)
            c = ind%w
            f[0,:,r,c] = qr[frame][0,:,r,c]


    # 理论上应该不需要先unfold再挑选的，但是第一个版本可以这样，后面再优化

    # im_col1_max = []
    # res = []
    # for i in range(nsearch, len(query_indexs)):
    #     # query_indexs[i] # torch.Size([1, 1, 27360]) 
    #     b,c,h,w = qr[i].size()
    #     tmp = torch.zeros(1,c,h*w)
    #     query = query_indexs[i]
    #     for j in range(h*w):
    #         r = query[0][0][j]/w
    #         c = query[0][0][j]%w
    #         tmp[:,:,j] = qr[i][:,:,r,c]
    #     res.append(tmp)

    
