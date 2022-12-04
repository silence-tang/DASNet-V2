###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']

# attention between local areas(each pixel pair)
# 建模long-range局部特征的上下文语音信息
class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim   # 512
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1) # 512,64
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)   # 512,64
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)    # 512,512
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size() # (B, 512, H, W)
        # Q
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # (B, HxW, 64)
        # K
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)   # (B, 64, HxW)
        # original attention matrix
        energy = torch.bmm(proj_query, proj_key)   # (B, HxW, HxW)
        # softmax让各区域间的差异更大
        attention = self.softmax(energy)   # (B, HxW, HxW)
        # V
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # (B, 512, HxW)
        # 计算由attention score加权的V matrix
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, 512, HxW)
        out = out.view(m_batchsize, C, height, width)  # (B, 512, H, W)
        out = self.gamma * out + x  # (B, 512, H, W)
        return out

# attention between channels(each channel pair, also can be regarded as each ground object pair)
# 建模通道间的语义信息
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

    def forward(self,x):
        m_batchsize, C, height, width = x.size()  
        proj_query = x.view(m_batchsize, C, -1)                    # (B, 512, HxW)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)     # (B, HxW, 512)
        energy = torch.bmm(proj_query, proj_key)                   # (B, 512, 512) 
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)                       # (B, 512, 512)
        proj_value = x.view(m_batchsize, C, -1)                    # (B, 512, HxW)
        out = torch.bmm(attention, proj_value)                     # (B, 512, HxW)
        out = out.view(m_batchsize, C, height, width)              # (B, 512, H, W) 
        out = self.gamma * out + x                                 # (B, 512, H, W)
        return out

