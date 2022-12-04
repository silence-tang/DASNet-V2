from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample, normalize
from .attention import PAM_Module
from .attention import CAM_Module
from .resbase import BaseNet
import torch.nn.functional as F

__all__ = ['DANet']


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4  # 2048 / 4 = 512
        # CBR block
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        # CBR block 
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        # PAM_Module & CAM_Module
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        # 空间注意力分支
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)                             # (B, 512, H, W)
        sa_conv = self.conv51(sa_feat)                       # (B, 512, H, W)
        sa_output = self.conv6(sa_conv)                      # (B, 512, H, W)
        # 通道注意力分支
        feat2 = self.conv5c(x)                               # (B, 512, H, W)
        sc_feat = self.sc(feat2)                             # (B, 512, H, W)
        sc_conv = self.conv52(sc_feat)                       # (B, 512, H, W)
        sc_output = self.conv7(sc_conv)                      # (B, 512, H, W)
        # 特征融合（先pixel-wise add，然后过conv8）
        feat_sum = sa_conv + sc_conv                         # (B, 512, H, W)
        sasc_output = self.conv8(feat_sum)                   # (B, 512, H, W)
        return sa_output, sc_output, sasc_output             # (B, 512, H, W)


class SiameseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SiameseDecoder, self).__init__()

        # UpBlock = Upsample + BatchNorm + LeakyRelu
        # in: 1x512x32x32, out: 1x256x64x64

        self.UpBlock1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels//2, padding=1, stride=2, kernel_size=(4,4)),
                                      nn.BatchNorm2d(in_channels//2),
                                      nn.LeakyReLU())
        
        # in: 1x256x64x64, out: 1x128x128x128
        self.UpBlock2 = nn.Sequential(nn.ConvTranspose2d(in_channels//2, in_channels//4, padding=1, stride=2, kernel_size=(4,4)),
                                      nn.BatchNorm2d(in_channels//4),
                                      nn.LeakyReLU())     
        
        # in: 1x128x128x128, out: 1x64x256x256
        self.UpBlock3 = nn.Sequential(nn.ConvTranspose2d(in_channels//4, out_channels, padding=1, stride=2, kernel_size=(4,4)),
                                      nn.BatchNorm2d(out_channels),
                                      nn.LeakyReLU())   
        

        # self.UpBlock1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, 3, padding=1, bias=False),
        #                               nn.BatchNorm2d(in_channels//2),
        #                               nn.ReLU(),
        #                               nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        #                             )     
        
        # # in: 1x256x64x64, out: 1x128x128x128
        # self.UpBlock2 = nn.Sequential(nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1, bias=False),
        #                               nn.BatchNorm2d(in_channels//4),
        #                               nn.ReLU(),
        #                               nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        #                             )
        
        # # in: 1x128x128x128, out: 1x32x256x256
        # self.UpBlock3 = nn.Sequential(nn.Conv2d(in_channels//4, out_channels, 3, padding=1, bias=False),
        #                               nn.BatchNorm2d(out_channels),
        #                               nn.ReLU(),
        #                               nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        #                             )

    def forward(self, x):
        for upblock in [self.UpBlock1, self.UpBlock2, self.UpBlock3]:
            x = upblock(x)
        return x


class DANet(BaseNet):
    """
    Paper: Fully Convolutional Networks for Semantic Segmentation
    Backbone: default:'resnet50'; 'resnet50', 'resnet101' or 'resnet152'
    """
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        # base_forward()是resnet backbone的前向函数
        _, _, _, c4 = self.base_forward(x) # c4是layer4的输出
        x = self.head(c4)
        x = list(x)
        return x[0], x[1], x[2]


def cnn():
    model = DANet(512, backbone='resnet50')
    return model


class SiameseNet(nn.Module):
    def __init__(self, norm_flag = 'l2'):
        super(SiameseNet, self).__init__()
        self.CNN = cnn()
        self.decoder = SiameseDecoder(512, 32)

        if norm_flag == 'l2':
           self.norm = F.normalize
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):
        # CNN(t0)返回的是before img的sa, sc, sasc
        out_t0_conv5, out_t0_fc7, out_t0_embedding = self.CNN(t0)
        # CNN(t1)返回的是after img的sa, sc, sasc
        # 1x512x32x32
        out_t1_conv5, out_t1_fc7, out_t1_embedding = self.CNN(t1)

        # decoder forward
        # 1x512x32x32  --> 1x64x256x256
        out_t0_conv5, out_t1_conv5 = self.decoder(out_t0_conv5), self.decoder(out_t1_conv5)
        out_t0_fc7, out_t1_fc7 = self.decoder(out_t0_fc7), self.decoder(out_t1_fc7)
        out_t0_embedding, out_t1_embedding = self.decoder(out_t0_embedding), self.decoder(out_t1_embedding)

        # 归一化t0 t1的sa
        out_t0_conv5_norm, out_t1_conv5_norm = self.norm(out_t0_conv5, 2, dim=1), self.norm(out_t1_conv5, 2, dim=1)
        # 归一化t0 t1的sc
        out_t0_fc7_norm, out_t1_fc7_norm = self.norm(out_t0_fc7, 2, dim=1), self.norm(out_t1_fc7, 2, dim=1)
        # 归一化t0 t1的sasc
        out_t0_embedding_norm, out_t1_embedding_norm = self.norm(out_t0_embedding, 2, dim=1), self.norm(out_t1_embedding, 2, dim=1)
        # 返回归一化完毕的t0 t1三种特征向量
        # 原1x512x32x32  --> 1x64x256x256
        return [out_t0_conv5_norm, out_t1_conv5_norm], [out_t0_fc7_norm, out_t1_fc7_norm], [out_t0_embedding_norm, out_t1_embedding_norm]


if __name__ == '__main__':
    model = SiameseNet(norm_flag='l2').cuda()
    print('gg')