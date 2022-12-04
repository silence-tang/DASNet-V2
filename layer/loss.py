import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# -------------------------------------------------------------------------------
class ContrastiveLoss1(nn.Module):
    def __init__(self, margin1 = 0.3, margin2=2.2, eps=1e-6):
        super(ContrastiveLoss1, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, y):
        diff = torch.abs(x1 - x2)
        dist_sq = torch.pow(diff + self.eps, 2).sum(dim=1)  # 加一个微小扰动
        dist = torch.sqrt(dist_sq)  # 求距离(L2)
        mdist_pos = torch.clamp(dist-self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2-dist, min=0.0)
        loss_pos = (1 - y) * (mdist_pos.pow(2))
        loss_neg = y * (mdist_neg.pow(2))
        loss = torch.mean(loss_pos + loss_neg)  # 没加权吗？
        return loss


# -------------------------------------------------------------------------------
class CLNew(nn.Module):
    def __init__(self, margin1 = 0.3, margin2=2.2, eps=1e-6):
        super(CLNew, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, y):
        diff = torch.abs(x1 - x2)  # 8x32x256x256
        dist_sq = torch.pow(diff + self.eps, 2).sum(dim=1)   # 加一个微小扰动 32x32 -> 256x256
        dist = torch.sqrt(dist_sq)                           # 求距离(L2)  32x32 -> 256x256
        mdist_pos = torch.clamp(dist-self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2-dist, min=0.0)
        w1 = 1 / 0.147
        w2 = 1 / (1 - 0.147)
        loss_pos = w2 * ((1 - y) * (mdist_pos.pow(2)))  # 逐元素乘 w2小，使网络较为不关注unchanged pairs
        loss_neg = w1 * (y * (mdist_neg.pow(2)))        # 逐元素乘 w1大，使网络重点关注changed pairs
        loss = torch.mean(loss_pos + loss_neg)
        return loss


class KLCoefficient(nn.Module):
    def __init__(self):
        super(KLCoefficient, self).__init__()

    def forward(self,hist1,hist2):
        kl = F.kl_div(hist1,hist2)
        dist = 1. / 1 + kl
        return dist