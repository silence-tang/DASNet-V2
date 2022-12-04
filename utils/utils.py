import torch
import numpy as np
import torch.nn as nn


def adjust_learning_rate(learning_rate, optimizer, step):
    """Adjust the initial_lr to be decayed every 20 epochs"""
    if step <= 200000:
        lr = learning_rate
    elif (step > 200000) & (step <= 400000):
        lr = 5e-5
    elif (step > 400000) & (step <= 500000):
        lr = 1e-5
    elif (step > 600000) & (step <= 700000):
        lr = 5e-6    
    else:
        lr = 1e-6
    
    optimizer.state_dict()['param_groups'][0]['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_metric_dict(thresh):
    metric_for_class = {}
    thresh = thresh # 阈值
    total_fp = np.zeros(thresh.shape)
    total_fn = np.zeros(thresh.shape)
    metric_for_class.setdefault('total_fp', total_fp)
    metric_for_class.setdefault('total_fn', total_fn)
    metric_for_class.setdefault('total_posnum', 0)
    metric_for_class.setdefault('total_negnum', 0)
    return metric_for_class


# 原始label可能尺寸大于feature，故插值构成新的label
def rz_label(label, size):
    gt_e = torch.unsqueeze(label, dim=1)
    interp = nn.functional.interpolate(gt_e, (size[0],size[1]), mode='bilinear', align_corners=True)
    gt_rz = torch.squeeze(interp, dim=1)
    return gt_rz
