import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import math



def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def Lovasz_Hinge(logits, labels, ignore_index=255): 
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    if ignore_index is not None:
        valid = (labels != ignore_index)
        logits = logits[valid]
        labels = labels[valid]

    if len(labels) == 0:
        return logits.sum() * 0.

    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]

    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss



def Focal_Loss(inputs, target, cls_weights, num_classes=2, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()

def Dice_loss(inputs, target, beta=1, smooth=1e-5):

    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size() # target is one-hot [B, H, W, 2]
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), dim=-1)
    temp_target = target.view(n, -1, ct)

    pred_fg = temp_inputs[..., 1]    # [N, pixels]
    target_fg = temp_target[..., 1]  # [N, pixels]

    # 计算前景 Dice
    tp = torch.sum(target_fg * pred_fg, dim=[0, 1])
    fp = torch.sum(pred_fg, dim=[0, 1]) - tp
    fn = torch.sum(target_fg, dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def CE_Loss(inputs, target, cls_weights, num_classes=2):

    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)


    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def DeepSupervision_Loss(outputs, pngs, labels, weights, num_classes,
                         lambda_boundary,  
                         boundary_loss_flag=True, 
                         epoch=None,
                         start_boundary=None):

    
    lambda_lovasz = lambda_boundary
    lovasz_loss_flag = boundary_loss_flag
    start_lovasz = start_boundary

    ds_weights = [0.2, 0.8] 
    assert len(outputs) <= len(ds_weights), "outputs overflow ds_weights length"

    loss_weights = {
        "ce": 0.2,     
        "dice": 0.3,   
        "lovasz": 0.5 
    }
    
    total_loss = 0.0

    for i, out in enumerate(outputs):
        if out.shape[-2:] != pngs.shape[-2:]:
            out = F.interpolate(out, size=pngs.shape[-2:], mode='bilinear', align_corners=False)

        ce = CE_Loss(out, pngs, weights, num_classes=num_classes)
        dice = Dice_loss(out, labels)
        loss = loss_weights["ce"] * ce + loss_weights["dice"] * dice

        if lovasz_loss_flag and epoch is not None and start_lovasz is not None:
            if epoch >= start_lovasz:
                if i == len(outputs) - 1: 
                    fg_logits = out[:, 1, :, :] - out[:, 0, :, :]
                    
                    loss_fg = Lovasz_Hinge(fg_logits, pngs)
                    loss_bg = Lovasz_Hinge(-fg_logits, 1 - pngs) # 背景的 logit 和 target 取反
                    
                    lovasz = 0.5 * (loss_fg + loss_bg)
                    loss += loss_weights["lovasz"] * lambda_lovasz * lovasz

        total_loss += ds_weights[i] * loss

    return total_loss












#这段代码定义了一个神经网络权重初始化函数 weights_init，用于初始化PyTorch模型的参数。
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
