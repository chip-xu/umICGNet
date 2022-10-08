import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sobel import SobelComputer
import matplotlib.pyplot as plt
import numpy as np
from utils import helpers
import os


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - dice_score.sum()/size

        return loss


class structure_loss(nn.Module):
    def __init__(self, weight=None):
        super(structure_loss, self).__init__()
        self.bce = BCELoss(weight)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = (diceloss + bceloss) / 2

        return loss


def Focal_loss(preds, labels, alpha=0.25, gamma=2):
    eps = 1e-7
    loss0 = -1 * alpha * torch.pow((1 - preds), gamma) * torch.log(preds + eps) * labels
    loss1 = -1 * (1 - alpha) * torch.pow(preds, gamma) * torch.log(1 - preds + eps) * (1 - labels)
    loss = loss0 + loss1

    return torch.mean(loss)


def Deep_Loss(pred, gt):
    p0, p1, p2, p3, p4, pc = pred[0:]

    loss0 = structure_loss()(p0, gt)
    variance0 = torch.log(1 + (p0 - gt) ** 2)
    exp_variance0 = torch.exp(-variance0)
    loss0 = torch.mean(loss0 * exp_variance0) + torch.mean(variance0)

    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    sobel_compute = SobelComputer()
    sobel_img, sobel_gt = sobel_compute(pc, gt)
    sobel_loss = Focal_loss(sobel_img, sobel_gt)

    loss1 = structure_loss()(p1, gt)
    variance1 = torch.log(1 + (p1 - gt) ** 2)
    exp_variance1 = torch.exp(-variance1)
    loss1 = torch.mean(loss1 * exp_variance1) + torch.mean(variance1)

    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = structure_loss()(p2, gt)
    variance2 = torch.log(1 + (p2 - gt) ** 2)
    exp_variance2 = torch.exp(-variance2)
    loss2 = torch.mean(loss2 * exp_variance2) + torch.mean(variance2)

    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = structure_loss()(p3, gt)
    variance3 = torch.log(1 + (p3 - gt) ** 2)
    exp_variance3 = torch.exp(-variance3)
    loss3 = torch.mean(loss3 * exp_variance3) + torch.mean(variance3)

    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = structure_loss()(p4, gt)
    variance4 = torch.log(1 + (p4 - gt) ** 2)
    exp_variance4 = torch.exp(-variance4)
    loss4 = torch.mean(loss4 * exp_variance4) + torch.mean(variance4)

    return (loss0 + loss1 + loss2 + loss3 + loss4) / 5 + sobel_loss
