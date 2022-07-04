import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon
import scipy.signal
import numpy as np
from tools.utils import skewiou_fun,calskewiou
import matplotlib.pyplot as plt
import os


def loss_plot(dirpath):
    trainloss_path = os.path.join(dirpath, "loss.txt")
    if os.path.exists(trainloss_path):
        trainloss = np.loadtxt(trainloss_path)[:, 0]
        plt.figure()
        plt.plot(trainloss, color="black", linestyle='-', linewidth=2)
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(dirpath, "loss.jpg"))

    valloss_path = os.path.join(dirpath, "val_loss.txt")
    if os.path.exists(valloss_path):
        valloss = np.loadtxt(valloss_path)
        plt.figure()
        plt.plot(valloss, color="black", linestyle='-', linewidth=2)
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(dirpath, "val_loss.jpg"))

class FocalLoss(nn.Module):
    # Reference: https://github.com/ultralytics/yolov5/blob/8918e6347683e0f2a8a3d7ef93331001985f6560/utils/loss.py#L32
    def __init__(self, alpha=0.25, gamma=2, reduction="none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # -log(pt)
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # pt
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        # αt
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        # (1-pt)^γ
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def bbox_iou_mix(pred_boxes, target_boxes, ifxywha):
    # 这里适用于pred_boxes和target_boxes不同shape
    if ifxywha:
        # xywha -> xyxya 转为正矩形框
        pred_boxes = torch.cat(
            [pred_boxes[..., :2] - pred_boxes[..., 2:4] / 2,
            pred_boxes[..., :2] + pred_boxes[..., 2:4] / 2,
            pred_boxes[..., 4:]], dim=-1)
        target_boxes = torch.cat(
            [target_boxes[..., :2] - target_boxes[..., 2:4] / 2,
            target_boxes[..., :2] + target_boxes[..., 2:4] / 2,
            target_boxes[..., 4:]], dim=-1)

    A = pred_boxes.size(0)
    B = target_boxes.size(0)
    max_xy = torch.min(pred_boxes[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       target_boxes[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(pred_boxes[:, :2].unsqueeze(1).expand(A, B, 2),
                       target_boxes[:, :2].unsqueeze(0).expand(A, B, 2))

    angle_a = pred_boxes[:, 4:5].unsqueeze(1).expand(A, B, 1)
    angle_b = target_boxes[:, 4:5].unsqueeze(0).expand(A, B, 1)

    da = angle_a-angle_b
    da = da[:, :, 0]
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((pred_boxes[:, 2]-pred_boxes[:, 0]) *
              (pred_boxes[:, 3]-pred_boxes[:, 1])).unsqueeze(1).expand(A, B)  # [A,B]
    area_b = ((target_boxes[:, 2]-target_boxes[:, 0]) *
              (target_boxes[:, 3]-target_boxes[:, 1])).unsqueeze(0).expand(A, B)  # [A,B]
    union = area_a + area_b - inter
    iou = inter / union
    return da, iou  # [A,B]


def bbox_iou(pred_boxes, target_boxes, ifxywha):
    assert pred_boxes.size() == target_boxes.size()
    # xywha -> xyxya 转为正矩形框
    if ifxywha:
        pred_boxes = torch.cat(
            [pred_boxes[..., :2] - pred_boxes[..., 2:4] / 2,
            pred_boxes[..., :2] + pred_boxes[..., 2:4] / 2,
            pred_boxes[..., 4:]], dim=-1)
        target_boxes = torch.cat(
            [target_boxes[..., :2] - target_boxes[..., 2:4] / 2,
            target_boxes[..., :2] + target_boxes[..., 2:4] / 2,
            target_boxes[..., 4:]], dim=-1)
    
    # 预测框wh
    w1 = pred_boxes[:, 2] - pred_boxes[:, 0]
    h1 = pred_boxes[:, 3] - pred_boxes[:, 1]
    # GT框wh
    w2 = target_boxes[:, 2] - target_boxes[:, 0]
    h2 = target_boxes[:, 3] - target_boxes[:, 1]
    # 计算预测框面积
    area1 = w1 * h1
    area2 = w2 * h2
    # 预测框中心坐标
    center_x1 = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
    center_y1 = (pred_boxes[:, 3] + pred_boxes[:, 1]) / 2
    # 目标框中心坐标
    center_x2 = (target_boxes[:, 2] + target_boxes[:, 0]) / 2
    center_y2 = (target_boxes[:, 3] + target_boxes[:, 1]) / 2
    # 交集
    inter_max_xy = torch.min(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    inter_min_xy = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    out_max_xy = torch.max(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    out_min_xy = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    # 外包围框
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    # 并集
    union = area1 + area2 - inter_area
      
    # IOU
    iou = inter_area / union

    # ArIoU
    angle_factor = torch.abs(torch.cos(pred_boxes[:, 4] - target_boxes[:, 4]))
    ariou = iou * angle_factor

    # GIOU
    outer_area = outer[:, 0]*outer[:, 1]
    giou = iou-(outer_area-union)/outer_area

    # CIoU
    # 外包框对角线距离
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    # 中心点直线距离
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    # CIOU补偿项
    u = inter_diag / outer_diag 
    # CIOU中的v-度量宽高比的一致性
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    # CIOU中的α权重系数
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    ciou = iou - (u + alpha * v)
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    return iou, ariou, giou, ciou

def bbox_xywha_skewiou(pred_boxes, target_boxes):
    skewiou = torch.zeros(len(pred_boxes),device=pred_boxes.device)
    for i in range(len(pred_boxes)):
        skewiou[i] = calskewiou(pred_boxes[i], target_boxes[i])
    return skewiou


