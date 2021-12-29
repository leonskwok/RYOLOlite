import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon
import numpy as np
from tools.utils import skewiou_fun,calskewiou


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


def bbox_xywha_ciou(pred_boxes, target_boxes):
    assert pred_boxes.size() == target_boxes.size()

    # xywha -> xyxya 转为正矩形框
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

    area1 = w1 * h1
    area2 = w2 * h2
    # 预测框中心坐标
    center_x1 = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
    center_y1 = (pred_boxes[:, 3] + pred_boxes[:, 1]) / 2
    # 目标框中心坐标
    center_x2 = (target_boxes[:, 2] + target_boxes[:, 0]) / 2
    center_y2 = (target_boxes[:, 3] + target_boxes[:, 1]) / 2

    inter_max_xy = torch.min(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    inter_min_xy = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    out_max_xy = torch.max(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    out_min_xy = torch.min(pred_boxes[:, :2], target_boxes[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    # A交B
    inter_area = inter[:, 0] * inter[:, 1]
    # 中心点直线距离
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    # 外包框对角线距离
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    # A并B
    union = area1 + area2 - inter_area
    # CIOU补偿项
    u = inter_diag / outer_diag
    # IOU
    iou = inter_area / union
    
    # CIOU中的v-度量宽高比的一致性
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)

    # CIOU中的α权重系数
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)

    ciou = iou - (u + alpha * v)
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    # 角度影响
    angle_factor = torch.abs(torch.cos(pred_boxes[:, 4] - target_boxes[:, 4]))
    ariou = iou * angle_factor

    # k=torch.pow(torch.abs(torch.sin(pred_boxes[:, 4] - target_boxes[:, 4])),2)
    # with torch.no_grad():
    #     alpha = v / (S + v + k)
    #     beta = k / (S + v + k)
    # ariou = iou - (u + alpha * v + beta * k)
    
    return ariou, ciou, iou

def bbox_xywha_skewiou(pred_boxes, target_boxes):
    skewiou = torch.zeros(len(pred_boxes),device=pred_boxes.device)
    for i in range(len(pred_boxes)):
        skewiou[i] = calskewiou(pred_boxes[i], target_boxes[i])
    return skewiou


