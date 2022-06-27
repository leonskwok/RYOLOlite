from ast import Bytes
import torch
from model.loss import *
from model.config import config
from model.loss import loss_plot

def cal_iou(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    angle_a = box_a[:, 4:5].unsqueeze(1).expand(A, B, 1)
    angle_b = box_b[:, 4:5].unsqueeze(0).expand(A, B, 1)

    da = angle_a-angle_b
    da = da[:, :, 0]
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
   
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand(A, B)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand(A, B)  # [A,B]
    union = area_a + area_b - inter
    iou = inter / union
    ariou = da*iou
    return iou,ariou # [A,B]

cfg = config()



quit()
target_boxes = torch.rand([3, 5]).cuda()
masked_anchors = torch.rand([3, 3]).cuda()
device =target_boxes.device
ga = target_boxes[:, 4]  # angle

num_target=len(target_boxes)
num_anchors = len(masked_anchors)
gt_box = torch.cat((torch.zeros((num_target, 2), device=device), target_boxes[:, 2:5]), dim=1)
# 构建零点anchor框[0,0,w,h,a]
anchor_shapes = torch.cat((torch.zeros((num_anchors, 2), device=device), masked_anchors), dim=1)

offset, _, arious = bbox_iou_mix(anchor_shapes, gt_box, True)

_, best_n = arious.max(0,keepdim=True)

print(target_boxes)
print(masked_anchors)
result=cal_iou(target_boxes,masked_anchors)
print(result[0])
print(result[1])





