import torch
import torch.nn as nn
import numpy as np

from model.backbone import *
from model.neck import *
from model.head import *
from model.utils import *
from model.yololayer import YoloLayer

# RYOLO-SG
class RTiny_SqueezeGhostPlus(nn.Module):

    def __init__(self, out_chs):
        super(RTiny_SqueezeGhostPlus, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_SqueezeGhost2()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1

# RYOLO-D
class RTiny_Mobile(nn.Module):

    def __init__(self, out_chs):
        super(RTiny_Mobile, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Mobile()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1


# RYOLO-G
class RTiny_Ghostplus(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_Ghostplus, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghost2()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1


# RYOLO
class RTiny(nn.Module):
    def __init__(self, out_chs):
        super(RTiny, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)
    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1,feat2)
        out0,out1 = self.head(feat1,feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1

# RYOLO-v4
class RYOLOv4(nn.Module):
    def __init__(self, out_chs):
        super(RYOLOv4, self).__init__()
        self.tiny = False
        self.backbone = Darknet53()
        self.neck = Neck_PanNet(256, 512, 1024)
        self.head = Headv4(out_chs)
    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2, feat3 = self.backbone(x)
        feat1, feat2, feat3 = self.neck(feat1, feat2, feat3)
        out1, out2, out3 = self.head(feat1, feat2,feat3)
        return out1, out2, out3

