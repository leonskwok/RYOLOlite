import torch
import torch.nn as nn
import numpy as np

from model.backbone import *
from model.neck import *
from model.head import *
from model.utils import *

# RYOLO-SG
class RTiny_SG(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_SG, self).__init__()
        self.tiny = True
        self.backbone = CSPDarkNet53tiny_SG("leaky")
        self.neck = Neck_tiny([256, 512], [384, 256], "leaky")
        self.head = Headv4_tiny([384,256],[out_chs,out_chs],"leaky")
    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out1: batchsize*((6+4)*3*6)*13*13
        return out0, out1


# RYOLO-D
class RTiny_D(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_D, self).__init__()
        self.tiny = True
        self.backbone = CSPDarkNet53tiny_D("leaky")
        self.neck = Neck_tiny([256, 512], [384, 256], "leaky")
        self.head = Headv4_tiny([384, 256], [out_chs, out_chs], "leaky")

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
class RTiny_G(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_G, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = CSPDarkNet53tiny_G("leaky")
        self.neck = Neck_tiny([256, 512], [384, 256], "leaky")
        self.head = Headv4_tiny([384, 256], [out_chs, out_chs], "leaky")

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
        self.backbone = CSPDarkNet53tiny("leaky")
        self.neck = Neck_tiny([256, 512], [384, 256], "leaky")
        self.head = Headv4_tiny([384, 256], [out_chs, out_chs], "leaky")

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1,feat2)
        out0,out1 = self.head(feat1,feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1




# RMobileYOLOv4
class RYOLOv3_M(nn.Module):
    def __init__(self, out_chs):
        super(RYOLOv3_M, self).__init__()
        self.tiny = False
        self.backbone = MobileNet("leaky")
        self.neck = Neck_PanNet([256, 512, 1024], [128, 256, 512], 'leaky')
        self.head = Headv4([128, 256, 512], [out_chs, out_chs, out_chs], "leaky")

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        feat1, feat2, feat3 = self.neck(feat1, feat2, feat3)
        out1, out2, out3 = self.head(feat1, feat2, feat3)
        return out1, out2, out3
    

                
# RGhostYOLOv4
class RYOLOv3_G(nn.Module):
    def __init__(self, out_chs):
        super(RYOLOv3_G, self).__init__()
        self.tiny = False
        self.backbone = GhostNet("leaky")
        self.neck = Neck_PanNet([40, 112, 160], [128, 256, 512], 'leaky')
        self.head = Headv4([128, 256, 512], [
                           out_chs, out_chs, out_chs], "leaky")

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        feat1, feat2, feat3 = self.neck(feat1, feat2, feat3)
        out1, out2, out3 = self.head(feat1, feat2, feat3)
        return out1, out2, out3


# RYOLO-v4
class RYOLOv4(nn.Module):
    def __init__(self, out_chs):
        super(RYOLOv4, self).__init__()
        self.tiny = False
        self.backbone = CSPDarknet53()  # 52*52*256, 26*26*512, 13*13*1024
        self.neck = Neck_PanNet([256, 512, 1024], [128, 256, 512], 'leaky')
        self.head = Headv4([128, 256, 512], [
                           out_chs, out_chs, out_chs], "leaky")
    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        feat1, feat2, feat3 = self.neck(feat1, feat2, feat3)
        out1, out2, out3 = self.head(feat1, feat2,feat3)
        return out1, out2, out3


class RYOLOv3(nn.Module):
    def __init__(self, out_chs):
        super(RYOLOv3, self).__init__()
        self.tiny = False
        self.backbone = Darknet53('leaky')  # 52*52*256, 26*26*512, 13*13*1024
        self.neck = Neck_FPN([256, 512, 1024], [128, 256, 512], 'leaky')
        self.head = Headv3([128, 256, 512], [out_chs, out_chs, out_chs], "leaky")

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        feat1, feat2, feat3 = self.neck(feat1, feat2, feat3)
        out1, out2, out3 = self.head(feat1, feat2, feat3)
        return out1, out2, out3
