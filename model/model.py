import torch
import torch.nn as nn
import numpy as np

from model.backbone import *
from model.neck import *
from model.head import *
from model.utils import *
from model.yololayer import YoloLayer


class RTiny_TripeAtte(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_TripeAtte, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny()
        self.neck = Neck_tiny_TripeAtte()
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



class RTiny_DCS(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_DCS, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_DCSP()
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


class RTiny_GhostBottle23(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle23, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle23()
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


class RTiny_GhostBottleAtte(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottleAtte, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_GhostbottleAtte()
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


class RTiny_GhostBottleTrip(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottleTrip, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_GhostbottleTrip()
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


class RTiny_GhostBottle(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle()
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


class RTiny_GhostBottle_FeaCASA(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_FeaCASA, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)
        self.atte1 = SpatialAttention()
        self.atte2 = ChannelAttention(512)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1 = self.atte1(feat1)*feat1
        feat2 = self.atte2(feat2)*feat2
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1


class RTiny_GhostBottleFeaTrip(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottleFeaTrip, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle()
        self.atte1 = TripletAttention()
        self.atte2 = TripletAttention()

        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1 = self.atte1(feat1)
        feat2 = self.atte2(feat2)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1




class RTiny_GhostBottle_all(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_all, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all()
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


class RTiny_GhostBottle_all_G(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_all_G, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all_G()
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


class RTiny_GhostBottle_all_G2(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_all_G2, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all_G2()
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


class RTiny_GhostBottle_all_G3(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_all_G3, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all_G3()
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


class RTiny_GhostBottle_G2_trip(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_G2_trip, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all_G2()
        self.atte2 = TripletAttention()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat2 = self.atte2(feat2)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1




class RTiny_GhostALLBottle(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostALLBottle, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_GhostALLbottle()
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


class RTiny_GhostALLBottle2(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostALLBottle2, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_GhostALLbottle2()
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


class RTiny_GhostBottle_all_G_FeaCASA(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_all_G_FeaCASA, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all_G()
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)
        self.atte1 = SpatialAttention()
        self.atte2 = ChannelAttention(512)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1 = self.atte1(feat1)*feat1
        feat2 = self.atte2(feat2)*feat2
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1





class RTiny_GhostBottle_all_FeaCASA(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottle_all_FeaCASA, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghostbottle_all()
        self.atte1 = SpatialAttention()
        self.atte2 = ChannelAttention(512)
        self.neck = Neck_tiny(None)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1 = self.atte1(feat1)*feat1
        feat2  =self.atte2(feat2)*feat2
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        # out0: batchsize*((6+4)*3*6)*26*26
        # out0: batchsize*((6+4)*3*6)*13*13
        return out0, out1


class RTiny_GhostBottleResAtte(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_GhostBottleResAtte, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_GhostbottleResAtte()
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


class RTiny_Ghost(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_Ghost, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghost()
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



class RTiny_Ghost_Atte(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_Ghost_Atte, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_Ghost_Atte()
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

class RTiny_Atte(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_Atte, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny_atte()
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


class RTiny_SE(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_SE, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny()
        self.neck = Neck_tiny(se_block)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        return out0, out1


class RTiny_CBMA(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_CBMA, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny()
        self.neck = Neck_tiny(cbam_block)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        return out0, out1


class RTiny_ECA(nn.Module):
    def __init__(self, out_chs):
        super(RTiny_ECA, self).__init__()
        # attention_block = [se_block, cbam_block, eca_block]
        self.tiny = True
        self.backbone = DarkNet53tiny()
        self.neck = Neck_tiny(eca_block)
        self.head = Head_tiny(out_chs)

    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2 = self.backbone(x)
        feat1, feat2 = self.neck(feat1, feat2)
        out0, out1 = self.head(feat1, feat2)
        return out0, out1

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

class RGhost(nn.Module):
    def __init__(self, out_chs):
        super(RGhost,self).__init__()
        self.tiny = False
        self.backbone = GhostNet()
        self.neck = Neck_PanNet(40, 112, 160)
        self.head = Headv4(out_chs)
    def forward(self, x):
        # feat1的shape为26,26,256
        # feat2的shape为13,13,512
        feat1, feat2, feat3 = self.backbone(x)
        feat1, feat2, feat3 = self.neck(feat1, feat2, feat3)
        out1, out2, out3 = self.head(feat1, feat2,feat3)
        return out1, out2, out3



#########################################################################

class YOLO(nn.Module):
    def __init__(self, ncls,block):
        super().__init__()
        # x,y,w,h,conf,a,cls 再乘3*6
        out_chs = (5 + 1 + ncls) * 3 * 6
        radian = np.pi / 180
        # angles=[-pi / 3, -pi / 6, 0, pi / 6, pi / 3, pi / 2]
        
        # 604
        # anchors_list = [[[12, 16], [19, 36], [40, 28]],
        #                 [[36, 75], [76, 55], [72, 146]],
        #                 [[142, 110], [192, 243], [459, 401]]]
        # 416
        anchors_list = [[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]]

        self.yolobody=block

        self.yolo1 = YoloLayer(num_classes=ncls, anchors=anchors_list[0],
                               angles=[-radian * 60, -radian * 30, 0,
                                       radian * 30, radian * 60, radian * 90],
                               stride=8, scale_x_y=1.2, ignore_thresh=0.6)
        self.yolo2 = YoloLayer(num_classes=ncls, anchors=anchors_list[1],
                               angles=[-radian * 60, -radian * 30, 0,
                                       radian * 30, radian * 60, radian * 90],
                               stride=16, scale_x_y=1.1, ignore_thresh=0.6)
        self.yolo3 = YoloLayer(num_classes=ncls, anchors=anchors_list[2],
                               angles=[-radian * 60, -radian * 30, 0,
                                       radian * 30, radian * 60, radian * 90],
                               stride=32, scale_x_y=1.05, ignore_thresh=0.6)      
        self.metrics={}

    def forward(self, x, target=None):

        feat1,feat2,feat3 = self.yolobody(x)

        # y1 [batch, grid*grid*num_anchors, (x,y,w,h,a,conf,cls)]
        y1, loss1 = self.yolo1(feat1, target) #grid = 76
        y2, loss2 = self.yolo2(feat2, target)  # grid = 38
        y3, loss3 = self.yolo3(feat3, target)  # grid = 19
        
        if target!=None:
            self.metrics={
                "loss": self.yolo1.metrics["loss"]+self.yolo2.metrics["loss"]+self.yolo3.metrics["loss"],
                "reg_loss": self.yolo1.metrics["reg_loss"]+self.yolo2.metrics["reg_loss"]+self.yolo3.metrics["reg_loss"],
                "conf_loss": self.yolo1.metrics["conf_loss"]+self.yolo2.metrics["conf_loss"]+self.yolo3.metrics["conf_loss"],
                "cls_loss": self.yolo1.metrics["cls_loss"]+self.yolo2.metrics["cls_loss"]+self.yolo3.metrics["cls_loss"],
                "cls_acc": self.yolo1.metrics["cls_acc"]+self.yolo2.metrics["cls_acc"]+self.yolo3.metrics["cls_acc"],
                "precision": self.yolo1.metrics["precision"]+self.yolo2.metrics["precision"]+self.yolo3.metrics["precision"],
            }

        return torch.cat([y1, y2, y3], 1), (loss1 + loss2 + loss3)


class TinyYolo(nn.Module):
    def __init__(self, ncls, block):
        super().__init__()
        # x,y,w,h,conf,a,cls 再乘3*6

        # 416
        anchors_list = [[[10, 14], [23, 27], [37, 58]],
                        [[81, 82], [135, 169], [344, 319]]]
        angles_list = [-np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2]

        self.yolobody = block
        self.yolo2 = YoloLayer(num_classes=ncls, anchors=anchors_list[0],
                               angles = angles_list,
                               stride=16, scale_x_y=1.1, ignore_thresh=0.6)
        self.yolo3 = YoloLayer(num_classes=ncls, anchors=anchors_list[1],
                               angles= angles_list,
                               stride=32, scale_x_y=1.05, ignore_thresh=0.6)
        self.metrics = {}

    def forward(self, i, target=None):
        p5, p3 = self.yolobody(i)
        # if(target == None):
        #     # y1 [batch, grid*grid*num_anchors, (x,y,w,h,a,conf,cls)] 绝对坐标
        #     y1 = self.yolo2(p3, target)  # grid = 76
        #     y2 = self.yolo3(p5, target)  # grid = 38
        #     return torch.cat([y1, y2], 1)

        # y1 [batch, grid*grid*num_anchors, (x,y,w,h,a,conf,cls)]
        y1, loss1 = self.yolo2(p3, target)  # grid = 76
        y2, loss2 = self.yolo3(p5, target)  # grid = 38

        if target != None:
            self.metrics = {
                "loss": self.yolo2.metrics["loss"]+self.yolo3.metrics["loss"],
                "reg_loss": self.yolo2.metrics["reg_loss"]+self.yolo3.metrics["reg_loss"],
                "conf_loss": self.yolo2.metrics["conf_loss"]+self.yolo3.metrics["conf_loss"],
                "cls_loss": self.yolo2.metrics["cls_loss"]+self.yolo3.metrics["cls_loss"],
                "cls_acc": self.yolo2.metrics["cls_acc"]+self.yolo3.metrics["cls_acc"],
                "precision": self.yolo2.metrics["precision"]+self.yolo3.metrics["precision"],
            }
        return torch.cat([y1, y2], 1), (loss1 + loss2)

class YOLO2_ghost(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # x,y,w,h,conf,a,cls 再乘3*6
        output_ch = (5 + 1 + n_classes) * 3 * 6
        radian = np.pi / 180

        # 416*416
        anchors_list = [[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]]

        self.backbone = GhostNet()
        # 对应三个分支的通道数
        self.neck = Neck_PanNet(40, 112, 160)
        self.head = Headv4(output_ch)
        self.yolo1 = YoloLayer(num_classes=n_classes, anchors=anchors_list[0],
                               angles=[-radian * 60, -radian * 30, 0,
                                       radian * 30, radian * 60, radian * 90],
                               stride=8, scale_x_y=1.2, ignore_thresh=0.6)
        self.yolo2 = YoloLayer(num_classes=n_classes, anchors=anchors_list[1],
                               angles=[-radian * 60, -radian * 30, 0,
                                       radian * 30, radian * 60, radian * 90],
                               stride=16, scale_x_y=1.1, ignore_thresh=0.6)
        self.yolo3 = YoloLayer(num_classes=n_classes, anchors=anchors_list[2],
                               angles=[-radian * 60, -radian * 30, 0,
                                       radian * 30, radian * 60, radian * 90],
                               stride=32, scale_x_y=1.05, ignore_thresh=0.6)
        self.metrics = {}

    def forward(self, i, target=None):
        if target is None:
            inference = True
        else:
            inference = False
        d3, d4, d5 = self.backbone(i)
        x20, x13, x6 = self.neck(d5, d4, d3, inference)
        x2, x10, x18 = self.head(x20, x13, x6)
        if(target == None):
            # y1 [batch, grid*grid*num_anchors, (x,y,w,h,a,conf,cls)]
            y1 = self.yolo1(x2, target)  # grid = 76
            y2 = self.yolo2(x10, target)  # grid = 38
            y3 = self.yolo3(x18, target)  # grid = 19
            return torch.cat([y1, y2, y3], 1)
        # y1 [batch, grid*grid*num_anchors, (x,y,w,h,a,conf,cls)]
        y1, loss1 = self.yolo1(x2, target)  # grid = 76
        y2, loss2 = self.yolo2(x10, target)  # grid = 38
        y3, loss3 = self.yolo3(x18, target)  # grid = 19
        self.metrics = {
            "loss": self.yolo1.metrics["loss"]+self.yolo2.metrics["loss"]+self.yolo3.metrics["loss"],
            "reg_loss": self.yolo1.metrics["reg_loss"]+self.yolo2.metrics["reg_loss"]+self.yolo3.metrics["reg_loss"],
            "conf_loss": self.yolo1.metrics["conf_loss"]+self.yolo2.metrics["conf_loss"]+self.yolo3.metrics["conf_loss"],
            "cls_loss": self.yolo1.metrics["cls_loss"]+self.yolo2.metrics["cls_loss"]+self.yolo3.metrics["cls_loss"],
            "cls_acc": self.yolo1.metrics["cls_acc"]+self.yolo2.metrics["cls_acc"]+self.yolo3.metrics["cls_acc"],
            "precision": self.yolo1.metrics["precision"]+self.yolo2.metrics["precision"]+self.yolo3.metrics["precision"],
        }
        return torch.cat([y1, y2, y3], 1), (loss1 + loss2 + loss3)





