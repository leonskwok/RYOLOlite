from model.utils import *

class Neck_tiny(nn.Module):
    # [256, 512]
    def __init__(self, attblock=None):
        super().__init__()
        # yolohead2分支
        self.conv1 = BasicConv(512, 256, 1)
        self.conv2 = BasicConv(256, 128, 1)
        self.upsample=nn.Upsample(scale_factor=2, mode='nearest')

        self.attblock = attblock
        if self.attblock != None:
            self.feat1_att = self.attblock(256)
            self.feat2_att = self.attblock(512)
            self.upsample_att = self.attblock(128)
        
    def forward(self, feat1, feat2):
        if self.attblock != None:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
        # conv1x1  13,13,512 -> 13,13,256
        x1 = self.conv1(feat2)
        # conv1x1  13, 13, 256 -> 13, 13, 128
        x2 = self.conv2(x1)
        # upsample 13,13,128 -> 26,26,128
        x3 = self.upsample(x2)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if self.attblock != None:
            x3 = self.upsample_att(x3)
        # contact
        x3 = torch.cat([x3, feat1], axis=1)
        return x3, x1


class Neck_tiny_TripeAtte(nn.Module):
    # [256, 512]
    def __init__(self):
        super().__init__()
        # yolohead2分支
        self.conv1 = BasicConv(512, 256, 1)
        self.conv2 = BasicConv(256, 128, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.attblock = TripletAttention()

    def forward(self, feat1, feat2):
        if self.attblock != None:
            feat1 = self.attblock(feat1)
            feat2 = self.attblock(feat2)
        # conv1x1  13,13,512 -> 13,13,256
        x1 = self.conv1(feat2)
        # conv1x1  13, 13, 256 -> 13, 13, 128
        x2 = self.conv2(x1)
        # upsample 13,13,128 -> 26,26,128
        x3 = self.upsample(x2)
        # 26,26,256 + 26,26,128 -> 26,26,384
        # if self.attblock != None:
        #     x3 = self.attblock(x3)
        # contact
        x3 = torch.cat([x3, feat1], axis=1)
        return x3, x1


class Neck_PanNet(nn.Module):
    # 1024 512,256
    def __init__(self, in_chs1,in_chs2,in_chs3):
        super().__init__()
        # conv x3
        self.conv1 = Conv(in_chs3, 512, 1, 1, 'leaky')
        self.conv2 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        # Concat + conv x3
        self.conv4 = Conv(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv(1024, 512, 1, 1, 'leaky')
        # conv + upsample
        self.conv7 = Conv(512, 256, 1, 1, 'leaky')
        # self.upsample1 = Upsample()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # 骨干网第二分支出来的conv
        self.conv8 = Conv(in_chs2, 256, 1, 1, 'leaky')
        # Concat + conv x5
        self.conv9 = Conv(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv(512, 256, 1, 1, 'leaky')

        # conv + upsample
        self.conv14 = Conv(256, 128, 1, 1, 'leaky')
        # self.upsample2 = Upsample()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # 骨干网第一分支出来的conv
        self.conv15 = Conv(in_chs1, 128, 1, 1, 'leaky')
        # Concat + conv x5
        self.conv16 = Conv(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv(256, 128, 1, 1, 'leaky')

        # 第二分支yolohead
        # downsample
        self.conv21 = Conv(128, 256, 3, 2, 'leaky')
        # concat + conv x5
        self.conv22 = Conv(512, 256, 1, 1, 'leaky')
        self.conv23 = Conv(256, 512, 3, 1, 'leaky')
        self.conv24 = Conv(512, 256, 1, 1, 'leaky')
        self.conv25 = Conv(256, 512, 3, 1, 'leaky')
        self.conv26 = Conv(512, 256, 1, 1, 'leaky')

        # 第三分支yolohead
        # downsample
        self.conv27 = Conv(256, 512, 3, 2, 'leaky')
        # concat + conv x5
        self.conv28 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv29 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv30 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv31 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv32 = Conv(1024, 512, 1, 1, 'leaky')

    def forward(self, feat1, feat2, feat3):
        # yolohead3分支
        # conv x3
        x1 = self.conv1(feat3)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        # Concat + conv x3
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # yolohead2分支
        # conv + upsample
        x7 = self.conv7(x6)
        # up = self.upsample1(x7, feat2.size(), inference)
        up = self.upsample1(x7)
        # 骨干网第二分支出来的conv
        # conv
        x8 = self.conv8(feat2)
        # concat + conv x5
        x8 = torch.cat([x8, up], dim=1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)

        # yolohead1 分支
        # conv + upsample
        x14 = self.conv14(x13)
        # up = self.upsample2(x14, feat1.size(), inference)
        up = self.upsample2(x14)

        # 骨干网第一分支出来的conv
        x15 = self.conv15(feat1)
        # concat + conv x5
        x15 = torch.cat([x15, up], dim=1)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)

        # 第二分支yolohead
        # downsample
        x21 = self.conv21(x20)
        # concat + conv x5
        x21 = torch.cat([x21, x13], dim=1)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        x25 = self.conv25(x24)
        x26 = self.conv26(x25)

        # 第三分支yolohead
        # downsample
        x27 = self.conv27(x26)
        # concat + conv x5
        x27 = torch.cat([x27, x6], dim=1)
        x28 = self.conv28(x27)
        x29 = self.conv29(x28)
        x30 = self.conv30(x29)
        x31 = self.conv31(x30)
        x32 = self.conv32(x31)

        return x20, x26, x32


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference):
        assert (x.data.dim() == 4)
        _, _, tw, th = target_size

        if inference:
            B, C, W, H = x.size()
            return x.view(B, C, W, 1, H, 1).expand(B, C, W, tw // W, H, th // H).contiguous().view(B, C, tw, th)
        else:
            return F.interpolate(x, size=(tw, th), mode='nearest')

