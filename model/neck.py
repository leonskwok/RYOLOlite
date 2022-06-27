from model.utils import *

class Neck_tiny(nn.Module):
    # [256, 512]
    def __init__(self, filter_in, filter_out,relu):
        super().__init__()      
        in_chs1, in_chs2 = filter_in
        out_chs1, out_chs2 = filter_out
        # 13*13
        self.conv1 = Conv(in_chs2, 256, 1, 1,  "bn", relu)
        self.conv2 = Conv(256, 128, 1, 1, "bn", relu)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
   
    def forward(self, feat1, feat2):
        # conv1x1  13,13,512 -> 13,13,256
        x1 = self.conv1(feat2)
        # conv1x1  13, 13, 256 -> 13, 13, 128
        x2 = self.conv2(x1)
        # upsample 13,13,128 -> 26,26,128
        x3 = self.upsample(x2)
        # 26,26,256 + 26,26,128 -> 26,26,384
        # contact
        x3 = torch.cat([x3, feat1], axis=1)
        return x3, x1


class Neck_PanNet(nn.Module):  
    def __init__(self, filter_in, filter_out, relu):
        super().__init__()
        # 256,512,1024
        in_chs0, in_chs1, in_chs2 = filter_in
        # 128,256,512
        out_chs0, out_chs1, out_chs2 = filter_out

        # conv x3
        self.conv1 = Conv(in_chs2, 512, 1, 1,'bn',relu)
        self.conv2 = Conv(512, 1024, 3, 1, 'bn',relu)
        self.conv3 = Conv(1024, 512, 1, 1, 'bn', relu)
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        # Concat + conv x3
        self.conv4 = Conv(2048, 512, 1, 1, 'bn', relu)
        self.conv5 = Conv(512, 1024, 3, 1,'bn', relu)
        self.conv6 = Conv(1024, 512, 1, 1, 'bn', relu)
        # conv + upsample
        self.conv7 = Conv(512, 256, 1, 1, 'bn', relu)
        # self.upsample1 = Upsample()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # 骨干网第二分支出来的conv
        self.conv8 = Conv(in_chs1, 256, 1, 1, 'bn', relu)
        # Concat + conv x5
        self.conv9 = Conv(512, 256, 1, 1, 'bn', relu)
        self.conv10 = Conv(256, 512, 3, 1, 'bn', relu)
        self.conv11 = Conv(512, 256, 1, 1, 'bn', relu)
        self.conv12 = Conv(256, 512, 3, 1,'bn', relu)
        self.conv13 = Conv(512, 256, 1, 1, 'bn', relu)

        # conv + upsample
        self.conv14 = Conv(256, 128, 1, 1, 'bn', relu)
        # self.upsample2 = Upsample()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # 骨干网第一分支出来的conv
        self.conv15 = Conv(in_chs0, 128, 1, 1, 'bn', relu)
        # Concat + conv x5
        self.conv16 = Conv(256, 128, 1, 1, 'bn', relu)
        self.conv17 = Conv(128, 256, 3, 1,'bn', relu)
        self.conv18 = Conv(256, 128, 1, 1, 'bn', relu)
        self.conv19 = Conv(128, 256, 3, 1, 'bn',relu)
        self.conv20 = Conv(256, out_chs0, 1, 1, 'bn', relu)

        # 第二分支yolohead
        # downsample
        self.conv21 = Conv(128, 256, 3, 2, 'bn', relu)
        # concat + conv x5
        self.conv22 = Conv(512, 256, 1, 1, 'bn',relu)
        self.conv23 = Conv(256, 512, 3, 1, 'bn', relu)
        self.conv24 = Conv(512, 256, 1, 1, 'bn', relu)
        self.conv25 = Conv(256, 512, 3, 1, 'bn', relu)
        self.conv26 = Conv(512, out_chs1, 1, 1, 'bn', relu)

        # 第三分支yolohead
        # downsample
        self.conv27 = Conv(256, 512, 3, 2, 'bn', relu)
        # concat + conv x5
        self.conv28 = Conv(1024, 512, 1, 1, 'bn', relu)
        self.conv29 = Conv(512, 1024, 3, 1, 'bn', relu)
        self.conv30 = Conv(1024, 512, 1, 1, 'bn', relu)
        self.conv31 = Conv(512, 1024, 3, 1, 'bn', relu)
        self.conv32 = Conv(1024, out_chs2, 1, 1, 'bn', relu)

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



class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()
    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out


class Neck_FPN(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, fileters_in, fileters_out,relu):
        super(Neck_FPN, self).__init__()
        # [256,512,1024]
        in_chs0, in_chs1, in_chs2 = fileters_in
        # [128,256,512]
        out_chs0, out_chs1, out_chs2 = fileters_out

        # 13*13
        self.__conv_set_0 = nn.Sequential(
            Conv(in_chs2, 512, 1, 1,  "bn", relu),
            Conv(512, 1024, 3, 1,  "bn", relu),
            Conv(1024, 512, 1, 1, "bn", relu),
            Conv(512, 1024, 3, 1, "bn", relu),
            Conv(1024, out_chs2, 1, 1,  "bn", relu),
        )

        self.__conv0 = Conv(out_chs2, 256, 1, 1,  "bn", relu)
        self.__upsample0 = nn.Upsample(scale_factor=2, mode="nearest")
        self.__route0 = Route()

        # 26*26
        self.__conv_set_1 = nn.Sequential(
            Conv(in_chs1+256, 256, 1, 1, "bn", relu),
            Conv(256, 512, 3, 1,  "bn", relu),
            Conv(512, 256, 1, 1,  "bn", relu),
            Conv(256, 512, 3, 1, "bn", relu),
            Conv(512, out_chs1, 1, 1, "bn", relu),
        )

        self.__conv1 = Conv(256, 128, 1, 1, "bn", relu)
        self.__upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.__route1 = Route()

        # 52*52
        self.__conv_set_2 = nn.Sequential(
            Conv(in_chs0+128, 128, 1, 1,  "bn", relu),
            Conv(128, 256, 3, 1,  "bn", relu),
            Conv(256, 128, 1, 1, "bn", relu),
            Conv(128, 256, 3, 1,  "bn", relu),
            Conv(256, out_chs0, 1, 1,  "bn", relu),
        )

    def forward(self, x0, x1, x2):
        # 13*13
        r2 = self.__conv_set_0(x2)

        # 26*26
        r1 = self.__conv0(r2)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
 
        # 52*52
        r0 = self.__conv1(r1)
        r0 = self.__upsample1(r0)
        x0 = self.__route1(x0, r0)
        r0 = self.__conv_set_2(x0)

        return r0, r1, r2 




