from model.utils import *


class Headv4_tiny(nn.Module):
    def __init__(self, in_filters, out_filters, relu):
        in_chs1, in_chs2 = in_filters
        out_chs2, out_chs2 = out_filters

        super().__init__()
        # 26*26
        self.conv1 = Conv(in_chs1, 256, 3, 1, "bn", relu)
        self.conv2 = Conv(256, out_chs2, 1, 1, None, None)
        # 13*13
        self.conv3 = Conv(in_chs2, 512, 3, 1, "bn", relu)
        self.conv4 = Conv(512, out_chs2, 1, 1, None, None)

    def forward(self, feat1, feat2):
        # 接yolohead1
        # 26,26,384 -> 26,26,256
        x1 = self.conv1(feat1)
        # 26,26,256 -> 26,26,outchs
        x2 = self.conv2(x1)
        # 接yolohead2
        # 13,13,256 -> 13,13,512
        x3 = self.conv3(feat2)
        # 13,13,512 -> 13, 13, outchs
        x4 = self.conv4(x3)
        return x2, x4


class Headv4(nn.Module):
    def __init__(self, in_filters, out_filters, relu):
        super().__init__()
        # 128,256,512
        in_chs0,in_chs1,in_chs2=in_filters
        out_chs0,out_chs1,out_chs2=out_filters
        # 52*52
        self.conv1 = Conv(in_chs0, 256, 3, 1, 'bn', relu)
        self.conv2 = Conv(256, out_chs0, 1, 1)
        # 26*26
        self.conv3 = Conv(in_chs1, 512, 3, 1, 'bn', relu)
        self.conv4 = Conv(512, out_chs1, 1, 1)
        # 13*13
        self.conv5 = Conv(in_chs2, 1024, 3, 1, 'bn', relu)
        self.conv6 = Conv(1024, out_chs2, 1, 1)

    def forward(self, input1, input2, input3):
        # 接yolohead1
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        # 接yolohead2
        x3 = self.conv3(input2)
        x4 = self.conv4(x3)
        # 接yolohead3
        x5 = self.conv5(input3)
        x6 = self.conv6(x5)
        return x2, x4, x6


class Headv3(nn.Module):
    def __init__(self, filters_in, filters_out, relu):
        super(Headv3, self).__init__()

        in_chs0, in_chs1, in_chs2 = filters_in
        out_chs0, out_chs1, out_chs2 = filters_out
        # 52*52
        self.__conv0_0 = Conv(in_chs0, 256, 3, 1, "bn", relu)
        self.__conv0_1 = Conv(256, out_chs2, 1, 1)
        # 26*26
        self.__conv1_0 = Conv(in_chs1, 512, 3, 1, "bn", relu)
        self.__conv1_1 = Conv(512, out_chs1, 1, 1)
        # 13*13
        self.__conv2_0 = Conv(in_chs2, 1024, 3, 1, "bn", relu)
        self.__conv2_1 = Conv(1024, out_chs0, 1, 1)

    def forward(self, x0, x1, x2):
        x0 = self.__conv0_0(x0)
        x0 = self.__conv0_1(x0)

        x1 = self.__conv1_0(x1)
        x1 = self.__conv1_1(x1)

        x2 = self.__conv2_0(x2)
        x2 = self.__conv2_1(x2)

        return x0,x1,x2
