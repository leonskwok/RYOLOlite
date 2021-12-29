from model.utils import *


class Head_tiny(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        # 第一分支yolohead
        self.conv1 = Conv(384, 256, 3, 1, 'leaky')
        self.conv2 = Conv(256, out_ch, 1, 1, 'linear', bn=False, bias=True)
        # 第二分支yolohead
        self.conv3 = Conv(256, 512, 3, 1, 'leaky')
        self.conv4 = Conv(512, out_ch, 1, 1, 'linear', bn=False, bias=True)

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
    def __init__(self, out_ch):
        super().__init__()
        # 第一分支yolohead
        self.conv1 = Conv(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv(256, out_ch, 1, 1, 'linear', bn=False, bias=True)

        # 第二分支yolohead
        self.conv3 = Conv(256, 512, 3, 1, 'leaky')
        self.conv4 = Conv(512, out_ch, 1, 1, 'linear', bn=False, bias=True)

        # 第三分支yolohead
        self.conv5 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv(1024, out_ch, 1, 1, 'linear', bn=False, bias=True)

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

