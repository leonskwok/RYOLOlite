from model.utils import *


class DarkNet53tiny_DCSP(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_DCSP, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)


        # 104,104,64 -> 52,52,128
        self.resblock_body1 = tinyResblock_DCS(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = tinyResblock_DCS(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = tinyResblock_DCS(256, 256)
        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class tinyResblock_DCS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tinyResblock_DCS, self).__init__()
        self.out_channels = out_channels
        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//4, out_channels//4, 3)
        self.conv4 = BasicConv(out_channels//4*3, out_channels, 1)

        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]

        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        x = torch.split(x, c//4, dim=1)[1]
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)

        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class DarkNet53tiny_Ghostbottle23(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghostbottle23, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GtinyResblockfirst(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GtinyResblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_GhostbottleResAtte(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_GhostbottleResAtte, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        spatialatte = SpatialAttention()
        self.resblock_body1 = GtinyResblockAtte(64, 64, spatialatte)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblockAtte(128, 128,None)
        # 26,26,256 -> 13,13,512
        channelatte = ChannelAttention(256)
        self.resblock_body3 = GtinyResblockAtte(256, 256,channelatte)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2











class DarkNet53tiny_GhostbottleAtte(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_GhostbottleAtte, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GtinyResblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GtinyResblock(256, 256)
        self.num_features = 1

        self.spatialatte = SpatialAttention()
        self.channelatte = ChannelAttention(512)

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        x = self.spatialatte(x)*x
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        x = self.channelatte(x)*x
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_GhostbottleTrip(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_GhostbottleTrip, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GtinyResblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GtinyResblock(256, 256)
        self.num_features = 1

        self.atteblock1 = TripletAttention()
        self.atteblock2 = TripletAttention()

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        x = self.atteblock1(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        x = self.atteblock2(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2



class DarkNet53tiny_Ghostbottle(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghostbottle, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GtinyResblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GtinyResblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_Ghostbottle_all(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghostbottle_all, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)
        # self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Gtinybottleblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Gtinybottleblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Gtinybottleblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_Ghostbottle_all_G(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghostbottle_all_G, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Gtinybottleblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Gtinybottleblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Gtinybottleblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_Ghostbottle_all_G2(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghostbottle_all_G2, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Gtinybottleblock2(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Gtinybottleblock2(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Gtinybottleblock2(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_SqueezeGhost(nn.Module):

    def __init__(self):
        super(DarkNet53tiny_SqueezeGhost, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3 = SqGhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = SqGhostCSPblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = SqGhostCSPblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = SqGhostCSPblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_SqueezeGhost2(nn.Module):

    def __init__(self):
        super(DarkNet53tiny_SqueezeGhost2, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3 = SqGhostConv2(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = SqGhostCSPblock2(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = SqGhostCSPblock2(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = SqGhostCSPblock2(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_Mobile(nn.Module):

    def __init__(self):
        super(DarkNet53tiny_Mobile, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3 = MobileBlock(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = MobileCSPblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = MobileCSPblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = MobileCSPblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2



class DarkNet53tiny_Ghostbottle_all_G3(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghostbottle_all_G3, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Gtinybottleblock3(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Gtinybottleblock3(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Gtinybottleblock3(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_GhostALLbottle(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_GhostALLbottle, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3_0 = BasicConv(512,256,1,1)
        self.conv3 = GhostConv(256, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Gtinyallbottleblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Gtinyallbottleblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Gtinyallbottleblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3_0(x)
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_GhostALLbottle2(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_GhostALLbottle2, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        # self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.conv3_0 = BasicConv(512, 256, 1, 1)
        self.conv3 = GhostConv(256, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Gtinyallbottleblock2(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Gtinyallbottleblock2(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Gtinyallbottleblock2(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3_0(x)
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2



class DarkNet53tiny_Ghost(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghost, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GtinyResblockfirst(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblockfirst(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GtinyResblockfirst(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class DarkNet53tiny_Ghost2(nn.Module):

    def __init__(self):
        super(DarkNet53tiny_Ghost2, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GhostResblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GhostResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GhostResblock(256, 256)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


#------------------------------------------------#
class DarkNet53tiny_Ghost_Atte(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_Ghost_Atte, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 =GhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = GtinyResblockfirst(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = GtinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = GtinyResblock(256, 256)
        self.num_features = 1

        self.spatialatte = SpatialAttention()
        self.channelatte = ChannelAttention(512)

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        x = self.spatialatte(x)*x
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        x = self.channelatte(x)*x
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class GtinyResblockfirst(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GtinyResblockfirst, self).__init__()
        self.out_channels = out_channels
        self.conv1_1 = GhostConv(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = BasicConv(out_channels, out_channels, 1)
        self.conv2_0 = GhostConv(out_channels//2, out_channels//2, 3)
        self.conv3 = GhostConv(out_channels//2, out_channels//2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class GhostResblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GhostResblock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = GhostConv(in_channels, out_channels, kernel_size=3)
        self.conv2 = GhostConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = GhostConv(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class GtinyResblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GtinyResblock, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels,in_channels//2,kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2,out_channels,kernel_size=3)
        self.conv1_2 = BasicConv(out_channels, out_channels, 1)
        self.conv2_0 = GhostConv(out_channels//2, out_channels//2, 3)
        self.conv3 = GhostConv(out_channels//2, out_channels//2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class Gtinybottleblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gtinybottleblock, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels, in_channels//2, kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2, out_channels, kernel_size=3)
        self.conv1_2 = BasicConv(out_channels, out_channels, 1)

        self.conv2_1 = BasicConv(out_channels//2,out_channels//4,kernel_size=1)
        self.conv2_0 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv3_0 = BasicConv(out_channels//2, out_channels//4,kernel_size=1)
        self.conv3 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_1(x)
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3_0(x)
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class Gtinybottleblock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gtinybottleblock2, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels, in_channels//2, kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2, out_channels, kernel_size=3)

        self.conv2_1 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv2_0 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv3_0 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv3 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_1(x)
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3_0(x)
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class SqGhostCSPblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqGhostCSPblock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = SqGhostConv(in_channels, out_channels, 3)
        self.conv2 = SqGhostConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = SqGhostConv(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class SqGhostCSPblock2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SqGhostCSPblock2, self).__init__()
        self.out_channels = out_channels
        self.conv1 = SqGhostConv2(in_channels, out_channels, 3)
        self.conv2 = SqGhostConv2(out_channels // 2, out_channels // 2, 3)
        self.conv3 = SqGhostConv2(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat



class MobileCSPblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MobileCSPblock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = MobileBlock(in_channels, out_channels, 3)
        self.conv2 = MobileBlock(out_channels // 2, out_channels // 2, 3)
        self.conv3 = MobileBlock(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat



class Gtinybottleblock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gtinybottleblock3, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels, in_channels//2, kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2, out_channels, kernel_size=3)

        self.conv2_1 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv2_0 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv3_0 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv3 = GhostConv(out_channels//4, out_channels//2, 3)

        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_1(x)
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3_0(x)
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat

class Gtinyallbottleblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gtinyallbottleblock, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels, in_channels//2, kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2, out_channels, kernel_size=3)

        self.conv2_1 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv2_0 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv3_0 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv3 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_1(x)
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3_0(x)
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class Gtinyallbottleblock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gtinyallbottleblock2, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels, in_channels//2, kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2, out_channels, kernel_size=3)
        self.conv1_2 = BasicConv(out_channels, out_channels, kernel_size=1)

        self.conv2_1 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv2_0 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv3_0 = BasicConv(
            out_channels//2, out_channels//4, kernel_size=1)
        self.conv3 = GhostConv(out_channels//4, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_1(x)
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3_0(x)
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat

class GtinyResblockAtte(nn.Module):
    def __init__(self, in_channels, out_channels,atteblock):
        super(GtinyResblockAtte, self).__init__()
        self.out_channels = out_channels
        # bottlenet结构
        self.conv1_0 = BasicConv(in_channels, in_channels//2, kernel_size=1)
        self.conv1_1 = GhostConv(in_channels//2, out_channels, kernel_size=3)
        self.conv1_2 = BasicConv(out_channels, out_channels, 1)
        self.conv2_0 = GhostConv(out_channels//2, out_channels//2, 3)
        self.conv3 = GhostConv(out_channels//2, out_channels//2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])
        self.atte = atteblock

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2_0(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        if self.atte is not None:
            x = self.atte(x)*x
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat

#------------------------------------------------#


class DarkNet53tiny_atte(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_atte, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = tinyResblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = tinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = tinyResblock(256, 256)
        self.num_features = 1

        self.spatialatte=SpatialAttention()
        self.channelatte=ChannelAttention(512)

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        x = self.spatialatte(x)*x
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        x = self.channelatte(x)*x
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


#------------------------------------------------#

class Darknet53_tiny_D(nn.Module):
    def __init__(self):
        super().__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        self.spatialatte = SpatialAttention()
        self.channelatte = ChannelAttention(512)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = ResBlock_D(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = ResBlock_D(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = tinyResblock(256, 256)

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x= self.resblock_body1(x)
        # x= self.spatialatte(x)
        # 52,52,128 -> 26,26,256
        x= self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # x=self.channelatte(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(inplanes, planes,kernel_size=1,stride=1,activation='relu',bn=True)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=2,activation='relu',bn=True)
        self.conv3 = Conv(planes, planes * self.expansion,kernel_size=1,stride=1,activation='linear',bn=True)

        self.avgpool=nn.AvgPool2d([2,2],[2,2])
        self.conv4 = Conv(inplanes,planes*self.expansion,kernel_size=1,stride=2,activation='linear',bn=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResBlock_D(nn.Module):
    def __init__(self,in_chs,out_chs):
        super().__init__()

        self.conv1 = BasicConv(in_chs, out_chs//2, kernel_size=1, stride=1)
        self.conv2 = BasicConv(out_chs//2, out_chs//2, kernel_size=3, stride=2)
        self.conv3 = BasicConv(out_chs//2, out_chs, kernel_size=1, stride=1)

        self.avgpool=nn.AvgPool2d([2,2],[2,2])
        self.conv4 = BasicConv(in_chs, out_chs, kernel_size=1, stride=1)

    def forward(self, x):
        route = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        route1=x
        x = self.avgpool(route)
        x = self.conv4(x)
        x = torch.cat([route1,x],dim=1)
        return x


#---------------------------------------------------#
#   Yolov4_Darknet53相关
#---------------------------------------------------#
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Yolov4_DownSampleFirst(3, 64)
        self.down2 = Yolov4_DownSample(64, 128, 2)
        self.down3 = Yolov4_DownSample(128, 256, 8)
        self.down4 = Yolov4_DownSample(256, 512, 8)
        self.down5 = Yolov4_DownSample(512, 1024, 4)

    def forward(self, i):
        d1 = self.down1(i)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return d3, d4, d5

class Yolov4_ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv(ch, ch, 1, 1, "mish"))
            resblock_one.append(Conv(ch, ch, 3, 1, "mish"))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class Yolov4_DownSampleFirst(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Yolov4_DownSampleFirst, self).__init__()
        self.conv0 = Conv(in_channels, 32, 3, 1, "mish")
        self.conv1 = Conv(32, 64, 3, 2, "mish")
        self.conv2 = Conv(64, 64, 1, 1, "mish")
        self.conv4 = Conv(64, 64, 1, 1, "mish")  # 這邊從1延伸出來
        self.conv5 = Conv(64, 32, 1, 1, "mish")
        self.conv6 = Conv(32, 64, 3, 1, "mish")  # 這邊有shortcut從4連過來
        self.conv8 = Conv(64, 64, 1, 1, "mish")
        # 這邊的input是conv2+conv8 所以有128
        self.conv10 = Conv(128, out_channels, 1, 1, "mish")

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        x4 = self.conv5(x3)
        x5 = x3 + self.conv6(x4)
        x6 = self.conv8(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x7 = self.conv10(x6)
        return x7


class Yolov4_DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks):
        super(Yolov4_DownSample, self).__init__()
        self.conv1 = Conv(in_channels, in_channels * 2, 3, 2, "mish")
        self.conv2 = Conv(in_channels * 2, in_channels, 1, 1, "mish")
        self.conv4 = Conv(in_channels * 2, in_channels,
                          1, 1, "mish")  # 這邊從1延伸出來
        self.resblock = Yolov4_ResBlock(ch=in_channels, nblocks=res_blocks)
        self.conv11 = Conv(in_channels, in_channels, 1, 1, "mish")
        self.conv13 = Conv(in_channels * 2, out_channels, 1,
                           1, "mish")  # 這邊的input是conv2+conv11 所以有128

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        r = self.resblock(x3)
        x4 = self.conv11(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv13(x4)
        return x5

#---------------------------------------------------#
#   yolov4_tiny的主干网
#---------------------------------------------------#
class DarkNet53tiny(nn.Module):
    def __init__(self):
        super(DarkNet53tiny, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = tinyResblock(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = tinyResblock(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = tinyResblock(256, 256)
        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2

class tinyResblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tinyResblock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = BasicConv(in_channels, out_channels, 3)
        self.conv2 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


#---------------------------------------------------#
#   引入ghost模块的yolov4_tiny主干网
#---------------------------------------------------#
class DarkNet53tiny_ghost(nn.Module):
    def __init__(self):
        super(DarkNet53tiny_ghost, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, 1, dw_size=3)
        # 104,104,64 -> 52,52,128
        self.resblock_body1 = tinyResblock_ghost(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = tinyResblock_ghost(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = tinyResblock_ghost(256, 256)
        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)
        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)
        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2

# 使用GhostConv代替Resblock的中间分支
class tinyResblock_ghost(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tinyResblock_ghost, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)
        self.conv2 = GhostConv(out_channels//2, out_channels//2, 1)
        self.conv3 = BasicConv(out_channels//2, out_channels, 1)

        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim=1)[1]
        # 对主干部分进行ghost卷积
        x = self.conv2(x)
        # 对相接后的结果进行1x1卷积
        x = self.conv3(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


#---------------------------------------------------#
#   GhostNet主干网
#---------------------------------------------------#
class GhostNet(nn.Module):
    def __init__(self, width=1.0):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t, c, SE, s
            # stage1
            [[3,  16,  16, 0, 1]],
            # stage2
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            # stage3
            [[5,  72,  40, 0.25, 2]],
            [[5, 120,  40, 0.25, 1]],
            # stage4
            [[3, 240,  80, 0, 2]],
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
                [3, 184,  80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1]
             ],
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1]
             ]
        ]
        # building first layer
        output_channel = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = make_divisible(c * width, 4)
                hidden_channel = make_divisible(exp_size * width, 4)
                layers.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        x = self.blocks[3](x)
        x = self.blocks[4](x)
        out1 = x

        x = self.blocks[5](x)
        x = self.blocks[6](x)
        out2 = x

        x = self.blocks[7](x)
        x = self.blocks[8](x)
        out3 = x

        # 52, 52, 40； 26, 26, 112； 13, 13, 160
        return out1, out2, out3


class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, se_ratio=0., act_layer=nn.ReLU,):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # Point-wise expansion
        self.ghost1 = GhostConv(in_chs, mid_chs, relu=True)
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size-1)//2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
        # Point-wise linear projection
        self.ghost2 = GhostConv(mid_chs, out_chs, relu=False)
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x
