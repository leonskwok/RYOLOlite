from model.utils import *

# for RTOLO-SG
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




# for RYOLO-D
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


# for RYOLO-G
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

# for RYOLO
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
        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)
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


# for RYOLO-v4
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
