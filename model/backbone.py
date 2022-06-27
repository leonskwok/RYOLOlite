from model.utils import *

# for RTOLO-SG
class CSPDarkNet53tiny_SG(nn.Module):
    def __init__(self, activate):
        super(CSPDarkNet53tiny_SG, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = Conv(3, 32, kernel_size=3, stride=2,norm="bn",activate=activate)
        self.conv2 = Conv(32, 64, kernel_size=3, stride=2,
                          norm="bn", activate=activate)
        # 13,13,512 -> 13,13,512
        self.conv3 = SepGhostConv(512, 512, kernel_size=3)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = CSPblock_SG(64, 64, activate)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = CSPblock_SG(128, 128, activate)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = CSPblock_SG(256, 256, activate)
        self.num_features = 1


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

class CSPblock_SG(nn.Module):
    def __init__(self, in_chs, out_chs, activate):
        super(CSPblock_SG, self).__init__()
        self.out_channels = out_chs
        self.conv1 = SepGhostConv(in_chs, out_chs, 3)
        self.conv2 = SepGhostConv(out_chs // 2, out_chs // 2, 3)
        self.conv3 = SepGhostConv(out_chs // 2, out_chs // 2, 3)
        self.conv4 = Conv(out_chs, out_chs, 1, 1, norm="bn", activate=activate)
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
class CSPDarkNet53tiny_D(nn.Module):

    def __init__(self, relu):
        super(CSPDarkNet53tiny_D, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = Conv(3, 32, 3, 2, "bn", relu)
        self.conv2 = Conv(32, 64, 3, 2,"bn", relu)
        # 13,13,512 -> 13,13,512
        self.conv3 = MobileConv(512, 512, 3, 1, relu)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = CSPblock_D(64, 64, relu)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = CSPblock_D(128, 128, relu)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = CSPblock_D(256, 256, relu)
        self.num_features = 1

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


class CSPblock_D(nn.Module):
    def __init__(self, in_chs, out_chs, relu):
        super(CSPblock_D, self).__init__()
        self.out_channels = out_chs
        self.conv1 = MobileConv(in_chs, out_chs, 3, 1, relu)
        self.conv2 = MobileConv(out_chs // 2, out_chs //2, 3, 1,relu)
        self.conv3 = MobileConv(out_chs // 2, out_chs // 2, 3, 1, relu)
        self.conv4 = Conv(out_chs, out_chs, 1, 1, "bn", relu)
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
class CSPDarkNet53tiny_G(nn.Module):
    def __init__(self, relu):
        super(CSPDarkNet53tiny_G, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = Conv(3, 32, 3, 2, "bn", relu )
        self.conv2 = Conv(32, 64, 3, 2, "bn",relu)
        # 13,13,512 -> 13,13,512
        self.conv3 = GhostConv(512, 512, 3, 2, 3, 1, relu)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = CSPblock_G(64, 64,  relu)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = CSPblock_G(128, 128,  relu)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = CSPblock_G(256, 256,  relu)
        self.num_features = 1

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

class CSPblock_G(nn.Module):
    def __init__(self, in_chs, out_chs, relu):
        super(CSPblock_G, self).__init__()
        self.out_channels = out_chs
        self.conv1 = GhostConv(in_chs, out_chs, 3, 2, 3, 1, relu)
        self.conv2 = GhostConv(out_chs // 2, out_chs // 2, 3, 2, 3, 1, relu)
        self.conv3 = GhostConv(out_chs // 2, out_chs // 2, 3, 2, 3, 1, relu)
        self.conv4 = Conv(out_chs, out_chs, 1, 1, "bn", relu)
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
class CSPDarkNet53tiny(nn.Module):
    def __init__(self, relu):
        super(CSPDarkNet53tiny, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = Conv(3, 32, 3, 2, "bn", relu)
        self.conv2 = Conv(32, 64, 3, 2, "bn", relu)
        # 13,13,512 -> 13,13,512
        self.conv3 = Conv(512, 512, 3, 1, "bn", relu)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = CSPblock(64, 64, relu)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = CSPblock(128, 128, relu)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = CSPblock(256, 256, relu)
        self.num_features = 1


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


class CSPblock(nn.Module):
    def __init__(self, in_chs, out_chs, relu):
        super(CSPblock, self).__init__()
        self.out_channels = out_chs
        self.conv1 = Conv(in_chs, out_chs, 3, 1, "bn", relu)
        self.conv2 = Conv(out_chs // 2, out_chs // 2, 3,
                          1, "bn", relu)
        self.conv3 = Conv(out_chs // 2, out_chs // 2, 3, 1, "bn", relu)
        self.conv4 = Conv(out_chs, out_chs, 1, 1, "bn", relu)
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
class CSPDarknet53(nn.Module):
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
        # 52*52*256, 26*26*512, 13*13*1024
        return d3, d4, d5

class Yolov4_ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv(ch, ch, 1, 1, norm ="bn", activate="mish"))
            resblock_one.append(Conv(ch, ch, 3, 1, norm ="bn", activate= "mish"))
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
        self.conv0 = Conv(in_channels, 32, 3, 1, "bn", "mish")
        self.conv1 = Conv(32, 64, 3, 2, "bn", "mish")
        self.conv2 = Conv(64, 64, 1, 1, "bn","mish")
        self.conv4 = Conv(64, 64, 1, 1, "bn","mish")  # 這邊從1延伸出來
        self.conv5 = Conv(64, 32, 1, 1, "bn","mish")
        self.conv6 = Conv(32, 64, 3, 1, "bn","mish")  # 這邊有shortcut從4連過來
        self.conv8 = Conv(64, 64, 1, 1, "bn", "mish")
        # 這邊的input是conv2+conv8 所以有128
        self.conv10 = Conv(128, out_channels, 1, 1, "bn", "mish")

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
    def __init__(self, in_chs, out_chs, res_blocks):
        super(Yolov4_DownSample, self).__init__()
        self.conv1 = Conv(in_chs, in_chs * 2, 3, 2, "bn","mish")
        self.conv2 = Conv(
            in_chs * 2, in_chs, 1, 1, "bn", "mish")
        self.conv4 = Conv(in_chs * 2, in_chs,
                          1, 1, "bn","mish")  # 這邊從1延伸出來
        self.resblock = Yolov4_ResBlock(ch=in_chs, nblocks=res_blocks)
        self.conv11 = Conv(
            in_chs, in_chs, 1, 1, "bn", "mish")
        self.conv13 = Conv(in_chs * 2, out_chs, 1,
                                    1, "bn", "mish")  # 這邊的input是conv2+conv11 所以有128

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        r = self.resblock(x3)
        x4 = self.conv11(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv13(x4)
        return x5

###################################################################
class MobileNet(nn.Module):
    def __init__(self,relu):
        super(MobileNet, self).__init__()
        self.conv = Conv(3,32,3,2,"bn",relu)

        self.Dw1 = MobileConv(32, 64, 3, 1, relu)
        self.Dw2 = MobileConv(64, 128, 3, 2, relu)
        self.Dw3 = MobileConv(128, 128, 3, 1, relu)
        self.Dw4 = MobileConv(128, 256, 3, 2, relu)
        self.Dw5 = MobileConv(256, 256, 3, 1, relu)
        self.Dw6 = MobileConv(256, 512, 3, 2, relu)
        self.Dw7 = MobileConv(512, 512, 3, 1, relu)
        self.Dw8 = MobileConv(512, 512, 3, 1, relu)
        self.Dw9 = MobileConv(512, 512, 3, 1, relu)
        self.Dw10 = MobileConv(512, 512, 3, 1, relu)
        self.Dw11 = MobileConv(512, 512, 3, 1, relu)
        self.Dw12 = MobileConv(512, 1024, 3, 2, relu)
        self.Dw13 = MobileConv(1024, 1024, 3, 1, relu)

    def forward(self, x):
        # 208
        d1 = self.conv(x)
        d1 = self.Dw1(d1)
        # 104
        d2 = self.Dw2(d1)
        d2 = self.Dw3(d2)
        # 52
        d3 = self.Dw4(d2)
        d3 = self.Dw5(d3)
        # 26
        d4 = self.Dw6(d3)
        d4 = self.Dw7(d4)
        d4 = self.Dw8(d4)
        d4 = self.Dw9(d4)
        d4 = self.Dw10(d4)
        d4 = self.Dw11(d4)
        # 13
        d5 = self.Dw12(d4)
        d5 = self.Dw13(d5)
        # 52*52*256  26*26*512  13*13*1024
        return d3, d4, d5

###################################################################
class GhostNet(nn.Module):
    def __init__(self,relu):
        super(GhostNet, self).__init__()
        # d1
        self.conv1 = Conv(3, 16, 3, 2, "bn", relu)
        self.GbBlock1 = GhostBottleneck(16, 16, 16, 3, 1, 0., relu)
        # d2
        self.GbBlock2 = GhostBottleneck(16, 48, 24, 3, 2, 0., relu)
        self.GbBlock3 = GhostBottleneck(24, 72, 24, 3, 1, 0., relu)
        # d3
        self.GbBlock4 = GhostBottleneck(24, 72, 40, 5, 2, 0.25, relu)
        self.GbBlock5 = GhostBottleneck(40, 120, 40, 5, 1, 0.25, relu)       
        # d4
        self.GbBlock6 = GhostBottleneck(40, 240, 80, 3, 2, 0., relu)
        self.GbBlock7 = GhostBottleneck(80, 200, 80, 3, 1, 0., relu)
        self.GbBlock8 = GhostBottleneck(80, 184, 80, 3, 1, 0., relu)
        self.GbBlock9 = GhostBottleneck(80, 184, 80, 3, 1, 0., relu)
        self.GbBlock10 = GhostBottleneck(80, 480, 112, 3, 1, 0.25, relu)
        self.GbBlock11 = GhostBottleneck(112, 672, 112, 3, 1, 0.25, relu)        
        # d5
        self.GbBlock12 = GhostBottleneck(112, 672, 160, 5, 2, 0.25, relu)
        self.GbBlock13 = GhostBottleneck(160, 960, 160, 5, 1, 0., relu)
        self.GbBlock14 = GhostBottleneck(160, 960, 160, 5, 1, 0.25, relu)
        self.GbBlock15 = GhostBottleneck(160, 960, 160, 5, 1, 0., relu)
        self.GbBlock16 = GhostBottleneck(160, 960, 160, 5, 1, 0.25, relu)
        # self.conv2 = Conv(160, 960, 1, 1, "bn", "relu")

    def forward(self, x):
        # 208
        d1 = self.conv1(x)
        d1 = self.GbBlock1(d1)
        # 104
        d2 = self.GbBlock2(d1)
        d2 = self.GbBlock3(d2)
        # 52
        d3 = self.GbBlock4(d2)
        d3 = self.GbBlock5(d3)
        # 26
        d4 = self.GbBlock6(d3)
        d4 = self.GbBlock7(d4)
        d4 = self.GbBlock8(d4)
        d4 = self.GbBlock9(d4)
        d4 = self.GbBlock10(d4)
        d4 = self.GbBlock11(d4)
        # 13
        d5 = self.GbBlock12(d4)
        d5 = self.GbBlock13(d5)
        d5 = self.GbBlock14(d5)
        d5 = self.GbBlock15(d5)
        d5 = self.GbBlock16(d5)
        # d5 = self.conv2(d5)
        # 52*52*40, 26*26*112, 13*13*160
        return d3, d4, d5

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size, stride, se_ratio, relu):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # Point-wise expansion
        self.ghost1 = GhostConv(in_chs, mid_chs, 1, 2, 3, 1, relu)
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = Conv(mid_chs, mid_chs, dw_kernel_size, stride, "bn", activate=None, group=mid_chs)
        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, act_layer=relu)
        else:
            self.se = None
        # Point-wise linear projection
        self.ghost2 = GhostConv(mid_chs, out_chs, 1, 2, 3, 1, relu=None)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                Conv(in_chs, in_chs, dw_kernel_size, stride, "bn", group=in_chs),
                Conv(in_chs, out_chs, 1, 1, "bn")
            )

    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


############################################################
class Darknet53(nn.Module):
    def __init__(self, relu):
        super(Darknet53, self).__init__()
        self.__conv = Conv(3, 32, 3, 1, 'bn', relu)
        self.__conv_5_0 = Conv(32, 64, 3, 2, 'bn', relu)
        self.__rb_5_0 = Residual_block(64, 64, 32, relu)
        self.__conv_5_1 = Conv(64, 128, 3, 2, 'bn',relu)
        self.__rb_5_1_0 = Residual_block(128, 128, 64,relu)
        self.__rb_5_1_1 = Residual_block(128, 128, 64,relu)

        self.__conv_5_2 = Conv(128, 256, 3, 2, 'bn', relu)
        self.__rb_5_2_0 = Residual_block(256, 256, 128, relu)
        self.__rb_5_2_1 = Residual_block(256, 256, 128,relu)
        self.__rb_5_2_2 = Residual_block(256, 256, 128, relu)
        self.__rb_5_2_3 = Residual_block(256, 256, 128,relu)
        self.__rb_5_2_4 = Residual_block(256, 256, 128,relu)
        self.__rb_5_2_5 = Residual_block(256, 256, 128,relu)
        self.__rb_5_2_6 = Residual_block(256, 256, 128,relu)
        self.__rb_5_2_7 = Residual_block(256, 256, 128, relu)

        self.__conv_5_3 = Conv(256, 512, 3, 2, 'bn', relu)
        self.__rb_5_3_0 = Residual_block(512, 512, 256,relu)
        self.__rb_5_3_1 = Residual_block(512, 512, 256,relu)
        self.__rb_5_3_2 = Residual_block(512, 512, 256, relu)
        self.__rb_5_3_3 = Residual_block(512, 512, 256, relu)
        self.__rb_5_3_4 = Residual_block(512, 512, 256,relu)
        self.__rb_5_3_5 = Residual_block(512, 512, 256,relu)
        self.__rb_5_3_6 = Residual_block(512, 512, 256,relu)
        self.__rb_5_3_7 = Residual_block(512, 512, 256, relu)

        self.__conv_5_4 = Conv(512, 1024, 3, 2,'bn', relu)
        self.__rb_5_4_0 = Residual_block(1024, 1024, 512, relu)
        self.__rb_5_4_1 = Residual_block(1024, 1024, 512,relu)
        self.__rb_5_4_2 = Residual_block(1024, 1024, 512, relu)
        self.__rb_5_4_3 = Residual_block(1024, 1024, 512, relu)

    def forward(self, x):
        x = self.__conv(x)

        x0_0 = self.__conv_5_0(x)
        x0_1 = self.__rb_5_0(x0_0)

        x1_0 = self.__conv_5_1(x0_1)
        x1_1 = self.__rb_5_1_0(x1_0)
        x1_2 = self.__rb_5_1_1(x1_1)

        x2_0 = self.__conv_5_2(x1_2)
        x2_1 = self.__rb_5_2_0(x2_0)
        x2_2 = self.__rb_5_2_1(x2_1)
        x2_3 = self.__rb_5_2_2(x2_2)
        x2_4 = self.__rb_5_2_3(x2_3)
        x2_5 = self.__rb_5_2_4(x2_4)
        x2_6 = self.__rb_5_2_5(x2_5)
        x2_7 = self.__rb_5_2_6(x2_6)
        x2_8 = self.__rb_5_2_7(x2_7)  # small

        x3_0 = self.__conv_5_3(x2_8)
        x3_1 = self.__rb_5_3_0(x3_0)
        x3_2 = self.__rb_5_3_1(x3_1)
        x3_3 = self.__rb_5_3_2(x3_2)
        x3_4 = self.__rb_5_3_3(x3_3)
        x3_5 = self.__rb_5_3_4(x3_4)
        x3_6 = self.__rb_5_3_5(x3_5)
        x3_7 = self.__rb_5_3_6(x3_6)
        x3_8 = self.__rb_5_3_7(x3_7)  # medium

        x4_0 = self.__conv_5_4(x3_8)
        x4_1 = self.__rb_5_4_0(x4_0)
        x4_2 = self.__rb_5_4_1(x4_1)
        x4_3 = self.__rb_5_4_2(x4_2)
        x4_4 = self.__rb_5_4_3(x4_3)  # large

        return x2_8, x3_8, x4_4


class Residual_block(nn.Module):
    def __init__(self, in_chs, out_chs, mid_chs, relu):

        super(Residual_block, self).__init__()
        self.__conv1 = Conv(in_chs, mid_chs, 1, 1, "bn", relu)
        self.__conv2 = Conv(mid_chs, out_chs, 3, 1, "bn", relu)

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r
        return out


