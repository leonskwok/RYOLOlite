from numpy.lib.arraypad import pad
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torch.autograd import Variable

import os
import scipy.signal
from tqdm import tqdm

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image

def fit_one_epoch(model_train, model, yolo_layer, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, weights_folder):
    loss = 0
    val_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets, paths = batch[0], batch[1], batch[2]

            device = 'cuda' if cuda else 'cpu'
            images = Variable(images.to(device), requires_grad=True)
            targets = Variable(targets.to(device), requires_grad=False)

            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(images)

            loss_value_all = 0
            loss_loc_all=0
            loss_conf_all=0
            loss_cls_all=0
            num_anchor_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item, loss_loc, loss_conf, loss_cls, total_anchors = yolo_layer(l, outputs[l], targets)

                loss_value_all += loss_item

                loss_loc_all+=loss_loc
                loss_conf_all+=loss_conf
                loss_cls_all+=loss_cls
                num_anchor_all+=total_anchors

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value_all.backward()
            optimizer.step()

            loss_history.append_trainloss(loss_value_all.item(),  loss_loc_all.item(
            ), loss_conf_all.item(), loss_cls_all.item())

            loss += loss_value_all.item()

            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                device='cuda' if cuda else 'cpu'
                images = Variable(images.to(device), requires_grad=True)
                targets = Variable(targets.to(device), requires_grad=False)

                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(images)

                loss_value_all = 0
                num_anchors_all=0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, loss_loc, loss_conf, loss_cls, total_anchors = yolo_layer(
                        l, outputs[l], targets)
                    loss_value_all += loss_item
                    num_anchors_all += total_anchors

                loss_history.append_valloss(loss_value_all.item())

            val_loss += loss_value_all.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')

    print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' %
          (loss / epoch_step, val_loss / epoch_step_val))
    if epoch+1==Epoch:
        torch.save(model.state_dict(), '%s/ep%03d-loss%.3f-val_loss%.3f.pth' %
                (weights_folder, epoch + 1, loss / epoch_step, val_loss / epoch_step_val))

class LossHistory():
    def __init__(self, log_dir):

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.save_path = log_dir
        self.losses = []
        self.val_loss = []

        self.loss_loc=[]
        self.loss_conf=[]
        self.loss_cls=[]

    def append_trainloss(self, loss, locloss,confloss,clsloss):
        self.losses.append(loss)
        self.loss_loc.append(locloss)
        self.loss_conf.append(confloss)
        self.loss_cls.append(clsloss)
        with open(os.path.join(self.save_path, "loss.txt"), 'a') as f:
            f.write(str(loss)+'\t'+str(locloss) +'\t' +str(confloss)+ '\t'+str(clsloss))
            f.write("\n")

    def append_valloss(self, val_loss):
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        # self.loss_plot()

class DwConv(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(DwConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3,
                               stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out

#-------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class GhostPointConv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, dw_size, kernel_size, stride=1):
        super(GhostPointConv, self).__init__()
        self.ghostconv = GhostConv(in_channels, out_channels, kernel_size, ratio, dw_size, stride)
        self.pointconv = nn.Conv2d(out_channels,out_channels,1,1,0,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.ghostconv(x)
        x = self.pointconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        # pad = (kernel_size - 1) // 2
        pad = kernel_size // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))

        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))

        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "linear":
            pass
        else:
            print("Convolution block seems to go wrong")
            exit(1)

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


#---------------------------------------------------#
#   注意力集中相关
#---------------------------------------------------#
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(
            kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = Conv(2, 1, kernel_size, stride=1, activation='linear', bn=True)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out




#---------------------------------------------------#
#   GhostNet相关
#---------------------------------------------------#
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GhostConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostConv, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size,
                      stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(0.1, inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(0.1, inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]



class SqGhostConv2(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 relu=True):
        super(SqGhostConv2, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.squeeze_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(0.1, inplace=True) if relu else nn.Sequential())

        self.dw_conv = nn.Sequential(
            nn.Conv2d(init_channels,
                      init_channels,
                      kernel_size,
                      stride,
                      padding=kernel_size // 2,
                      groups=init_channels,
                      bias=False), nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(0.1, inplace=True) if relu else nn.Sequential())


        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels,
                      new_channels,
                      dw_size,
                      1,
                      dw_size // 2,
                      groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(0.1, inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.squeeze_conv(x)
        x1 = self.dw_conv(x1)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class MobileBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(MobileBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size,
                               stride=stride,
                               padding=1,
                               groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1,inplace=True)
        self.conv2 = nn.Conv2d(in_planes,
                               out_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out