# References: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

from numpy.core.fromnumeric import squeeze
from torch import tensor
from torch._C import device
from model.loss import *

def to_cpu(tensor):
    return tensor.detach().cpu()

def anchor_wh_iou(wh1, wh2):
    """
    :param wh1: width and height of ground truth boxes
    :param wh2: width and height of anchor boxes
    :return: iou
    """
    # 转置wh2
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


class YoloLossLayer(nn.Module):
    def __init__(self, ncls, anchors, angles, anchors_mask, input_shape, ignore_iouthresh, ignore_angthresh):
        super(YoloLossLayer, self).__init__()
        self.ncls = ncls
        self.anchors = anchors
        self.angles = angles
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.ignore_iouthresh = ignore_iouthresh
        self.ignore_angthresh = ignore_angthresh

        self.lambda_coord = 1.0
        self.lambda_conf_scale = 1.0
        self.lambda_cls_scale = 1.0

    def build_targets(self, pred_boxes, pred_cls, target, stride, masked_anchors):
        # pred_boxes:[b,k,w,h,5] 相对坐标
        # pre_cls[b,k,w,h,ncls]
        # target:[cx,cy,w,h,a,cls,idx] 相对坐标
        # masked_anchors[w,h,a] 在特征图上的值
        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        # pred_cls[b,k,w,h,ncls]
        nB, nA, nG, _, nC = pred_cls.size()

        # obj_mask：1表示该位置是物体
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        # noobj_mask： 1表示该位置没有物体
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        ariou_loss = FloatTensor(nB, nA, nG, nG).fill_(0)
        ciou_loss = FloatTensor(nB, nA, nG, nG).fill_(0)

        ta = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
        # riou_loss = FloatTensor(nB, nA, nG, nG).fill_(0)
        # iou_loss = FloatTensor(nB, nA, nG, nG).fill_(0)

        # 转化为特征图直接坐标
        target_boxes = torch.cat((target[:, :4] * nG, target[:, 4:5]), dim=-1)

        # 提取target_boxs x,y,w,h,a
        gxy = target_boxes[:, :2]  # cx,cy
        gwh = target_boxes[:, 2:4]  # w,h
        ga = target_boxes[:, 4]  # angle

        arious = []
        offset = []

        # 构建零点GT框[0,0,w,h,a]
        gt_box = FloatTensor(
            torch.cat((torch.zeros((target_boxes.size(0), 2), device=target_boxes.device), target_boxes[:, 2:5]), 1))
        # 构建零点anchor框[0,0,w,h,a]
        anchor_shapes = FloatTensor(
            torch.cat((torch.zeros((len(masked_anchors), 2), device=masked_anchors.device), masked_anchors), 1))

        for anchor in anchor_shapes:
            # 在特征图坐标系上进行计算
            ariou = skewiou_fun(anchor, gt_box)
            arious.append(ariou)
            # 角度偏置
            offset.append(torch.abs(torch.sub(anchor[4], ga)))

        arious = torch.stack(arious).to('cuda')
        offset = torch.stack(offset)

        # best_n：与每个target匹配的anchor的序号，根据anchor与GT的形状匹配来确定best_n,即anchor序号
        _, best_n = arious.max(0)

        # target:[cx,cy,w,h,a,cls,idx]
        idx = target[:, 6].long().t()
        target_labels = target[:, 5].long().t()
        gi, gj = gxy.long().t()  # 转置

        # 确定哪些位置上的anchor有物体
        obj_mask[idx, best_n, gj, gi] = 1
        # 确定哪些位置上的anchor没有物体
        noobj_mask[idx, best_n, gj, gi] = 0

        for i, (anchor_ious, angle_offset) in enumerate(zip(arious.t(), offset.t())):
            # 如果某些位置anchor的iou值大于了ignore_thres且角度偏置小于ignore_angthresh，不计算相应的损失函数，在noobj_mask对应位置置0
            noobj_mask[idx[i], (anchor_ious > self.ignore_iouthresh[0]),gj[i], gi[i]] = 0
            noobj_mask[idx[i], (anchor_ious >self.ignore_iouthresh[1]) & (angle_offset < self.ignore_angthresh), gj[i], gi[i]] = 0

        # angle的损失值,只计算obj的angle损失值
        ta[idx, best_n, gj, gi] = ga - masked_anchors[best_n][:, 2]

        # 将相对应的类别置1
        tcls[idx, best_n, gj, gi, target_labels] = 1

        # 置信度
        tconf = obj_mask.float()

        # 这里pred_boxes和target_boxes都是特征图上的直接坐标
        ariou, ciou, iou = bbox_xywha_ciou(pred_boxes[idx, best_n, gj, gi], target_boxes)
        
        # rskewiou = bbox_xywha_skewiou(pred_boxes[idx, best_n, gj, gi], target_boxes)

        with torch.no_grad():
            img_size = stride * nG
            bbox_loss_scale = 2.0 - 1.0 * \
                gwh[:, 0] * gwh[:, 1] / (img_size ** 2)

        # pred_cls->(batch_size, nanchor, gridw, gridh, conf, cls) argmax(-1)得到预测分类的值
        # class_mask 判断预测是否正确
        class_mask[idx, best_n, gj, gi] = (
            pred_cls[idx, best_n, gj, gi].argmax(-1) == target_labels).float()

        iou_scores[idx, best_n, gj, gi] = ariou
        # ariou_loss[idx, best_n, gj, gi] = 1 - ariou
        ariou_loss[idx, best_n, gj, gi] = torch.exp(1 - ariou) - 1
        ciou_loss[idx, best_n, gj, gi] = 1-ciou

        obj_mask = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        return iou_scores, ariou_loss, ciou_loss,class_mask, obj_mask, noobj_mask, ta, tcls, tconf

    def forward(self, l, input, target=None):

        bs, grid_size = input.size(0), input.size(2)

        stride = self.input_shape / grid_size

        masked_anchors = [(a_w / stride, a_h / stride, a) for a_w,
                          a_h in np.array(self.anchors)[self.anchors_mask[l]] for a in self.angles]

        FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
        num_anchors = len(masked_anchors)
        # 读取预测值 predictio[b,k,w,h, 6 + ncls])
        prediction = (input.view(bs, num_anchors, self.ncls + 6,
                      grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())
        pred_x = torch.sigmoid(prediction[..., 0])
        pred_y = torch.sigmoid(prediction[..., 1])
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_a = prediction[..., 4]
        pred_conf = torch.sigmoid(prediction[..., 5])
        pred_cls = torch.sigmoid(prediction[..., 6:])

        # 生成网格坐标[1,1,gtidx,gridy,1]
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size]).type(FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size]).type(FloatTensor)

        # masked_anchors[(w,h,a)]
        masked_anchors = FloatTensor(masked_anchors)
        anchor_w = masked_anchors[:, 0].view([1, num_anchors, 1, 1])
        anchor_h = masked_anchors[:, 1].view([1, num_anchors, 1, 1])
        anchor_a = masked_anchors[:, 2].view([1, num_anchors, 1, 1])

        # 解码预测值
        pred_boxes = FloatTensor(prediction[..., :5].shape)
        pred_boxes[..., 0] = (pred_x + grid_x)
        pred_boxes[..., 1] = (pred_y + grid_y)
        pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w)
        pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h)
        pred_boxes[..., 4] = pred_a + anchor_a

        # 转化为原图上的直接坐标 output[batch, h*w*k, 6+4]
        output = torch.cat(
            (
                torch.cat([pred_boxes[..., :4] * stride,
                          pred_boxes[..., 4:]], dim=-1).view(bs, -1, 5),
                pred_conf.view(bs, -1, 1),
                pred_cls.view(bs, -1, self.ncls),
            ),
            -1,
        )

        if target is None:
            return output
        else:
            iou_scores, ariou_loss, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf = self.build_targets(
                pred_boxes, pred_cls, target, stride, masked_anchors)

            iou_const = ariou_loss[obj_mask]

            # angle_loss 使用reduction="none"
            angle_loss = F.smooth_l1_loss(
                pred_a[obj_mask], ta[obj_mask], reduction="none")

            reg_loss = angle_loss + ciou_loss[obj_mask]
            
            # reg_loss = reg_loss.mean()

            with torch.no_grad():
                reg_const = iou_const / reg_loss
            reg_loss = (reg_loss * reg_const).mean()


            # Focal Loss for object's prediction
            # FOCAL = FocalLoss(reduction="mean")
            # conf_loss = (
            #     FOCAL(pred_conf[obj_mask], tconf[obj_mask])
            #     + FOCAL(pred_conf[noobj_mask], tconf[noobj_mask])
            # )
            # 使用交叉熵损失函数计算conf_loss
            conf_loss = (
                F.binary_cross_entropy(
                    pred_conf[obj_mask], tconf[obj_mask], reduction="mean")
                + F.binary_cross_entropy(pred_conf[noobj_mask],
                                         tconf[noobj_mask], reduction="mean")
            )
            # 使用交叉熵损失函数计算cls_loss
            cls_loss = F.binary_cross_entropy(
                pred_cls[obj_mask], tcls[obj_mask], reduction="mean")

            # Loss scaling
            # 1
            reg_loss = self.lambda_coord * reg_loss
            # 1
            conf_loss = self.lambda_conf_scale * conf_loss
            # 1
            cls_loss = self.lambda_cls_scale * cls_loss
            total_anchors = grid_size*grid_size*num_anchors
            total_loss = reg_loss + conf_loss + cls_loss

            return total_loss*bs, reg_loss*bs, conf_loss*bs, cls_loss*bs, total_anchors


######################################################################################

# input_shape[416,416]
class YOLOLoss(nn.Module):
    def __init__(self, anchors, angles, ncls, input_shape, cuda, anchors_mask, label_smoothing=0):
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[81,82],[135,169],[344,319]
        #   26x26的特征层对应的anchor是[10,14],[23,27],[37,58]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.angles=angles
        self.ncls = ncls
        self.bbox_attrs = 6 + ncls
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + \
            (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - \
            (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        #----------------------------------------------------#
        #   求出真实框左上角右下角
        #----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        #----------------------------------------------------#
        #   求真实框和预测框所有的iou
        #----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(
            intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        #----------------------------------------------------#
        #   计算中心的差距
        #----------------------------------------------------#
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        #----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        #----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins,
                               torch.zeros_like(intersect_maxes))
        #----------------------------------------------------#
        #   计算对角线距离
        #----------------------------------------------------#
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / \
            torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(
            b1_wh[..., 1], min=1e-6)) - torch.atan(b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou

    def box_skewiou(self, b1, b2):
   
        b1=b1.view(-1,5)
        b2=b2.view(-1,5)
        skewious=torch.zeros(len(b1),1).cuda()
        for i in range(len(skewious)):
            skewious[i]=calskewiou(b1[i],b2[i])
        
        return skewious

    #---------------------------------------------------#
    #   平滑标签
    #---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None):
        #----------------------------------------------------#
        #   l 代表使用的是第几个有效特征层
        #   input的shape为  bs, (6+num_classes)*3*6, 13, 13
        #                   bs, (6+num_classes)*3*6, 26, 26
        #   targets 真实框的标签情况 [batch_size, num_gt, 6]
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #--------------------------------#
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   stride_h = stride_w = 32、16
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h, a)
                          for a_w, a_h in np.array(self.anchors)[self.anchors_mask[l]] for a in self.angles]
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3*6 * (6+num_classes), 13, 13 => bs, 3*6, 6 + num_classes, 13, 13 => batch_size, 3*6, 13, 13, 6 + num_classes

        #   batch_size, 3*6, 13, 13, 6 + num_classes
        #   batch_size, 3*6, 26, 26, 6 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l])*len(self.angles), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        #-----------------------------------------------#
        #   先验框的中心位置的调整参数
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #-----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        #-----------------------------------------------#
        #   获得角度调整参数
        #-----------------------------------------------#
        a = prediction[..., 4]
        #-----------------------------------------------#
        #   获得置信度，是否有物体
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 5])
        #-----------------------------------------------#
        #   种类置信度
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 6:])

        #-----------------------------------------------#
        #   获得网络应该有的预测结果
        #   y_true[bs,k,j,i,6+ncls]
        #   noobj_mask[bs,k,j,i]
        #   box_loss_scale[bs,k,j,i]
        #-----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(
            l, targets, scaled_anchors, in_h, in_w)

        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #   noobj_mask[bs, k, j, i]
        #----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(
            l, x, y, h, w, a, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()
        #-----------------------------------------------------------#
        #   reshape_y_true[...,2:3]和reshape_y_true[...,3:4]
        #   表示真实框的宽高，二者均在0-1之间
        #   真实框越大，比重越小，小框的比重更大。
        #-----------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale

        #---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU
        #   y_true[bs, k, j, i, 6+ncls] (x,y,w,h,a,conf)
        #   pred_boxes[bs, 3*6, h, w, 5] (x,y,w,h,a)
        #   ciou = (1-ciou)*box_loss_scale
        #----------------------------------------------------------------#
        objmask = (y_true[..., 5] == 1) # [bs,k,j,i]
        # ciouloss=1-ciou
        ciou_loss = (1 - self.box_ciou(pred_boxes[objmask], y_true[..., :5][objmask])) * box_loss_scale[objmask]
        # skewiouloss=exp(1-skewiou)-1
        skewiou_loss = (torch.exp(1 - self.box_skewiou(pred_boxes[objmask], y_true[..., :5][objmask])) - 1).squeeze(1)
        
        angle_loss = F.smooth_l1_loss(
            pred_boxes[..., 4][objmask], y_true[..., 4][objmask],reduction='none')
        
        reg_loss=angle_loss+ciou_loss

        with torch.no_grad():
            reg_const=skewiou_loss/reg_loss
        reg_loss=reg_loss*reg_const
        
        loss_loc = torch.mean(reg_loss)
        #-----------------------------------------------------------#
        #   计算置信度的loss
        #-----------------------------------------------------------#
        noobj_mask = noobj_mask == 1

        loss_conf = torch.mean(self.BCELoss(conf[objmask], y_true[..., 5][objmask])) + \
            torch.sum(self.BCELoss(conf[noobj_mask], y_true[..., 5][noobj_mask]))
        #-----------------------------------------------------------#
        #   计算分类loss
        #-----------------------------------------------------------#
        loss_cls = torch.sum(self.BCELoss(pred_cls[objmask], self.smooth_labels(y_true[..., 6:][objmask], self.label_smoothing, self.ncls)))

        loss = loss_loc + loss_conf + loss_cls
        # 正例数
        num_pos = torch.sum(y_true[..., 5])
        return loss, num_pos, loss_loc, loss_conf, loss_cls

    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / \
            2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / \
            2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / \
            2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / \
            2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:,
                                                     3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:,
                                                     3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(
            A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(
            A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3] -
                  box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3] -
                  box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def calculate_skewiou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = _box_a.size(0)
        B = _box_b.size(0)

        iou = torch.zeros((A,B))
        for i in range(A):
            for j in range(B):
                iou[i,j]=calskewiou(_box_a[i],_box_b[j])

        return iou

    def get_target(self, l, targets, anchors, in_h, in_w):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(targets)
        #-----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        #   noobj_mask[b,k,h,w]
        #-----------------------------------------------------#
        noobj_mask = torch.ones(
            bs, len(self.anchors_mask[l])*len(self.angles), in_h, in_w, requires_grad=False)
        #-----------------------------------------------------#
        #   让网络更加去关注小目标
        #-----------------------------------------------------#
        box_loss_scale = torch.zeros(
            bs, len(self.anchors_mask[l])*len(self.angles), in_h, in_w, requires_grad=False)
        #-----------------------------------------------------#
        #   batch_size, 3*6, 13, 13, 6 + num_classes
        #-----------------------------------------------------#
        y_true = torch.zeros(
            bs, len(self.anchors_mask[l])*len(self.angles), in_h, in_w, self.bbox_attrs, requires_grad=False)
        
        FloatTensor =  torch.FloatTensor

        # skewious=[]

        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            # target[x,y,w,h,a,cls]
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #   batch_target[cx,cy,w,h,a,cls]
            #-------------------------------------------------------#
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target[:, 5] = targets[b][:, 5]
            batch_target = batch_target.cpu()

            #-------------------------------------------------------#
            #   将真实框转换一个形式[0,0,w,h,a]
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box = FloatTensor(torch.cat((torch.zeros((batch_target.size(
                0), 2)), batch_target[:, 2:5]), 1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式[0,0,w,h,a]
            #   9, 4
            #-------------------------------------------------------#
            anchor_shapes = FloatTensor(torch.cat((torch.zeros(
                (len(anchors), 2)), FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 18]每一个真实框和18个先验框的重合情况
            #   best_ns:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            #-------------------------------------------------------#
            skewiou = self.calculate_skewiou(gt_box, anchor_shapes)
            # skewious.append(skewiou)
            best_ns = torch.argmax(skewiou, dim=-1)
            #----------------------------------------#
            #   获得真实框属于哪个网格点
            #   batch_target[cx,cy,w,h,a,cls]
            #----------------------------------------#
            i = torch.floor(batch_target[:, 0]).long()
            j = torch.floor(batch_target[:, 1]).long()

            #----------------------------------------#
            #   noobj_mask代表无目标的特征点
            #----------------------------------------#
            noobj_mask[b, best_ns, j, i] = 0   
            noobj_mask[b, (skewiou > self.ignore_threshold).squeeze(0), j, i] = 0
            #----------------------------------------#
            #   tx、ty代表中心调整参数的真实值
            #   y_true[batch_size, 3*6, 13, 13, 6 + num_classes]
            #                           (x,y,w,h,a,conf,cls)
            #----------------------------------------#
            y_true[b, best_ns, j, i, 0] = batch_target[:, 0]  # cx
            y_true[b, best_ns, j, i, 1] = batch_target[:, 1]  # cy
            y_true[b, best_ns, j, i, 2] = batch_target[:, 2]  # w
            y_true[b, best_ns, j, i, 3] = batch_target[:, 3]  # h
            y_true[b, best_ns, j, i, 4] = batch_target[:, 4]  # a
            y_true[b, best_ns, j, i, 5] = 1 # conf
            c = torch.floor(batch_target[:, 5]).long()
            y_true[b, best_ns, j, i, c + 6] = 1  # cls

            box_loss_scale[b, best_ns, j, i] = batch_target[:, 2] * batch_target[:, 3] / in_w / in_h
           
        # skewious=torch.stack(skewious)
        return y_true, noobj_mask, box_loss_scale

    # x,y,w,h,a都是预测框的值，格式[bs,3*6,h,w,1]
    # scaled_anchors[w,h,a]
    def get_ignore(self, l, x, y, h, w, a, targets, scaled_anchors, in_h, in_w, noobj_mask,skewious):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   grid_x[bs,3*6,h,w,1]
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])*len(self.angles)), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])*len(self.angles)), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高 scaled_anchors_l[w,h,a]
        scaled_anchors_l = FloatTensor(np.array(scaled_anchors))

        # index_select(dim,idx) 在dim维挑选idx列/行
        anchor_w = scaled_anchors_l.index_select(1, LongTensor([0]))  # w
        anchor_h = scaled_anchors_l.index_select(1, LongTensor([1]))  # h
        anchor_a = scaled_anchors_l.index_select(1, LongTensor([2]))  # a
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(
            1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(
            1, 1, in_h * in_w).view(h.shape)
        anchor_a = anchor_a.repeat(bs, 1).repeat(
            1, 1, in_h * in_w).view(a.shape)

        # offset = torch.abs(torch.sub(anchor_a, a))  # 角度偏置
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #   pred_boxes[bs,3*6,h,w,5] (x,y,w,h,a)
        #-------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes_a = torch.unsqueeze(a + anchor_a, -1)
        pred_boxes = torch.cat(
            [pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h, pred_boxes_a], dim=-1)

        return noobj_mask, pred_boxes


######################################################################################
class YoloLayer(nn.Module):
    def __init__(self, num_classes, anchors, angles, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.angles = angles
        self.num_anchors = len(anchors) * len(angles)
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh

        # 每个anchor得数据格式w,h,a，anchor数量是 3x6=18
        self.masked_anchors = [(a_w / self.stride, a_h / self.stride, a) for a_w, a_h in self.anchors for a in
                               self.angles]
        self.reduction = "mean"

        self.lambda_coord = 2.0
        self.lambda_conf_scale = 1.0
        self.lambda_cls_scale = 1.0
        self.metrics = {}

    def build_targets(self, pred_boxes, pred_cls, target):
        # target:[img_idx,cls,cx,cy,w,h,a] 特征图上的相对坐标
        # pred_boxes:[bx, by, bw, bh, ba],特征图上的直接坐标
        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
        
        # nB:batch size
        # nA:anchor数目        
        # nG：输出特征图的大小
        # _ : nG
        # nC:类别数目
        nB, nA, nG, _, nC = pred_cls.size()

        # Output tensors
        # obj_mask：1表示该位置是物体，0表示该位置不是物体
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        # noobj_mask： 1表示该位置没有物体，0表示该位置是物体或者不参与计算的anchor
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        skew_iou = FloatTensor(nB, nA, nG, nG).fill_(0)
        ciou_loss = FloatTensor(nB, nA, nG, nG).fill_(0)
        ta = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert ground truth position to position that relative to the size of box (grid size)
        # target[img_idx, label, x, y, w, h, a]
        # target[:, 2:6] * nG 将相对坐标转化为在特征图上的直接坐标
        # target[:, 6:]是angle，不需要转化
        target_boxes = torch.cat((target[:, 2:6] * nG, target[:, 6:]), dim=-1)
        
        gxy = target_boxes[:, :2] # 中心坐标x,y
        gwh = target_boxes[:, 2:4] # w，h
        ga = target_boxes[:, 4] # angle

        # Get anchors with best iou and their angle difference with ground truths
        arious = []
        offset = []
        for anchor in self.masked_anchors:
            # # anchor: (w,h,a)
            # # ariou: anchor和真实框的IOU
            # ariou = anchor_wh_iou(anchor[:2], gwh)         
            # # 角度使用减法运算并cos转换到0-1
            # cos = torch.abs(torch.cos(torch.sub(anchor[2], ga))) 
            # arious.append(ariou * cos)
            ariou = skewiou_fun(anchor[:3], target_boxes[:, 2:5])
            arious.append(ariou)
            offset.append(torch.abs(torch.sub(anchor[2], ga)))# 角度偏置

        # arious: iou*cos
        arious = torch.stack(arious)
        # 角度编制 cos  0-1
        offset = torch.stack(offset)

        # best_n：与每个target匹配的anchor的序号，根据anchor与GT的形状匹配来确定best_n,即anchor序号
        best_ious, best_n = arious.max(0)

        # Separate target values
        # target[:, :2]是img_idx和cls_lable
        # b:img_idx, target_labels: cls_label
        b, target_labels = target[:, :2].long().t()
        gi, gj = gxy.long().t()# 转置

        # Set masks to specify object's location
        # GT与所在cell的anchor相匹配
        # 确定哪些位置上的anchor有物体
        obj_mask[b, best_n, gj, gi] = 1
        # 确定哪些位置上的anchor没有物体
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold     
        for i, (anchor_ious, angle_offset) in enumerate(zip(arious.t(), offset.t())):
            # 如果在某些位置anchor的iou值大于了ignore_thres(0.6)，不计算相应的损失，在noobj_mask对应位置置0
            noobj_mask[b[i], (anchor_ious > self.ignore_thresh), gj[i], gi[i]] = 0
            # 如果某些位置anchor的iou值大于了ignore_thres(0.6)且角度偏置小于15度，不计算相应的损失函数，在noobj_mask对应位置置0
            # noobj_mask[b[i], (anchor_ious > 0.4) & (angle_offset < (np.pi / 12)), gj[i], gi[i]] = 0

        # angle的损失值
        ta[b, best_n, gj, gi] = ga - self.masked_anchors[best_n][:, 2]

        # 将相对应的类别置1
        tcls[b, best_n, gj, gi, target_labels] = 1

        # 置信度
        tconf = obj_mask.float()

        # Calculate ciou loss
        # 这里pred_boxes和target_boxes都是特征图上的直接坐标
        # 返回skew_iou和ciou_loss
        # iou, ciou = bbox_xywha_ciou(pred_boxes[b, best_n, gj, gi], target_boxes)
        # 这里的iou为旋转矩形框的交并集， 而不是使用正矩形框的交并集*angle
        iou, ciou = bbox_xywha_skewiou(pred_boxes[b, best_n, gj, gi], target_boxes)
        
        with torch.no_grad():
            img_size = self.stride * nG
            # bbox_loss_scale = 2 - 相对面积，值的范围是(1~2)，边界框的尺寸越小，
            # bbox_loss_scale 的值就越大。box_loss_scale可以弱化边界框尺寸对损失值的影响
            bbox_loss_scale = 2.0 - 1.0 * gwh[:, 0] * gwh[:, 1] / (img_size ** 2)
        
        # ciouloss
        ciou = bbox_loss_scale * (1.0 - ciou)
        # Compute label correctness and iou at best anchor

        # pred_cls->(batch_size, nanchor, gridw, gridh, conf, cls) argmax(-1)得到预测分类的值
        # class_mask 判断预测是否正确
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        
        # 这里的iou是iou*cos(angle)
        iou_scores[b, best_n, gj, gi] = iou

        # magnitude for reg loss
        skew_iou[b, best_n, gj, gi] = torch.exp(1 - iou) - 1
        # skew_iou[b, best_n, gj, gi] = 1 - iou

        # unit vector for reg loss
        ciou_loss[b, best_n, gj, gi] = ciou

        obj_mask = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        # iou_scores  iou*angle
        # skew_iou    exp(1 - iou) - 1
        # ciou_loss   ciou时的loss
        # class_mask
        # obj_mask
        # noobj_mask
        # ta
        # tcls
        # tconf
        return iou_scores, skew_iou, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf

    def forward(self, output, target=None):
        # anchors = [12, 16, 19, 36, 40, 28,// 36, 75, 76, 55, 72, 146,// 142, 110, 192, 243, 459, 401]
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]
        # anchor_step = len(anchors) // num_anchors

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor

        # output.shape-> [batch_size, num_anchors * (num_classes + 6), grid_size, grid_size]
        batch_size, grid_size = output.size(0), output.size(2)

        # prediction.shape-> torch.Size([batchsize, num_anchors, grid_size, grid_size, 6 + num_classes])
        prediction = (
            output.view(batch_size, self.num_anchors, self.num_classes + 6, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )
        # 预测的偏移值 tx, ty, tw, th, ta. tconf, tcls 
        pred_x = torch.sigmoid(prediction[..., 0]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_y = torch.sigmoid(prediction[..., 1]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_a = prediction[..., 4]
        pred_conf = torch.sigmoid(prediction[..., 5])
        # 使用sigmod
        pred_cls = torch.sigmoid(prediction[..., 6:])

        # grid.shape-> [1, 1, 52, 52, 1]
        # 預測出來的(pred_x, pred_y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)

        # anchor.shape-> [1, 3, 1, 1, 1]
        # masked_anchors[i,(w,h,a)]
        self.masked_anchors = FloatTensor(self.masked_anchors)
        anchor_w = self.masked_anchors[:, 0].view([1, self.num_anchors, 1, 1])
        anchor_h = self.masked_anchors[:, 1].view([1, self.num_anchors, 1, 1])
        anchor_a = self.masked_anchors[:, 2].view([1, self.num_anchors, 1, 1])

        # decode
        # 预测得到的bounding box
        # bx, by, bw, bh, ba, 在特征图上的直接坐标
        # pred_boxes[batchsize, num_anchors,grid_size, grid_size,5]
        pred_boxes = FloatTensor(prediction[..., :5].shape)
        pred_boxes[..., 0] = (pred_x + grid_x)
        pred_boxes[..., 1] = (pred_y + grid_y)
        pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w)
        pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h)
        pred_boxes[..., 4] = pred_a + anchor_a

        # 转化为原图上的直接坐标
        # output[batch, grid*grid*num_anchors, (x,y,w,h,a,conf,ncls...)]
        output = torch.cat(
            (
                torch.cat([pred_boxes[..., :4] * self.stride, pred_boxes[..., 4:]], dim=-1).view(batch_size, -1, 5),
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )

        if target is None:
            return output
        else:
            # iou_scores:   iou*cos(angle)
            # skew_iou:     exp(1 - iou) - 1
            # cious_loss:   ciou_loss即1-ciou
            # class_mask:   类别预测是否正确
            # ta:      ga-anchor的角度
            iou_scores, skew_iou, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes, pred_cls=pred_cls, target=target
            )
            # --------------------
            # - Calculating Loss 构建损失函数 -
            # --------------------

            # Reg Loss for bounding box prediction
            iou_const = skew_iou[obj_mask]

            # angle_loss 使用reduction="none"
            angle_loss = F.smooth_l1_loss(pred_a[obj_mask], ta[obj_mask], reduction="none")
            
            # regloss=angleloss+ciouloss
            # angleloss=smoothl1(dangle)
            # ciosloss= bbox_loss_scale * (1.0 - ciou)
            reg_loss = angle_loss + ciou_loss[obj_mask]

            with torch.no_grad():
                reg_const = iou_const / reg_loss
            reg_loss = (reg_loss * reg_const).mean()

            # Focal Loss for object's prediction
            # FOCAL = FocalLoss(reduction=self.reduction)
            # conf_loss = (
            #     FOCAL(pred_conf[obj_mask], tconf[obj_mask])
            #     + FOCAL(pred_conf[noobj_mask], tconf[noobj_mask])
            # )
            # 使用交叉熵损失函数计算conf_loss
            conf_loss = (
                F.binary_cross_entropy(
                    pred_conf[obj_mask], tconf[obj_mask], reduction="mean")
                + F.binary_cross_entropy(pred_conf[noobj_mask],
                                         tconf[noobj_mask], reduction="mean")
            )

            # 使用交叉熵损失函数计算cls_loss
            cls_loss = F.binary_cross_entropy(
                pred_cls[obj_mask], tcls[obj_mask], reduction="mean")
            
            # Loss scaling
            # 2
            reg_loss = self.lambda_coord * reg_loss
            # 1
            conf_loss = self.lambda_conf_scale * conf_loss
            # 1
            cls_loss = self.lambda_cls_scale * cls_loss

            total_loss = reg_loss + conf_loss + cls_loss

            # --------------------
            # -   Logging Info   -
            # --------------------
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "reg_loss": to_cpu(reg_loss).item(),
                "conf_loss": to_cpu(conf_loss).item(),
                "cls_loss": to_cpu(cls_loss).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
            }
            # output[x, y, w, h, a, conf, cls]
            return output, total_loss * batch_size
