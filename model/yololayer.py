# References: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

from model.loss import *

def to_cpu(tensor):
    return tensor.detach().cpu()


class YoloLossLayer(nn.Module):
    def __init__(self, ncls, anchors, angles, anchors_mask, input_shape, ignore_iouthresh, ignore_angthresh, reg_type):
        super(YoloLossLayer, self).__init__()
        self.ncls = ncls
        self.anchors = anchors
        self.angles = angles
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.ignore_iouthresh = ignore_iouthresh
        self.ignore_angthresh = ignore_angthresh
        self.reg_type = reg_type

    def forward(self, l, input, target=None):
        device = input.device
        # input:[nB,(6+ncls)*nA, nG, nG],  6-(x,y,w,h,a,conf)
        batch_size, grid_size = input.size(0), input.size(2)
        
        # masked_anchors:[nA,(w,h,a)] 特征图上直接坐标
        stride = self.input_shape / grid_size
        scaled_anchors = [(a_w / stride, a_h / stride, a) for a_w,
                          a_h in np.array(self.anchors)[self.anchors_mask[l]] for a in self.angles]

        FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
        ByteTensor = torch.cuda.ByteTensor if input.is_cuda else torch.ByteTensor

        num_anchors = len(scaled_anchors)#3*6=18
        # prediction:[nB,nA,(6+ncls),nG,nG]
        prediction = (input.view(batch_size, num_anchors, self.ncls + 6,
                      grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())
        
        # prediction:[nB,nA,nG,nG,(6+ncls)]  6-(x,y,w,h,a,conf)
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

        # scaled_anchors[nA,(w,h,a)]
        scaled_anchors = FloatTensor(scaled_anchors)
        scaled_anchor_w = scaled_anchors[:, 0].view([1, num_anchors, 1, 1])
        scaled_anchor_h = scaled_anchors[:, 1].view([1, num_anchors, 1, 1])
        scaled_anchor_a = scaled_anchors[:, 2].view([1, num_anchors, 1, 1])

        # 解码预测值，特征图上直接坐标
        # pred_boxes:[nB,nA,nG,nG,5]  5-(x,y,w,h,a)
        pred_boxes = FloatTensor(prediction[..., :5].shape)
        pred_boxes[..., 0] = (pred_x + grid_x)
        pred_boxes[..., 1] = (pred_y + grid_y)
        pred_boxes[..., 2] = (torch.exp(pred_w) * scaled_anchor_w)
        pred_boxes[..., 3] = (torch.exp(pred_h) * scaled_anchor_h)
        pred_boxes[..., 4] = pred_a + scaled_anchor_a

        # 原图上的直接坐标
        # output:[nB,nA,nG,nG,6+ncls] 6-(x,y,w,h,a,conf)
        output = torch.cat(
            (
                torch.cat([pred_boxes[..., :4] * stride,
                          pred_boxes[..., 4:]], dim=-1).view(batch_size, -1, 5),
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.ncls),
            ),
            -1,
        )
        # 训练
        if target is not None:
            # pred_boxes:[nB,nA,nG,nG,(x,y,w,h,a)] 特征图上直接坐标
            # pre_cls[nB,nA,nG,nG,nC]
            # target:[nGtBox, (cx,cy,w,h,a,cls,idx)] 输入图上的相对坐标
            # masked_anchors[nG,nG,nA] 在特征图上的值
            
            # pred_cls[nB,nA,bG,bG,nC]
            nB, nA, nG, _, nC = pred_cls.size()

            num_target = len(target)

            # obj_mask：1表示该位置是物体
            # obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
            # noobj_mask： 1表示该位置没有物体
            noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)

            tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
            ta = FloatTensor(nB, nA, nG, nG).fill_(0)
            tw = FloatTensor(nB, nA, nG, nG).fill_(0)
            th = FloatTensor(nB, nA, nG, nG).fill_(0)
            tx = FloatTensor(nB, nA, nG, nG).fill_(0)
            ty = FloatTensor(nB, nA, nG, nG).fill_(0)
            tconf = FloatTensor(nB, nA, nG, nG).fill_(0)

            # 转化为特征图直接坐标
            # target_boxes：[nGtBox,(x,y,w,h,a)]
            scaled_target = torch.cat((target[:, :4] * nG, target[:, 4:5]), dim=-1)

            # 提取target_boxs x,y,w,h,a
            gxy = scaled_target[:, :2]  # cx,cy
            gwh = scaled_target[:, 2:4]  # w,h
            ga = scaled_target[:, 4]  # angle
           
            # 构建零点GT框[0,0,w,h,a]
            gt_box = torch.cat((torch.zeros((num_target,2),device=device), scaled_target[:, 2:5]),dim=1)
            # 构建零点anchor框[0,0,w,h,a]
            anchor_shapes = torch.cat((torch.zeros((num_anchors,2),device=device),scaled_anchors),dim=1)

            # 计算Anchors与GT框的交并集，特征图上的直接坐标
            # anchors_da, anchors_iou = bbox_iou_mix(anchor_shapes, gt_box, ifxywha=True)
            # anchors_da = torch.abs(anchors_da)
            # anchors_ariou = anchors_da*anchors_iou

            anchors_skewious = []
            offset = []
            for anchor in anchor_shapes:
                # 在特征图坐标系上进行计算
                anchors_skewiou = skewiou_fun(anchor, gt_box)
                anchors_skewious.append(anchors_skewiou)
                # 角度偏置
                offset.append(torch.abs(torch.sub(anchor[4], ga)))
            anchors_skewious = torch.stack(anchors_skewious).to(device)
            offset = torch.stack(offset)
            anchors_ariou=anchors_skewious
            anchors_da = offset

            # best_n：与每个target匹配的anchor的序号，根据anchor与GT的形状匹配来确定best_n,即anchor序号
            _, best_n = anchors_ariou.max(0)

            # target:[:,(cx,cy,w,h,a,cls,idx)]
            idx = target[:, 6].long().t()
            target_labels = target[:, 5].long().t()
            # 转置,取整获得网格坐标
            gi, gj = gxy.long().t()  

            # 确定哪些位置上的anchor有物体
            # obj_mask[idx, best_n, gj, gi] = 1
            obj_mask = [idx, best_n, gj, gi]

            # 确定哪些位置上的anchor没有物体
            noobj_mask[idx, best_n, gj, gi] = 0
            # 如果某些位置anchor的iou值大于了ignore_thres且角度偏置小
            # 于ignore_angthresh，不计算相应的损失函数，在noobj_mask对应位置置0
            for i, (anchor_ariou, anchor_da) in enumerate(zip(anchors_ariou.t(), anchors_da.t())):
                noobj_mask[idx[i], (anchor_ariou > self.ignore_iouthresh[0]), gj[i], gi[i]] = 0
                noobj_mask[idx[i], (anchor_ariou > self.ignore_iouthresh[1]) & (
                    anchor_da%(0.5*np.pi) < self.ignore_angthresh), gj[i], gi[i]] = 0

            # 真实角度偏差
            ta[idx, best_n, gj, gi] = ga - scaled_anchors[best_n][:, 2]
            # 真实宽高修正偏差
            tw[idx, best_n, gj, gi] = torch.log(gwh[:, 0]/scaled_anchors[best_n][:, 0])
            th[idx, best_n, gj, gi] = torch.log(gwh[:, 1]/scaled_anchors[best_n][:, 1])
            # 真实中心坐标修正偏差
            tx[idx, best_n, gj, gi] = gxy[:, 0] - gi
            ty[idx, best_n, gj, gi] = gxy[:, 1] - gj
            # 真实类别
            tcls[idx, best_n, gj, gi, target_labels] = 1

            # 真实置信度
            # tconf = obj_mask.float()
            tconf[obj_mask] = 1.0

            with torch.no_grad():
                bbox_loss_scale = 2.0 - 1.0 * gwh[:, 0] * gwh[:, 1] / (nG ** 2)
         
            # obj_mask = obj_mask.type(torch.bool)

            noobj_mask = noobj_mask.type(torch.bool)


            # 这里pred_boxes和target_boxes都是特征图上的直接坐标
            iou, anchors_skewiou, giou, ciou = bbox_iou(pred_boxes[obj_mask], scaled_target,True)
            
            # 计算回归损失函数
            if self.reg_type=='const_factor':  
                # 使用交叉熵损失函数计算conf_loss
                conf_loss_obj = F.binary_cross_entropy(
                    pred_conf[obj_mask], tconf[obj_mask], reduction="mean")
                conf_loss_noobj = F.binary_cross_entropy(
                    pred_conf[noobj_mask], tconf[noobj_mask], reduction="mean")
                conf_loss = conf_loss_obj + conf_loss_noobj

                # 使用交叉熵损失函数计算cls_loss
                cls_loss = F.binary_cross_entropy(
                    pred_cls[obj_mask], tcls[obj_mask], reduction="mean")

                # 回归损失
                angle_loss = F.smooth_l1_loss(pred_a[obj_mask], ta[obj_mask],reduction="none")
                ciou_loss = 1-ciou
                reg_loss = angle_loss + ciou_loss
                ariou_loss = torch.exp(1 - anchors_skewiou) - 1
                iou_const = ariou_loss
                with torch.no_grad():
                    reg_const = iou_const / reg_loss
                reg_loss = torch.mean(reg_loss * reg_const)

            elif self.reg_type=='ciou_l1':               
                # 使用交叉熵损失函数计算conf_loss
                conf_loss_obj = F.binary_cross_entropy(
                    pred_conf[obj_mask], tconf[obj_mask], reduction="mean")
                conf_loss_noobj = F.binary_cross_entropy(
                    pred_conf[noobj_mask], tconf[noobj_mask], reduction="mean")
                conf_loss = conf_loss_obj + conf_loss_noobj

                # 使用交叉熵损失函数计算cls_loss
                cls_loss = F.binary_cross_entropy(
                    pred_cls[obj_mask], tcls[obj_mask], reduction="mean")

                angle_loss = F.smooth_l1_loss(pred_a[obj_mask], ta[obj_mask], reduction="none")
                ciou_loss = 1-ciou
                reg_loss = torch.mean(angle_loss + ciou_loss)
  
            elif self.reg_type=='l1':
                # 使用交叉熵损失函数计算conf_loss
                conf_loss_obj = F.binary_cross_entropy(
                    pred_conf[obj_mask], tconf[obj_mask], reduction="mean")
                conf_loss_noobj = F.binary_cross_entropy(
                    pred_conf[noobj_mask], tconf[noobj_mask], reduction="mean")
                conf_loss = conf_loss_obj + conf_loss_noobj

                # 使用交叉熵损失函数计算cls_loss
                cls_loss = F.binary_cross_entropy(
                    pred_cls[obj_mask], tcls[obj_mask], reduction="mean")

                # pred_a_inrange = torch.frac(pred_a / (0.5*np.pi))
                pred_a_inrange = pred_a
                angle_loss = F.smooth_l1_loss(
                    pred_a_inrange[obj_mask], ta[obj_mask], reduction="mean")
                xy_loss = F.binary_cross_entropy(pred_x[obj_mask], tx[obj_mask], reduction="mean")+F.binary_cross_entropy(
                    pred_y[obj_mask], ty[obj_mask], reduction="mean")
                wh_loss = F.mse_loss(pred_w[obj_mask], tw[obj_mask], reduction="mean")+F.mse_loss(
                    pred_h[obj_mask], th[obj_mask], reduction="mean")
                reg_loss = angle_loss+xy_loss+wh_loss

            # Loss scaling
            reg_loss = 1. * reg_loss
            conf_loss = 1. * conf_loss
            cls_loss = 1. * cls_loss
            loss = reg_loss + conf_loss + cls_loss

            return loss, reg_loss, conf_loss, cls_loss
        else:
            return output