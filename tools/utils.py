from os import path
from pickle import decode_long
import numpy as np
import torch
import tqdm
from shapely.geometry import Polygon
from numpy.core.fromnumeric import mean
# from test import num9060,num6030,num3010,num1000


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loss, window,  update_type):
    viz.line(
        X=torch.ones((1,)).cpu() * iteration,
        Y=torch.Tensor([loss]).cpu(),
        win=window,
        update=update_type
    )




def R(theta):
    """
    Args:
        theta: must be radian
    Returns: rotation matrix
    """
    r = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])
    return r


def T(x, y):
    """
    Args:
        x, y: values to translate
    Returns: translation matrix
    """
    t = torch.tensor([[1, 0, x], [0, 1, y], [0, 0, 1]])
    return t


def rotate(center_x, center_y, a, p):
    P = torch.matmul(T(center_x, center_y), torch.matmul(R(a), torch.matmul(T(-center_x, -center_y), p)))
    return P[:2]


def xywha2xyxyxyxy(p):
    """
    Args:
        p: 1-d tensor which contains (x, y, w, h, a)
    Returns: bbox coordinates (x1, y1, x2, y2, x3, y3, x4, y4) which is transferred from (x, y, w, h, a)
    """
    x, y, w, h, a = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]

    x1, y1, x2, y2 = x + w / 2, y - h / 2, x + w / 2, y + h / 2
    x3, y3, x4, y4 = x - w / 2, y + h / 2, x - w / 2, y - h / 2

    P1 = torch.tensor((x1, y1, 1)).reshape(3, -1)
    P2 = torch.tensor((x2, y2, 1)).reshape(3, -1)
    P3 = torch.tensor((x3, y3, 1)).reshape(3, -1)
    P4 = torch.tensor((x4, y4, 1)).reshape(3, -1)
    P = torch.stack((P1, P2, P3, P4)).squeeze(2).T
    P = rotate(x, y, a, P)
    X1, X2, X3, X4 = P[0]
    Y1, Y2, Y3, Y4 = P[1]

    return X1, Y1, X2, Y2, X3, Y3, X4, Y4


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def skewiou_fun(box1, box2):
    iou = []
    g = torch.stack(xywha2xyxyxyxy(box1))
    g = Polygon(g.reshape((4, 2)))
    for i in range(len(box2)):
        p = torch.stack(xywha2xyxyxyxy(box2[i]))
        p = Polygon(p.reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            value=0
        else:
            inter = Polygon(g).intersection(Polygon(p)).area
            union = g.area + p.area - inter
            if union ==0:
                value=0
            else:
                value = inter / (union + 1e-16)
        iou.append(torch.tensor(value))
    return torch.stack(iou)


def calskewiou(box1, box2):
    g = torch.stack(xywha2xyxyxyxy(box1))
    g = Polygon(g.reshape((4, 2)))
    p = torch.stack(xywha2xyxyxyxy(box2))
    p = Polygon(p.reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union==0:
        return 0
    else:
        skewiou = torch.tensor(inter / (union + 1e-16))
        return skewiou


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    # 将conf从小到大排序，返回对应的索引
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    


    # Find unique classes
    # 收集所有图片中出现的工件类
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    # ap平均精确度，p精确率，r召回率 
    # 精确率 = 正确预测样本中实际正样本数 / 所有的正样本数   precision = TP/(TP+FP)
    # 召回率 = 正确预测样本中实际正样本数 /实际的正样本数    Recall = TP/(TP+FN)

    ap, p, r = [], [], []
    # tqdm进度条
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        # 提取预测的所有的c类别框
        i = pred_cls == c
        # c类目标框总数
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        # c类预测框总数
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:

            # 错误预测的样本数，cumsum累加
            fpc = (1 - tp[i]).cumsum()
            # 正确预测的样本数，cumsum累加
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / n_gt
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    # targets[img_idx, label, x, y, w, h, a], 注意已经转为绝对坐标
    # outputs:[x,y,w,h,a,conf,cls_conf,cls_idx]
    batch_metrics = []
    # 轮询每一张图片
    for sample_i in range(len(outputs)):
        # 没有检测到目标
        if outputs[sample_i] is None:
            continue
        # output是其中一幅图的检测到的所有目标
        output = outputs[sample_i]
        pred_boxes = output[:, :5]
        pred_scores = output[:, 5]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])
        # annotations[label, x, y, w, h, a]
        # 在targets中找到对应的图片
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        # 对应图片的目标的lable集合
        target_labels = annotations[:, 0] if len(annotations) else []
        
        # 图片中存在目标框
        if len(annotations):
            # 记录已经成功预测的目标框的idx集合
            detected_boxes = []
            # 目标框集合
            target_boxes = annotations[:, 1:]

            # 轮询单张图片中所有的预测框
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                # 已经找到所有的目标框，跳出循环
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                # 预测框的类别不在target中，直接跳过
                if pred_label not in target_labels:
                    continue

                # 找到与预测框交并集最大的目标框
                iou, box_index = skewiou_fun(pred_box, target_boxes).max(0)

                # 预测框与目标框的iou大于threhold，则认为准确预测并记录
                if iou >= iou_threshold and box_index not in detected_boxes:
                    # true_positives预测框中的正例或负例，1为正例，0为负例
                    true_positives[pred_i] = 1
                    # 记录已经成功预测的目标框的idx集合
                    detected_boxes += [box_index]
        true_positives.sort()
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
        # batch_metrics[true_positives,pred_scores，pred_labels]
    return batch_metrics

def new_get_batch_statistics(outputs, targets, paths, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    # tensor targets[:(x, y, w, h, a, cls, idx)], 注意已经转为绝对坐标
    # 数组   outputs:[bs:,(x,y,w,h,a,conf,cls_conf,cls_idx)]
    batch_metrics = []

    rec_hardsample=[]
    recdata=[]

    # 轮询每一张图片
    for sample_i in range(len(outputs)):
        # 没有检测到目标
        if outputs[sample_i] is None:
            continue
        # output是其中一幅图的检测到的所有目标
        output = outputs[sample_i]
        pred_boxes = output[:, :5]
        pred_scores = output[:, 5]
        pred_labels = output[:, -1]

        # 预测框的数目
        true_positives = np.zeros(pred_boxes.shape[0])
        # 在targets中找到对应的图片  annotations[x, y, w, h, a, cls]
        annotations = targets[targets[:, 6] == sample_i][:, :6]
        # 对应图片的目标的lable集合
        target_labels = annotations[:, 5] if len(annotations) else []


        # 图片中存在目标框
        if len(annotations):
            # 记录已经成功预测的目标框的idx集合
            detected_boxes = []
            # 目标框集合
            target_boxes = annotations[:, :5]

            # 轮询单张图片中所有的预测框
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # 已经找到所有的目标框，跳出循环
                if len(detected_boxes) == len(annotations):
                    break

                # 预测框的类别不在target中，直接跳过
                if pred_label not in target_labels:
                    continue

                # 找到与预测框交并集最大的目标框
                iou, box_index = skewiou_fun(pred_box, target_boxes).max(0)
                if iou>1:
                    print(box_index)
                
                gtclass = target_labels[box_index]
                preclass = pred_label


                # 预测框与目标框的iou大于threhold，则认为准确预测并记录
                if iou >= iou_threshold and box_index not in detected_boxes and preclass==gtclass:
                    # true_positives预测框中的正例或负例，1为正例，0为负例
                    true_positives[pred_i] = 1

                    # 记录已经成功预测的目标框的idx集合
                    detected_boxes += [box_index]
                    
                    px, py, pw, ph, pa = pred_box
                    gx, gy, gw, gh, ga = target_boxes[box_index]
                    
                    recdata.append([px, py, pw, ph, pa, gx, gy,
                                   gw, gh, ga, iou, preclass, gtclass])

                    da=(torch.abs(pa-ga)/np.pi*180)%90

                    # 记录预测过大的图片路径 
                    if da>30:
                        rec_hardsample.append([paths[sample_i],da])
                    
                    dxy = torch.sqrt((px-gx)*(px-gx)+(py-gy)*(py-gy))
                    if dxy>1.35:
                        rec_hardsample.append([paths[sample_i], dxy])

                                               
        # true_positives.sort()
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
        # batch_metrics[true_positives,pred_scores，pred_labels,pred_angles]
    return batch_metrics, rec_hardsample, recdata


def datastatistical(data, AP, ncls, classname, APpath, Catchpath):
    px = data[:, 0]
    py = data[:, 1]
    pw = data[:, 2]
    ph = data[:, 3]
    pa = data[:, 4]
    gx = data[:, 5]
    gy = data[:, 6]
    gw = data[:, 7]
    gh = data[:, 8]
    ga = data[:, 9]
    iou = data[:, 10]

    dx = torch.abs(px-gx)
    dy = torch.abs(py-gy)
    dxy = torch.sqrt(dx*dx+dy*dy)
    dw = torch.abs(pw-gw)
    dh = torch.abs(ph-gh)
    da = (torch.abs(pa-ga)/torch.pi*180) % 90
    
    
    n00_05 = 0
    n05_10 = 0
    n10_30 = 0
    n30_60 = 0
    n60_90 = 0
    total = len(da)
    for a in da:
      if a <= 5:
          n00_05 += 1
      elif a > 5 and a <= 10:
          n05_10 += 1
      elif a > 10 and a <= 30:
          n10_30 += 1
      elif a > 30 and a < 60:
          n30_60 += 1
      else:
          n60_90 += 1

    


    with open(APpath, 'a+') as f:
        f.writelines('%d\n'%(AP))
        f.writelines('dw: % .3f\t % .3f \n' % (torch.mean(dw), torch.max(dw)))
        f.writelines('dh: % .3f\t % .3f \n' % (torch.mean(dh), torch.max(dh)))
        f.writelines('da: % .3f\t % .3f \n' % (torch.mean(da), torch.max(da)))
        f.writelines('iou: % .3f\t % .3f \n' %
                     (torch.mean(iou), torch.min(iou)))
        f.writelines('dxy: % .3f\t % .3f \n' %
                     (torch.mean(dxy), torch.max(dxy)))
        f.writelines('% .3f\t % .3f\t % .3f\t % .3f\t % .3f \n' %
                        (n00_05/total*100, n05_10/total*100, n10_30/total*100, n30_60/total*100, n60_90/total*100))
        f.writelines('\n\n\n')
    
    rec = [[] for _ in range(ncls)]
    with open(Catchpath, 'a+') as f:
        f.writelines('%d\n' % (AP))
        for i in range(ncls):
            clsdata = data.clone()
            # 分类
            clsdata = clsdata[clsdata[:, 11] == i]

            sum = 500 if len(clsdata) < 500 else len(clsdata)
            clsdata = clsdata[clsdata[:, 12] == i]
            # 角度
            pa = clsdata[:, 4]
            ga = clsdata[:, 9]
            clsdata = clsdata[torch.abs(pa-ga)/np.pi*180 % 90 < 5]

            # 定位误差
            px = clsdata[:, 0]
            py = clsdata[:, 1]
            gx = clsdata[:, 5]
            gy = clsdata[:, 6]
            dx = px-gx
            dy = py-gy
            dxy = torch.sqrt(dx*dx+dy*dy)
            clsdata = clsdata[dxy <= 1.5]

            ph = clsdata[:, 3]
            gh = clsdata[:, 8]
            dh = torch.abs(ph-gh)
            clsdata = clsdata[dh <= 1.5]

            rec[i] = len(clsdata)/sum
            f.writelines('-class %d(%s): %.4f\n' % (i, classname[i], len(clsdata)/sum))

            # pw = cls[:, 2]
            # gw = cls[:, 7]
        rec = torch.Tensor(rec)
        f.writelines('平均成功率：%.4f' % (rec.mean().item()))
        f.writelines('\n\n\n')
