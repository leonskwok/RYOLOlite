import torch
from tools.utils import skewiou_fun

def post_process(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Args:
        prediction: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 10] in my case
                    last dimension-> [x, y, w, h, a, conf, num_classes]
    Returns:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # prediction[batch_size, (76*76+38*38+19*19)*3*6, (x,y,w,h,a,conf,ncls...)]
    output = [None for _ in range(len(prediction))]
    # 轮询每一幅图
    for batch, image_pred in enumerate(prediction):
        # image_pred[(76*76+38*38+19*19)*3*6, (x,y,w,h,a,conf,ncls...)]
        # 过滤掉低置信度的anchor
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]

        if not image_pred.size(0):
            continue

        # max(1)[0]返回所有行中的最大值,即最大类别置信度
        # score=clsconf*conf，conf为是否存在物体置信度，两者乘积即为该anchor的得分
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        # 根据score从大到小排列image_pred，即排列所有anchor框
        image_pred = image_pred[(-score).argsort()]
        # 每个anchor框对应的最大类别置信度即其类别索引(0-ncls)
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)
        # detection即为经过刷选和置信度排序的anchor框，detections[numanchor:(x,y,w,h,a,conf,cls_conf,cls_idx)]
        detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1)

        # non-maximum suppression
        keep_boxes = []      
        labels = detections[:, -1].unique()
        # 轮询每个分类
        for label in labels:
            # 筛选目标类别[:,(x,y,w,h,a,conf,cls_conf,cls_idx)]
            detect = detections[detections[:, -1] == label]
            while len(detect):
                # 求最高score的预测框与其他预测框的iou,overlap筛选得到预测同一目标的所有预测框
                large_overlap = skewiou_fun(detect[0, :5], detect[:, :5]) > nms_thres
                # 筛选得到iou满足nms_thres的预测框的conf和clsconf
                weights = detect[large_overlap, 5:6]
                # 边界框融合策略,用几个预测框共同拟合出一个预测框
                detect[0, :4] = (weights * detect[large_overlap, :4]).sum(0) / weights.sum()
                # detach()返回一个从当前图中分离的Variable,不会更新梯度
                keep_boxes += [detect[0].detach()]
                # nms,消除检测个共同目标的所有预测框
                detect = detect[~large_overlap]
            if keep_boxes:
                # 保存每幅图中的所有目标框[:,(x,y,w,h,a,conf,cls_conf,cls_idx)]
                output[batch] = torch.stack(keep_boxes)
    return output
