import torch
import numpy as np
from torch.autograd import Variable
from tools.plot import load_class_names
from tools.post_process import post_process
# from tools.utils import get_batch_statistics, ap_per_class
from tools.utils import get_batch_statistics, ap_per_class, new_get_batch_statistics
from tools.load import split_data
import os
from tqdm import tqdm

from torchstat import stat
from model.config import config
import time


cfg = config()

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

n60_90 = 0
n30_60 = 0
n10_30 = 0
n05_10 = 0
n00_05 = 0

if __name__ == "__main__":

    model = cfg.model
    test_folder = cfg.test_folder
    weights_path = cfg.weights_path
    class_path = cfg.class_path
    conf_thres = cfg.conf_thres
    nms_thres = cfg.nms_thres
    iou_thres = np.array(cfg.iou_thres).mean()
    testbatchsize = cfg.testbatchsize
    img_size = cfg.img_size
    ncls = cfg.ncls
    CUDA = cfg.CUDA
    yolo_layer = cfg.yolo_layer
    hardsamplepath = cfg.hardsample_path

    if CUDA:
        FloatTensor = torch.cuda.FloatTensor
        model = model.cuda()
        device = 'cuda'
    else:
        FloatTensor = torch.FloatTensor
        device = 'cpu'

    class_names = load_class_names(class_path)
    pretrained_dict = torch.load(
        weights_path, map_location=torch.device('cuda'))

    model.load_state_dict(pretrained_dict)
    model.eval()

    test_dataset, test_dataloader = split_data(
        test_folder, ncls, img_size, testbatchsize, shuffle=True)

    print("Compute mAP...")
    averageAP = []

    for thre in range(50, 100, 5):

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)

        rec_time = []
        recdata = []
        rec_hardsample = []

        length = len(test_dataset)

        with tqdm(total=length, desc=f'%0.2f' % (thre/100), postfix=dict, mininterval=0.3) as pbar:
            for batch_i, (imgs, targets, paths) in enumerate(test_dataloader):
                targets = targets.to(device)
                # 提取各图片的类别信息
                labels += targets[:, 5].tolist()
                # 相对坐标转为绝对坐标
                # [x,y,w,h,theta,cls]
                targets[:, :4] *= img_size
                imgs = torch.autograd.Variable(
                    imgs.type(FloatTensor), requires_grad=False)

                with torch.no_grad():
                    time_start = time.time()

                    outputs = model(imgs)
                    prediction = []
                    for l in range(len(outputs)):
                        prediction.append(yolo_layer(l, outputs[l]))
                    prediction = torch.cat(
                        (prediction[0], prediction[1]), dim=1)
                    # nms后处理 outputs:[:,(x,y,w,h,a,conf,cls_conf,cls_idx)] 绝对坐标
                    outputs = post_process(
                        prediction, conf_thres=conf_thres, nms_thres=nms_thres)

                    time_end = time.time()
                    rec_time.append(time_end-time_start)

                # sample_metrics[true_positives, pred_scores, pred_labels]
                # sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=args.iou_thres)
                for thre in range(50, 100, 5):
                    sample_metric, hardsample, samplerecdata = new_get_batch_statistics(
                        outputs, targets, paths, thre/100)
                    sample_metrics += sample_metric

                    recdata.extend(samplerecdata)
                    rec_hardsample.extend((hardsample))

                pbar.update(1)

        # true_positives, pred_scores, pred_labels
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)
        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        print(thre)
        print(f"mAP: {AP.mean()}")

        averageAP.append(AP.mean())

    print(f"averageAP: {np.array(averageAP).mean()}")

    # np.savetxt('statistical/'+cfg.model._get_name() + '.txt',  torch.Tensor(recdata).numpy())

    with open(hardsamplepath, 'w+', encoding='utf-8') as f:
        for path in rec_hardsample:
            f.writelines(path+'\n')
