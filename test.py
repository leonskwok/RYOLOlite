import torch
import numpy as np
from torch.autograd import Variable
from tools.post_process import post_process
# from tools.utils import get_batch_statistics, ap_per_class
from tools.utils import ap_per_class, batch_statistics, datastatistical, get_batch_statistics
from tools.load import split_data
import os
from tqdm import tqdm

from torchstat import stat
from model.config import config
import time
import shutil

cfg = config()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

def test():
    model = cfg.model
    test_folder = cfg.test_folder
    weights_path = cfg.weights_path
    conf_thres = cfg.conf_thres
    nms_thres = cfg.nms_thres
    testbatchsize = cfg.testbatchsize
    img_size = cfg.img_size
    ncls = cfg.ncls
    CUDA = cfg.CUDA
    yolo_layer = cfg.yolo_layer
    hardsamplepath = cfg.hardsample_txt
    # 保存文件路径
    savedir = os.path.dirname(cfg.weights_path)
    # 记录检测误差
    Angfile = os.path.join(savedir, 'Ang.txt')
    if os.path.exists(Angfile):
        os.remove(Angfile)
    # 记录抓取成功率
    Catchfile = os.path.join(savedir, 'Catch.txt')
    if os.path.exists(Catchfile):
        os.remove(Catchfile)
    # 记录平均准确率
    APfile = os.path.join(savedir, 'AP.txt')
    if os.path.exists(APfile):
        os.remove(APfile)

    # 记录各阈值下的检测数据
    Statdir = os.path.join(savedir, 'stat')
    if os.path.exists(Statdir):
        shutil.rmtree(Statdir)
    os.makedirs(Statdir)

    if CUDA:
        FloatTensor = torch.cuda.FloatTensor
        model = model.cuda()
        device = "cuda"
    else:
        FloatTensor = torch.FloatTensor
        divice = "cpu"
        
    # 加载类别名
    class_names = cfg.class_names
    # 加载已训练模型
    pretrained_dict = torch.load(
        weights_path, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.eval()

    # 加载测试集
    test_dataset, test_dataloader = split_data(
        test_folder, ncls, img_size, testbatchsize, shuffle=True)

    print("Compute mAP...")
    averageAP = []
    # IoU阈值
    IoUthre = list(range(50, 100, 5))

    # List of tuples (TP, confs, pred)
    sample_metrics = [[] for _ in range(len(IoUthre))]
    # 记录困难样本
    rec_hardsample = [[] for _ in range(len(IoUthre))]
    #
    recdata = [[] for _ in range(len(IoUthre))]

    labels = []
    rec_time = []

    num_imgs = len(test_dataset)

    with tqdm(total=num_imgs, desc=f'', postfix=dict, mininterval=0.3) as pbar:
        for batch_i, (imgs, targets, paths) in enumerate(test_dataloader):
            targets = targets.to(device)
            # 提取各图片的类别信息
            labels += targets[:, 5].tolist()
            # 相对坐标转为绝对坐标
            # [x,y,w,h,theta,cls]
            targets[:, :4] *= img_size
            imgs = Variable(imgs.type(FloatTensor), requires_grad=False)

            t = time.time()
            with torch.no_grad():              
                outputs = model(imgs)
                prediction = []
                for l in range(len(outputs)):
                    prediction.append(yolo_layer(l, outputs[l]))
                prediction = torch.cat(prediction, dim=1)
                # nms后处理 outputs:[:,(x,y,w,h,a,conf,cls_conf,cls_idx)] 绝对坐标
                outputs = post_process(prediction, conf_thres, nms_thres)
            rec_time.append(time.time()-t)

            # sample_metrics[true_positives, pred_scores, pred_labels]
            # sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=args.iou_thres)
            for i in range(len(IoUthre)):
                sample_metric, hardsample, samplerecdata = batch_statistics(
                    outputs, targets, paths, IoUthre[i]/100)
                sample_metrics[i] += sample_metric
                recdata[i].extend(samplerecdata)
                rec_hardsample[i].extend((hardsample))

            pbar.update(1)

    # true_positives, pred_scores, pred_labels
    for i in range(len(IoUthre)):
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics[i]))]
        precision, recall, AP, f1, ap_class = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)

        # AP.txt
        with open(APfile, 'a+') as f:
            f.writelines('%d\n' % (IoUthre[i]))
            for j, c in enumerate(ap_class):
                f.writelines(
                    f"+ Class '{c}' ({class_names[c]}) - AP: {AP[j]}\n")
            f.writelines(f"mAP: {AP.mean()}\n")
            f.writelines('\n\n\n')

        # Ang.txt, Catch.txt
        datastatistical(torch.Tensor(
            recdata[i]), IoUthre[i], ncls, class_names, Angfile, Catchfile, num_imgs)

        # stat/APxx.txt  stat/Hardxx.txt
        np.savetxt(os.path.join(Statdir, 'AP%d.txt' %
                (IoUthre[i])),  torch.Tensor(recdata[i]).numpy())
        averageAP.append(AP.mean())

        # hardsample.txt
        with open(os.path.join(Statdir, 'Hard%d.txt' % (IoUthre[i])), 'a+', encoding='utf-8') as f:
            for path, da in rec_hardsample[i]:
                f.writelines(path+'\t%0.2f\n' % (da))

    with open(APfile, 'a+') as f:
        f.writelines(f"averageAP: {np.array(averageAP).mean()}")
    print(f"averageAP: {np.array(averageAP).mean()}")


if __name__ == "__main__":
    test()