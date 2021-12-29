import torch
import numpy as np
from torch.autograd import Variable
from tools.plot import load_class_names
from tools.post_process import post_process
# from tools.utils import get_batch_statistics, ap_per_class
from tools.utils import ap_per_class, new_get_batch_statistics, datastatistical, get_batch_statistics
from tools.load import split_data
import os
from tqdm import tqdm

from torchstat import stat
from model.config import config
import time
import shutil


cfg = config()

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU


if __name__ == "__main__":

    model = cfg.model
    test_folder=cfg.test_folder
    weights_path=cfg.weights_path
    class_path = cfg.class_path
    conf_thres = cfg.conf_thres
    nms_thres=cfg.nms_thres
    iou_thres = np.array(cfg.iou_thres).mean()
    testbatchsize=cfg.testbatchsize
    img_size=cfg.img_size
    ncls = cfg.ncls
    CUDA=cfg.CUDA
    yolo_layer=cfg.yolo_layer
    hardsamplepath=cfg.hardsample_path

    savedir=os.path.dirname(cfg.weights_path)
    

    Angfile = os.path.join(savedir,'Ang.txt')
    if os.path.exists(Angfile):
        os.remove(Angfile)

    Catchfile = os.path.join(savedir, 'Catch.txt')
    if os.path.exists(Catchfile):
        os.remove(Catchfile)

    APfile = os.path.join(savedir, 'AP.txt')
    if os.path.exists(APfile):
        os.remove(APfile)

    Statdir = os.path.join(savedir, 'stat')
    if os.path.exists(Statdir):
        shutil.rmtree(Statdir)
    os.makedirs(Statdir)

    if CUDA:
        FloatTensor = torch.cuda.FloatTensor
        model = model.cuda()
        device='cuda'
    else:
        FloatTensor = torch.FloatTensor
        device='cpu'

    class_names = load_class_names(class_path)
    pretrained_dict = torch.load(weights_path, map_location=torch.device('cuda'))
  
    model.load_state_dict(pretrained_dict)
    model.eval()

    test_dataset, test_dataloader = split_data(test_folder, ncls, img_size, testbatchsize,shuffle=True)

    print("Compute mAP...")
    averageAP = []

    thre = list(range(50, 100, 5))
    accumulate = [[] for _ in range(len(thre))]
    # List of tuples (TP, confs, pred)
    sample_metrics = [[] for _ in range(len(thre))]
    rec_hardsample = [[] for _ in range(len(thre))]
    recdata = [[] for _ in range(len(thre))]

    labels = []
    rec_time=[]
      
    length = len(test_dataset)

    with tqdm(total=length, desc=f'', postfix=dict, mininterval=0.3) as pbar:      
        for batch_i, (imgs, targets, paths) in enumerate(test_dataloader):
            targets = targets.to(device)
            # 提取各图片的类别信息
            labels += targets[:, 5].tolist()
            # 相对坐标转为绝对坐标
            # [x,y,w,h,theta,cls]
            targets[:, :4] *= img_size
            imgs = torch.autograd.Variable(imgs.type(FloatTensor), requires_grad=False)

            with torch.no_grad():           
                time_start = time.time()    

                outputs = model(imgs)
                prediction = []
                for l in range(len(outputs)):
                    prediction.append(yolo_layer(l, outputs[l]))
                prediction = torch.cat((prediction[0], prediction[1]), dim=1)
                # nms后处理 outputs:[:,(x,y,w,h,a,conf,cls_conf,cls_idx)] 绝对坐标
                outputs = post_process(prediction, conf_thres=conf_thres, nms_thres=nms_thres)

                time_end = time.time()
                rec_time.append(time_end-time_start)
            
            # sample_metrics[true_positives, pred_scores, pred_labels]
            # sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=args.iou_thres)
            for i in range(len(thre)):
                sample_metric, hardsample ,samplerecdata = new_get_batch_statistics(outputs, targets, paths, thre[i]/100)
                
                sample_metrics[i] += sample_metric
                recdata[i].extend(samplerecdata)
                rec_hardsample[i].extend((hardsample))
                   
            pbar.update(1)

    # true_positives, pred_scores, pred_labels
    for i in range(len(thre)):
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics[i]))]   
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        # AP.txt
        with open(APfile,'a+') as f:
            f.writelines('%d\n'%(thre[i]))
            for j, c in enumerate(ap_class):
                f.writelines(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[j]}\n")
            f.writelines(f"mAP: {AP.mean()}\n")
            f.writelines('\n\n\n')
        
        # Ang.txt, Catch.txt
        datastatistical(torch.Tensor(recdata[i]), thre[i], ncls, class_names, Angfile, Catchfile)

        # 不同阈值下的检测结果记录APxx.txt
        np.savetxt(os.path.join(Statdir,'AP%d.txt'%(thre[i])),  torch.Tensor(recdata[i]).numpy())
        averageAP.append(AP.mean())

        # hardsample.txt
        with open(os.path.join(Statdir, 'Hard%d.txt' % (thre[i])), 'a+', encoding='utf-8') as f:
            for path,da in rec_hardsample[i]:
                f.writelines(path+'\t%0.2f\n'%(da))

    with open(APfile,'a+') as f:
        f.writelines(f"averageAP: {np.array(averageAP).mean()}")
    print(f"averageAP: {np.array(averageAP).mean()}")
   










        
            


