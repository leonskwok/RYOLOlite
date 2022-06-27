import time
import shutil
import os
import numpy as np
from model.config import config
import torch
from tools.plot import plot_boxes
from tools.post_process import post_process
from tools.load import ImageDataset
from torch.autograd import Variable

cfg = config()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

def detect():
    # 清理数据
    shutil.rmtree('./outputs/')
    os.makedirs('./outputs/')

    model = cfg.model
    weights_path = cfg.weights_path
    img_size = cfg.img_size
    batch_size = cfg.testbatchsize

    conf_thres = cfg.conf_thres
    nms_thres = cfg.nms_thres
    output_folder = cfg.output_folder
    yolo_layer = cfg.yolo_layer
    hardsampledetect = cfg.hardsampledetect
    hardsample_txt = cfg.hardsample_txt

    # 检测困难样本
    if hardsampledetect:
        target_folder = cfg.hardsample_folder
        with open(hardsample_txt, 'r') as f:
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder)
            os.makedirs(target_folder)
            paths = f.read().splitlines()
            for path in paths:
                file, error = path.split('\t')
                fpath, fname = os.path.split(file)
                shutil.copy(file, os.path.join(target_folder, fname))
    else:
        target_folder = cfg.detect_folder

    FloatTensor = torch.cuda.FloatTensor if cfg.CUDA else torch.FloatTensor
    device = "cuda" if cfg.CUDA else "cpu"

    # 加载类别
    class_names = cfg.class_names
    # 检测模型数据
    pretrained_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model = model.to(device)
    model.load_state_dict(pretrained_dict)

    model.eval()

    dataset = ImageDataset(target_folder, img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

    boxes = []
    imgs = []

    t = time.time()
    time_infer = []
    for img_path, img in dataloader:
        img = Variable(img.type(FloatTensor))

        with torch.no_grad():
            temp = time.time()
            outputs = model(img)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]
            temp2 = time.time()

            prediction = []
            for l in range(len(outputs)):
                prediction.append(yolo_layer(l, outputs[l]))
            prediction = torch.cat(prediction,dim=1)
            outputs = post_process(prediction, conf_thres, nms_thres)        

        print('-----------------------------------')
        num = 0
        # b是单张输入图的所有预测框
        for b in outputs:
            # 有预测框
            if b is not None:
                num += len(b)
        print("{}-> {} objects found".format(img_path, num))
        time_infer.append(temp2 - temp)
        print('-----------------------------------')

        # 记录所有预测框
        boxes.extend(outputs)
        imgs.extend(img_path)

    print('-----------------------------------')
    print("Total detecting time : ", round(time.time() - t, 5))
    print('-----------------------------------')
    print('mean_inference_time: ', round(np.mean(time_infer), 5))
    print('FPS: ', round(1/np.mean(time_infer),5))

    for i, (img_path, outputs) in enumerate(zip(imgs, boxes)):
        plot_boxes(img_path, outputs, class_names, img_size, output_folder)

if __name__ == "__main__":
    detect()
