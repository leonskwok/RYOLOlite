# encoding: utf-8

import time
import shutil
import os
import torch
import numpy as np
from model.config import config

import argparse
from model import yololayer
from tools.plot import load_class_names, plot_boxes
from tools.post_process import post_process
from tools.load import ImageDataset

cfg = config()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

if __name__ == "__main__":
    # 清理数据
    shutil.rmtree('./outputs/')
    os.makedirs('./outputs/')

    model = cfg.model
    class_path = cfg.class_path
    weights_path = cfg.weights_path
    img_size = cfg.img_size
    batch_size = cfg.testbatchsize

    conf_thres = cfg.conf_thres
    nms_thres = cfg.nms_thres
    output_folder = cfg.output_folder
    yolo_layer = cfg.yolo_layer
    hardsampledetect = cfg.hardsampledetect
    hardsamplepath = cfg.hardsample_path
    if hardsampledetect:
        detect_folder = cfg.hardsample_folder
        with open(hardsamplepath, 'r') as f:
            shutil.rmtree(detect_folder)
            os.makedirs(detect_folder)
            paths = f.read().splitlines()
            for path in paths:
                fpath, fname = os.path.split(path)
                shutil.copy(path, os.path.join(detect_folder, fname))
    else:
        detect_folder = cfg.detect_folder

    if cfg.CUDA:
        FloatTensor = torch.cuda.FloatTensor
        device = 'cuda'
    else:
        FloatTensor = torch.FloatTensor
        device = 'cpu'
    class_names = load_class_names(class_path)
    pretrained_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model = model.to(device)
    model.load_state_dict(pretrained_dict)

    model.eval()

    dataset = ImageDataset(detect_folder, img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

    boxes = []
    imgs = []

    start = time.time()

    time_infer = []
    for img_path, img in dataloader:
        img = torch.autograd.Variable(img.type(FloatTensor))

        with torch.no_grad():
            temp = time.time()
            outputs = model(img)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]
            temp2 = time.time()
            prediction = []
            for l in range(len(outputs)):
                prediction.append(yolo_layer(l, outputs[l]))
            prediction = torch.cat((prediction[0], prediction[1]), dim=1)
            box = post_process(prediction, conf_thres, nms_thres)
            boxes.extend(box)
            print('-----------------------------------')
            num = 0
            for b in box:
                num += len(b)
            print("{}-> {} objects found".format(img_path, num))
            time_infer.append(temp2 - temp)
            print('-----------------------------------')

        imgs.extend(img_path)

    end = time.time()
    print('-----------------------------------')
    print("Total detecting time : ", round(end - start, 5))
    print('-----------------------------------')
    print('mean_inference_time: ', round(np.mean(time_infer), 5))
    print('FPS: ', round(1/np.mean(time_infer),5))

    for i, (img_path, box) in enumerate(zip(imgs, boxes)):
        plot_boxes(img_path, box, class_names, img_size, output_folder)
