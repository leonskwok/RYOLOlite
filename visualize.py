import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tools.augments import resize, pad_to_square
from model.config import config
import torch.nn as nn
import os


def PlotFeature():
    cfg = config()
    model = cfg.model
    weights_path = cfg.weights_path
    img_size = cfg.img_size
    
    # 加载模型
    pretrained_dict = torch.load(
        weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)

       
    # 读取图像并预处理
    img_path = "data/process_5workpiece/others/4-2.jpg"
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    img, _ = pad_to_square(img, 0)
    img = resize(img, img_size)
    img = img.unsqueeze(0)
    

    # 模型推理到指定层
    img = model.backbone.conv1(img)
    img = model.backbone.conv2(img)
    # img = model.backbone.resblock_body1(img)[0]
    img = img.squeeze(0)
    img=img.detach()

    # 画特征图
    feature_num = img.shape[0]
    row_num = int(np.ceil(np.sqrt(feature_num)))
    plt.figure(figsize=(15,15))
    for index in range(1, feature_num+1):       
        plt.subplot(row_num, row_num, index)  
        plt.imshow(img[index-1], cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace = 0, wspace=0)
    plt.savefig(os.path.join(os.path.dirname(img_path),"feature.png"))
    quit()

if __name__ == '__main__':
    PlotFeature()

