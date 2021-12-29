# Reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/datasets.py

import glob
import random
import os
import numpy as np
import cv2 as cv
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from tools.plot import xywha2xyxyxyxy
from tools.augments import vertical_flip, horisontal_flip, rotate, gaussian_noise
from model.utils import preprocess_input


# 图片填充到方形分辨率
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding，pad的四个数分别代表左右上下的填充维度
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageDataset(Dataset):
    def __init__(self, folder_path, img_size):
        self.files = sorted(glob.glob("%s/*.jpg" % folder_path))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #  Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #  Width in {320, 416, 512, 608, ... 320 + 96 * m}
        img_path = self.files[index % len(self.files)]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        #transform = transforms.ToPILImage(mode="RGB")
        #image = transform(img)
        #image.show()
        return img_path, img


class ListDataset(Dataset):
    def __init__(self, list_path, labels, input_size):
        
        self.img_files = list_path
        # 找到与各图片相对应的lable路径
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.inputsize = input_size
        self.labels = labels

    def __getitem__(self, index):
        # 读取图片，填充为正方形,黑色填充 pad[左，右，上，下]
        img_path = self.img_files[index]
        image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(image.shape) != 3:
            image = image.unsqueeze(0)
            image = image.expand((3, image.shape[1:]))

        _, h, w = image.shape

        image, pad = pad_to_square(image, 0)
        # 延展后的宽高
        _, padded_h, padded_w = image.shape

        #  Label，x1, y1, x2, y2, x3, y3, x4, y4，theta, x, y, w, h
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            # 得到所有的GT框
            box = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 13))
            # 得到所有的类别
            label = torch.from_numpy(np.array(self.labels[index]))
            # 框的四个角坐标，直接坐标系
            x1, y1, x2, y2, x3, y3, x4, y4 = \
                box[:, 0], box[:, 1], box[:, 2], box[:, 3], box[:, 4], box[:, 5], box[:, 6], box[:, 7]           
            num_targets = len(box)
            # 中心点坐标
            x = ((x1 + x3) / 2 + (x2 + x4) / 2) / 2
            y = ((y1 + y3) / 2 + (y2 + y4) / 2) / 2
            # 长宽
            w = torch.sqrt(torch.pow((x1 - x2), 2) + torch.pow((y1 - y2), 2))
            h = torch.sqrt(torch.pow((x2 - x3), 2) + torch.pow((y2 - y3), 2))
            # 角度计算， 逆时针方向
            theta = ((y2 - y1) / (x2 - x1 + 1e-16) + (y3 - y4) / (x3 - x4 + 1e-16)) / 2
            # theta:[-pi/2,pi/2]
            theta = torch.atan(theta)

            # ###############################################################
            # 这里调整到[-pi/2,pi/2)
            # theta = torch.stack([t if t != (np.pi / 2) else t - np.pi for t in theta])
            # for i in range(num_targets):
            #     if theta[i]>0:
            #         temp1, temp2 = h[i].clone(), w[i].clone()
            #         w[i], h[i] = temp1, temp2
            #         # 调整角度到[-pi/2,0)
            #         theta[i] -= np.pi/2
            #################################################################

            # 调整角度到(-pi/2,pi/2]
            theta = torch.stack([t if t != -(np.pi / 2) else t + np.pi for t in theta])
            for t in theta:
                assert -(np.pi / 2) < t <= (np.pi / 2), "angle: " + str(t)

            for i in range(num_targets):
                # 如果是竖条状，转为横条状，角度作相应变换
                if w[i] < h[i]:
                    temp1, temp2 = h[i].clone(), w[i].clone()
                    w[i], h[i] = temp1, temp2
                    if theta[i] > 0:
                        theta[i] = theta[i] - np.pi / 2
                    else:
                        theta[i] = theta[i] + np.pi / 2
            assert (-np.pi / 2 < theta).all() or (theta <= np.pi / 2).all()

            # 填补为方形后的新中心坐标 pad[左右上下]
            x += pad[0]
            y += pad[2]

            # Returns (x, y, w, h)
            # x,y,w,h之于图片的相对位置，即相对坐标
            x /=  padded_w
            y /=  padded_h
            w /=  padded_w
            h /=  padded_h
            
            targets = torch.zeros((len(box), 7))
            targets[:, 0] = x
            targets[:, 1] = y
            targets[:, 2] = w
            targets[:, 3] = h
            targets[:, 4] = theta
            targets[:, 5] = label
        else:
            targets = torch.zeros((1, 7))
            #没有真实框是lable为-1
            targets[:, 5] = -1
            return image, targets, img_path

        # target里面是相对坐标
        return  image, targets, img_path

    def collate_fn(self, batch):
        imgs, targets, paths = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 6] = i
        # 重新组合targets为tensor
        targets = torch.cat(targets, 0)
        # label以转化为相对坐标，resize图像也不用转化label
        imgs = torch.stack([resize(img, self.inputsize) for img in imgs])
        return imgs, targets, paths

    def __len__(self):
        return len(self.img_files)


def split_data(data_dir, ncls, img_size, batch_size, shuffle=True):
  
    dataset = ImageFolder(data_dir)

    classes = [[] for _ in range(ncls)]
    # x是路径，y是文件夹名，也即是lable
    for x, y in dataset.samples:
        classes[int(y)].append(x)

    train_inputs, train_labels = [], []

    # 讀取每個類別中所有的檔名 (i: label, data: filename)
    for i, data in enumerate(classes):  
        for x in data:
            train_inputs.append(x)
            train_labels.append(i)

    train_dataset = ListDataset(train_inputs, train_labels, input_size=img_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                                   pin_memory=True, collate_fn=train_dataset.collate_fn)

    return train_dataset, train_dataloader


if __name__ == "__main__":
    train_dataset, train_dataloader = split_data("data/test", 416)
    for i, (img_path, imgs, targets) in enumerate(train_dataloader):
        img = imgs.squeeze(0).numpy().transpose(1, 2, 0)
        img = img.copy()
        print(img_path)

        # targets[img_idx, label, x, y, w, h, a],注意是相对坐标
        for p in targets:
            x, y, w, h, theta = p[2] * img.shape[1], p[3] * img.shape[1], p[4] * img.shape[1], p[5] * img.shape[1], p[6]

            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = xywha2xyxyxyxy(np.array([x, y, w, h, theta]))
            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = int(X1), int(Y1), int(X2), int(Y2), int(X3), int(Y3), int(X4), int(Y4)

            cv.line(img, (X1, Y1), (X2, Y2), (255, 0, 0), 1)
            cv.line(img, (X2, Y2), (X3, Y3), (255, 0, 0), 1)
            cv.line(img, (X3, Y3), (X4, Y4), (255, 0, 0), 1)
            cv.line(img, (X4, Y4), (X1, Y1), (255, 0, 0), 1)

        cv.imshow('My Image', img)
        # 0-1转为0-255
        img[:, 1:] = img[:, 1:] * 255.0
        if img_path[0].split('/')[-2] == str(1):
            path = "data/augmentation/plane_" + img_path[0].split('/')[-1]
        else:
            path = "data/augmentation/car_" + img_path[0].split('/')[-1]
        cv.imwrite(path, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
