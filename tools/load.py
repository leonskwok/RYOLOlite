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
from tools.augments import vertical_flip, horisontal_flip, rotate, gaussian_noise, resize, mixup, hsv, pad_to_square
from model.utils import preprocess_input


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
        temp = Image.open(img_path)

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

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


class BaseDataset(Dataset):
    def __init__(self, img_size=608, sample_size=600, augment=False, mosaic=False, multiscale=True, normalized_labels=False):
        self.img_size = img_size
        # 图像增强
        self.augment = augment
        # 马赛克
        self.mosaic = mosaic
        self.mosaic_sample_size = sample_size
        self.mosaic_border = [-self.mosaic_sample_size //
                              2, -self.mosaic_sample_size // 2]
        # 多尺度
        self.multiscale = multiscale
        # 平滑标签
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.img_files = None
        self.label_files = None
        self.category = None

    def __getitem__(self, index):
        # 马赛克增强
        if self.mosaic:
            img, targets = self.load_mosaic(index)
            if np.random.random() < 0.1:
                img2, targets2 = self.load_mosaic(index)
                # 两幅图混合
                img, targets = mixup(img, targets, img2, targets2)
            img = transforms.ToTensor()(img)

        else:
            img, (h, w) = self.load_image(index)
            img = transforms.ToTensor()(img)
            img, pad = pad_to_square(img, 0)

            label_factor = (h, w) if self.normalized_labels else (1, 1)
            padded_size = img.shape[1:]
            boundary = (0, w, 0, h)

            targets = self.load_target(
                index, label_factor, pad, padded_size, boundary)

        # Apply augmentations
        if self.augment:
            # 随机旋转
            if np.random.random() < 0.5:
                img, targets = rotate(img, targets)
            # 随机水平移动
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            # 随机垂直移动
            if np.random.random() < 0.5:
                img, targets = vertical_flip(img, targets)

        return self.img_files[index], img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        img_path = self.img_files[index]

        # Extract image as PyTorch tensor
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w, c = img.shape

        # Handle images with less than three channels
        if c != 3:
            img = np.transpose(np.stack(np.array([img, img, img])), (1, 2, 0))

        if self.augment:
            #img = gaussian_noise(img) # np.random.normal(mean, var ** 0.5, image.shape) would increase run time significantly
            hsv(img)

        return img, (h, w)

    def load_mosaic(self, index):
        """
        Loads 1 image + 3 random images into a 4-image mosaic.
        Each image is cropped based on the sameple_size.
        A larger sample size means more information in each image would be used.
        """

        labels4 = []
        s = self.mosaic_sample_size
        yc, xc = [int(random.uniform(-x, 2 * s + x))
                  for x in self.mosaic_border]  # mosaic center x, y
        # 3 additional image indices
        indices = [index] + \
            [random.randint(0, len(self.img_files) - 1) for _ in range(3)]
        random.shuffle(indices)

        h_padded, w_padded = s * 2, s * 2
        padded_size = (h_padded, w_padded)

        for i, index in enumerate(indices):
            img, (h, w) = self.load_image(index)
            label_factor = (h, w) if self.normalized_labels else (1, 1)

            # place img in img4
            if i == 0:  # top left
                img4 = np.zeros(
                    (h_padded, w_padded, img.shape[2]), dtype=np.uint8)
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                xt, yt = int(random.uniform((x2a - x1a), w)
                             ), int(random.uniform((y2a - y1a), h))
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x2b
                padh = yc - y2b
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                xt, yt = int(random.uniform((x2a - x1a), w)
                             ), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x1b
                padh = yc - y2b
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                xt, yt = int(random.uniform((x2a - x1a), w)
                             ), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x2b
                padh = yc - y1b
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, s * 2), min(s * 2, yc + h)
                xt, yt = int(random.uniform((x2a - x1a), w)
                             ), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x1b
                padh = yc - y1b

            # img4[ymin:ymax, xmin:xmax]
            img4[y1a:y2a, x1a:x2a, :] = img[y1b:y2b, x1b:x2b, :]

            # Labels
            pad = (padw, padw, padh, padh)
            boundary = (x1b, x2b, y1b, y2b)
            targets = self.load_target(
                index, label_factor, pad, padded_size, boundary)
            labels4.append(targets)

        # Concat labels
        if len(labels4):
            labels4 = torch.cat(labels4, 0)

        return img4, labels4

    def load_target(self, index, label_factor, pad, padded_size, boundary):
        """
        Args:
            index: index of label files going to be load
            label_factor: factor that resize labels to the same size with images
            pad: the amount of zero pixel value that are padded beside images
            padded_size: the size of images after padding
            boundary: the boundary of targets

        Returns:
            Normalized labels of objects -> [batch_index, label, x, y, w, h, theta] -> torch.Size([num_targets, 7])
        """
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            x, y, w, h, theta, label, num_targets = self.load_files(label_path)

            # Return zero length tersor if there is no object in the image
            if not num_targets:
                return torch.zeros((0, 7))

            # Check whether theta of oriented bounding boxes are within the boundary or not
            if not ((-np.pi / 2 < theta).all() or (theta <= np.pi / 2).all()):
                raise AssertionError(
                    "Theta of oriented bounding boxes are not within the boundary (-pi / 2, pi / 2]")

            # Make the height of bounding boxes always smaller then it's width
            for i in range(num_targets):
                if w[i] < h[i]:
                    temp1, temp2 = h[i].clone(), w[i].clone()
                    w[i], h[i] = temp1, temp2
                    if theta[i] > 0:
                        theta[i] = theta[i] - np.pi / 2
                    else:
                        theta[i] = theta[i] + np.pi / 2

            # Make the scale of coordinates to the same size of images
            h_factor, w_factor = label_factor
            x *= w_factor
            y *= h_factor
            w *= w_factor
            h *= h_factor

            # Remove objects that exceed the size of images or the cropped area when doing mosaic augmentation
            left_boundary, right_boundary, top_boundary, bottom_boundary = boundary
            mask = torch.ones_like(x)
            mask = torch.logical_and(mask, x > left_boundary)
            mask = torch.logical_and(mask, x < right_boundary)
            mask = torch.logical_and(mask, y > top_boundary)
            mask = torch.logical_and(mask, y < bottom_boundary)

            label = label[mask]
            x = x[mask]
            y = y[mask]
            w = w[mask]
            h = h[mask]
            theta = theta[mask]

            # Relocalize coordinates based on images padding or mosaic augmentation
            x1 = (x - w / 2) + pad[0]
            y1 = (y - h / 2) + pad[2]
            x2 = (x + w / 2) + pad[1]
            y2 = (y + h / 2) + pad[3]

            # Normalized coordinates
            padded_h, padded_w = padded_size
            x = ((x1 + x2) / 2) / padded_w
            y = ((y1 + y2) / 2) / padded_h
            w /= padded_w
            h /= padded_h

            targets = torch.zeros((len(label), 7))
            targets[:, 1] = label
            targets[:, 2] = x
            targets[:, 3] = y
            targets[:, 4] = w
            targets[:, 5] = h
            targets[:, 6] = theta
            return targets

        else:
            print(label_path)
            assert False, "Label file not found"

    def load_files(self):
        raise NotImplementedError


class UCASAODDataset(BaseDataset):
    def __init__(self, data_dir, class_names, img_size=416, sample_size=600, augment=False, mosaic=False, multiscale=True, normalized_labels=False):
        super().__init__(img_size, sample_size, augment,
                         mosaic, multiscale, normalized_labels)
        # 测试集/训练集所有的图片
        self.img_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        # 所有的标签
        self.label_files = [path.replace(".png", ".txt")
                            for path in self.img_files]
        self.category = {}
        # 标签序号
        for i, name in enumerate(class_names):
            self.category[name.replace(" ", "-")] = i

    def load_files(self, label_path):
        # （label,x1,y1,x2,y2,x3,y3,x4,y4,theta,x,y,width,height）
        lines = open(label_path, 'r').readlines()

        x1, y1, x2, y2, x3, y3, x4, y4, label = [], [], [], [], [], [], [], [], []
        for line in lines:
            line = line.split('\t')
            x1.append(float(line[1]))
            y1.append(float(line[2]))
            x2.append(float(line[3]))
            y2.append(float(line[4]))
            x3.append(float(line[5]))
            y3.append(float(line[6]))
            x4.append(float(line[7]))
            y4.append(float(line[8]))
            label.append(self.category[line[0]])

        num_targets = len(label)
        if not num_targets:
            return None, None, None, None, None, None, 0

        x1 = torch.tensor(x1)
        y1 = torch.tensor(y1)
        x2 = torch.tensor(x2)
        y2 = torch.tensor(y2)
        x3 = torch.tensor(x3)
        y3 = torch.tensor(y3)
        x4 = torch.tensor(x4)
        y4 = torch.tensor(y4)
        label = torch.tensor(label)

        x = ((x1 + x3) / 2 + (x2 + x4) / 2) / 2
        y = ((y1 + y3) / 2 + (y2 + y4) / 2) / 2
        w = torch.sqrt(torch.pow((x1 - x2), 2) + torch.pow((y1 - y2), 2))
        h = torch.sqrt(torch.pow((x2 - x3), 2) + torch.pow((y2 - y3), 2))

        theta = ((y2 - y1) / (x2 - x1 + 1e-16) +
                 (y3 - y4) / (x3 - x4 + 1e-16)) / 2
        theta = torch.atan(theta)
        theta = torch.stack(
            [t if t != -(np.pi / 2) else t + np.pi for t in theta])

        return x, y, w, h, theta, label, num_targets


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

# def load_data(data_dir, dataset, action, img_size=416, sample_size=600, batch_size=4, shuffle=True, augment=False, mosaic=False, multiscale=False):
#     # 加载类别名
#     class_names = load_class_names(os.path.join(data_dir, "class.names"))
#     # 训练集/测试集目录
#     data_dir = os.path.join(data_dir, action)

#     if dataset == "UCAS_AOD":
#         dataset = UCASAODDataset(data_dir, class_names, img_size=img_size,
#                                  sample_size=sample_size, augment=augment, mosaic=mosaic, multiscale=multiscale)

#     elif dataset == "custom":
#         dataset = CustomDataset(data_dir, img_size=img_size, augment=augment,
#                                 sample_size=sample_size, mosaic=mosaic, multiscale=multiscale)

#     else:
#         raise NotImplementedError


#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                                              pin_memory=True, collate_fn=dataset.collate_fn)

#     return dataset, dataloader


# if __name__ == "__main__":
#     train_dataset, train_dataloader = split_data("data/test", 416)
#     for i, (img_path, imgs, targets) in enumerate(train_dataloader):
#         img = imgs.squeeze(0).numpy().transpose(1, 2, 0)
#         img = img.copy()
#         print(img_path)

#         # targets[img_idx, label, x, y, w, h, a],注意是相对坐标
#         for p in targets:
#             x, y, w, h, theta = p[2] * img.shape[1], p[3] * img.shape[1], p[4] * img.shape[1], p[5] * img.shape[1], p[6]

#             X1, Y1, X2, Y2, X3, Y3, X4, Y4 = xywha2xyxyxyxy(np.array([x, y, w, h, theta]))
#             X1, Y1, X2, Y2, X3, Y3, X4, Y4 = int(X1), int(Y1), int(X2), int(Y2), int(X3), int(Y3), int(X4), int(Y4)

#             cv.line(img, (X1, Y1), (X2, Y2), (255, 0, 0), 1)
#             cv.line(img, (X2, Y2), (X3, Y3), (255, 0, 0), 1)
#             cv.line(img, (X3, Y3), (X4, Y4), (255, 0, 0), 1)
#             cv.line(img, (X4, Y4), (X1, Y1), (255, 0, 0), 1)

#         cv.imshow('My Image', img)
#         # 0-1转为0-255
#         img[:, 1:] = img[:, 1:] * 255.0
#         if img_path[0].split('/')[-2] == str(1):
#             path = "data/augmentation/plane_" + img_path[0].split('/')[-1]
#         else:
#             path = "data/augmentation/car_" + img_path[0].split('/')[-1]
#         cv.imwrite(path, img)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
