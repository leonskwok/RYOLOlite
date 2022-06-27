import torch
import numpy as np
import torch.nn.functional as F
import cv2


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

def mixup(img, labels, img2, labels2):
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    img = (img * r + img2 * (1 - r)).astype(np.uint8)
    labels = torch.cat((labels, labels2), 0)
    return img, labels


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image


def hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * \
            [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
            sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=img)


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def gaussian_noise(images, mean, std):
    return images + (torch.randn(images.size()) * 0.2) * std + mean


def rotate(images, targets):
    # targets[img_idx, label, x, y, w, h, a], 注意是相对坐标
    # 这里是否应该换成(-90，90)
    # (0-1)*90
    degree = np.random.rand() * 90
    radian = np.pi / 180 * degree
    # 旋转矩阵
    R = torch.stack([
        torch.stack([torch.cos(torch.tensor(-radian)), -torch.sin(torch.tensor(-radian)), torch.tensor(0)]),
        torch.stack([torch.sin(torch.tensor(-radian)), torch.cos(torch.tensor(-radian)), torch.tensor(0)]),
        torch.stack([torch.tensor(0), torch.tensor(0), torch.tensor(1)])]).reshape(3, 3)
    # 移到图片中心点
    T1 = torch.stack([
        torch.stack([torch.tensor(1), torch.tensor(0), torch.tensor(-0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(1), torch.tensor(-0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(0), torch.tensor(1)])]).reshape(3, 3)
    # 移回图片左上角
    T2 = torch.stack([
        torch.stack([torch.tensor(1), torch.tensor(0), torch.tensor(0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(1), torch.tensor(0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(0), torch.tensor(1)])]).reshape(3, 3)

    images = images.unsqueeze(0)
    rot_mat = get_rot_mat(radian)[None, ...].repeat(images.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, images.size(), align_corners=True)
    images = F.grid_sample(images, grid, align_corners=True)
    images = images.squeeze(0)

    # x,y中心坐标
    # [x,y,1]
    points = torch.cat([targets[:, 2:4], torch.ones(len(targets), 1)], dim=1)
    points = points.T
    points = torch.matmul(T2, torch.matmul(R, torch.matmul(T1, points))).T
    targets[:, 2:4] = points[:, :2]

    targets = targets[targets[:, 2] < 1]
    targets = targets[targets[:, 2] > 0]
    targets = targets[targets[:, 3] < 1]
    targets = targets[targets[:, 3] > 0]
    assert (targets[:, 2:4] > 0).all() or (targets[:, 2:4] < 1).all()

    targets[:, 6] = targets[:, 6] - radian
    targets[:, 6][targets[:, 6] <= -np.pi / 2] = targets[:, 6][targets[:, 6] <= -np.pi / 2] + np.pi

    assert (-np.pi / 2 < targets[:, 6]).all() or (targets[:, 6] <= np.pi / 2).all()
    return images, targets


def vertical_flip(images, targets):
    images = torch.flip(images, [0, 1])
    targets[:, 3] = 1 - targets[:, 3]
    targets[:, 6] = - targets[:, 6]

    return images, targets


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    targets[:, 6] = - targets[:, 6]

    return images, targets
