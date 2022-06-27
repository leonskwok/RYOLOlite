#-*- coding: UTF-8 -*-
from PIL import Image, ImageChops
import random
import os
import numpy as np
import math
import cv2
from numpy.lib.npyio import loadtxt

import shutil
import xml.etree.cElementTree as ET


def Get_files(file_dir, type):
    Dists = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == type:
                # fileNm = os.path.splitext(file)[0]
                # dist = (os.path.join(root, fileNm), str(fileNm))
                # Dists.append(dist)
                Dists.append(file)
    # Dists.sort(key=lambda x: x[1])
    return Dists


def rotate(angle, x, y):
    """
    基于原点的弧度旋转
    :param angle:   弧度
    :param x:       x
    :param y:       y
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey


def xy_rorate(theta, x, y, centerx, centery):
    """
    针对中心点进行旋转
    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx+r_x, centery+r_y

def rec_rotate(x, y, width, height, theta):
    """
    传入矩形的x,y和宽度高度，弧度，转成QUAD格式
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = x + width / 2
    centery = y + height / 2

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x+width, y, centerx, centery)
    x3, y3 = xy_rorate(theta, x, y+height, centerx, centery)
    x4, y4 = xy_rorate(theta, x+width, y+height, centerx, centery)

    return x1, y1, x2, y2, x4, y4, x3, y3


def xml_to_list(xml_path, txt_path):
    #获得所有的XML文件列表
    dir_list = [dir for dir in os.listdir(
        xml_path) if dir.split('.')[1] == 'xml']
    for xml in dir_list:
        # 打开xml文档
        tree = ET.parse(os.path.join(xml_path, xml.strip()))
        # 获得root节点
        root = tree.getroot()
        filename = root.find('filename').text
        # 写文件
        file_object = open(os.path.join(
            txt_path, filename + ".txt"), 'w')
        # file_object_log = open(filename + ".log", 'w') #写文件
        flag = False
        for size in root.findall('size'):  # 找到root节点下的size节点
            width = size.find('width').text  # 子节点下节点width的值
            height = size.find('height').text  # 子节点下节点height的值

        # 找到root节点下的所有object节点
        for object in root.findall('object'):
            name = object.find('name').text
            robndbox = object.find('robndbox')
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)

            x = cx - w/2
            y = cy - h/2
            # 保留6位小数
            if angle < 1.57:
                theta = round(angle, 6)  # 顺时针为正
            else:
                theta = round(angle - np.pi, 6)  # 逆时针为负
            x1, y1, x2, y2, x4, y4, x3, y3 = rec_rotate(x, y, w, h, theta)
            file_object.write(str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x4)+' '+str(y4)+' '+str(x3)+' '+str(y3)+' ' +
                              str(theta)+' ' + str(cx)+' ' + str(cy)+' ' + str(w)+' ' + str(h))
            file_object.write('\n')
        file_object.close()


def draw(image_data, line):
    line = line.strip()
    points_list = line.split(' ')[:-5]
    if points_list:
        points = [int(float(point)) for point in points_list]
        points = np.reshape(points, (-1, 2))
        location = np.array([list(p) for p in points])
        cv2.drawContours(
            image_data, [location], -1, (0, 255, 0), 2)


def visualize(image_root, gt_root, output_root):
    def read_gt_file(image_name):
        gt_file = os.path.join(gt_root, '%s.txt' %
                               (image_name.split('.jpg')[0]))
        with open(gt_file, 'r') as gt_f:
            return gt_f.readlines()

    def read_image_file(image_name):
        img_file = image_root+image_name
        img_array = cv2.imread(img_file)
        return img_array

    for image_name in os.listdir(image_root):
        if(os.path.splitext(image_name)[1] != '.jpg'):
            continue
        image_data = read_image_file(image_name)
        gt_image_data = image_data.copy()

        gt_list = read_gt_file(image_name)
        for gt in gt_list:
            draw(gt_image_data, gt)
            cv2.imwrite(output_root + image_name, gt_image_data)


root = '/home/guoliangliang/Documents/R-YOLOv4-main/data/'
processdir=root+'process_workpiece/'
img_dir = processdir + 'img/'
xml_dir = processdir + 'xml/'
txt_dir = processdir + 'txt/'
out_dir = processdir + 'out/'
vis_dir = processdir + 'vis/'

datasetdir = root + 'dataset_workpiece/'
detect_dir = datasetdir + "detect/"
train_dir = datasetdir + "train/"
test_dir = datasetdir + "test/"
val_dir = datasetdir + "val/"


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out


def pad_to_square(img, pad_value):
    h, w, c = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding，pad的四个数分别代表左右上下的填充维度
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    if h<=w:
        img = cv2.copyMakeBorder(
            img, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        img = cv2.copyMakeBorder(
            img, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=pad_value)


    return img, pad

def processimgtest(imgpath,txtpath,savedir):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

    cv2.imwrite(savedir+'origin.jpg', img)

    txt = loadtxt(txtpath)
    mB = np.mean(img[:, :, 0])
    mG=np.mean(img[:,:,1])
    mR = np.mean(img[:, :, 2])

    imgh, imgw, imgc = img.shape
    [x1, y1, x2, y2, x3, y3, x4, y4, theta, cx, cy, w, h] = txt

    cx = (x1+x3)/2
    cy = (y1+y3)/2

    ratio = random.randint(5, 12)/10
    ratio = 1.2
    affineM = np.array(
        [[ratio, 0, cx*(1-ratio)], [0, ratio, cy*(1-ratio)]], np.float32)
    kimg = cv2.warpAffine(img, affineM, (imgw, imgh),
                          borderValue=[mB, mG, mR])
    cv2.imwrite(savedir+'size.jpg', kimg)
    x1 = int(cx+ratio*(x1-cx))
    x2 = int(cx+ratio*(x2-cx))
    x3 = int(cx+ratio*(x3-cx))
    x4 = int(cx+ratio*(x4-cx))
    y1 = int(cy+ratio*(y1-cy))
    y2 = int(cy+ratio*(y2-cy))
    y3 = int(cy+ratio*(y3-cy))
    y4 = int(cy+ratio*(y4-cy))
    points =np.reshape([x1,y1,x2,y2,x3,y3,x4,y4],(-1,2))
    location = np.array([list(p) for p in points])
    tempimg = kimg.copy()
    cv2.drawContours(tempimg,[location],-1,(0,255,0),2)
    cv2.imwrite(savedir+'label.jpg', tempimg)


    # 获得弧度偏移 顺时针方向为正
    rand_theta = np.pi/180*random.randint(-90, 90)-theta
    rand_theta = np.pi/180*30-theta

    # 坐标旋转
    x1, y1 = xy_rorate(rand_theta, x1, y1, cx, cy)
    x2, y2 = xy_rorate(rand_theta, x2, y2, cx, cy)
    x3, y3 = xy_rorate(rand_theta, x3, y3, cx, cy)
    x4, y4 = xy_rorate(rand_theta, x4, y4, cx, cy)

    M = cv2.getRotationMatrix2D(
        (cx, cy), -rand_theta/np.pi*180, 1)

    # 图片旋转
    kimg = cv2.warpAffine(kimg, M, (imgw, imgh), borderMode=cv2.BORDER_CONSTANT, borderValue=[mB, mG, mR])

    cv2.imwrite(savedir+'rotate.jpg', kimg)

    # 获得最小/最大偏移量
    xoff_min = int(min(x1, x2, x3, x4))
    xoff_max = int(imgw - max(x1, x2, x3, x4))
    yoff_min = int(min(y1, y2, y3, y4))
    yoff_max = int(imgh - max(y1, y2, y3, y4))

    # 获得随机偏移量
    rand_x = random.randint(-xoff_min, xoff_max)
    rand_y = random.randint(-yoff_min, yoff_max)
    rand_x = 0.75*xoff_max
    # 图片偏移
    M = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
    kimg = cv2.warpAffine(kimg, M, (imgw, imgh),borderMode=cv2.BORDER_WRAP)

    cv2.imwrite(savedir+'move.jpg', kimg)

    kimg = gasuss_noise(kimg, 0, 0.003)
    cv2.imwrite(savedir + 'noise.jpg', kimg)

    kimg,pad = pad_to_square(kimg,0)
    cv2.imwrite(savedir + 'pad.jpg', kimg)

    cv2.resize(kimg,(416,416),kimg,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(savedir+'resize.jpg', kimg)


def creat(img_dir, txt_dir, targetpath, times, islabel, isgaosi):
    # 获得所有图片
    ImgFiles = Get_files(img_dir, '.jpg')
    # TxtFiles = Get_JpgTxtfiles(PreTxtPath, '.txt')

    for i in range(len(ImgFiles)):
        filename = os.path.splitext(ImgFiles[i])[0]
        type = filename.split('-')[0]
        #读取指定img和txt
        txt = loadtxt(txt_dir + os.path.splitext(ImgFiles[i])[0]+'.txt')

        img = cv2.imread(img_dir + ImgFiles[i], cv2.IMREAD_COLOR)
        imgh, imgw, imgc = img.shape
        mB=np.mean(img[:,:,0])
        mG=np.mean(img[:,:,1])
        mR = np.mean(img[:, :, 2])

        # 对每个数据标签偏移
        for k in range(times):
            kimg=img
            if(k % 100 == 0):
                print(str(i)+'\t'+str(k))

            [x1, y1, x2, y2, x3, y3, x4, y4, theta, cx, cy, w, h] = txt

            cx = (x1+x3)/2
            cy = (y1+y3)/2

            # 随机缩放
            ratio = random.randint(80,120)/100
            affineM = np.array([[ratio, 0, cx*(1-ratio)], [0, ratio, cy*(1-ratio)]], np.float32)
            kimg = cv2.warpAffine(kimg, affineM, (imgw, imgh), borderValue=[mB, mG, mR])
            x1 = int(cx+ratio*(x1-cx))
            x2 = int(cx+ratio*(x2-cx))
            x3 = int(cx+ratio*(x3-cx))
            x4 = int(cx+ratio*(x4-cx))
            y1 = int(cy+ratio*(y1-cy))
            y2 = int(cy+ratio*(y2-cy))
            y3 = int(cy+ratio*(y3-cy))
            y4 = int(cy+ratio*(y4-cy))

            # 获得弧度偏移 顺时针方向为正
            rand_theta = np.pi/180*random.randint(-90, 90)-theta
            # 坐标旋转
            x1, y1 = xy_rorate(rand_theta, x1, y1, cx, cy)
            x2, y2 = xy_rorate(rand_theta, x2, y2, cx, cy)
            x3, y3 = xy_rorate(rand_theta, x3, y3, cx, cy)
            x4, y4 = xy_rorate(rand_theta, x4, y4, cx, cy)
            # 获得图像的旋转矩阵
            M = cv2.getRotationMatrix2D(
                (cx, cy), -rand_theta/np.pi*180, 1)

            # 图片旋转
            kimg = cv2.warpAffine(kimg, M, (imgw, imgh),borderMode=cv2.BORDER_CONSTANT,borderValue=[mB,mG,mR])
            # kimg = cv2.warpAffine(
            #     kimg, M, (imgw, imgh), borderMode=cv2.BORDER_WRAP)

            # 获得最小/最大偏移量
            xoff_min = int(min(x1, x2, x3, x4))
            xoff_max = int(imgw - max(x1, x2, x3, x4))
            yoff_min = int(min(y1, y2, y3, y4))
            yoff_max = int(imgh - max(y1, y2, y3, y4))

            # 获得随机偏移量
            rand_x = random.randint(-xoff_min, xoff_max)
            rand_y = random.randint(-yoff_min, yoff_max)

            # 图片偏移
            M = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            kimg = cv2.warpAffine(kimg, M, (imgw, imgh),borderMode=cv2.BORDER_WRAP)


            # rimg = Image.fromarray(np.uint8(rimg))
            # rimg = ImageChops.offset(rimg, rand_x, rand_y)

            # 标签偏移
            x1 = x1 + rand_x
            y1 = y1 + rand_y
            x2 = x2 + rand_x
            y2 = y2 + rand_y
            x3 = x3 + rand_x
            y3 = y3 + rand_y
            x4 = x4 + rand_x
            y4 = y4 + rand_y
            cx = cx + rand_x
            cy = cy + rand_y

            # 椒盐噪声
            # img = sp_noise(img, 0.01)
            # 高斯噪声
            if isgaosi:
                kimg = gasuss_noise(kimg, 0, 0.001)

            img_neme = (os.path.splitext(ImgFiles[i])[
                        0]+'_'+str(k)).zfill(7) + '.jpg'
            txt_name = (os.path.splitext(ImgFiles[i])[
                        0]+'_'+str(k)).zfill(7) + '.txt'

            # test
            # xmin = int(min(x1, x2, x3, x4))
            # ymin = int(min(y1, y2, y3, y4))
            # xmax = int(max(x1, x2, x3, x4))
            # ymax = int(max(y1, y2, y3, y4))
            # if(xmin < 0) or (ymin < 0) or (xmax > imgw) or (ymax > imgh):
            #     print(txt_name)

            if islabel:
                clsdir = targetpath + type
                if not os.path.exists(clsdir):
                    os.makedirs(clsdir)
                cv2.imwrite(clsdir+'/' + img_neme,kimg)
                with open(clsdir+'/' + txt_name, 'w') as wstream:
                    wstream.write(str(x1)+' ' + str(y1) + ' '+str(x2) + ' '+str(y2) + ' '+str(x3) + ' '+str(y3) + ' ' +
                                    str(x4)+' '+str(y4) + ' ' + str(rand_theta)+' '+str(cx)+' ' + str(cy)+' '+str(w)+' '+str(h))

            else:
                cv2.imwrite(targetpath + img_neme,kimg)


if __name__ == '__main__':
    if os.path.exists(txt_dir):
        shutil.rmtree(txt_dir)
    os.makedirs(txt_dir)
    xml_to_list(xml_dir, txt_dir)

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    creat(img_dir, txt_dir, train_dir, 500,  islabel=True, isgaosi=True)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    creat(img_dir, txt_dir, test_dir, 50,  islabel=True, isgaosi=False)
    
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)
    creat(img_dir, txt_dir, val_dir, 20,  islabel=True, isgaosi=False)

    if os.path.exists(detect_dir):
        shutil.rmtree(detect_dir)
    os.makedirs(detect_dir)
    creat(img_dir, txt_dir, detect_dir, 10, islabel=False, isgaosi=False)

    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    visualize(val_dir + str(1)+'/', val_dir + str(1)+'/', vis_dir)
    processimgtest(processdir+'img/0-1.jpg',
                   processdir+'txt/0-1.txt',
                   processdir+'imgtest/')
