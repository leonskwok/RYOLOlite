#-*- coding: UTF-8 -*-
from PIL import Image, ImageChops
import random
import os
import numpy as np
import time


def Get_JpgTxtfiles(file_dir, type):
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

if __name__ == '__main__':
    for type in range(2):
        root = os.getcwd()+'/data/picProcess'
        # 原始图片和标签地址
        PreImgPath = os.path.join(root, 'before_img/'+str(type)+'/')
        PreTxtPath = os.path.join(root, 'before_label/'+str(type)+'/')

        # 偏移后图片和标签路径
        # Aft_ImgPath = os.path.join(root, 'after_img/'+str(type)+'/')
        # Aft_TxtPath = os.path.join(root, 'after_label/'+str(type)+'/')
        # Aft_ImgPath = "/home/guoliangliang/Documents/R-YOLOv4-main/data/train/"+str(type)+'/'
        # Aft_TxtPath = "/home/guoliangliang/Documents/R-YOLOv4-main/data/train/"+str(type)+'/'
        Aft_ImgPath = "/home/guoliangliang/Documents/R-YOLOv4-main/data/detect/"
            
        Aft_TxtPath = "/home/guoliangliang/Documents/R-YOLOv4-main/data/detect/"



        # 获得所有图片
        ImgFiles = Get_JpgTxtfiles(PreImgPath, '.jpg')
        # TxtFiles = Get_JpgTxtfiles(PreTxtPath, '.txt')

        # 指定倍数
        time = 1
        for i in range(len(ImgFiles)):
            #读取指定img和txt
            img = Image.open(PreImgPath + ImgFiles[i])
            # x1,y1,x2,y2,x3,y3,x4,y4,theta,cx,cy,w,h
            txt = np.loadtxt(PreTxtPath + os.path.splitext(ImgFiles[i])[0]+'.txt')

            w = img.width
            h = img.height

            # 获得最小/最大偏移量
            xoff_min = int(min(txt[0], txt[2], txt[4], txt[6]))
            xoff_max = int(w - max(txt[0], txt[2], txt[4], txt[6]))
            yoff_min = int(min(txt[1], txt[3], txt[5], txt[7]))
            yoff_max = int(h - max(txt[1], txt[3], txt[5], txt[7]))

            for k in range(time):
                # 获得随机偏移量
                rand_x = random.randint(-xoff_min, xoff_max)
                rand_y = random.randint(-yoff_min, yoff_max)

                # 图片偏移
                im_off = ImageChops.offset(img, rand_x, rand_y)

                # 标签偏移
                x1 = txt[0] + rand_x
                y1 = txt[1] + rand_y
                x2 = txt[2] + rand_x
                y2 = txt[3] + rand_y
                x3 = txt[4] + rand_x
                y3 = txt[5] + rand_y   
                x4 = txt[6] + rand_x
                y4 = txt[7] + rand_y
                cx = txt[9] + rand_x
                cy = txt[10] + rand_y

                xmin = int(min(x1, x2, x3, x4))
                ymin = int(min(y1, y2, y3, y4))
                xmax = int(max(x1, x2, x3, x4))
                ymax = int(max(y1, y2, y3, y4))
                           
                img_neme = (os.path.splitext(ImgFiles[i])[0]+'_'+str(k)).zfill(10) + '.jpg'
                txt_name = (os.path.splitext(ImgFiles[i])[0]+'_'+str(k)).zfill(10)+  '.txt'

                if(xmin<0) or (ymin<0) or (xmax>w) or (ymax>h):
                    print(txt_name)

                im_off.save(Aft_ImgPath + img_neme)
                # with open(Aft_TxtPath + txt_name, 'w') as wstream:                                 
                #     wstream.write(str(x1)+' '+ str(y1)+ ' '+str(x2)+ ' '+str(y2)+ ' '+str(x3)+ ' '+str(y3)+ ' '+\
                #         str(x4)+' '+str(y4)+ ' '+ str(txt[8])+' '+str(cx)+' '+ str(cy)+' '+str(txt[11])+' '+str(txt[12]))

