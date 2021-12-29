import cv2
import numpy as np
import os
import shutil
import time


# def Get_JpgTxtfiles(file_dir, type):
#     Dists = []
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == type:
#                 fileNm = os.path.splitext(file)[0]
#                 dist = (os.path.join(root, fileNm), str(fileNm))
#                 Dists.append(dist)
#     return Dists


# dir = 'data/detect_new'
# outdir = 'opencvmethod/'
# shutil.rmtree(outdir)
# os.makedirs(outdir)

# imglist = Get_JpgTxtfiles(dir,'.jpg')
# paths = []
# rects = []
# time_infer=[]
# for i in range(len(imglist)):
#     temp = time.time()
#     path = imglist[i][0]+'.jpg'
#     img = cv2.imread(path)
#     # 灰度图像
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 二值图像
#     ret, img = cv2.threshold(img, 37, 255, cv2.THRESH_BINARY)
#     # 开运算
#     g = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, g)
#     # 找边界
#     contours, hierarchy = cv2.findContours(
#         img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour = []
#     for cont in contours:
#         contour.extend(cont)
#     # 最小外接矩形
#     min_rect = cv2.minAreaRect(np.array(contour))
#     temp2 = time.time()
#     paths.append(path)
#     rects.append(min_rect)
#     time_infer.append(temp2 - temp)

# for i, (img_path, box) in enumerate(zip(paths, rects)):
#     image=cv2.imread(img_path)
#     file = os.path.split(img_path)[1]

#     box = cv2.boxPoints(box)
#     box = np.int0(box)
#     cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
#     cv2.imwrite(outdir+file, image)

# print('mean_inference_time: ', round(np.mean(time_infer), 5))
# print('FPS: ', round(1/np.mean(time_infer), 5))


########################################################
dir = 'data/detect_new/'
outdir = 'opencvtest/'
shutil.rmtree(outdir)
os.makedirs(outdir)

# 00000004_7.jpg
image = cv2.imread('data/process/img/6.jpg')
cv2.imwrite(outdir+'1origin.jpg', image)

img = image.copy()

# 灰度图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(outdir+'2gray.jpg',img)

# 二值图像
ret, thresh = cv2.threshold(img, 37, 255, cv2.THRESH_BINARY)
cv2.imwrite(outdir+'3binary.jpg', thresh)

# 开运算
g = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, g)
cv2.imwrite(outdir+'4open.jpg', img_open)

# 找轮廓
contours, hierarchy = cv2.findContours(
    img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_cont = image.copy()
cv2.drawContours(img_cont, contours, -1, (0, 0, 255), 3)

contour = []
for cont in contours:
    contour.extend(cont)
    
cv2.imwrite(outdir+'5contour.jpg', img_cont)


min_rect = cv2.minAreaRect(np.array(contour))
box = cv2.boxPoints(min_rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
cv2.imwrite(outdir+'6rect.jpg', image)





# img_connect,labels,states,centroid = cv2.connectedComponentsWithStats(img_open,connectivity=8)
# rect = cv2.minAreaRect(img_connect)

# # 得到最小矩形的坐标
# box = cv2.boxPoints(rect)

# # 标准化坐标到整数
# box = np.int0(box)

# # 画出边界
# cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
# cv2.imwrite(outdir+'6minrect.jpg', image)





