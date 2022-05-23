import cv2
import numpy as np
import os
import shutil
import time

def Get_JpgTxtfiles(file_dir, type):
    Dists = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == type:
                fileNm = os.path.splitext(file)[0]
                dist = (os.path.join(root, fileNm), str(fileNm))
                Dists.append(dist)
    return Dists
def opencvtest():
    dir = 'data/detect_new'
    outdir = 'opencvmethod/opencvresult'
    shutil.rmtree(outdir)
    os.makedirs(outdir)

    imglist = Get_JpgTxtfiles(dir,'.jpg')
    paths = []
    rects = []
    time_infer=[]
    for i in range(len(imglist)):
        temp = time.time()
        path = imglist[i][0]+'.jpg'
        img = cv2.imread(path)
        # 灰度图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值图像
        ret, img = cv2.threshold(img, 37, 255, cv2.THRESH_BINARY)
        # 开运算
        g = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, g)
        # 找边界
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = []
        for cont in contours:
            contour.extend(cont)
        # 最小外接矩形
        min_rect = cv2.minAreaRect(np.array(contour))
        temp2 = time.time()
        paths.append(path)
        rects.append(min_rect)
        time_infer.append(temp2 - temp)

    for i, (img_path, box) in enumerate(zip(paths, rects)):
        image=cv2.imread(img_path)
        file = os.path.split(img_path)[1]

        box = cv2.boxPoints(box)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
        cv2.imwrite(outdir+file, image)

    print('mean_inference_time: ', round(np.mean(time_infer), 5))
    print('FPS: ', round(1/np.mean(time_infer), 5))

def opencvprocess():
    dir = 'data/detect_new/'
    outdir = 'opencvtest/opencvprocess'

    ptsfile = outdir + 'pts.txt'
    centerfile = outdir + 'center.txt'
    ctrlptsfile = outdir + 'ctrlpts.txt'

    # 以完成插补
    if False:
        pts = np.loadtxt(ptsfile, dtype=np.int32)
        center = np.loadtxt(centerfile, dtype=np.int32)
        ctrlpts = np.loadtxt(ctrlptsfile, dtype=np.int32)
        img = cv2.imread(outdir + 'figs/7rect.jpg')
        # cv2.drawContours(img, pts, -1, (0, 0, 255), 3)
        for pt in pts:
            cv2.circle(img, tuple(pt), 3, (255, 0, 0), 3)
        for pt in center:
            cv2.circle(img, tuple(pt), 15, (0, 0, 255), -1)

        h, w, c = img.shape
        hdata = ctrlpts[:, 1]
        wdata = ctrlpts[:, 0]
        padw = 0
        padh = 0
        if min(hdata) < 0 or max(hdata) > h:
            padh = max((0 - min(hdata)), (max(hdata) - h))
            padh += 20

        if min(wdata) < 0 or max(wdata) > w:
            padw = max((0 - min(wdata)), (max(wdata) - w))
            padw += 20
        img = cv2.copyMakeBorder(img,
                                padh,
                                padh,
                                padw,
                                padw,
                                cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))
        for pt in ctrlpts:
            pt[0] += padw
            pt[1] += padh
            cv2.circle(img, tuple(pt), 15, (255, 0, 0), -1)

        # for i in range(len(ctrlpts)-1):
        #     stapt=tuple(ctrlpts[i])
        #     endpt=tuple(ctrlpts[i+1])
        #     cv2.line(img,stapt,endpt,(0,255,0),5)

        cv2.imwrite(outdir + 'figs/traj.jpg', img)
        quit()

    shutil.rmtree(outdir + 'figs/')
    os.makedirs(outdir + 'figs/')

    # 1读取图像
    # img = cv2.imread('D:/USER_NAME/DESKTOP/6.jpg')
    img = cv2.imread('data/process/img/1.jpg')
    origon = img.copy()
    # img = cv2.imread('data/process/img/0.jpg')
    cv2.imwrite(outdir + 'figs/origin.jpg', img)

    img = cv2.blur(img, (5, 5))
    cv2.imwrite(outdir + 'figs/blur.jpg', img)

    # 2灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(outdir + 'figs/gray.jpg', img)

    # 3阈值处理
    # ret1, thresh = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
    # print(ret1)
    ret2, thresh = cv2.threshold(img, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img = thresh
    cv2.imwrite(outdir + 'figs/binary.jpg', img)

    # 腐蚀处理
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(outdir + 'figs/erode.jpg', img)

    # 闭运算
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, g)
    cv2.imwrite(outdir + 'figs/close.jpg', img)

    # # 开运算
    # g = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, g)
    # cv2.imwrite(outdir + 'figs/open.jpg', img)

    # # 膨胀处理
    # kernel = np.ones((15, 15), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # cv2.imwrite(outdir + 'figs/dilate.jpg', img)

    # # 开运算
    # g = cv2.getStructuringElement(cv2.MORPH_RECT, (100,100))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, g)
    # cv2.imwrite(outdir + 'figs/open.jpg', img)

    # canny = cv2.Canny(img_open, 200, 255, L2gradient=True)
    # cv2.imwrite(outdir + '2canny.jpg', canny)

    # 6轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    img_cont = origon.copy()
    cv2.drawContours(img_cont, contours, -1, (0, 0, 255), 3)
    cv2.imwrite(outdir + 'figs/6contour.jpg', img_cont)

    center = []
    center.append((0, 0))
    # 7最小外接矩形
    for cont in contours:
        min_rect = cv2.minAreaRect(np.array(cont))
        center.append(min_rect[0])
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)
        cv2.drawContours(origon, [box], 0, (0, 0, 255), 3)
        cv2.circle(origon, tuple(np.int0(min_rect[0])), 15, (0, 0, 255), -1)
    cv2.imwrite(outdir + 'figs/7rect.jpg', origon)

    h, w, c = origon.shape
    center.append((w, h))

    center = np.array(center)
    i = np.argsort(center[:, 0])
    center = center[i]

    centerfile = os.path.join(outdir, 'center.txt')
    if os.path.exists(centerfile):
        os.remove(centerfile)
    with open(centerfile, 'w+') as f:
        for i in range(len(center)):
            f.writelines('%f\t%f' % (center[i][0], center[i][1]))
            f.writelines('\n')