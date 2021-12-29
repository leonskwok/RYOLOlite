import numpy as np
import os
import torch

from torch.nn.functional import fold

def datastatistical(data):
    data = torch.tensor(data)
    rec = [[] for _ in range(4)]
    for i in range(4):
        cls = data.clone()
        # 分类
        cls = cls[cls[:,11]==i]

        sum = 500 if len(cls)<500 else len(cls)     
        cls = cls[cls[:,12]==i]
        # 角度
        pa = cls[:, 4]
        ga = cls[:, 9]
        cls = cls[torch.abs(pa-ga)/np.pi*180 % 90 < 5]

        # 定位误差
        px = cls[:, 0]
        py = cls[:, 1]
        gx = cls[:, 5]
        gy = cls[:, 6]
        dx = px-gx
        dy = py-gy
        dxy = torch.sqrt(dx*dx+dy*dy)
        cls = cls[dxy<=1.5]
    
        ph = cls[:, 3]
        gh = cls[:, 8]
        dh = torch.abs(ph-gh)
        cls = cls[dh<=1.5]
    
        rec[i] = len(cls)/sum
        print('%.4f' % (len(cls)/sum))

        # pw = cls[:, 2]
        # gw = cls[:, 7]
    rec = torch.Tensor(rec)
    print('%.4f'%(rec.mean().item()))


if __name__ == "__main__":
    data = np.loadtxt('results/1_yolov4_smoothl1/stat/AP95.txt')
    datastatistical(data)

