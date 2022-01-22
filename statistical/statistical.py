import numpy as np
import os
import torch

from torch.nn.functional import fold

def datastatistical(data):
    data = torch.tensor(data)
    # rec = [[] for _ in range(4)]
    # for i in range(4):
    #     cls = data.clone()
    #     # 分类
    #     cls = cls[cls[:,11]==i]

    #     sum = 500 if len(cls)<500 else len(cls)     
    #     cls = cls[cls[:,12]==i]
    #     # 角度
    #     pa = cls[:, 4]
    #     ga = cls[:, 9]
    #     cls = cls[torch.abs(pa-ga)/np.pi*180 % 90 < 5]

    #     # 定位误差
    #     px = cls[:, 0]
    #     py = cls[:, 1]
    #     gx = cls[:, 5]
    #     gy = cls[:, 6]
    #     dx = px-gx
    #     dy = py-gy
    #     dxy = torch.sqrt(dx*dx+dy*dy)
    #     cls = cls[dxy<=1.5]
    
    #     ph = cls[:, 3]
    #     gh = cls[:, 8]
    #     dh = torch.abs(ph-gh)
    #     cls = cls[dh<=1.5]
    
    #     rec[i] = len(cls)/sum
    #     print('%.4f' % (len(cls)/sum))

    #     # pw = cls[:, 2]
    #     # gw = cls[:, 7]
    # rec = torch.Tensor(rec)
    # print('%.4f'%(rec.mean().item()))
    px = data[:, 0]
    py = data[:, 1]
    pw = data[:, 2]
    ph = data[:, 3]
    pa = data[:, 4]
    gx = data[:, 5]
    gy = data[:, 6]
    gw = data[:, 7]
    gh = data[:, 8]
    ga = data[:, 9]
    iou = data[:, 10]

    dx = torch.abs(px-gx)
    dy = torch.abs(py-gy)
    dxy = torch.sqrt(dx*dx+dy*dy)
    dw = torch.abs(pw-gw)
    dh = torch.abs(ph-gh)
    da = (torch.abs(pa-ga)/np.pi*180) % 90

    print('dw: % .3f\t % .3f \n' % (torch.mean(dw), torch.max(dw)))
    print('dh: % .3f\t % .3f \n' % (torch.mean(dh), torch.max(dh)))
    print('da: % .3f\t % .3f \n' % (torch.mean(da), torch.max(da)))
    print('iou: % .3f\t % .3f \n' %
                    (torch.mean(iou), torch.min(iou)))
    print('dxy: % .3f\t % .3f \n' %
                    (torch.mean(dxy), torch.max(dxy)))  
    print('\n\n\n')


if __name__ == "__main__":
    data = np.loadtxt('results/4_yolov4_ghostbottle/stat/AP50.txt')
    datastatistical(data)

