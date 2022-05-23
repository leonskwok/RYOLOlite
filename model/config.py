import datetime
from model.model import *
from model.yololayer import YOLOLoss, YoloLossLayer

class config():
    def __init__(self):
        self.ncls = 4
        numanchors = 3
        angles = [-60, -30, 0, 30, 60, 90]
        # angles = [-90, -75,  -60, -45, -30, -15]
        # angles = [-90, -60, -30]
        self.angles = [t/180*np.pi for t in angles]
        outchs = (5 + 1 + self.ncls) * numanchors * len(self.angles)
        self.ifreg_const=False

        # self.model = RTiny(outchs)#标准卷积
        # self.model = RTiny_Ghostplus(outchs)#Ghost卷积
        # self.model = RTiny_Mobile(outchs)#深度可分离卷积
        self.model = RTiny_SqueezeGhostPlus(outchs)  #可分离幻影卷积

        # self.model = RYOLOv4(outchs)

        self.start_epoch = 0
        self.end_epoch = 60
        self.lr = 0.001
        self.img_size = 416

        self.trainbatchsize = 64
        self.valbatchsize=16
        self.testbatchsize=1

        self.conf_thres=0.7
        self.nms_thres=0.2
        self.iou_thres=[0.6, 0.4]
        self.angle_thres=np.pi/12

        datasetdir = 'data/dataset_new/'
        self.train_folder = datasetdir +'train'
        self.test_folder = datasetdir +'test'
        self.val_folder = datasetdir + 'val'
        self.detect_folder = datasetdir + 'detect'
        self.hardsample_folder = 'data/hardsample'
        self.output_folder='outputs'
        self.hardsample_path = 'data/hardsample.txt'

        self.hardsampledetect=False

        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%m-%d_%H-%M')

        if self.ifreg_const:
            self.weights_folder = 'weights/'+self.model._get_name()+time_str
        else:
            self.weights_folder = 'weights/' + self.model._get_name()+'(L1)'+ time_str

        self.log_folder = self.weights_folder
        self.class_path = 'data/coco.names'
        self.trainedmodel=''
        self.weights_path = 'results/5_RTiny_SqueezeGhostPlus01-20_10-05/ep060-loss1.251-val_loss0.942.pth'
        self.CUDA = True
        self.GPU = '1,2,3'

        if self.model.tiny:
            # self.anchor = [[10, 14], [23, 27], [37, 58],
            #             [81, 82], [135, 169], [344, 319]]
            # self.anchor_mask = [[3, 4, 5], [0, 1, 2]]
            if numanchors==3:
                self.anchor = [[14, 10], [27, 23], [58, 37],
                            [82, 81], [169, 135], [344, 319]]
                self.anchor_mask = [[0,1,2], [3,4,5]]
            elif numanchors==2:
                # self.anchor = [[76, 51], [152, 101],
                #                 [230, 153], [357, 238]]
                # self.anchor_mask = [[0, 1], [2, 3]]
                self.anchor = [[47, 37], [140, 112],
                                [233, 186], [326, 260]]
                self.anchor_mask = [[0, 1], [2, 3]]

        else:
            self.anchor = [[10, 13], [16, 30], [33, 23], [30, 61], [
                62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
            self.anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # self.yolo_loss = YOLOLoss(
        #     self.anchor, self.angles, self.ncls, [self.img_size, self.img_size], self.CUDA, self.anchor_mask)
        self.yolo_layer = YoloLossLayer(self.ncls, self.anchor, self.angles,
                                        self.anchor_mask, self.img_size,
                                        self.iou_thres, self.angle_thres,self.ifreg_const)
