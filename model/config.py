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

        # self.model = RTiny(outchs)
        # self.model = RTiny_Atte(outchs)
        # self.model = RTiny_Ghost_Atte(outchs)
        # self.model = RTiny_Ghost(outchs)
        # self.model = RTiny_Ghostplus(outchs)
        # self.model  =RTiny_SqueezeGhost(outchs)
        self.model = RTiny_SqueezeGhostPlus(outchs)
        # self.model = RTiny_Mobile(outchs)
        # self.model = RTiny_GhostBottle(outchs)
        # self.model = RTiny_GhostBottle_all(outchs)
        # self.model = RTiny_GhostBottle_all_G_FeaCASA(outchs)
        # self.model = RTiny_GhostBottle_all_G(outchs)
        # self.model = RTiny_GhostBottle_all_G2(outchs)

        # self.model = RTiny_GhostBottle_all_G3(outchs)
        # self.model = RTiny_GhostBottle_G2_trip(outchs)
        # self.model = RTiny_GhostALLBottle(outchs)
        # self.model = RTiny_GhostALLBottle2(outchs)
        # self.model = RTiny_GhostBottle_all_FeaCASA(outchs)
        # self.model = RTiny_GhostBottle_FeaCASA(outchs)
        # self.model = RTiny_GhostBottleFeaTrip(outchs)
        # self.model = RTiny_GhostBottleResAtte(outchs)
        # self.model=  RTiny_GhostBottleAtte(outchs)
        # self.model = RTiny_GhostBottle23(outchs)
        # self.model = RTiny_DCS(outchs)
        # self.model = RTiny_TripeAtte(outchs)
        # self.model = RTiny_GhostBottleTrip(outchs)

        # self.model = RTiny_SE(outchs)
        # self.model = RTiny_CBMA(outchs)
        # self.model = RTiny_ECA(outchs)
        # self.model = RYOLOv4(outchs)
        # self.model = RGhost(outchs)

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

        self.train_folder = 'data/train_new'
        self.test_folder = 'data/test_new'
        self.val_folder = 'data/val_new'
        self.detect_folder='data/detect_new'
        self.hardsample_folder = 'data/hardsample'
        self.output_folder='outputs'
        self.hardsample_path = 'others/hardsample.txt'

        self.hardsampledetect=True


        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%m-%d_%H-%M')

        self.log_folder = 'logs/'+self.model._get_name()+time_str
        self.weights_folder = 'weights/'+self.model._get_name()+time_str
        self.class_path = 'data/coco.names'
        self.trainedmodel=''
        self.weights_path = 'weights/RTiny_SqueezeGhostPlus01-21_10-23/ep060-loss1.247-val_loss0.819.pth'
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
        self.yolo_layer=YoloLossLayer(self.ncls,self.anchor,self.angles,self.anchor_mask,self.img_size,self.iou_thres,self.angle_thres)
