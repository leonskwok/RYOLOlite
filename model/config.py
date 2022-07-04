import datetime
from model.model import *
from model.yololayer import YoloLossLayer
from tools.plot import load_class_names

class config():
    def __init__(self):
        # datasetdir = 'data/dataset_5workpiece'
        datasetdir = 'data/dataset_UCAS_AOD'
        self.train_folder = os.path.join(datasetdir, "train")
        self.test_folder = os.path.join(datasetdir, "test")
        self.val_folder = os.path.join(datasetdir, "val")
        self.detect_folder = os.path.join(datasetdir, "detect")
        self.hardsample_folder = os.path.join(datasetdir, "hardsample")
        self.output_folder = 'outputs'

        self.hardsample_txt = 'data/hardsample.txt'
        self.hardsampledetect = False

        self.class_names = load_class_names(os.path.join(datasetdir, 'class.names'))
        self.ncls = len(self.class_names)

        angles = [-60, -30, 0, 30, 60, 90]
        self.angles = [t/180*np.pi for t in angles]
        outchs = (5 + 1 + self.ncls) * 3 * len(self.angles)

        # 常数因子
        # self.reg_type = "l1"
        # self.reg_type = "ciou_l1"
        self.reg_type = "const_factor"

        # self.model = RTiny(outchs)#标准卷积   
        # self.model = RTiny_D(outchs)#深度可分离卷积
        # self.model = RTiny_G(outchs)#Ghost卷积
        # self.model = RTiny_SG(outchs)  #可分离幻影卷积

        self.model = RYOLOv3_M(outchs)
        # self.model = RYOLOv3_G(outchs)

        # self.model = RYOLOv4(outchs)
        # self.model = RYOLOv3(outchs)

        self.lr = 0.001
        self.img_size = 608
     
        # self.start_epoch = 0
        # self.end_epoch = 60
        # self.trainbatchsize = 64
        # self.valbatchsize = 16
        # self.testbatchsize = 1
        
        self.start_epoch = 0
        self.end_epoch = 80
        self.trainbatchsize = 16
        self.valbatchsize = 1
        self.testbatchsize = 1

        self.conf_thres=0.7
        self.nms_thres=0.2
        self.iou_thres=[0.6, 0.4]
        self.angle_thres=np.pi/12

        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%m-%d_%H:%M')

        filtname = "{}_{}_{}".format(self.model._get_name(),self.reg_type,time_str)
        self.weights_folder = os.path.join('weights', filtname)

        self.log_folder = self.weights_folder

        self.trainedmodel=''
        self.weights_path = 'weights/RYOLOv3_M_l1_07-01_14:43/ep080-loss0.018-val_loss0.000.pth'
        self.CUDA = True
        self.GPU = '1,2,3'

        if self.model.tiny:
            self.anchor = [[14, 10], [27, 23], [58, 37],
                        [82, 81], [169, 135], [344, 319]] 
            self.anchor_mask = [[0, 1, 2], [3, 4, 5]]

        else:
            self.anchor = [[10, 13], [16, 30], [23, 33], 
                            [30, 61], [45, 62], [59, 119], 
                            [90, 116], [156, 198], [326, 373]]
            self.anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        self.yolo_layer = YoloLossLayer(self.ncls, self.anchor, self.angles,
                                        self.anchor_mask, self.img_size,
                                        self.iou_thres, self.angle_thres,self.reg_type)
