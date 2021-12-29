import torch.optim as optim
import time
import random
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torchvision
from tools.load import split_data
import torch.backends.cudnn as cudnn
import os

from model.utils import LossHistory,fit_one_epoch,weights_init
from model.config import config

cfg = config()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    lr = cfg.lr
    ncls=cfg.ncls
    start_epoch = cfg.start_epoch
    end_epoch = cfg.end_epoch

    batchsize = cfg.trainbatchsize
    valbatchsize=cfg.valbatchsize
    imgsize = cfg.img_size
    CUDA = cfg.CUDA

    val_folder=cfg.val_folder
    train_folder = cfg.train_folder
    weights_folder=cfg.weights_folder
    trainedmodel = cfg.trainedmodel
    class_path = cfg.class_path
    log_folder = cfg.log_folder

    model = cfg.model
    yolo_layer=cfg.yolo_layer

    init()
    weights_init(model)

    if trainedmodel != '':
        print('Load weights {}.'.format(trainedmodel))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(trainedmodel, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items(
        ) if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if CUDA:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory(log_folder)

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch-start_epoch, eta_min=1e-5)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    train_dataset, gen = split_data(
        train_folder,ncls, imgsize, batchsize, shuffle=True)
    val_dataset, gen_val = split_data(
        val_folder,ncls, imgsize, valbatchsize, shuffle=True)

    

    num_train=len(train_dataset)
    num_val=len(val_dataset)

    epoch_step = num_train // batchsize
    epoch_step_val = num_val // valbatchsize

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_layer, loss_history, optimizer, epoch,
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA,weights_folder)
        lr_scheduler.step()

