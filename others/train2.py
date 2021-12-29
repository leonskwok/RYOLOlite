import torch.optim as optim
import time
import random
import numpy as np
import torch
from torch.autograd import Variable
from tools.load import split_data
import torch.backends.cudnn as cudnn
from tools.logger import *
import os
# from model.utils import LossHistory,fit_one_epoch

from model.config import cfg


os.environ["CUDA_VISIBLE_DEVICES"] = cfg['nGPU']


def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cudnn.deterministic = True
    cudnn.benchmark = False


if __name__ == "__main__":
    train_folder = "data/train"
    weights_path = ""
    epochs = cfg['epochs']
    class_path = "data/coco.names"
    lr = 5e-4
    batch_size = 64
    img_size = 416
    Cuda = True
    Init_Epoch = 0

    init()

    device = cfg['device']
    ncls = cfg['ncls']
    model = cfg['model']
    filename = cfg['filename']
    print(filename)
    if cfg['ifLog']:
        logger = Logger("logs/"+filename)

    model.apply(weights_init_normal)  # 權重初始化

    # 多GPU训练
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    # model.to(device)

    train_dataset, train_dataloader = split_data(
        train_folder, img_size, batch_size, shuffle=True, augment=True, multiscale=True)

    num_iters_per_epoch = len(train_dataloader)
    scheduler_iters = round(epochs * len(train_dataloader) / batch_size)
    total_step = num_iters_per_epoch * epochs

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_iters, eta_min=1e-5)

    for epoch in range(epochs):
        total_loss = 0.0
        start_time = time.time()
        print("\n---- [Epoch %d/%d] ----\n" % (epoch + 1, epochs))
        model.train()

        for batch, (_, imgs, targets) in enumerate(train_dataloader):
            global_step = num_iters_per_epoch * epoch + batch + 1
            imgs = Variable(imgs.to(device), requires_grad=True)
            targets = Variable(targets.to(device), requires_grad=False)

            optimizer.zero_grad()
            outputs, loss = model(imgs, targets)

            loss.backward()
            total_loss += loss.item()

            # if global_step % args.subdivisions == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     scheduler.step()

            optimizer.step()

            scheduler.step()

            # ---------------------
            # -      logging      -
            # ---------------------
            tensorboard_log = []
            print("Step: %d/%d" % (global_step, total_step))

            for name, metric in model.metrics.items():
                tensorboard_log += [(f"{name}", metric)]

            if cfg['ifLog']:
                logger.list_of_scalars_summary(tensorboard_log, global_step)

        # reset epoch loss counters
        print("Total Loss: %f, Runtime %f" %
              (total_loss, time.time() - start_time))

    torch.save(model.state_dict(), "weights/"+filename+".pth")
