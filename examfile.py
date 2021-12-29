import torch
from torch.functional import Tensor
from PIL import Image
import numpy as np
import sys
import torch
import os
import shutil
from model.config import config
from torchstat import stat
from model.backbone import tinyResblock
from model.backbone import GtinyResblock



if __name__ == "__main__":

    m = GtinyResblock(64, 64)
    stat(m, (64, 416, 416))




