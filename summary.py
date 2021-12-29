import torch
from torchsummary import summary
from torchstat import stat
from model.config import config

cfg=config()

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = cfg.model
    # stat(m.to(device), input_size=(3, 416, 416))
    stat(m, (3, 416, 416))

