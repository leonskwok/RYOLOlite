
import torch
import os
from model.config import config
import torch.backends.cudnn as cudnn


cfg = config()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

if __name__ == "__main__":
    imgsize = cfg.img_size
    CUDA = cfg.CUDA
    weights_path = cfg.weights_path
    model = cfg.model

    device = torch.device('cuda' if CUDA else 'cpu')

    pretrained_dict = torch.load(
        weights_path, map_location=torch.device('cuda'))

    model.load_state_dict(pretrained_dict)
    model.eval()

    if CUDA:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    dummy_input = torch.randn((1, 3, imgsize, imgsize), device=device)
    input_name = ["input"]
    if model.tiny:
        output_name = ["output_26", "output_13"]
    else:
        output_name = ["output_52", "output_26", "output_13"]
    torch.onnx.export(model, dummy_input, model._get_name()+'.onnx',
                    input_names=input_name, output_names=output_name, opset_version=12)
