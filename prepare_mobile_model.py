# Build a Mock Model in PyTorch with a convolution and a reduceMean layer
import os
import sys
from argparse import Namespace


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import torch.onnx as torch_onnx
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from models.psp import pSp

def build_model():
    # Load configs
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    # opts['device'] = 'cpu'

    opts = Namespace(**opts)

    # Init model
    model = pSp(opts)
    # model.eval()
    return model


if __name__ == '__main__':
    # model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model = build_model()
    model.eval()

    scripted_module = torch.jit.script(model)
    # Export full jit version model (not compatible mobile interpreter), leave it here for comparison
    scripted_module.save("deeplabv3_scripted.pt")
    # Export mobile interpreter version model (compatible with mobile interpreter)
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")