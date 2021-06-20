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
import torch.onnx as torch_onnx
# from torch2trt import torch2trt

sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from models.psp import pSp


def run():
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
    model.eval()
    model.cuda()

    with torch.no_grad():
        
        # Use this an input trace to serialize the model
        # input_shape = (1, 256, 256)
        # x = torch.randn(1, 1, 256, 256, requires_grad=True)
        x = torch.randn(1, 1, 256, 256, requires_grad=True).cuda()

        # TensorRT
        # Export the model to an ONNX file
        # orward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                    # inject_latent=None, return_latents=False, alpha=None):
        # dummy_input = (x, True, None, False, False, None, False, None)
        dummy_input = x
        model_onnx_path = "psp_sketch_model_default.onnx"

        # tensorRT
        # model_rt_path = "./psp_sketch_model_tensorRT.pth"

        # convert to TensorRT feeding sample data as input
        # model_trt = torch2trt(model, [x])
        # torch.save(model_trt.state_dict(), model_rt_path)
        # print("Export of torch_model.tensorRT complete!")
        output = torch_onnx.export(model, 
                                dummy_input, 
                                model_onnx_path, 
                                input_names=["x"],
                                output_names=["images"],
                                opset_version=13,          # the ONNX version to export the model to
                                export_params=True,
                                verbose=True)

    # output = torch_onnx.export(model, 
    #                         dummy_input, 
    #                         model_onnx_path, 
    #                         input_names=["x","resize","latent_mask","input_code","randomize_noise","inject_latent","return_latents","alpha"],
    #                         output_names=["images"],
    #                         verbose=True)
    print("Export of torch_model.onnx complete!")


if __name__ == '__main__':
    run()
