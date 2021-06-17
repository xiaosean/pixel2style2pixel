import os
from argparse import Namespace
import sys
import time
import random

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import streamlit as st
import yaml
from torchvision.transforms import ToPILImage
import hydra
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from utils.common import tensor2im
from attrdict import AttrDict

def model_init(opts, seed=42):
    print(f"Model init time = {time.time()}")
    # CUDNN SETTING
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True 

    # update test options with options used during training
    ckpt = torch.load(opts.checkpoint_path, map_location='cpu')
    net = pSp(opts)
    net.eval()
    net.cuda()

    return net



def run():
    # ===============================
    # Define the used variables
    # ===============================
    IMAGE_DISPLAY_SIZE = (512, 512)
    MODEL_INFERENCE_SIZE = (256, 256)
    IMAGE_DIR = 'demo_photo'
    TEAM_DIR = 'team'
    MODEL_CONFIG = "./configs/demo_site.yaml"


    # ==============
    # Set up model
    # ==============
    # Load the model args
    with open(MODEL_CONFIG, "r") as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
        opts = AttrDict(opts)


    net = model_init(opts)

    # Set up the transformer for input image
    inference_transform = transforms.Compose([
        transforms.Resize((256, 256)),    
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    # ===============================
    # Construct demo site by streamlit
    # ===============================
    st.title('Welcome to mvclab Sketch2Real')
    st.write(" ------ ")

    st.sidebar.warning('\
        Please upload SINGLE-FACE images. For best results, please also ALIGN the face in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("NTUST EdgeAI Course")

    # For demo columns
    
    # Load demo images
    demo_edge = Image.open('./demo_image/fakeface_edge.jpg')
    demo_pridiction = Image.open('./demo_image/fakeface_prediction.jpg')
    
    # Create demo columns on website
    demo_left_column, demo_right_column = st.beta_columns(2)
    demo_left_column.image(demo_edge,  caption = "Demo edge image")
    demo_right_column.image(demo_pridiction, caption = "Demo generate image")
    
    # Create a img upload button
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    left_column, right_column = st.beta_columns(2)
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        left_column.image(input_image.resize(IMAGE_DISPLAY_SIZE, Image.ANTIALIAS), caption = "Your protrait image(edge only)")

        # print(f"input_image.shape =, {input_image.size}, input_image type = {input_image.type}" )
        tensor_img = inference_transform(input_image).cuda().float().unsqueeze(0)
        print(f"tensor_img.shape =, {tensor_img.shape}, tensor_img type = {tensor_img.type}" )
        result = run_on_batch(tensor_img, net, opts)
        print(f"result.shape =, {result.shape}, result type = {result.type}" )
        result = tensor2im(result[0]).resize(IMAGE_DISPLAY_SIZE, Image.ANTIALIAS)
        right_column.image(result,  caption = "Generated image")

    # Demo result image
    # scatter_img = Image.fromarray(scatter)
    # skeleton_img = Image.fromarray(skeleton)

    # right_column.image(scatter_img,  caption = "Predicted Keypoints")
    # st.image(skeleton_img, caption = 'FINAL: Predicted Pose')
    



#         with torch.no_grad():
#             input_cuda = input_batch.cuda().float()
#             tic = time.time()
#             result_batch = run_on_batch(input_cuda, net, opts)
#             toc = time.time()
#             global_time.append(toc - tic)

#         for i in range(opts.test_batch_size):
#             result = tensor2im(result_batch[i])
#             im_path = dataset.paths[global_i]

#             if opts.couple_outputs or global_i % 100 == 0:
#                 input_im = log_input_image(input_batch[i], opts)
#                 resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
#                 if opts.resize_factors is not None:
#                     # for super resolution, save the original, down-sampled, and output
#                     source = Image.open(im_path)
#                     res = np.concatenate([np.array(source.resize(resize_amount)),
#                                           np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
#                                           np.array(result.resize(resize_amount))], axis=1)
#                 else:
#                     # otherwise, save the original and output
#                     res = np.concatenate([np.array(input_im.resize(resize_amount)),
#                                           np.array(result.resize(resize_amount))], axis=1)
#                 Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

#             im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
#             Image.fromarray(np.array(result)).save(im_save_path)

#             global_i += 1

#     stats_path = os.path.join(opts.exp_dir, 'stats.txt')
#     result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
#     print(result_str)

#     with open(stats_path, 'w') as f:
#         f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # For style mixing
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            # Get W+
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
