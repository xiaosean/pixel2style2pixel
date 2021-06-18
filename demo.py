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
    with st.sidebar:
        st.header("TUST EdgeAI Course")
        with st.form(key="grid_reset"):
            n_photos = st.slider("Number of generate photos:", 2, 16, 8)
            n_cols = st.number_input("Number of columns", 2, 8, 4)
            mixed_alpha = st.slider("Number of mixed style(0~1)", 0, 100, 0)
            latent_code_range = st.slider('Select a range of values', 1, 18, (1, 18))
            st.form_submit_button(label="Generate new images")



    st.title('Welcome to mvclab Sketch2Real')
    st.write(" ------ ")

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
        tensor_img = inference_transform(input_image).cuda().float().unsqueeze(0)
        result = run_on_batch(tensor_img, net, opts)
        result = tensor2im(result[0]).resize(IMAGE_DISPLAY_SIZE, Image.ANTIALIAS)
        right_column.image(result,  caption = "Generated image")
        # Create grid
        n_rows = 1 + n_photos // n_cols
        rows = [st.beta_container() for _ in range(n_rows)]
        cols_per_row = [r.beta_columns(n_cols) for r in rows]

        for image_index in range(n_photos):
            with rows[image_index // n_cols]:
                # Generate with alpha and latent code attributes.
                result = run_on_batch(tensor_img, net, opts, latent_code=latent_code_range, mixed_alpha=mixed_alpha/100)
                result = tensor2im(result[0]).resize(IMAGE_DISPLAY_SIZE, Image.ANTIALIAS)
                cols_per_row[image_index // n_cols][image_index % n_cols].image(result)



def run_on_batch(inputs, net, opts, latent_code=None, mixed_alpha=0):
    # latent_mask = [int(l) for l in opts.latent_mask.split(",")]
    latent_mask = [i for i in range(18)]
    if latent_code:
        latent_mask = [i for i in range(latent_code[0]-1, latent_code[1])]
    # print(f"selected latent mask = {latent_mask} alpha = {mixed_alpha}")
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
                    alpha=mixed_alpha,
                    resize=opts.resize_outputs)
        result_batch.append(res)
    result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
