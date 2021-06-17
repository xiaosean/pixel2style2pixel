import os
from argparse import Namespace
import sys
import time

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import streamlit as st
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


# @st.cache 
def run():
    # ============== Variable setting
    IMAGE_DISPLAY_SIZE = (256, 256)
    IMAGE_DIR = 'demo_photo'
    TEAM_DIR = 'team'

    st.title('Welcome to mvclab Sketch2Real')
    st.write(" ------ ")

    st.sidebar.warning('\
        Please upload SINGLE-FACE images. For best results, please also ALIGN the face in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("NTUST EdgeAI Course")


if __name__ == '__main__':
    run()
