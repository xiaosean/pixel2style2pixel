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
        input_image = Image.open(uploaded_file).resize(IMAGE_DISPLAY_SIZE, Image.ANTIALIAS)
        left_column.image(input_image, caption = "Your protrait image(edge only)")
    # Demo result image
    # scatter_img = Image.fromarray(scatter)
    # skeleton_img = Image.fromarray(skeleton)

    # right_column.image(scatter_img,  caption = "Predicted Keypoints")
    # st.image(skeleton_img, caption = 'FINAL: Predicted Pose')

    #  ----------------------
    # test_opts = TestOptions().parse()

    # if test_opts.resize_factors is not None:
    #     assert len(
    #         test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
    #     out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
    #                                     'downsampling_{}'.format(test_opts.resize_factors))
    #     out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
    #                                     'downsampling_{}'.format(test_opts.resize_factors))
    # else:
    #     out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    #     out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    # os.makedirs(out_path_results, exist_ok=True)
    # os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
#     ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
#     opts = ckpt['opts']
#     opts.update(vars(test_opts))
#     if 'learn_in_w' not in opts:
#         opts['learn_in_w'] = False
#     if 'output_size' not in opts:
#         opts['output_size'] = 1024
#     opts = Namespace(**opts)
#     # opts.learn_in_w = True
#     print(f"opts option = {opts}")
#     net = pSp(opts)
#     net.eval()
#     net.cuda()

#     print('Loading dataset for {}'.format(opts.dataset_type))
#     dataset_args = data_configs.DATASETS[opts.dataset_type]
#     transforms_dict = dataset_args['transforms'](opts).get_transforms()
#     dataset = InferenceDataset(root=opts.data_path,
#                                transform=transforms_dict['transform_inference'],
#                                opts=opts)
#     dataloader = DataLoader(dataset,
#                             batch_size=opts.test_batch_size,
#                             shuffle=False,
#                             num_workers=int(opts.test_workers),
#                             drop_last=False)

#     if opts.n_images is None:
#         opts.n_images = len(dataset)

#     global_i = 0
#     global_time = []
#     for input_batch in tqdm(dataloader):
#         if global_i >= opts.n_images:
#             break
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


# def run_on_batch(inputs, net, opts):
#     if opts.latent_mask is None:
#         result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
#     else:
#         latent_mask = [int(l) for l in opts.latent_mask.split(",")]
#         result_batch = []
#         for image_idx, input_image in enumerate(inputs):
#             # For style mixing
#             # get latent vector to inject into our input image
#             vec_to_inject = np.random.randn(1, 512).astype('float32')
#             # Get W+
#             _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
#                                       input_code=True,
#                                       return_latents=True)
#             # get output image with injected style vector
#             res = net(input_image.unsqueeze(0).to("cuda").float(),
#                       latent_mask=latent_mask,
#                       inject_latent=latent_to_inject,
#                       alpha=opts.mix_alpha,
#                       resize=opts.resize_outputs)
#             result_batch.append(res)
#         result_batch = torch.cat(result_batch, dim=0)
#     return result_batch


if __name__ == '__main__':
    run()
