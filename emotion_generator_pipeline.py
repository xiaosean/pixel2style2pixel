import os
import time
import glob
from argparse import Namespace
import pdb
import random

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import cv2
import numpy as np
import torch
from torchvision.ops import roi_align
from configs import transforms_config
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

sys.path.append(".")
sys.path.append("..")

from DSFD import face_detection
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image, tensor2numpy
from options.test_options import TestOptions
from models.psp import pSp


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def crop_image(im, bboxes):
    faces = []
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        faces += [im[y0:y1, x0:x1]]
    return faces

def image_to_torch(image, device):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    else:
        assert image.dtype == np.float32
    image = np.rollaxis(image, 2)
    image = image[None, :, :, :]
    image = torch.from_numpy(image).to(device)
    return image

def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
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


if __name__ == "__main__":
    
    # Custom default setting 
    INPUT_VIDEO = "./ir_0312.mp4"
    # INPUT_VIDEO = "./ir_0312_last1min.mp4"
    # INPUT_VIDEO = "./ir_0312_1min.mp4"
    DEVICE = "cuda:0"
    SEED = 42
    VISUAL_STEP = 500
    SKIP_FRAME = 5
    # CUDNN SETTING
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True 
    # torch.backends.cudnn.deterministic = False
    

    # pspnet default setting
    test_opts = TestOptions().parse()
    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
 
    
    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 256
    opts = Namespace(**opts)
    # Data preprocess
    print('Loading dataset for {}'.format(opts.dataset_type))

    inference_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1)])
                # transforms.ToTensor()])
                # transforms.Normalize(mean=[0.5], std=[0.5])])
    # Create pspnet
    net = pSp(opts)
    net.eval()
    net.cuda()

    
    # Process video

    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"opencv get length = {length}, fps ={fps}")
    out_filename = INPUT_VIDEO[:-4]
    output_movie = cv2.VideoWriter(out_filename + "_out.mp4", fourcc, 30, (width, height))

    cnt = -1
    t = time.time()
    start_time = time.time()
    while(cap.isOpened()):
        cnt += 1
        ret1, im = cap.read()   
        
        if cnt % SKIP_FRAME > 0:
            output_movie.write(last_im)
            continue

        if im is None:
            print(f"im is None, cnt = {cnt}")
            break 
        
        with torch.no_grad():
            # Face detector
            dets = detector.detect(
                im[:, :, ::-1]
            )[:, :4]

            tensor_im = image_to_torch(im, DEVICE)
            dets = torch.tensor(dets, device=DEVICE).view(-1, 4)
            faces = roi_align(input=tensor_im, boxes=[dets],output_size=(256,256))
            del tensor_im
            
            total_n = len(faces)
            if total_n == 0:
                output_movie.write(im)
                continue
            # batch processing
            face_proc_cnt = 0
            # Normalization
            faces /= 255
            faces = inference_transform(faces)   
            n_batches = total_n//opts.test_batch_size
            if total_n / opts.test_batch_size > n_batches:
                n_batches += 1
            # Processing loop
            for i in range(n_batches):
                start_idx = i*opts.test_batch_size
                end_idx = (i+1)*opts.test_batch_size
                result_batch = run_on_batch(faces[start_idx:end_idx], net, opts)
                for j in range(opts.test_batch_size):
                    if face_proc_cnt >= total_n:
                        break
                    top_h, top_w, down_h, down_w = dets[face_proc_cnt]
                    top_h, top_w, down_h, down_w = int(top_h), int(top_w), int(down_h), int(down_w)
                    # Handle bounding box out of image
                    down_h = min(im.shape[1], down_h)
                    down_w = min(im.shape[0], down_w)
                    img_h, img_w = int(down_h)-int(top_h), int(down_w)-int(top_w)
                    result = F.interpolate(result_batch[j].unsqueeze(0), size=(img_w, img_h))  #The resize operation on tensor.
                    result.squeeze_(0)
                    result = np.array(tensor2im(result))[:,:,::-1]
                    face_proc_cnt += 1
                    im[top_w:down_w, top_h:down_h, :] = result
            # Write frame to video
            last_im = im.copy()
            output_movie.write(im)

        # Visualize the proegess
        if cnt % VISUAL_STEP == 0 and cnt > 0:
            cost_time = time.time() - start_time
            remaining_time = length / cnt * cost_time - cost_time
            print(f"Progress {cnt}|{length} --- Remaining {int(remaining_time/60)}m/{int(remaining_time%60)}s")
        # if cnt > 101:
            # break
    # Close videos  
    print(f"cnt = {cnt}")
    output_movie.release()
    cap.release()
    total_cost_time = time.time() - start_time
    print("total_cost_time =", total_cost_time, "each frame cost =", total_cost_time / cnt)

