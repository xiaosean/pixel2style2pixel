# docker run -t -i --name style2pix --gpus all -v /home/xiaosean/:/data -p 8007:8007 -p 8008:8008 --shm-size 120G  pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime /bin/bash
docker run -t -i --name v19_pix2style --gpus all -v /home/xiaosean/:/data -p 8051:8051 --shm-size 120G pytorch_v19 /bin/bash
# docker run -t -i --name style2pix --gpus all -v /home/xiaosean/:/data -p 8007:8007 -p 8008:8008 --shm-size 120G pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.8 /bin/bash
# apt update
# wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# export PATH=$PATH:~/miniconda3/bin/
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# --- apt install libglib2.0-0
# apt install libgtk2.0-dev
# --- conda install -c conda-forge opencv 
# --- apt-get install build-essential
# apt install libgl1-mesa-glx
# conda env update -n base --file ./environment/psp_env.yaml

# glib problem
# sudo apt update
# sudo apt install wget gcc-8 unzip libssl1.0.0 software-properties-common
# sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# sudo apt update
# sudo apt-get install --only-upgrade libstdc++6
