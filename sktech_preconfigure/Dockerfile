FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# Install some basic utilities
RUN apt-get update 
RUN apt-get install -y \
    git \
    zip \
    sudo \
    libx11-6 \
    build-essential \
    ca-certificates \
    wget \
    curl \
    tmux \
    htop \
    nano \
    vim 

# ####################################################################################
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Install Miniconda
WORKDIR /home/user
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/Miniconda3-py38_4.9.2-Linux-x86_64.sh\
 && ~/Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p ~/miniconda \
 && rm ~/Miniconda3-py38_4.9.2-Linux-x86_64.sh
ENV PATH=/home/user/miniconda/condabin/:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda init

# Create a Python 3.8 environment
RUN conda create -y --name py38 python=3.8 && conda clean -ya
RUN echo "conda activate py38" >> .bashrc
ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Opencv3
RUN sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libgl1-mesa-glx \
    build-essential
RUN conda install -y -c conda-forge opencv

# Basic Deep Leaning
RUN conda install pytorch=0.4.1 cuda90 -c pytorch
RUN pip install tensorboard

RUN pip install easydict
RUN pip install scikit-image
RUN pip install pyyaml
RUN pip install requests


# ####################################################################################
# EXPOSE 22
# RUN sudo apt-get install -y \
#     openssh-server

# RUN sudo rm -rf /var/lib/apt/lists/*
# RUN echo `echo PATH=$PATH`:\$PATH >> .bashrc
# RUN echo "conda activate py38" >> .bashrc
CMD bash