# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends\
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    sudo \
    unzip \
    zip \
    libgomp1 \
    ca-certificates \
    libgl1-mesa-glx \
    gnupg \
    lsb-release \ 
    gedit net-tools iputils-ping \
    python3 \
    python3-pip \
    python3-wstool \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the NVIDIA package repositories
RUN curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - && \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update pip to >=20.3
RUN pip install -U pip>=20.3

# Create a symbolic link for python3 -> python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set up Git global configuration
RUN git config --global user.name "Aman Chulawala" && \
    git config --global user.email "aman.chulawala@gmail.com"

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Set the working directory
WORKDIR /home
# Clone the working repository
RUN git clone --recurse-submodules https://github.com/FarStryke21/RL_CoveragePlanning.git

# Install the Environments
WORKDIR /home/RL_CoveragePlanning/viewpointPlaygroundEnv
RUN pip install --ignore-installed -e .

# Install BaselineZoo
WORKDIR /home/RL_CoveragePlanning/rl-baselines3-zoo
RUN pip install --ignore-installed -e .

# Get Tensorboard
RUN pip install tensorboard

WORKDIR /home/dir

# Set the default command to run when starting the container
CMD ["/bin/bash"]
