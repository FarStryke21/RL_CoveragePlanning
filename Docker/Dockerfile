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
    swig \
    build-essential \
    zip \
    libgomp1 \
    ca-certificates \
    libgl1-mesa-glx \
    gnupg \
    lsb-release \ 
    gedit net-tools iputils-ping \
    python3 \
    python3-pip python3-dev \
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


# # Install the RL Zoo
# WORKDIR /home/packages
# RUN git clone https://github.com/FarStryke21/rl-baselines3-zoo.git
# WORKDIR /home/packages/rl-baselines3-zoo
# RUN pip install --ignore-installed -e .

# # Install Environment
# WORKDIR /home/packages
# RUN git clone https://github.com/FarStryke21/viewpointPlaygroundEnv.git
# WORKDIR /home/packages/viewpointPlaygroundEnv
# RUN pip install --ignore-installed -e .

# Install the requisite python packages
RUN pip install --no-cache-dir --ignore-installed \
    gym==0.26.2 \
    "stable-baselines3[extra_no_roms,tests,docs]>=2.3.0a1,<3.0" \
    pybullet_envs_gymnasium>=0.4.0 \
    cloudpickle>=2.2.1 \
    plotly \
    wandb \
    moviepy \
    gymnasium \
    numpy \
    open3d==0.18.0 \
    scipy \
    matplotlib
# Install box2d-py separately with verbose output. 
# It likes to cause problems so this just as a precaution!
RUN pip install --no-cache-dir box2d-py==2.3.8 -v

# Get Tensorboard
RUN pip install --upgrade tensorflow tensorboard

# Copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint script to be executed
ENTRYPOINT ["/entrypoint.sh"]

# Set Working Directory
WORKDIR /home/dir

# Set the default command to run when starting the container
CMD ["/bin/bash"]
