FROM osrf/ros:noetic-desktop-full AS base

SHELL ["/bin/bash", "-c"]

# Minimal setup
ENV ROS_DISTRO noetic
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
    locales \
    lsb-release \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN dpkg-reconfigure locales

# # Install ROS Noetic Desktop Full
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ros-noetic-desktop-full \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    build-essential \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep fix-permissions \
    && rosdep update --rosdistro $ROS_DISTRO

# source setup.bash on startup
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

FROM base AS dev

# Install general dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip3 install \ 
    pyserial \
    pymodbus===2.1.0 \
    numpy \
    scipy 

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-noetic-realsense2-camera ros-noetic-realsense2-description ros-noetic-rqt-controller-manager \
    libspnav-dev spacenavd libhidapi-dev\
    && pip3 install numpy-quaternion open3d pyspacemouse 

# Install ROS dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-noetic-moveit \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /root/catkin_ws

# Copy the 'src' directory from 'catkin_ws' to the container's workspace
COPY ./catkin_ws/src ./src

RUN source /opt/ros/noetic/setup.bash && \
    apt-get update && rosdep install -q -y \
    --from-paths ./src \
    --ignore-src \
    --rosdistro noetic \
    && rm -rf /var/lib/apt/lists/*

# Build the ROS workspace
RUN source /opt/ros/noetic/setup.bash && \
    catkin build

# Source the workspace setup files on container startup
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

# After building the ROS workspace, add the GUI control configuration for Gazebo
RUN mkdir -p ~/.gazebo && \
    echo "[spacenav]" >> ~/.gazebo/gui.ini && \
    echo "deadband_x = 0.1" >> ~/.gazebo/gui.ini && \
    echo "deadband_y = 0.1" >> ~/.gazebo/gui.ini && \
    echo "deadband_z = 0.1" >> ~/.gazebo/gui.ini && \
    echo "deadband_rx = 0.1" >> ~/.gazebo/gui.ini && \
    echo "deadband_ry = 0.1" >> ~/.gazebo/gui.ini && \
    echo "deadband_rz = 0.1" >> ~/.gazebo/gui.ini && \
    echo "topic=~/spacenav/remapped_joy_topic_to_something_not_used" >> ~/.gazebo/gui.ini
