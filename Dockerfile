#############################
FROM ros:humble-ros-base AS base
ENV ROS_DISTRO=${ROS_DISTRO}
SHELL ["/bin/bash", "-c"]

# ROS
RUN apt-get update && apt-get install -y \
        build-essential \
        python3-colcon-common-extensions \
        python3-pip \
        python3-pybind11 \
        python3-pytest-cov \
        python3-rosdep \
        python3-rosinstall-generator \
        python3-vcstool \
        wget \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

# Install gz garden
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    gnupg \
 && curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
      --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
      http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list \
 && apt-get update \
 && apt-get install -y gz-garden \
 && rm -rf /var/lib/apt/lists/*

RUN echo "export GZ_VERSION=garden" >> ~/.bashrc
RUN source ~/.bashrc

# Additional dependencies for the VRX simulator
RUN apt-get update && apt install -y python3-sdformat13 ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-ros-gzgarden

# Installing python dependencies.
RUN pip install tensorflow==2.14.0
RUN pip install keras==2.14.0
RUN pip install matplotlib==3.5.1
RUN pip install scipy==1.8.0
RUN pip install torch==2.5.1
RUN pip install ultralytics==8.3.69
RUN pip install numpy==1.23.5
RUN pip install transforms3d

RUN apt update && apt install -y ros-humble-tf-transformations


# 
RUN apt-get update && apt-get install -y \
    libgl1 libglvnd0 libglx0 libxext6 libx11-6 mesa-utils \
 && rm -rf /var/lib/apt/lists/*

# # Create Colcon workspace with external dependencies
RUN mkdir -p /vrx_ws/src
WORKDIR /vrx_ws/src
COPY /src /vrx_ws/src

# Build the base Colcon workspace
WORKDIR /vrx_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
  && apt-get update -y \
  && colcon build --merge-install

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc
RUN echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
RUN echo "source /vrx_ws/install/setup.bash" >> ~/.bashrc

# # Set up the entrypoint
WORKDIR /vrx_ws
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]

CMD ["bash"]
