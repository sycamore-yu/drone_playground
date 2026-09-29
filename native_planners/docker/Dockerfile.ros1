FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-yaml \
    python3-pip \
    python3-scipy \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libdw-dev \
    libeigen3-dev \
    libelf-dev \
    libopencv-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    ros-noetic-cmake-modules \
    ros-noetic-cv-bridge \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-image-transport \
    ros-noetic-laser-geometry \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    ros-noetic-std-srvs \
    ros-noetic-tf \
    ros-noetic-tf2-eigen \
    ros-noetic-tf2-ros \
    ros-noetic-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir grpcio==1.62.3 protobuf==5.29.5

RUN mkdir -p /opt/drone_playground/bridge /opt/drone_playground/planners

CMD ["sleep", "infinity"]
