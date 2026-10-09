ARG BASE_IMAGE=ros:noetic-ros-base-focal@sha256:72b8bc59035dc0a5b8e07aae28c16caa84192971d72d207c72ed734fb1d5e97d
FROM ${BASE_IMAGE} AS dependencies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y -o Acquire::Retries=3 --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates patch pkg-config \
    libeigen3-dev libpcl-dev libopencv-dev libyaml-cpp-dev libdw-dev \
    libgrpc++-dev protobuf-compiler-grpc protobuf-compiler libprotobuf-dev \
    ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-tf2-ros \
    && rm -rf /var/lib/apt/lists/*
FROM dependencies AS worker
ENV EGO_COMMIT=bfda51284c8c1b476043255a8145ef925a3778a5
ENV SUPER_COMMIT=2ad3419c127a617c6d7df6925e81a14175a9c096
RUN mkdir -p /opt/upstream && cd /opt/upstream \
    && git init ego-planner && cd ego-planner \
    && git remote add origin https://github.com/ZJU-FAST-Lab/ego-planner.git \
    && git config core.repositoryformatversion 1 && git config extensions.partialClone origin \
    && git config remote.origin.promisor true && git config remote.origin.partialclonefilter blob:none \
    && git sparse-checkout init --cone && git sparse-checkout set src/planner \
    && git fetch --filter=blob:none --depth 1 origin "$EGO_COMMIT" && git checkout --detach FETCH_HEAD \
    && cd .. && git init SUPER && cd SUPER \
    && git remote add origin https://github.com/hku-mars/SUPER.git \
    && git config core.repositoryformatversion 1 && git config extensions.partialClone origin \
    && git config remote.origin.promisor true && git config remote.origin.partialclonefilter blob:none \
    && git sparse-checkout init --cone && git sparse-checkout set super_planner rog_map \
    && git fetch --filter=blob:none --depth 1 origin "$SUPER_COMMIT" && git checkout --detach FETCH_HEAD
COPY patches /opt/ros_planner_worker/patches
RUN patch -d /opt/upstream/ego-planner -p1 < /opt/ros_planner_worker/patches/ego-sync.patch
RUN patch -d /opt/upstream/SUPER -p1 < /opt/ros_planner_worker/patches/super-state.patch
RUN patch -d /opt/upstream/ego-planner -p1 < /opt/ros_planner_worker/patches/ego-gcc9.patch
RUN patch -d /opt/upstream/SUPER -p1 < /opt/ros_planner_worker/patches/super-reset.patch
RUN patch -d /opt/upstream/ego-planner -p1 < /opt/ros_planner_worker/patches/ego-release.patch
RUN patch -d /opt/upstream/SUPER -p1 < /opt/ros_planner_worker/patches/super-release.patch
COPY CMakeLists.txt /opt/ros_planner_worker/
COPY proto /opt/ros_planner_worker/proto
COPY src /opt/ros_planner_worker/src
ARG BUILD_JOBS=2
RUN source /opt/ros/noetic/setup.bash \
    && cmake -S /opt/ros_planner_worker -B /opt/ros_planner_worker/build -G Ninja -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    && cmake --build /opt/ros_planner_worker/build -j "$BUILD_JOBS"
COPY config /opt/ros_planner_worker/config
COPY scripts/entrypoint.sh /opt/ros_planner_worker/entrypoint.sh
EXPOSE 50051
ENTRYPOINT ["/opt/ros_planner_worker/entrypoint.sh"]
