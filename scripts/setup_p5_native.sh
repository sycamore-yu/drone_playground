#!/usr/bin/env bash
# Build pinned native ROS1 planners in a private directory of an existing Noetic container.
set -euo pipefail
P5_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P5_CONTAINER="${P5_ROS_CONTAINER:-flightbench}"
P5_SUPER_SOURCE="${P5_SUPER_SOURCE:-$P5_PROJECT_ROOT/../../../reference_repos/SUPER}"
P5_EGO_SOURCE="${P5_EGO_SOURCE:-$P5_PROJECT_ROOT/tmp/p5-refsrc/ego-planner}"
P5_SUPER_REV=2ad3419c127a617c6d7df6925e81a14175a9c096
P5_EGO_REV=bfda51284c8c1b476043255a8145ef925a3778a5
mkdir -p "$P5_PROJECT_ROOT/tmp/p5-native"
for method in ego super; do
    if [[ "$method" == ego ]]; then
        source_path="$P5_EGO_SOURCE"; revision="$P5_EGO_REV"; target=ego-planner
    else
        source_path="$P5_SUPER_SOURCE"; revision="$P5_SUPER_REV"; target=SUPER
    fi
    git -C "$source_path" archive --output="$P5_PROJECT_ROOT/tmp/p5-native/$method.tar" "$revision"
    docker exec "$P5_CONTAINER" mkdir -p "/tmp/p5-native/$method/src/$target"
    docker cp "$P5_PROJECT_ROOT/tmp/p5-native/$method.tar" "$P5_CONTAINER:/tmp/p5-native/$method.tar"
    docker exec "$P5_CONTAINER" tar -xf "/tmp/p5-native/$method.tar" -C "/tmp/p5-native/$method/src/$target"
done
docker cp "$P5_PROJECT_ROOT/scripts/p5_ros_bridge.py" "$P5_CONTAINER:/tmp/p5-native/ros_bridge.py"
docker exec "$P5_CONTAINER" bash -c '
set -e
source /opt/ros/noetic/setup.bash
cd /tmp/p5-native/ego
catkin_make -j4 -DCATKIN_WHITELIST_PACKAGES="cmake_utils;quadrotor_msgs;traj_utils;plan_env;path_searching;bspline_opt;ego_planner" -DCMAKE_BUILD_TYPE=Release
' > "$P5_PROJECT_ROOT/tmp/p5-native/ego-build.log" 2>&1
docker exec "$P5_CONTAINER" bash -c '
set -e
source /tmp/p5-native/ego/devel/setup.bash
cd /tmp/p5-native
apt-get download libdw-dev libelf-dev
mkdir -p sysroot
for pkg in *.deb; do dpkg-deb -x "$pkg" sysroot; done
ln -sf /usr/lib/x86_64-linux-gnu/libdw.so.1 sysroot/usr/lib/x86_64-linux-gnu/libdw.so
ln -sf /usr/lib/x86_64-linux-gnu/libelf.so.1 sysroot/usr/lib/x86_64-linux-gnu/libelf.so
export CPLUS_INCLUDE_PATH=/tmp/p5-native/sysroot/usr/include
export LIBRARY_PATH=/tmp/p5-native/sysroot/usr/lib/x86_64-linux-gnu
cd super
bash src/SUPER/scripts/select_ros_version.sh ROS1
# Build the runtime target; optional upstream log-viewer/tuning GUIs are not needed.
catkin_make -j3 -DCATKIN_WHITELIST_PACKAGES="quadrotor_msgs;rog_map;super_planner" -DCMAKE_BUILD_TYPE=Release fsm_node
' > "$P5_PROJECT_ROOT/tmp/p5-native/super-build.log" 2>&1
docker inspect "$P5_CONTAINER" --format '{{.Image}}' > "$P5_PROJECT_ROOT/tmp/p5-native/container-image.txt"
printf '%s\n' "EGO $P5_EGO_REV" "SUPER $P5_SUPER_REV"

