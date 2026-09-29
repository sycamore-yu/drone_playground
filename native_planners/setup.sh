#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_ROOT="$ROOT/native_planners"
source "$NATIVE_ROOT/versions.env"

mkdir -p "$NATIVE_ROOT/sources" "$NATIVE_ROOT/.build" "$ROOT/tmp/native-planners"

ensure_source() {
    local name="$1" repository="$2" revision="$3"
    local target="$NATIVE_ROOT/sources/$name"
    if [[ ! -d "$target/.git" ]]; then
        git clone --filter=blob:none "$repository" "$target"
    fi
    if ! git -C "$target" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -C "$target" fetch --depth 1 origin "$revision"
    fi
    git -C "$target" checkout --detach "$revision"
    test "$(git -C "$target" rev-parse HEAD)" = "$revision"
}

ensure_source ego-planner "$EGO_REPOSITORY" "$EGO_REVISION"
ensure_source SUPER "$SUPER_REPOSITORY" "$SUPER_REVISION"

docker build \
    -f "$NATIVE_ROOT/docker/Dockerfile.ros1" \
    -t "$ROS_IMAGE" \
    "$NATIVE_ROOT/docker"

if docker container inspect "$ROS_CONTAINER" >/dev/null 2>&1; then
    docker rm -f "$ROS_CONTAINER" >/dev/null
fi
docker run -d --name "$ROS_CONTAINER" --init "$ROS_IMAGE" >/dev/null

docker exec "$ROS_CONTAINER" mkdir -p \
    "$RUNTIME_ROOT/bridge" \
    "$RUNTIME_ROOT/planners/ego/src/ego-planner" \
    "$RUNTIME_ROOT/planners/super/src/SUPER"

git -C "$NATIVE_ROOT/sources/ego-planner" archive \
    --output="$NATIVE_ROOT/.build/ego.tar" "$EGO_REVISION"
git -C "$NATIVE_ROOT/sources/SUPER" archive \
    --output="$NATIVE_ROOT/.build/super.tar" "$SUPER_REVISION"

docker cp "$NATIVE_ROOT/.build/ego.tar" "$ROS_CONTAINER:$RUNTIME_ROOT/ego.tar"
docker cp "$NATIVE_ROOT/.build/super.tar" "$ROS_CONTAINER:$RUNTIME_ROOT/super.tar"
docker cp "$NATIVE_ROOT/patches" "$ROS_CONTAINER:$RUNTIME_ROOT/patches"

docker exec "$ROS_CONTAINER" bash -lc "
set -eo pipefail
tar -xf '$RUNTIME_ROOT/ego.tar' -C '$RUNTIME_ROOT/planners/ego/src/ego-planner'
cd '$RUNTIME_ROOT/planners/ego/src/ego-planner'
git apply '$RUNTIME_ROOT/patches/ego-3d-goals.patch'
source /opt/ros/noetic/setup.bash
cd '$RUNTIME_ROOT/planners/ego'
catkin_make -j4 \
  -DCATKIN_WHITELIST_PACKAGES='cmake_utils;quadrotor_msgs;traj_utils;plan_env;path_searching;bspline_opt;ego_planner' \
  -DCMAKE_BUILD_TYPE=Release
"

docker exec "$ROS_CONTAINER" bash -lc "
set -eo pipefail
tar -xf '$RUNTIME_ROOT/super.tar' -C '$RUNTIME_ROOT/planners/super/src/SUPER'
source '$RUNTIME_ROOT/planners/ego/devel/setup.bash'
cd '$RUNTIME_ROOT/planners/super/src/SUPER'
git apply '$RUNTIME_ROOT/patches/super-control-initial-time.patch'
bash scripts/select_ros_version.sh ROS1
cd '$RUNTIME_ROOT/planners/super'
catkin_make -j3 \
  -DCATKIN_WHITELIST_PACKAGES='quadrotor_msgs;rog_map;super_planner' \
  -DCMAKE_BUILD_TYPE=Release \
  quadrotor_msgs_generate_messages_cpp quadrotor_msgs_generate_messages_py
catkin_make -j3 \
  -DCATKIN_WHITELIST_PACKAGES='quadrotor_msgs;rog_map;super_planner' \
  -DCMAKE_BUILD_TYPE=Release fsm_node
"

{
    printf 'container=%s\n' "$ROS_CONTAINER"
    printf 'image=%s\n' "$ROS_IMAGE"
    printf 'image_id=%s\n' "$(docker image inspect "$ROS_IMAGE" --format '{{.Id}}')"
    printf 'ego=%s\n' "$EGO_REVISION"
    printf 'super=%s\n' "$SUPER_REVISION"
} | tee "$ROOT/tmp/native-planners/runtime.txt"
