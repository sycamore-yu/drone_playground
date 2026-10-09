#!/bin/bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
set -u
mkdir -p /tmp/ros-planner-worker/log
roscore > /tmp/ros-planner-worker/roscore.log 2>&1 &
roscore_pid=$!
trap 'kill "$roscore_pid" 2>/dev/null || true' EXIT
for i in $(seq 1 100); do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep 0.1
done
/opt/ros_planner_worker/build/ros_planner_worker "${1:-0.0.0.0:50051}" &
service_pid=$!
trap 'kill "$service_pid" "$roscore_pid" 2>/dev/null || true' TERM INT EXIT
wait "$service_pid"
