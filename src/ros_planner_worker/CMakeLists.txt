cmake_minimum_required(VERSION 3.16)
project(drone_playground_ros_planner_worker LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(EGO_ROOT /opt/upstream/ego-planner CACHE PATH "Pinned EGO checkout")
set(SUPER_ROOT /opt/upstream/SUPER CACHE PATH "Pinned SUPER checkout")
find_package(catkin REQUIRED COMPONENTS roscpp nav_msgs sensor_msgs cv_bridge pcl_ros tf2_ros)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED IMPORTED_TARGET grpc++)

set(PROTO_REL drone_playground/simulation/ros_planner/planner)
set(GENERATED ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED}/ego_planner)
add_custom_command(OUTPUT ${GENERATED}/${PROTO_REL}.pb.cc ${GENERATED}/${PROTO_REL}.pb.h
                          ${GENERATED}/${PROTO_REL}.grpc.pb.cc ${GENERATED}/${PROTO_REL}.grpc.pb.h
  COMMAND protobuf::protoc -I ${CMAKE_CURRENT_SOURCE_DIR}/proto
    --cpp_out=${GENERATED} --grpc_out=${GENERATED} --plugin=protoc-gen-grpc=/usr/bin/grpc_cpp_plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO_REL}.proto
  DEPENDS proto/${PROTO_REL}.proto)
add_library(native_proto ${GENERATED}/${PROTO_REL}.pb.cc ${GENERATED}/${PROTO_REL}.grpc.pb.cc)
target_include_directories(native_proto PUBLIC ${GENERATED})
target_link_libraries(native_proto PUBLIC protobuf::libprotobuf PkgConfig::GRPC)

execute_process(COMMAND /usr/bin/python3 /opt/ros/noetic/lib/gencpp/gen_cpp.py
  ${EGO_ROOT}/src/planner/plan_manage/msg/DataDisp.msg -p ego_planner
  -Istd_msgs:/opt/ros/noetic/share/std_msgs/msg -o ${GENERATED}/ego_planner
  -e /opt/ros/noetic/share/gencpp RESULT_VARIABLE MSG_RESULT)
if(NOT MSG_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot generate upstream EGO ROS message")
endif()
include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(EGO ${EGO_ROOT}/src/planner)
add_library(ego_adapter STATIC src/ego.cpp
  ${EGO}/plan_manage/src/planner_manager.cpp
  ${EGO}/plan_env/src/grid_map.cpp ${EGO}/plan_env/src/raycast.cpp
  ${EGO}/path_searching/src/dyn_a_star.cpp
  ${EGO}/bspline_opt/src/bspline_optimizer.cpp ${EGO}/bspline_opt/src/uniform_bspline.cpp
  ${EGO}/traj_utils/src/planning_visualization.cpp ${EGO}/traj_utils/src/polynomial_traj.cpp)
target_include_directories(ego_adapter PRIVATE ${EGO}/plan_manage/include ${EGO}/plan_env/include
  ${EGO}/path_searching/include ${EGO}/bspline_opt/include ${EGO}/traj_utils/include)
target_link_libraries(ego_adapter PUBLIC native_proto ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

file(GLOB SUPER_SOURCES ${SUPER_ROOT}/super_planner/src/super_core/*.cpp
  ${SUPER_ROOT}/super_planner/src/traj_opt/*.cpp ${SUPER_ROOT}/super_planner/src/utils/*.cpp
  ${SUPER_ROOT}/rog_map/src/rog_map/*.cpp ${SUPER_ROOT}/rog_map/src/rog_map_ros/*.cpp)
list(FILTER SUPER_SOURCES EXCLUDE REGEX "/fsm.cpp$")
add_library(super_adapter STATIC src/super.cpp ${SUPER_SOURCES}
  ${SUPER_ROOT}/rog_map/include/rog_map/rog_map_core/raycaster.cpp)
target_include_directories(super_adapter PRIVATE ${SUPER_ROOT}/super_planner/include ${SUPER_ROOT}/rog_map/include)
target_compile_definitions(super_adapter PRIVATE USE_ROS1 ORIGIN_AT_CORNER FMT_HEADER_ONLY
  ROOT_DIR="/tmp/ros-planner-worker/" ROS_PLANNER_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
target_link_libraries(super_adapter PUBLIC native_proto ${catkin_LIBRARIES} ${PCL_LIBRARIES} yaml-cpp dw)
add_executable(ros_planner_worker src/service.cpp)
target_link_libraries(ros_planner_worker PRIVATE ego_adapter super_adapter)
