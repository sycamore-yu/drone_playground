cmake_minimum_required(VERSION 3.24)
project(drone_native VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../src/drone_playground/rpc/proto")
set(GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED}")
add_custom_command(
  OUTPUT "${GENERATED}/algorithm.pb.cc" "${GENERATED}/algorithm.pb.h"
         "${GENERATED}/algorithm.grpc.pb.cc" "${GENERATED}/algorithm.grpc.pb.h"
  COMMAND protobuf::protoc --proto_path=${PROTO_ROOT}
          --proto_path=$<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>
          --cpp_out=${GENERATED} --grpc_out=${GENERATED}
          --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
          "${PROTO_ROOT}/algorithm.proto"
  DEPENDS "${PROTO_ROOT}/algorithm.proto"
  VERBATIM)
add_library(drone_native STATIC src/service.cc
  "${GENERATED}/algorithm.pb.cc" "${GENERATED}/algorithm.grpc.pb.cc")
target_include_directories(drone_native PUBLIC include "${GENERATED}")
target_link_libraries(drone_native PUBLIC gRPC::grpc++ protobuf::libprotobuf)

option(DRONE_NATIVE_BUILD_TESTS "Build protocol interoperability fixtures" OFF)
if(DRONE_NATIVE_BUILD_TESTS)
  add_executable(interop_server tests/interop_server.cc)
  target_link_libraries(interop_server PRIVATE drone_native)
endif()
