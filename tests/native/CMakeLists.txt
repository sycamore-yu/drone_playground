cmake_minimum_required(VERSION 3.24)
project(drone_rpc_interop LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
get_target_property(PROTOBUF_INCLUDES protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
list(GET PROTOBUF_INCLUDES 0 PROTOBUF_INCLUDE)

set(PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../src/drone_playground/integrations/rpc/proto")
set(GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED}")
add_custom_command(
  OUTPUT "${GENERATED}/algorithm.pb.cc" "${GENERATED}/algorithm.pb.h"
         "${GENERATED}/algorithm.grpc.pb.cc" "${GENERATED}/algorithm.grpc.pb.h"
  COMMAND protobuf::protoc --proto_path=${PROTO_ROOT}
          --proto_path=${PROTOBUF_INCLUDE}
          --cpp_out=${GENERATED} --grpc_out=${GENERATED}
          --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
          "${PROTO_ROOT}/algorithm.proto"
  DEPENDS "${PROTO_ROOT}/algorithm.proto"
  VERBATIM)
add_executable(interop_server interop_server.cc
  "${GENERATED}/algorithm.pb.cc" "${GENERATED}/algorithm.grpc.pb.cc")
target_include_directories(interop_server PRIVATE "${GENERATED}")
target_link_libraries(interop_server PRIVATE gRPC::grpc++ protobuf::libprotobuf)
