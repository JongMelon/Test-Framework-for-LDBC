cmake_minimum_required(VERSION 3.9)

project(Test LANGUAGES CXX CUDA)

find_package(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

aux_source_directory(src DIR_SRCS)

include_directories(include)

add_executable(${PROJECT_NAME} ${DIR_SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
