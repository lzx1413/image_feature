# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(vladtest)
set(CMAKE_MODULE_PATH "/home/zuoxin/workplace/vladtest_advance/cmake-modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O4")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(./ SRC_LIST)
#find_package(MKL REQUIRED)
include_directories(/home/zuoxin/usr/local/include)
link_directories(/home/zuoxin/usr/local/lib)
#link_directories(${MKL_LIBRARIES})
set(CMAKE_CXX_COMPILER "g++")
# Declare the executable target built from your sources
add_executable(vladtest ${SRC_LIST})

# Link your application with OpenCV libraries
target_link_libraries(vladtest opencv_core opencv_imgproc opencv_highgui opencv_calib3d opencv_cudaarithm opencv_cudabgsegm opencv_cudacodec opencv_cudafeatures2d opencv_cudafilters opencv_cudaimgproc opencv_cudalegacy opencv_cudaobjdetect opencv_cudaoptflow opencv_cudawarping opencv_cudev opencv_features2d opencv_flann opencv_hal opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_shape opencv_stitching opencv_superres opencv_video vl)
