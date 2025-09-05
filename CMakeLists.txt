cmake_minimum_required(VERSION 3.16)
project(VideoSketchEffect)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(sketch_video sketch_video.cpp)
target_link_libraries(sketch_video ${OpenCV_LIBS})
