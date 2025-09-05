cmake_minimum_required(VERSION 3.16)
project(VideoSketchEffect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs)
find_package(Threads REQUIRED)

add_executable(sketch_video sketch_video.cpp)

target_link_libraries(sketch_video 
    ${OpenCV_LIBS} 
    Threads::Threads
)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sketch_video OpenMP::OpenMP_CXX)
endif()
