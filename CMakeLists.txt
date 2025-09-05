cmake_minimum_required(VERSION 3.16)
project(VideoSketchEffect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable aggressive optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mavx2 -mfma")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
endif()

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs)
find_package(Threads REQUIRED)

# Check for CUDA support
find_package(OpenCV COMPONENTS cudaimgproc cudaarithm QUIET)
if(OpenCV_cudaimgproc_FOUND)
    message(STATUS "OpenCV CUDA support found")
    add_definitions(-DHAVE_OPENCV_CUDAIMGPROC)
endif()

# Check for Intel TBB
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "Intel TBB found - enabling parallel optimizations")
    add_definitions(-DHAVE_TBB)
endif()

add_executable(sketch_video sketch_video.cpp)

target_link_libraries(sketch_video 
    ${OpenCV_LIBS} 
    Threads::Threads
)

if(TBB_FOUND)
    target_link_libraries(sketch_video TBB::tbb)
endif()

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sketch_video OpenMP::OpenMP_CXX)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(sketch_video PRIVATE -fopenmp -ftree-vectorize)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(sketch_video PRIVATE -fopenmp -fvectorize)
elseif(MSVC)
    target_compile_options(sketch_video PRIVATE /openmp /arch:AVX2)
endif()
