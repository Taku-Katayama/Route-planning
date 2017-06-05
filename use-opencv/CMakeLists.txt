set(OpenCV_DIR "C:/opencv-3.1.0/build/install")

include_directories(${OpenCV_INCLUDE_DIRS})
cmake_minimum_required(VERSION 2.6)
project(Route_Planning)
find_package(OpenCV REQUIRED)
add_executable(Route_Planning main.cpp)
TARGET_LINK_LIBRARIES(Route_Planning ${OpenCV_LIBS})
add_compile_options(-std=c++11)
