cmake_minimum_required(VERSION 3.0)

project(paper_mobile_manipulator)

set(CMAKE_CXX_STANDARD 11)

include_directories("/usr/include/eigen3")
include_directories("/usr/include/ompl-1.5")

add_executable(paper_mm_simple paper_mm_simple.cpp)
target_link_libraries(paper_mm_simple PUBLIC ompl yaml-cpp)
