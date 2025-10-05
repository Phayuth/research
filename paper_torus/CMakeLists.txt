cmake_minimum_required(VERSION 3.0)

project(paper_torus)

set(CMAKE_CXX_STANDARD 17)

include_directories("/usr/include/eigen3")
include_directories("/usr/include/ompl-1.5")

# Include directories for robotics_cpp libraries
include_directories("../../robotics_cpp/spatial_geometry/include")
include_directories("../../robotics_cpp/robot/include")

# Add subdirectories for external libraries from robotics_cpp project
# Going up one level from workspace to access robotics_cpp
add_subdirectory("../../robotics_cpp/spatial_geometry" "${CMAKE_BINARY_DIR}/spatial_geometry")
add_subdirectory("../../robotics_cpp/robot" "${CMAKE_BINARY_DIR}/robot")

# r2s space
add_executable(r2s_altgoal src/r2s_altgoal.cpp)
target_link_libraries(r2s_altgoal PUBLIC ompl robot spatial_geometry yaml-cpp)

add_executable(r2s_cartgoal src/r2s_cartgoal.cpp)
target_link_libraries(r2s_cartgoal PUBLIC ompl robot spatial_geometry yaml-cpp)

find_package(Boost REQUIRED COMPONENTS serialization)
add_executable(r2s_prm src/r2s_prm.cpp)
target_link_libraries(r2s_prm PUBLIC ompl robot spatial_geometry yaml-cpp Boost::serialization)

add_executable(r2s_snggoal src/r2s_snggoal.cpp)
target_link_libraries(r2s_snggoal PUBLIC ompl robot spatial_geometry yaml-cpp)


# r6s space
add_executable(r6s_altgoal src/r6s_altgoal.cpp)
target_link_libraries(r6s_altgoal PUBLIC ompl robot spatial_geometry yaml-cpp)

# add_executable(r6s_cartgoal_tsrrt src/r6s_cartgoal_tsrrt.cpp)
# target_link_libraries(r6s_cartgoal_tsrrt PUBLIC ompl robot spatial_geometry yaml-cpp)

add_executable(r6s_cartgoal src/r6s_cartgoal.cpp)
target_link_libraries(r6s_cartgoal PUBLIC ompl robot spatial_geometry yaml-cpp)

add_executable(r6s_snggoal src/r6s_snggoal.cpp)
target_link_libraries(r6s_snggoal PUBLIC ompl robot spatial_geometry yaml-cpp)


# so2 space
add_executable(so2s_snggoal src/so2s_snggoal.cpp)
target_link_libraries(so2s_snggoal PUBLIC ompl robot spatial_geometry yaml-cpp)

# bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(my_module test/bindpy.cpp)
