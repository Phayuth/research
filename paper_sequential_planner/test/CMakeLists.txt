cmake_minimum_required(VERSION 3.10)
project(ur5e_collision_checker)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(pinocchio REQUIRED)
find_package(coal REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost COMPONENTS graph REQUIRED)

# Include directories
include_directories(${pinocchio_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories("/usr/include/ompl-1.5")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

add_executable(ur5e_col ur5e_col.cpp)
target_link_libraries(ur5e_col pinocchio::pinocchio yaml-cpp)

add_executable(tsp_solver_nnaprox tsp_solver_nnaprox.cpp)

add_executable(r6s_omplsnggoal r6s_omplsnggoal.cpp)
target_link_libraries(r6s_omplsnggoal ompl yaml-cpp)

# Print configuration information
message(STATUS "Pinocchio include dirs: ${pinocchio_INCLUDE_DIRS}")
message(STATUS "Pinocchio libraries: ${pinocchio_LIBRARIES}")
message(STATUS "Collision libraries: ${COLLISION_LIBS}")
