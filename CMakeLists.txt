cmake_minimum_required(VERSION 3.16)
project(plane-lio CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)
# Debian's VTK (pulled in by PCL io) references MPI::MPI_C even when MPI is absent.
find_package(MPI QUIET COMPONENTS C)
if(NOT TARGET MPI::MPI_C)
  add_library(MPI::MPI_C INTERFACE IMPORTED)
endif()
find_package(PCL REQUIRED COMPONENTS common io filters kdtree registration)
# Debian's GTSAM CMake export references a missing libCppUnitLite.a, so locate GTSAM directly.
find_package(Boost REQUIRED COMPONENTS serialization system filesystem thread timer chrono)
find_package(TBB QUIET)
find_library(GTSAM_LIB gtsam REQUIRED)
find_path(GTSAM_INC gtsam/geometry/Pose3.h REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP)

# --- lio_core: ROS-free LiDAR-inertial odometry (IESKF + voxel map + residuals)
add_library(lio_core
  core/src/ieskf.cpp
  core/src/imu_processor.cpp
  core/src/voxel_map.cpp
  core/src/lio_builder.cpp)
target_include_directories(lio_core PUBLIC core/include ${PCL_INCLUDE_DIRS})
target_link_libraries(lio_core PUBLIC Eigen3::Eigen ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(lio_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- lio_mapping: keyframes, loop closure (iSAM2) and multi-session merge
add_library(lio_mapping_lib
  mapping/src/keyframe.cpp
  mapping/src/loop_closure.cpp
  mapping/src/map_merge.cpp)
target_include_directories(lio_mapping_lib PUBLIC mapping/include)
target_link_libraries(lio_mapping_lib PUBLIC lio_core ${GTSAM_LIB} Boost::serialization Boost::system Boost::filesystem Boost::thread Boost::timer)
target_include_directories(lio_mapping_lib PUBLIC ${GTSAM_INC})
if(TBB_FOUND)
  target_link_libraries(lio_mapping_lib PUBLIC TBB::tbb)
endif()

# --- offline apps
add_executable(lio_mapping apps/lio_mapping.cpp)
target_link_libraries(lio_mapping lio_mapping_lib yaml-cpp)
add_executable(map_merge apps/map_merge.cpp)
target_link_libraries(map_merge lio_mapping_lib)
