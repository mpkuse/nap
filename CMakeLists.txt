cmake_minimum_required(VERSION 2.8.3)
project(nap)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

set( CMAKE_CXX_FLAGS "-fpermissive" )

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  NapMsg.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)


add_executable( ls_pose_graph_opt src/ceres_try.cpp )
add_executable( toy_vertigo src/ceres_vertigo.cpp )
add_executable( toy_vertigo_dcs src/ceres_vertigo_dyn_cov_scl.cpp )
target_link_libraries( ls_pose_graph_opt ${CERES_LIBRARIES}  ${catkin_LIBRARIES} )
target_link_libraries( toy_vertigo ${CERES_LIBRARIES}  ${catkin_LIBRARIES} )
target_link_libraries( toy_vertigo_dcs ${CERES_LIBRARIES}  ${catkin_LIBRARIES} )
