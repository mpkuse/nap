cmake_minimum_required(VERSION 2.8.3)
project(nap)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)



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
)
