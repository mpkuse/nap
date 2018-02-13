cmake_minimum_required(VERSION 2.8.3)
project(nap)

set( CMAKE_CXX_STANDARD 11 )
find_package(Threads)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
  image_transport
  cv_bridge

)

list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake")
find_package(Eigen3 REQUIRED)
#find_package(Ceres REQUIRED)
#find_package(DBoW3 REQUIRED )
find_package(OpenCV 3 REQUIRED)
find_package(ZLIB REQUIRED) #for cnpy.cpp

set( CMAKE_CXX_FLAGS "-fpermissive -std=c++11" )

#set( "DBOW_CPP_FILES" "src/DBoW3/BowVector.cpp src/DBoW3/Database.cpp src/DBoW3/DescManip.cpp src/DBoW3/FeatureVector.cpp src/DBoW3/QueryResults.cpp src/DBoW3/ScoringObject.cpp src/DBOW3/Vocabulary.cpp src/DBOW3/BowVector.cpp src/DBoW3/Database.cpp src/DBoW3/DescManip.cpp src/DBoW3/FeatureVector.cpp src/DBoW3/QueryResults.cpp src/DBoW3/ScoringObject.cpp src/DBoW3/Vocabulary.cpp ")
#set( "DBOW_H_FILES" "BowVector.h DBoW3.h Database.h DescManip.h FeatureVector.h QueryResults.h ScoringObject.h Vocabulary.h exports.h quicklz.h timers.h" )




## Generate messages in the 'msg' folder
add_message_files(
  FILES
  NapMsg.msg
  NapNodeMsg.msg
  NapVisualEdgeMsg.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs

)



catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime image_transport cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
#  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
#  ${DBoW3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


#add_executable( ls_pose_graph_opt src/ceres_try.cpp )
#add_executable( toy_vertigo src/ceres_vertigo.cpp )
#add_executable( toy_vertigo_dcs src/ceres_vertigo_dyn_cov_scl.cpp )
#
add_executable( dbow3_test
                    src/dbow3_test.cpp
                    src/DBoW3/BowVector.cpp
                    src/DBoW3/Database.cpp
                    src/DBoW3/DescManip.cpp
                    src/DBoW3/FeatureVector.cpp
                    src/DBoW3/QueryResults.cpp
                    src/DBoW3/ScoringObject.cpp
                    src/DBoW3/Vocabulary.cpp
                    src/DBoW3/BowVector.cpp
                    src/DBoW3/Database.cpp
                    src/DBoW3/DescManip.cpp
                    src/DBoW3/quicklz.c
                )


add_executable( dbow3_naive
                    src/dbow3_naive.cpp
                    src/cnpy.cpp

                    src/DBoW3/BowVector.cpp
                    src/DBoW3/Database.cpp
                    src/DBoW3/DescManip.cpp
                    src/DBoW3/FeatureVector.cpp
                    src/DBoW3/QueryResults.cpp
                    src/DBoW3/ScoringObject.cpp
                    src/DBoW3/Vocabulary.cpp
                    src/DBoW3/BowVector.cpp
                    src/DBoW3/Database.cpp
                    src/DBoW3/DescManip.cpp
                    src/DBoW3/quicklz.c

                    ) #cnpy is for writing npy. borrowed from
#
add_executable( pose_graph_opt_node
                    src/pose_graph_opt_node.cpp
                    src/Node.cpp
                    src/Edge.cpp
                    src/PinholeCamera.cpp
                    src/DataManager_core.cpp
                    src/DataManager_rviz_visualization.cpp
                    src/DataManager_looppose_computation.cpp
                    src/DataManager_ceres.cpp
                    src/cnpy.cpp
              )

#target_link_libraries( ls_pose_graph_opt ${CERES_LIBRARIES}  ${catkin_LIBRARIES} )
#target_link_libraries( toy_vertigo ${CERES_LIBRARIES}  ${catkin_LIBRARIES} )
#target_link_libraries( toy_vertigo_dcs ${CERES_LIBRARIES}  ${catkin_LIBRARIES} )
#
target_link_libraries( dbow3_test  ${OpenCV_LIBRARIES} )
target_link_libraries( dbow3_naive  ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} )
target_link_libraries( pose_graph_opt_node  ${OpenCV_LIBRARIES}  ${catkin_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}  )
