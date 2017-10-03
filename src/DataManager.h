#pragma once
/** DataManager.h


      (in DataManager_core.cpp)
      This class handles (and syncs) the subscribers for image, odometry,
      point-cloud, loop closure msg. Internally it builds up a pose-graph
      with class Edge and class Node.

      (in DataManager_rviz_visualization.cpp)
      pose-graph as Marker msg for visualization.

      (in DataManager_looppose_computation.cpp)
      Another critically important part of this is the computation of relative
      pose of loop closure msg. Thus, it also has the camera-instrinsic class PinholeCamera

      (in DataManager_ceres.cpp)
      TODO Yet Another important part is going to be call to solve the
      pose graph optimization problem with CERES.


      Author  : Manohar Kuse <mpkuse@connect.ust.hk>
      Created : 3rd Oct, 2017
*/


#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>


#include <thread>
#include <mutex>
#include <atomic>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>


#include <ros/ros.h>
#include <ros/package.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <nap/NapMsg.h>



#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

using namespace std;


// CLasses In this Node
#include "Node.h"
#include "Edge.h"
#include "PinholeCamera.h"



class DataManager
{
public:
  DataManager( ros::NodeHandle &nh );
  void setCamera( PinholeCamera& camera );
  void setVisualizationTopic( string rviz_topic );

  ~DataManager();  //< Writes pose graph to file and deallocation

  // //////////////// //
  //    Callbacks     //
  // //////////////// //

  /// These should be same images you feed to place-recognition system
  /// Subscribes to images and associates these with the pose-graph.
  /// This is not required for actual computation, but is used for debugging
  void image_callback( const sensor_msgs::ImageConstPtr& msg );

  /// Subscribes to pointcloud. pointcloud messages are sublish by the
  /// visual-innertial odometry system from Qin Tong. These are used for
  /// loop-closure relative pose computation
  void point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg );


  /// Subscribes to odometry messages. These are used to make the nodes.
  /// Everytime a new odometry is received, a new node is created.
  /// The newly created node has a timestamp and a pose (wrt to world ie. ^{w}T_c )
  /// make sure to subscribe to camera_pose without loop,
  void camera_pose_callback( const nav_msgs::Odometry::ConstPtr msg );


  /// Subscribes to loop-closure messages
  void place_recog_callback( const nap::NapMsg::ConstPtr& msg  );


  // ////////////////   //
  //  Visualization     //
  // ////////////////   //
  // All 3 publish with handle `pub_pgraph`
  void publish_once(); //< Calls the next 2 functions successively
  void publish_pose_graph_nodes(); //< Publishes all nNodes
  void publish_pose_graph_edges( const std::vector<Edge*>& x_edges ); //< publishes the given edge set



  //
  // Core Data variables
  //
  // TODO: Consider making these private
  vector<Node*> nNodes; //list of notes
  vector<Edge*> odometryEdges; //list of odometry edges
  vector<Edge*> loopClosureEdges; //List of closure edges

private:

  // Buffer Utilities
  int find_indexof_node( ros::Time stamp );

  std::queue<cv::Mat> unclaimed_im;
  std::queue<ros::Time> unclaimed_im_time;
  void flush_unclaimed_im();

  std::queue<Matrix<double,3,Dynamic>> unclaimed_pt_cld;
  std::queue<ros::Time> unclaimed_pt_cld_time;
  void flush_unclaimed_pt_cld();



  ros::NodeHandle nh; //< Node Handle
  ros::Publisher pub_pgraph; //< Visualization Marker handle

  PinholeCamera camera; //< Camera Intrinsics. See corresponding class

};
