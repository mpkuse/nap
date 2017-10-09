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


  // ////////////////////////////   //
  //  Relative Pose Computation     //
  // ////////////////////////////   //

  void pose_from_2way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& p_T_c );
  void pose_from_3way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& p_T_c );






private:

  //
  // Core Data variables
  //
  vector<Node*> nNodes; //list of notes
  vector<Edge*> odometryEdges; //list of odometry edges
  vector<Edge*> loopClosureEdges; //List of closure edges

  //
  // Buffer Utilities
  //
  int find_indexof_node( ros::Time stamp );

  std::queue<cv::Mat> unclaimed_im;
  std::queue<ros::Time> unclaimed_im_time;
  void flush_unclaimed_im();

  std::queue<Matrix<double,3,Dynamic>> unclaimed_pt_cld;
  std::queue<ros::Time> unclaimed_pt_cld_time;
  void flush_unclaimed_pt_cld();

  void print_cvmat_info( string msg, const cv::Mat& A );
  string type2str( int );

  //
  // rel pose computation utils
  //

  // msg --> Received with callback
  // mat_ptr_curr, mat_pts_prev, mat_pts_curr_m --> 2xN outputs
  void extract_3way_matches_from_napmsg( const nap::NapMsg::ConstPtr& msg,
        cv::Mat&mat_pts_curr, cv::Mat& mat_pts_prev, cv::Mat& mat_pts_curr_m );

  // image, 2xN, image, 2xN, image 2xN, out_image
  // or
  // image, 1xN 2 channel, image 1xN 2 channel, image 1xN 2 channel
  void plot_3way_match( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                        const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                        const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                        cv::Mat& dst);

  // image, 2xN.
  // If mat is more than 2d will only take 1st 2 dimensions as (x,y) ie (cols,rows)
  void plot_point_sets( const cv::Mat& im, const cv::Mat& pts_set, cv::Mat& dst, const cv::Scalar& color );


  // My wrapper for cv2.triangulatePoints()
  // [Input]
  // ix_curr        : index of node corresponding to curr
  // mat_pts_curr   : 2xN matrix representing point matches in curr
  // ix_curr_m      : index of node corresponding to curr-1
  // mat_pts_curr_m : 2xN matrix representing point matches in curr-1
  // [Output]
  // c_3dpts        : 3D points in co-ordinate frame of curr
  void triangulate_points( int ix_curr, const cv::Mat& mat_pts_curr,
                           int ix_curr_m, const cv::Mat& mat_pts_curr_m,
                           cv::Mat& c_3dpts );


  // My wrapper for cv2.solvePnP().
  // [Input]
  // c_3dpts : 3d points. 3xN, 1-channel. It is also ok to pass 4xN. Last row will be ignored
  // pts2d   : 2d Points 2xN 1-channel
  // [Output]
  // im_T_c  : Pose of model-cordinates (system in which 3d pts are specified) wrt to camera in which the 2d points are specified
  void estimatePnPPose( const cv::Mat& c_3dpts, const cv::Mat& pts2d,
                        Matrix4d& im_T_c  );


  void _to_homogeneous( const cv::Mat& in, cv::Mat& out );
  void _from_homogeneous( const cv::Mat& in, cv::Mat& out );
  void _perspective_divide_inplace( cv::Mat& in );
  double _diff_2d( const cv::Mat&A, const cv::Mat&B ); //< 2xM, 2xM,  RMS of these 2 matrices

  void convert_rvec_eigen4f( const cv::Mat& rvec, const cv::Mat& tvec, Matrix4f& Tr );


  // Debug file in opencv format utils (in DataManager_core.cpp)
  // void open_debug_xml( const string& fname );
  // const cv::FileStorage& get_debug_file_ptr();
  // void close_debug_xml();
  // cv::FileStorage debug_fp;

  // END 'rel pose computation utils'


  ros::NodeHandle nh; //< Node Handle
  ros::Publisher pub_pgraph; //< Visualization Marker handle

  PinholeCamera camera; //< Camera Intrinsics. See corresponding class

};
