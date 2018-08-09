#pragma once
/** DataManager.h


      (in DataManager_core.cpp)
      This class handles (and syncs) the subscribers for image, odometry,
      point-cloud, loop closure msg. Internally it builds up a pose-graph
      with class Edge and class Node.

      (in DataManager_rviz_visualization.cpp)
      pose-graph as Marker msg for visualization.

      This data will only maintain a vector of nodes. External classes are to be
      implemented for geometry computation of each type. DataManager::place_recog_callback()
      gets a NapMsg from nap_multiproc.py node (neural network and feature-matching-node).

      Currently Corvus and LocalBundle classes are implemented. These are
      instantiated in DataManager::place_recog_callback().

      Author  : Manohar Kuse <mpkuse@connect.ust.hk>
      Created : 3rd Oct, 2017
      Major Update : Jun, 2018
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


#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


#include <ceres/ceres.h>

using namespace std;

// CLasses In this Node
#include "Node.h"
#include "Edge.h"
#include "PinholeCamera.h"
#include "LocalBundle.h"
#include "Corvus.h"

#include "Feature3dInvertedIndex.h"

#include "tic_toc.h"

class EdgeManager; // fwd declaration to break circular dependency
class REdge;

class DataManager
{
public:
  DataManager( ros::NodeHandle &nh );
  DataManager(const DataManager &obj);

  void setCamera( const PinholeCamera& camera );
  void setVisualizationTopics( string rviz_topic );
  void setOpmodesToProcess( const vector<int>& _enabled_opmode );
  void setDebugOutputFolder( const string& debug_output_dir ) {
      this->BASE__DUMP=debug_output_dir;
      debug_directory_is_set=true;
      ROS_INFO( "DEBUG Directory :: %s",  BASE__DUMP.c_str() );
  }





private:
    vector<int> enabled_opmode;
    string BASE__DUMP;
    bool debug_directory_is_set = false;

public:

  ~DataManager();  //< Writes pose graph to file and deallocation

  // //////////////// //
  //    Callbacks     //
  // //////////////// //

  /// These should be same images you feed to place-recognition system
  /// Subscribes to images and associates these with the pose-graph.
  /// This is not required for actual computation, but is used for debugging
  void image_callback( const sensor_msgs::ImageConstPtr& msg );

  /// Nap Cluster assignment in raw format. mono8 type image basically a 60x80 array of numbers with intensity as cluster ID
  /// This is used for daisy matching
  void raw_nap_cluster_assgn_callback( const sensor_msgs::ImageConstPtr& msg );


  /// Subscribes to pointcloud. pointcloud messages are sublish by the
  /// visual-innertial odometry system from Qin Tong. These are used for
  /// loop-closure relative pose computation
  void point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg ); //< 3d
  void tracked_features_callback( const sensor_msgs::PointCloudConstPtr& msg ); //< 2d tracked features


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
  // void publish_pose_graph_nodes(); //< Publishes all nNodes
  void publish_pose_graph_nodes_original_poses(); // Publish into pub_pgraph_org
  void publish_pose_graph_edges( const std::vector<Edge*>& x_edges ); //< publishes the given edge set
  void publish_node_pointcloud();



  const vector<Node*>& getNodesRef() { return nNodes; }
  const vector<REdge*>& getREdgesRef() { return r_edges; }
  const Feature3dInvertedIndex * getTFIDFRef() { return tfidf; }
  const PinholeCamera& getCameraRef() { return camera;}
  const ros::Publisher& getMarkerPublisher() { return pub_pgraph; }

  void getREdgesLock();
  void getREdgesUnlock();
  int getREdgesSize();


  void getNodesLock();
  void getNodesUnlock();
  int getNodesSize() ;


private:

    // /////////////////////////////////////////////// //
    // Republish                                       //
    // /////////////////////////////////////////////// //
    ros::Publisher pub_chatter_colocation;
    void republish_nap( const nap::NapMsg::ConstPtr& msg );
public:
    void republish_nap( const ros::Time& t_c, const ros::Time& t_p, const Matrix4d& p_T_c, int32_t op_mode, float goodness=0.001 );

private:
  //
  // Core Data variables
  //
  vector<Node*> nNodes; //list of notes
  vector<Edge*> odometryEdges; //list of odometry edges
  vector<Edge*> loopClosureEdges; //List of closure edges
  vector<REdge*> r_edges; //< list of edges opmode12.
  std::mutex m_r_edges;
  std::mutex m_nNodes;


  Feature3dInvertedIndex  * tfidf; // TF-IDF (inverted index of 3d points by globalidx)

  //
  // Buffer Utilities
  //
  // int find_indexof_node( ros::Time stamp );
  int find_indexof_node( ros::Time stamp, bool print_info=false );


  std::queue<cv::Mat> unclaimed_im;
  std::queue<ros::Time> unclaimed_im_time;
  void flush_unclaimed_im();

  std::queue<cv::Mat> unclaimed_napmap;
  std::queue<ros::Time> unclaimed_napmap_time;
  void flush_unclaimed_napmap();

  // std::queue<Matrix<double,3,Dynamic>> unclaimed_pt_cld;
  std::queue<MatrixXd> unclaimed_pt_cld; //4xN
  std::queue<MatrixXd> unclaimed_pt_cld_unvn; //3xN
  std::queue<MatrixXd> unclaimed_pt_cld_uv; //3xN
  std::queue< VectorXi  > unclaimed_pt_cld_globalid; // N
  std::queue<ros::Time> unclaimed_pt_cld_time; // N
  void flush_unclaimed_pt_cld();

  std::queue<Matrix<double,3,Dynamic>> unclaimed_2d_feat;
  std::queue<ros::Time> unclaimed_2d_feat_time;
  void flush_unclaimed_2d_feat();


  void print_cvmat_info( string msg, const cv::Mat& A );
  string type2str( int );


  // image, 2xN.
  // If mat is more than 2d will only take 1st 2 dimensions as (x,y) ie (cols,rows)
  void plot_point_sets( const cv::Mat& im, const cv::Mat& pts_set, cv::Mat& dst, const cv::Scalar& color, const string& msg=string("") );
  void plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst, const cv::Scalar& color, const string& msg=string("") );
  std::vector<std::string> split( std::string const& original, char separator );



  // void convert_rvec_eigen4f( const cv::Mat& rvec, const cv::Mat& tvec, Matrix4f& Tr );
  bool if_file_exist( string fname ); //in DataManager_rviz_visualization.cpp
  bool if_file_exist( char * fname );

  string matrix4f_to_string( const Matrix4f& M );
  string matrix4d_to_string( const Matrix4d& M );


  ros::NodeHandle nh; //< Node Handle
  ros::Publisher pub_pgraph; //< Visualization Marker handle, nodes will have curr pose
  ros::Publisher pub_pgraph_org; //< Publishes Original (unoptimized pose graph)
  ros::Publisher pub_bundle; //< Info related to bundle (opmode28)
  ros::Publisher pub_3dpoints; //< Analysis of 3d points.
  ros::Publisher pub_place_recog_status_image;



  PinholeCamera camera; //< Camera Intrinsics. See corresponding class



  // Publishing helpers
  void eigenpointcloud_2_ros_markermsg( const MatrixXd& M, visualization_msgs::Marker& marker, const string& ns );
  void eigenpointcloud_2_ros_markertextmsg( const MatrixXd& M,
                    vector<visualization_msgs::Marker>& marker_ary, const string& ns );

  void printMatrixInfo( const string& msg, const MatrixXd& M );


  // Utils
  void write_image( string fname, const cv::Mat& img);
  template <typename Derived>
  void write_EigenMatrix(const string& filename, const MatrixBase<Derived>& a);
  void write_Matrix2d( const string& filename, const double * D, int nRows, int nCols );
  void write_Matrix1d( const string& filename, const double * D, int n  );

  // Plots a point set and marks the index. optionally can append a status image
  void plot_point_on_image( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
            const cv::Scalar& color, bool annotate, bool enable_status_image,
            const string& msg ,
            cv::Mat& dst );

  void plot_point_on_image( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
          const cv::Scalar& color, vector<string> annotate, bool enable_status_image,
          const string& msg ,
          cv::Mat& dst );

  void publish_image( const ros::Publisher& pub, const cv::Mat& img );
  void publish_text_as_image( const ros::Publisher& pub, const string& colon_separated_text );

};
