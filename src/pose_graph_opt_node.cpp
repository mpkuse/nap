/** pose_graph_opt_node.cpp

      This node will subscribes to odometry message and napMsg (place recognition module).
      The napMsg is the edge message containing basically 2 timestamps of places it thinks as similar
      In the future possibly the relative transform of 2 timestamps also be embedded.

      Internally it will construct the pose graph.
      CERES for pose-graph optimization solver

      Author  : Manohar Kuse <mpkuse@connect.ust.hk>
      Created : 7th July, 2017
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
#include "DataManager.h"
#include "MeshObject.h"

namespace Color {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };

    Modifier red(FG_RED);
    Modifier green(FG_GREEN);
    Modifier def(FG_DEFAULT);
    Modifier blue(FG_BLUE);
}



void print_matrix( string msg, const Eigen::Ref<const MatrixXd>& M, const Eigen::IOFormat& fmt )
{
  cout << msg<< M.rows() << "_" << M.cols() << "=\n" << M.format(fmt) << endl;

}

void split(const std::string &s, char delim, vector<string>& vec_of_items)
{
    std::stringstream ss(s);
    std::string item;
    // vector<string> vec_of_items;
    vec_of_items.clear();
    while (std::getline(ss, item, delim)) {
        // *(result++) = item;
        vec_of_items.push_back( item );
    }
}



int main(int argc, char ** argv )
{
  //--- ROS INIT ---//
  ros::init( argc, argv, "pose_graph_opt_node" );
  ros::NodeHandle nh("~");


  //--- Config File ---//
  string config_file;
  nh.getParam( "config_file", config_file );
  ROS_WARN( "Config File Name : %s", config_file.c_str() );
  PinholeCamera camera = PinholeCamera( config_file );


  //--- DataManager ---//
  DataManager dataManager = DataManager(nh);
  dataManager.setCamera(camera);

  //--- Pose Graph Visual Marker ---//
  string rviz_pose_graph_topic = string( "/mish/pose_nodes" );
  ROS_INFO( "Publish Pose Graph Visual Marker to %s", rviz_pose_graph_topic.c_str() );
  dataManager.setVisualizationTopic( rviz_pose_graph_topic );



  //--- Subscribers ---//
  //
  // TODO To compare my pose-graph-optimization with qin-tong's might be useful.
  // string camera_pose_topic = string("/vins_estimator/camera_pose_no_loop");
  string camera_pose_topic = string("/vins_estimator/camera_pose");
  ROS_INFO( "Subscribe to %s", camera_pose_topic.c_str() );
  ros::Subscriber sub_odometry = nh.subscribe( camera_pose_topic, 1000, &DataManager::camera_pose_callback, &dataManager );


  string place_recognition_topic = string("/raw_graph_edge");
  // string place_recognition_topic = string("/colocation");
  ROS_INFO( "Subscribed to %s", place_recognition_topic.c_str() );
  ros::Subscriber sub_place_recognition = nh.subscribe( place_recognition_topic, 1000, &DataManager::place_recog_callback, &dataManager );


  // 3d points
  // string point_cloud_topic = string( "/vins_estimator/point_cloud_no_loop" );
  // ROS_INFO( "Subscribed to %s", point_cloud_topic.c_str() );
  // ros::Subscriber sub_pcl_topic = nh.subscribe( point_cloud_topic, 1000, &DataManager::point_cloud_callback, &dataManager );


#if defined _DEBUG_3WAY || defined _DEBUG_PNP
  // 2d features in normalized cords
  string features_tracked_topic = string( "/feature_tracker/feature" );
  ROS_INFO( "Subscribed to %s", features_tracked_topic.c_str() );
  ros::Subscriber sub_features_tracked_topic = nh.subscribe( features_tracked_topic, 1000, &DataManager::tracked_features_callback, &dataManager );


  //
  //   This is not a requirement for core computation. But is subscribed for debug reasons. Especially to verify correctness of 3way matches
  string image_topic = string( "/vins_estimator/keyframe_image");
  ROS_INFO( "Subscribed to %s", image_topic.c_str() );
  ros::Subscriber sub_image = nh.subscribe( image_topic, 1000, &DataManager::image_callback, &dataManager );

  //
  // Nap Cluster assignment in raw format. mono8 type image basically a 60x80 array of numbers with intensity as cluster ID
  // This is used for daisy matching
  string nap_cluster_assgn_topic = string( "/nap/cluster_assignment" );
  ROS_INFO( "Subscribed to %s", nap_cluster_assgn_topic.c_str() );
  ros::Subscriber sub_nap_cl_asgn = nh.subscribe( nap_cluster_assgn_topic, 1000, &DataManager::raw_nap_cluster_assgn_callback, &dataManager );


#endif


#if defined _DEBUG_AR
  //
  //  Subscribers for path (2 paths)
  // string path_vio_topic = string( "/vins_estimator/path"); //imu path
  string path_vio_topic = string( "/vins_estimator/camera_path"); //camera path
  ROS_INFO( "Subscribe to %s", path_vio_topic.c_str() );
  ros::Subscriber sub_path_vio = nh.subscribe( path_vio_topic, 1000, &DataManager::path_vio_callback, &dataManager );

  // string path_posegraph_topic = string( "/pose_graph/pose_graph_path" ); //imu path (after loop closure)
  string path_posegraph_topic = string( "/pose_graph/pose_graph_camera_path" );
  ROS_INFO( "Subscribe to %s", path_vio_topic.c_str() );
  ros::Subscriber sub_path_posegraph = nh.subscribe( path_posegraph_topic, 1000, &DataManager::path_posegraph_callback, &dataManager );

  string mesh_pose_topic = string( "/object_mesh_pose");
  ROS_INFO( "Subscribe to %s", mesh_pose_topic.c_str() );
  ros::Subscriber sub_mesh_pose = nh.subscribe( mesh_pose_topic, 1000, &DataManager::mesh_pose_callback, &dataManager );
#endif


  //--- END Subscribes ---//
  std::cout<< Color::green <<  "Visualization Node by mpkuse!" << Color::def << endl;

#if defined _DEBUG_AR
  std::cout<< Color::blue <<  "Loading OBJ Meshes" << Color::def << endl;
  vector<string> splitted_;
  split( "1.obj;chair.obj",';',splitted_ );

  // string obj_name = string( "1.obj" ); // assuming objects are kept in
  // dataManager.add_new_meshobject( obj_name );
  // dataManager.add_new_meshobject( "chair.obj" );

  for( int q=0 ; q<splitted_.size() ; q++ )
  {
    dataManager.add_new_meshobject( splitted_[q] );
  }
  std::cout<< Color::blue <<  "Done...loading meshes" << Color::def << endl;
  // return 0;

  // std::cout << "Listen to Keyboard thread\n";
  // std::thread keyboard_command_process{dataManager.make_AR_movie};
  // std::thread keyboard_command_process( &DataManager::threaded_keyboard_listener, &dataManager );

#endif


  // Setup ceres thread
  // std::thread ceres_th( &DataManager::ceres_main, &dataManager );



  ros::Rate loop_rate(40);
  while( ros::ok() )
  {
    dataManager.publish_once();
    // ROS_INFO( "spinOnce");

    ros::spinOnce();
    loop_rate.sleep();
  }
  // dataManager.bool_publish_all = false;

  // ceres_th.join();



  //---------DONE
  return 0;
}
