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

#include "cnpy.h"
// #include "SolvePoseGraph.h"

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
}



void print_matrix( string msg, const Eigen::Ref<const MatrixXd>& M, const Eigen::IOFormat& fmt )
{
  cout << msg<< M.rows() << "_" << M.cols() << "=\n" << M.format(fmt) << endl;

}

// Writes the data to file (debug)
void write_nodes_debug_data( const string& base_path, const DataManager&  dataManager )
{
    cout << "nap/pose_graph_opt_node (geometry node)/write_nodes_debug_data\n";

    vector<Node*> all_nodes = dataManager.getNodesRef();
    cout << "Total Nodes: " << all_nodes.size() << endl;


    vector<unsigned int> shape;
    int N = all_nodes.size();
    shape={1};
    cnpy::npz_save( base_path+"/vins_3d_points.npz", "N", &N, &shape[0], 1, "w" );

    for( int i=0 ; i<all_nodes.size() ; i++ ) //loop over every node
    {
        MatrixXd w_X = all_nodes[i]->getPointCloud(); // 4xN
        VectorXi id_w_X = all_nodes[i]->getPointCloudGlobalIds(); // N
        cout << "w_X.shape="<<w_X.rows() << " " << w_X.cols() << "; id_size"<< id_w_X.size() << endl;

        if( id_w_X.size() == 0 )
            continue;



        // remeber that eigen stores raw data in col major format and not the usual row major format.
        // shape = { w_X.rows(), w_X.cols() };
        shape = { w_X.cols(), w_X.rows() }; // this is not a bug. Careful when using Eigen::data().
        cnpy::npz_save( base_path+"/vins_3d_points.npz", "wvio_X"+to_string(i), w_X.data(), &shape[0], 2, "a" );

        shape = { id_w_X.size() };
        cnpy::npz_save( base_path+"/vins_3d_points.npz", "id_w_X"+to_string(i), id_w_X.data(), &shape[0], 1, "a" );

    }



    //
    // Print First and Last Matrixces for verification.
    {
    MatrixXd w_X = all_nodes[0]->getPointCloud(); // 4xN
    VectorXi id_w_X = all_nodes[0]->getPointCloudGlobalIds(); // N
    cout << "wvio_X0\n" << w_X.transpose() << endl;
    cout << "id_w_X\n" << id_w_X << endl;

    int N = all_nodes.size()/2;
    cout << "N="<< N << endl;
    w_X = all_nodes[N]->getPointCloud(); // 4xN
    id_w_X = all_nodes[N]->getPointCloudGlobalIds(); // N
    cout << "wvio_X"<< N << "\n" << w_X.transpose() << endl;
    cout << "id_w_X"<< N << "\n" << id_w_X << endl;
    }

    cout << "Done with `write_nodes_debug_data`\n";
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
  string point_cloud_topic = string( "/vins_estimator/keyframe_point" );
  ROS_INFO( "Subscribed to %s", point_cloud_topic.c_str() );
  ros::Subscriber sub_pcl_topic = nh.subscribe( point_cloud_topic, 1000, &DataManager::point_cloud_callback, &dataManager );

#if defined _DEBUG_BUNDLE

  //
  //   This is not a requirement for core computation. But is subscribed for debug reasons. Especially to verify correctness of 3way matches
  string image_topic = string( "/vins_estimator/keyframe_image");
  ROS_INFO( "Subscribed to %s", image_topic.c_str() );
  ros::Subscriber sub_image = nh.subscribe( image_topic, 1000, &DataManager::image_callback, &dataManager );

#endif

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


  //--- END Subscribes ---//
  std::cout<< Color::green <<  "Pose Graph Optimization Node by mpkuse!" << Color::def << endl;



  //
  // Setup thread for solving the pose graph slam. Uses std::thread
  // SolvePoseGraph solve(dataManager );
  // solve.setFileOut( "/dev/pts/18" );
  // std::thread solver_thread( &SolvePoseGraph::sayHi, solve );





  ros::Rate loop_rate(40);
  while( ros::ok() )
  {
    dataManager.publish_once();
    // ROS_INFO( "spinOnce");

    ros::spinOnce();
    loop_rate.sleep();
  }
  // dataManager.bool_publish_all = false;
  write_nodes_debug_data( "/home/mpkuse/Desktop/bundle_adj/pose_graph_analyis", dataManager );


  // solver_thread.join();


  //---------DONE
  cout << "//---------Done";
  return 0;
}
