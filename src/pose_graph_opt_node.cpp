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
#include "EdgeManager.h"
#include "Feature3dInvertedIndex.h"

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


// write images to disk
#define ____main__poses_file 1
void write_kf_debug_data( const string& base_path, const DataManager&  dataManager )
{
    cout << "write all the keyframes to :"<< base_path << endl;
    vector<Node*> all_nodes = dataManager.getNodesRef();
    cout << "Total Nodes: " << all_nodes.size() << endl;
    int c = 0;

    ofstream myfile;
    char txtfname[200];
    snprintf( txtfname, 200, "%s/kf/poses.txt", base_path.c_str() );
    cout << "Open poses file: "<< txtfname << endl;
    #if ____main__poses_file
    myfile.open ( string(txtfname), ios::out );
    assert( myfile.is_open() );
    #endif


    for( int i=0 ; i<all_nodes.size() ; i+=3 ) //loop over every node
    {
        // cout << i << " valid_image " << all_nodes[i]->valid_image() << endl;
        if( i > 1 )
        {
            cout << "del_time: " << all_nodes[i]->time_image - all_nodes[i-1]->time_image << endl;
        }

        if( all_nodes[i]->valid_image() )
        {
            cv::Mat im = all_nodes[i]->getImageRef();

            char fname[200];
            snprintf( fname, 200, "%s/kf/%06d.jpg", base_path.c_str(), c );
            cout << i << " write image: "<< fname << endl;
            c++;
            cv::imwrite( fname, im  );


            #if ____main__poses_file
            // Write pose in row-major. write only 3 rows (12 doubles)
            Matrix4d M;
            all_nodes[i]->getOriginalTransform( M );
            for( int row=0 ; row<3; row++ )
            {
                for( int col=0 ; col<4 ; col++ )
                {
                    myfile << M(row,col) << " ";
                    // myfile << 0.0 << " ";
                }
            }
            myfile << "\n";
            #endif
        }
        else
        {
            cout << i << " invalid image\n";
        }
    }

    #if ____main__poses_file
    myfile.close();
    cout << "Close poses file: "<< txtfname << endl;
    #endif

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
        cout << i <<  " w_X.shape="<<w_X.rows() << " " << w_X.cols() << "; id_size"<< id_w_X.size() << endl;

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
    // MatrixXd w_X = all_nodes[0]->getPointCloud(); // 4xN
    // VectorXi id_w_X = all_nodes[0]->getPointCloudGlobalIds(); // N
    // cout << "wvio_X0\n" << w_X.transpose() << endl;
    // cout << "id_w_X\n" << id_w_X << endl;
    //
    // int N = all_nodes.size()/2;
    // cout << "N="<< N << endl;
    // w_X = all_nodes[N]->getPointCloud(); // 4xN
    // id_w_X = all_nodes[N]->getPointCloudGlobalIds(); // N
    // cout << "wvio_X"<< N << "\n" << w_X.transpose() << endl;
    // cout << "id_w_X"<< N << "\n" << id_w_X << endl;
    }

    cout << "Done with `write_nodes_debug_data`\n";
}

int main(int argc, char ** argv )
{
  //--- ROS INIT ---//
  ros::init( argc, argv, "pose_graph_opt_node" );
  ros::NodeHandle nh("~");
  // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);


  //-- Set debug directory --//
  string debug_output_dir;
  nh.getParam( "debug_output_dir", debug_output_dir );
  ROS_WARN( "debug_output_dir : %s", debug_output_dir.c_str() );


  //--- Config File ---//
  string config_file;
  nh.getParam( "config_file", config_file );
  ROS_WARN( "Config File Name : %s", config_file.c_str() );
  PinholeCamera camera = PinholeCamera( config_file );


  //--- DataManager ---//
  DataManager dataManager = DataManager(nh);
  dataManager.setCamera(camera);
  dataManager.setDebugOutputFolder( debug_output_dir );

  //--- Pose Graph Visual Marker ---//
  string rviz_pose_graph_topic = string( "/mish/pose_nodes" );
  ROS_INFO( "Publish Pose Graph Visual Marker to %s", rviz_pose_graph_topic.c_str() );
  dataManager.setVisualizationTopics( rviz_pose_graph_topic );


  //-- Tell the place_recog_callback() which opmodes to process. opmodes not in this list will be ignored.
  vector<int> enabled_opmode;
  // enabled_opmode.push_back(10); // republish t_curr and t_prev
  // enabled_opmode.push_back(29); // 3way (not in use)
  // enabled_opmode.push_back(20); // contains t_curr, t_prev and matched-tracked points.
  // enabled_opmode.push_back(18); // Corvus
  // enabled_opmode.push_back(28); // LocalBundle
  // enabled_opmode.push_back(17);    // Corvus with tfidf.
  enabled_opmode.push_back(12);    // Raw edge (just a result of thresholding on descriptor score)
  dataManager.setOpmodesToProcess( enabled_opmode );


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


#if LOCALBUNDLE_DEBUG_LVL > 0 || CORVUS_DEBUG_LVL > 0 || __Feature3dInvertedIndex_DEBUG_LVL >0 || __EdgeManager_DEBUG_LVL > 0
  //
  //   This is not a requirement for core computation. But is subscribed for debug reasons. Especially to verify correctness of 3way matches
  string image_topic = string( "/vins_estimator/keyframe_image");
  ROS_INFO( "Subscribed to %s", image_topic.c_str() );
  ros::Subscriber sub_image = nh.subscribe( image_topic, 1000, &DataManager::image_callback, &dataManager );

#endif

// #if defined _DEBUG_3WAY || defined _DEBUG_PNP // these #defs were removed. Below code kept for reference.
#if 0
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
  std::cout<< Color::green <<  "Coordinates to Pose Processor Node by mpkuse!" << Color::def << endl;


  // Edge Processor Thread Launch
  EdgeManager * edge_manager = new EdgeManager( &dataManager );
  std::thread edge_proc_thread( &EdgeManager::loop, *edge_manager );



  ros::Rate loop_rate(40);
  while( ros::ok() )
  {
    dataManager.publish_once();
    // ROS_INFO( "spinOnce");

    ros::spinOnce();
    loop_rate.sleep();
  }


  edge_proc_thread.join();


  // dataManager.bool_publish_all = false;
  // write_nodes_debug_data( "/home/mpkuse/Desktop/bundle_adj/pose_graph_analyis", dataManager );
  // write_nodes_debug_data( debug_output_dir, dataManager );
  // write_kf_debug_data( debug_output_dir, dataManager  );


  // dataManager.getTFIDFRef()->sayHi();


  // solver_thread.join();


  //---------DONE
  cout << "//---------Done";
  return 0;
}
