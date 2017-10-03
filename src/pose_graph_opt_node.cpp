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

// NOTE: Mark for removal

/*
class PinholeCamera {

public:
  PinholeCamera()
  {
    mValid = false;
  }
  PinholeCamera( string config_file )
  {

    cv::FileStorage fs( config_file, cv::FileStorage::READ );
    if( !fs.isOpened() )
    {
      ROS_ERROR( "Cannot open config file : %s", config_file.c_str() );
      ROS_ERROR( "Quit");
      mValid = false;
      exit(1);
    }
    this->config_file_name = string(config_file);

    cout << "---Camera Config---\n";
    fs["model_type"] >> config_model_type;     cout << "config_model_type:"<< config_model_type << endl;
    fs["camera_name"] >> config_camera_name;   cout << "config_camera_name:" << config_camera_name << endl;
    fs["image_width"] >> config_image_width;   cout << "config_image_width:" << config_image_width << endl;
    fs["image_height"] >> config_image_height; cout << "config_image_height:" << config_image_height << endl;


    fs["projection_parameters"]["fx"] >> _fx;
    fs["projection_parameters"]["fy"] >> _fy;
    fs["projection_parameters"]["cx"] >> _cx;
    fs["projection_parameters"]["cy"] >> _cy;
    cout << "projection_parameters :: " << _fx << " " << _fy << " " << _cx << " " << _cy << " " << endl;

    fs["distortion_parameters"]["k1"] >> _k1;
    fs["distortion_parameters"]["k2"] >> _k2;
    fs["distortion_parameters"]["p1"] >> _p1;
    fs["distortion_parameters"]["p2"] >> _p2;
    cout << "distortion_parameters :: " << _k1 << " " << _k2 << " " << _p1 << " " << _p2 << " " << endl;
    cout << "---    ---\n";

    // Define the 3x3 Projection matrix eigen and/or cv::Mat.
    m_K = cv::Mat::zeros( 3,3,CV_32F );
    m_K.at<float>(0,0) = _fx;
    m_K.at<float>(1,1) = _fy;
    m_K.at<float>(0,2) = _cx;
    m_K.at<float>(1,2) = _cy;
    m_K.at<float>(2,2) = 1.0;

    m_D = cv::Mat::zeros( 4, 1, CV_32F );
    m_D.at<float>(0,0) = _k1;
    m_D.at<float>(1,0) = _k2;
    m_D.at<float>(2,0) = _p1;
    m_D.at<float>(3,0) = _p2;
    cout << "m_K" << m_K << endl;
    cout << "m_D" << m_D << endl;

    // Define 4x1 vector of distortion params eigen and/or cv::Mat.
    e_K << _fx, 0.0, _cx,
          0.0,  _fy, _cy,
          0.0, 0.0, 1.0;

    e_D << _k1 , _k2, _p1 , _p2;
    cout << "e_K" << m_K << endl;
    cout << "e_D" << m_D << endl;
    mValid = true;

  }

  cv::Mat m_K; //3x3
  cv::Mat m_D; //4x1

  Matrix3d e_K;
  Vector4d e_D;

  bool isValid()
  {
    return mValid;
  }

  double fx() { return _fx; }
  double fy() { return _fy; }
  double cx() { return _cx; }
  double cy() { return _cy; }
  double k1() { return _k1; }
  double k2() { return _k2; }
  double p1() { return _p1; }
  double p2() { return _p2; }

  string getModelType() { return config_model_type; }
  string getCameraName() { return config_camera_name; }
  string getConfigFileName() { return config_file_name; }
  int getImageWidth() { return config_image_width; }
  int getImageHeight() { return config_image_height; }

  int getImageRows() { return this->getImageHeight(); }
  int getImageCols() { return this->getImageWidth(); }

  // Given N 3D points,  3xN matrix, or 4xN matrix (homogeneous)
  // project these points using this camera.
  // TODO: Extend this function to include extrinsics (given as arguments)
  void perspectiveProject3DPoints( cv::Mat& _3dpts )
  {
      cout << "Not Implemented";
  }

  // Given
  // void triangulatePoints( const Node * nC,  const cv::Mat& mat_pts_curr,
  //                         const Node * nCM, const cv::Mat& mat_pts_curr_m )
  // {
  //   // node curr (nC) is assumed to be at [I|0]
  //   // K [ I | 0 ]
  //   MatrixXd I_0;
  //   I_0 = Matrix4d::Identity().topLeftCorner<3,4>();
  //   MatrixXd P1 = camera.e_K * I_0; //3x4
  //
  // }



private:
  string config_model_type, config_camera_name;
  string config_file_name;
  int config_image_width, config_image_height;

  double _fx, _fy, _cx, _cy;
  double _k1, _k2, _p1, _p2;

  bool mValid;



  // // given a 2xN input cv::Mat converts to 1xN 2-channel output. also assumes CV_32F type
  // void _1channel_to_2channel( const cv::Mat& input, cv::Mat& output )
  // {
  //   assert( input.rows() == 2 && input.channels()==1 );
  //   output = cv::Mat( 1, input.cols, CV_32FC2 );
  //   for( int l=0 ; l<input.cols ; l++ )
  //   {
  //     output.at<cv::Vec2f>(0,l)[0] = input.at<float>(0,l);
  //     output.at<cv::Vec2f>(0,l)[1] = input.at<float>(1,l);
  //   }
  //
  // }
  //
  //
  //
  // // given a 1xN 2-channel input cv::Mat converts to 2xN,also assumes CV_32F type
  // void _2channel_to_1channel( const cv::Mat& input, cv::Mat& output )
  // {
  //   assert( input.rows() == 1 && input.channels()==2 );
  //   output = cv::Mat( 2, input.cols, CV_32FC );
  //   for( int l=0 ; l<input.cols ; l++ )
  //   {
  //     output.at<float>(0,l) = input.at<cv::Vec2f>(0,l)[0];
  //     output.at<float>(1,l) = input.at<cv::Vec2f>(0,l)[1];
  //   }
  //  }

};
*/

// NOTE: Mark for removal

/*
class Node
{
public:
  Node( ros::Time time_stamp, geometry_msgs::Pose pose )
  {
    this->time_stamp = ros::Time(time_stamp);
    this->time_pose = ros::Time(time_stamp);
    this->pose = geometry_msgs::Pose(pose);

    // TODO
    // Consider also storing original poses. e_p and e_q can be evolving poses (ie. optimization variables)
    // Basically need to revisit this when getting it to work with ceres

    // opt_position = new double[3];
    // opt_position[0] = pose.pose.position.x;
    // opt_position[1] = pose.pose.position.y;
    // opt_position[2] = pose.pose.position.z;
    // e_p << pose.pose.position.x, pose.pose.position.y,pose.pose.position.z;
    e_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );

    // opt_quat = new double[4];
    // opt_quat[0] = pose.pose.orientation.x;
    // opt_quat[1] = pose.pose.orientation.y;
    // opt_quat[2] = pose.pose.orientation.z;
    // opt_quat[3] = pose.pose.orientation.w;
    e_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
    // TODO extract covariance
  }



  ros::Time time_stamp; //this is timestamp of the pose
  geometry_msgs::Pose pose;

  ros::Time time_pcl, time_pose, time_image;

  //optimization variables
  // double *opt_position; //translation component ^wT_1
  // double *opt_quat;     //rotation component ^wR_1

  Vector3d e_p;
  Quaterniond e_q;

  void getCurrTransform(Matrix4d& M)
  {
    M = Matrix4d::Zero();
    M.col(3) << e_p, 1.0;
    Matrix3d R = e_q.toRotationMatrix();
    M.topLeftCorner<3,3>() = e_q.toRotationMatrix();

    // cout << "e_p\n" << e_p << endl;
    // cout << "e_q [w,x,y,z]\n" << e_q.w() << " " << e_q.x() << " " << e_q.y() << " " << e_q.z() << " " << endl;
    // cout << "R\n" << R << endl;
    // cout << "M\n"<< M << endl;
  }


  // 3d point cloud
  Matrix<double,3,Dynamic> ptCld;
  void setPointCloud( ros::Time time, const vector<geometry_msgs::Point32> & points )
  {
    ptCld = Matrix<double,3,Dynamic>(3,points.size());
    for( int i=0 ; i<points.size() ; i++ )
    {
      ptCld(0,i) = points[i].x;
      ptCld(1,i) = points[i].y;
      ptCld(2,i) = points[i].z;
    }
    this->time_pcl = ros::Time(time);
  }

  void setPointCloud( ros::Time time, const Matrix<double,3,Dynamic>& e )
  {
    ptCld = Matrix<double,3,Dynamic>( e );
    this->time_pcl = ros::Time(time);
  }

  const Matrix<double,3,Dynamic>& getPointCloud( )
  {
    return ptCld;
  }

  void getPointCloudHomogeneous( MatrixXd& M )
  {
    M = MatrixXd(4, ptCld.cols() );
    for( int i=0 ; i<ptCld.cols() ; i++ )
    {
      M(0,i) = ptCld(0,i);
      M(1,i) = ptCld(1,i);
      M(2,i) = ptCld(2,i);
      M(3,i) = 1.0;
    }
  }

  // image
  void setImage( ros::Time time, const cv::Mat& im )
  {
    image = cv::Mat(im.clone());
    this->time_image = ros::Time(time);
  }

  const cv::Mat& getImageRef()
  {
    return image;
  }

private:
  cv::Mat image;

};
*/

// NOTE: Mark for removal

/*
#define EDGE_TYPE_ODOMETRY 0
#define EDGE_TYPE_LOOP_CLOSURE 1

#define EDGE_TYPE_LOOP_SUBTYPE_BASIC 10 //Has enough sparse matches
#define EDGE_TYPE_LOOP_SUBTYPE_3WAY 11 //need 3 way matching, not enough sparse-feature matches
class Edge {
public:
  Edge( const Node *a, int a_id, const Node * b, int b_id, int type )
  {
    this->a = a;
    this->b = b;
    this->type = type;

    // edge_rel_position = new double[3];
    // edge_rel_quat = new double[3];
    this->a_id = a_id;
    this->b_id = b_id;


  }

  void setEdgeTimeStamps( ros::Time time_a, ros::Time time_b )
  {
    this->a_timestamp = time_a;
    this->b_timestamp = time_b;
  }

  // Given the pose in frame of b. In other words, relative pose of a in frame
  // of reference of b. This is a 4x4 matrix, top-left 3x3 represents rotation part.
  // 4th col represents translation.
  void setEdgeRelPose( const Matrix4d& b_T_a )
  {
    e_p << b_T_a(0,3), b_T_a(1,3), b_T_a(2,3);
    e_q = Quaterniond( b_T_a.topLeftCorner<3,3>() );
  }

  // Convert the stored pose into matrix and return. Note that the
  // stored pose is b_T_a, ie. pose of a in reference frame b.
  void getEdgeRelPose( Matrix4d& M )
  {
    M = Matrix4d::Zero();
    M.col(3) << e_p, 1.0;
    Matrix3d R = e_q.toRotationMatrix();
    M.topLeftCorner<3,3>() = e_q.toRotationMatrix();
  }


  void setLoopEdgeSubtype( int sub_type )
  {
    if( this->type == EDGE_TYPE_LOOP_CLOSURE )
    {
      this->sub_type = sub_type;
    }
  }

const Node *a, *b; //nodes
int type;
int sub_type;
int a_id, b_id;
ros::Time a_timestamp, b_timestamp;

Vector3d e_p;
Quaterniond e_q;
};
*/

// NOTE: Mark for removal. Careful! Some might might be recyclable

/*
class DataManager
{
public:
  DataManager(ros::NodeHandle &nh )
  {
      this->nh = nh;
      pub_pgraph = nh.advertise<visualization_msgs::Marker>( "/mish/pose_nodes", 0 );

  }

  void setCamera( PinholeCamera& camera )
  {
    this->camera = camera;

    cout << "--- Camera Params from DataManager ---\n";
    cout << "K\n" << this->camera.e_K << endl;
    cout << "D\n" << this->camera.e_D << endl;
    cout << "--- END\n";
  }


  ~DataManager()
  {
    cout << "In ~DataManager\n";

    string file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.nodes.csv";
    ofstream fp_nodes;
    fp_nodes.open( file_name );
    cout << "Write file (" <<  file_name << ") with " << nNodes.size() << " entries\n";


    fp_nodes << "#i, t, x, y, z, q.x, q.y, q.z, q.w\n";
    for( int i=0 ; i<nNodes.size() ; i++ )
    {
      Node * n = nNodes[i];
      fp_nodes <<  i << ", " << n->time_stamp  << endl;
                // << e_p[0] << ", " << e_p[1] << ", "<< e_p[2] << ", "
                // << e_q.x() << ", "<< e_q.y() << ", "<< e_q.x() << ", "<< e_q.x() << endl;
    }
    fp_nodes.close();


    // Write Odometry Edges
    file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.odomedges.csv";
    ofstream fp_odom_edge;
    fp_odom_edge.open( file_name );
    cout << "Write file (" <<  file_name << ") with " << odometryEdges.size() << " entries\n";

    fp_odom_edge << "#i, i_c, i_p, t_c, t_p\n";
    for( int i=0 ; i<odometryEdges.size() ; i++ )
    {
      Edge * e = odometryEdges[i];
      fp_odom_edge << i << ", "<< e->a_id << ", "<< e->a_timestamp
                        << ", "<< e->b_id << ", "<< e->b_timestamp << endl;
    }
    fp_odom_edge.close();


    // Write Loop Closure Edges
    file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.loopedges.csv";
    ofstream fp_loop_edge;
    fp_loop_edge.open( file_name );
    cout << "Write file (" <<  file_name << ") with " << loopClosureEdges.size() << " entries\n";

    fp_loop_edge << "#i, i_c, i_p, t_c, t_p\n";
    for( int i=0 ; i<loopClosureEdges.size() ; i++ )
    {
      Edge * e = loopClosureEdges[i];
      fp_loop_edge << i << ", "<< e->a_id << ", "<< e->a_timestamp
                        << ", "<< e->b_id << ", "<< e->b_timestamp << endl;
    }
    fp_loop_edge.close();


  }

  // No Loop Path Call back
  void noloop_path_callback( const nav_msgs::PathConstPtr & msg )
  {
    ROS_INFO( "Received - Path - %d", (int)msg->poses.size() );
    int N_poses = msg->poses.size();



    Node * n = new Node(msg->poses[N_poses-1].header.stamp, msg->poses[N_poses-1].pose);
    nNodes.push_back( n );

    // ALSO add odometry edges to 1 previous.
    int N = nNodes.size();
    int prev_k = 2;
    if( N <= prev_k )
      return;

    //add conenction from `current` to `current-1`.
    // Edge * e = new Edge( nNodes[N-1], N-1, nNodes[N-2], N-2 );
    // odometryEdges.push_back( e );

    for( int i=0 ; i<prev_k ; i++ )
    {
      Edge * e = new Edge( nNodes[N-1], N-1, nNodes[N-2-i], N-2-i, EDGE_TYPE_ODOMETRY );
      odometryEdges.push_back( e );
    }

    // TODO add relative transform as edge-inferred (using poses from corresponding edges)


    // Using 3d points, Matches between 2 images and PnP determine relative pose of the edge (observed)

  }

  // These should be same images you feed to place-recognition system
  void image_callback( const sensor_msgs::ImageConstPtr& msg )
  {
    int i_ = find_indexof_node(msg->header.stamp);
    ROS_DEBUG( "Received - Image - %d", i_ );

    cv::Mat image, image_resized;
    try{
      image = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image;
      cv::resize( image, image_resized, cv::Size(320,240) );
    }
    catch( cv_bridge::Exception& e)
    {
      ROS_ERROR( "cv_bridge exception: %s", e.what() );
    }

    if( i_ < 0 )
    {
      // unclaimed_im.push( image.clone() );
      unclaimed_im.push( image_resized.clone() );
      unclaimed_im_time.push( ros::Time(msg->header.stamp) );
      flush_unclaimed_im();
    }
    else
    {
      // nNodes[i_]->setImage( msg->header.stamp, image );
      nNodes[i_]->setImage( msg->header.stamp, image_resized );

    }

  }

  void point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg )
  {
    int i_ = find_indexof_node(msg->header.stamp);
    ROS_INFO( "Received - PointCloud - %d", i_);

    if( i_ < 0 )
    {
      // 1. msg->points to eigen matrix
      Matrix<double,3,Dynamic> ptCld;
      ptCld = Matrix<double,3,Dynamic>(3,msg->points.size());
      for( int i=0 ; i<msg->points.size() ; i++ )
      {
        ptCld(0,i) = msg->points[i].x;
        ptCld(1,i) = msg->points[i].y;
        ptCld(2,i) = msg->points[i].z;
      }

      // 2. Put this eigen matrix to queue
      unclaimed_pt_cld.push( ptCld );
      unclaimed_pt_cld_time.push( msg->header.stamp );
      flush_unclaimed_pt_cld();
    }
    else
    {
      // Corresponding node exist
      nNodes[i_]->setPointCloud( msg->header.stamp, msg->points );

    }

  }


  void place_recog_callback( const nap::NapMsg::ConstPtr& msg  )
  {
    ROS_INFO( "Received - NapMsg");
    // cout << msg->c_timestamp << " " << msg->prev_timestamp << endl;

    //
    // Look it up in nodes list (iterate over nodelist)
    int i_curr = find_indexof_node(msg->c_timestamp);
    int i_prev = find_indexof_node(msg->prev_timestamp);
    cout << i_curr << "<-->" << i_prev << endl;
    // cout <<  msg->c_timestamp << "<-->" << msg->prev_timestamp << endl;
    cout <<  msg->c_timestamp-nNodes[0]->time_stamp << "<-->" << msg->prev_timestamp-nNodes[0]->time_stamp << endl;
    cout << "Last Node timestamp : "<< nNodes.back()->time_stamp - nNodes[0]->time_stamp << endl;
    if( i_curr < 0 || i_prev < 0 )
      return;


    //
    // make vector of loop closure edges
    Edge * e = new Edge( nNodes[i_curr], i_curr, nNodes[i_prev], i_prev, EDGE_TYPE_LOOP_CLOSURE );
    e->setEdgeTimeStamps(msg->c_timestamp, msg->prev_timestamp);


    ///////////////////////////////////
    // Relative Pose Computation     //
    //////////////////////////////////
    cout << "n_sparse_matches : " << msg->n_sparse_matches << endl;
    cout << "3way match sizes : " << msg->curr.size() << " " << msg->prev.size() << " " << msg->curr_m.size() << endl;

      ////////////////////
      //---- case-a : If 3way matching is empty : do ordinary way to compute relative pose. Borrow code from Qin
      ///////////////////
    if( msg->n_sparse_matches >= 200 )
    {
      // TODO: Put Qin Tong's code here. ie. rel pose computation when we have sufficient number of matches
      e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_BASIC);
      cout << "Do Qin Tong's Code here, for computation of relative pose\n";
    }
    ////////////////////
    //---- case-b : If 3way matching is not empty : i) Triangulate curr-1 and curr. ii) pnp( 3d pts from (i) ,  prev )
    ////////////////////
    else if( msg->n_sparse_matches < 200 && msg->curr.size() > 0 && msg->curr.size() == msg->prev.size() && msg->curr.size() == msg->curr_m.size()  )
    {
      e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_3WAY);

#define DEBUG_3WAY_POSE_COMPUTATION

      // Collect 2d points from the msg, corresponding to 3-way matching
      // and make them as  cv::Mat 2xN. Note that these points are (x,y) and not (rows, cols)
      cv::Mat mat_pts_curr, mat_pts_prev, mat_pts_curr_m;
      this->extract_3way_matches_from_napmsg(msg, mat_pts_curr, mat_pts_prev, mat_pts_curr_m);



      // Undistort observed points
      cv::Mat distorted_pts_curr, distorted_pts_prev, distorted_pts_curr_m;
      this->extract_3way_matches_from_napmsg_as2channel(msg, distorted_pts_curr, distorted_pts_prev, distorted_pts_curr_m);

      cv::Mat undistort_pts_curr, undistort_pts_prev, undistort_pts_curr_m;
      cv::Mat undistort_R, undistort_P;
      cout << "distorted_pts_curr " << distorted_pts_curr.rows << " "<< distorted_pts_curr.cols << " "<< distorted_pts_curr.channels() << endl;
      cout << "distorted_pts_prev " << distorted_pts_prev.rows << " "<< distorted_pts_prev.cols << " "<< distorted_pts_prev.channels() << endl;
      cout << "distorted_pts_curr_m " << distorted_pts_curr_m.rows << " "<< distorted_pts_curr_m.cols << " "<< distorted_pts_curr_m.channels() << endl;
      cv::undistortPoints(  distorted_pts_curr, undistort_pts_curr, camera.m_K, camera.m_D);
      cv::undistortPoints(  distorted_pts_prev, undistort_pts_prev, camera.m_K, camera.m_D);
      cv::undistortPoints(  distorted_pts_curr_m, undistort_pts_curr_m, camera.m_K, camera.m_D, undistort_R, undistort_P);
      cout << "undistort_pts_curr " << undistort_pts_curr.rows << " "<< undistort_pts_curr.cols << " " << undistort_pts_curr.channels() << endl;
      cout << "undistort_pts_prev " << undistort_pts_prev.rows << " "<< undistort_pts_prev.cols << " " << undistort_pts_prev.channels() << endl;
      cout << "undistort_pts_curr_m " << undistort_pts_curr_m.rows << " "<< undistort_pts_curr_m.cols << " " << undistort_pts_curr_m.channels() << endl;



      // Get nodes from the timestamps
      int ix_curr = find_indexof_node(msg->t_curr);
      int ix_prev = find_indexof_node(msg->t_prev);
      int ix_curr_m = find_indexof_node(msg->t_curr_m);




      //DEBUG: Get the 3 images, we have 3way matches. plot these points on images.

#ifdef DEBUG_3WAY_POSE_COMPUTATION
      //TODO: Assert that ix_* are not -1
      if( ix_curr < 0 || ix_prev < 0 || ix_curr_m < 0 ){
        cout << "ERROR in place_recog_callback. Possibly invalid timestamps in napMsg. ie. cannot find the timestamps specified in the message in the pose-graph\n";
        exit(1);
      }

      cv::Mat curr_im, prev_im, curr_m_im;

      // Collect Images - for debug
      curr_im = this->nNodes[ix_curr]->getImageRef();
      prev_im = this->nNodes[ix_prev]->getImageRef();
      curr_m_im = this->nNodes[ix_curr_m]->getImageRef();


      cv::Mat dst, dst2;
      this->plot_3way_match( curr_im, mat_pts_curr, prev_im, mat_pts_prev, curr_m_im, mat_pts_curr_m, dst );
      char fname[200];
      sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d.jpg", i_curr, i_prev );
      cout << "Writing file : " << fname << endl;
      cv::imwrite( fname, dst );

      this->plot_3way_match( curr_im, distorted_pts_curr, prev_im, distorted_pts_prev, curr_m_im, distorted_pts_curr_m, dst2 );
      sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_2ch_%d_%d.jpg", i_curr, i_prev );
      cout << "Writing file : " << fname << endl;
      cv::imwrite( fname, dst2 );


          //              //
          //  END DEBUG   //
#endif

      //TODO: Computation of pose using curr, prev, curr-1 dense-point matches
      //
      //    step-1: Triangulation using, ix_curr, ix_curr_m
      //      Input  : pts_curr, pts_curr_m, Camera_Intrinsic, curr_T_{curr_m}
      //      Output : vector<cv::Point3d> of above points.

      // K [ I | 0 ]
      MatrixXd I_0;
      I_0 = Matrix4d::Identity().topLeftCorner<3,4>();
      MatrixXd P1 = camera.e_K * I_0; //3x4

      // K [ ^{c-1}T_c ] ==> K [ inv( ^wT_{c-1} ) * ^wT_c ]
      Matrix4d w_T_cm;
      nNodes[ix_curr_m]->getCurrTransform(w_T_cm);//4x4
      Matrix4d w_T_c;
      nNodes[ix_curr]->getCurrTransform(w_T_c);//4x4

      MatrixXd Tr;
      Tr = w_T_cm.inverse() * w_T_c; //relative transform
      MatrixXd P2 = camera.e_K * Tr.topLeftCorner<3,4>(); //3x4


      cv::Mat out3dPts;
      //remember, eigen stores matrix as col-major and opencv stores as row-major. Becareful with opencv-eigen conversions
      cv::Mat xP1(3,4,CV_64F );
      cv::Mat xP2(3,4,CV_64F );
      cv::eigen2cv( P1, xP1 );
      cv::eigen2cv( P2, xP2 );
      cv::Mat xP1_cv32fc, xP2_cv32fc;
      xP1.convertTo(xP1_cv32fc, CV_32FC1 );
      xP2.convertTo(xP2_cv32fc, CV_32FC1 );


      // cv::triangulatePoints( cv::eigen2cv(P1), cv::eigen2cv(P2),  pts_curr, pts_curr_m,   out3dPts );
      // cv::triangulatePoints( xP1, xP2,  pts_curr, pts_curr_m,   out3dPts );
      cv::triangulatePoints( xP1, xP2,  mat_pts_curr, mat_pts_curr_m,   out3dPts );



      #ifdef DEBUG_3WAY_POSE_COMPUTATION
      //       //
      // DEBUG - Write OpenCV YAML File //
      cout << "xP1.type" << xP1.type() << endl;
      cout << "xP2.type" << xP2.type() << endl;
      cout << "xP1_cv32fc.type" << xP1_cv32fc.type() << endl;
      cout << "xP1_cv32fc.type" << xP1_cv32fc.type() << endl;
      cout << "pts_curr.type" << mat_pts_curr.type() << endl;
      cout << "pts_curr_m.type" << mat_pts_curr_m.type() << endl;
      cout << "out3dPts.type" << out3dPts.type() << endl;
      cout << "camera.m_K.type" << camera.m_K.type() << endl;

      sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d.opencv", i_curr, i_prev );
      cout << "Writing : " << fname << endl;
      cv::FileStorage tmp_fp( fname, cv::FileStorage::WRITE );
      tmp_fp << "out3dPts" << out3dPts;

      tmp_fp << "reproj_out3dPts_on_C" <<  (xP1_cv32fc * out3dPts);
      tmp_fp << "reproj_out3dPts_on_CM" <<  (xP2_cv32fc * out3dPts);
      tmp_fp << "xP1" << xP1;
      tmp_fp << "xP2" << xP2;


      tmp_fp << "pts_curr" << mat_pts_curr;
      tmp_fp << "pts_curr_m" << mat_pts_curr_m;


      tmp_fp << "distorted_pts_curr" << distorted_pts_curr;
      tmp_fp << "distorted_pts_curr_m" << distorted_pts_curr_m;
      tmp_fp << "undistort_pts_curr" << undistort_pts_curr;
      tmp_fp << "undistort_pts_curr_m" << undistort_pts_curr_m;
      tmp_fp << "undistort_R" << undistort_R;
      tmp_fp << "undistort_P" << undistort_P;


      cv::Mat grey_curr_im, grey_curr_m_im;
      cv::cvtColor(curr_im, grey_curr_im, cv::COLOR_BGR2GRAY);
      cv::cvtColor(curr_m_im, grey_curr_m_im, cv::COLOR_BGR2GRAY);
      tmp_fp << "curr_im" << grey_curr_im;
      tmp_fp << "curr_m_im" << grey_curr_m_im;

      tmp_fp << "K" << camera.m_K;
      tmp_fp << "D" << camera.m_D;

      cv::Mat __w_T_c( w_T_c.rows(), w_T_c.cols(), CV_64F );
      cv::eigen2cv( w_T_c, __w_T_c );
      tmp_fp << "w_T_c" << __w_T_c;
      cv::Mat __w_T_cm( w_T_cm.rows(), w_T_cm.cols(), CV_64F );
      cv::eigen2cv( w_T_cm, __w_T_cm );
      tmp_fp << "w_T_cm" << __w_T_cm;
      tmp_fp.release();

      //           //
      // END DEBUG //
      #endif


      //
      //    step-2: pnp. Using 3d pts from step1 and pts_prev
      //

    }
    else {
      ROS_ERROR( "in place_recog_callback: Error computing rel pose. Edge added without pose. This might be fatal!");
    }




    loopClosureEdges.push_back( e );

  }

  // void camera_pose_callback( const geometry_msgs::PoseStamped::ConstPtr& msg )
  void camera_pose_callback( const nav_msgs::Odometry::ConstPtr msg )
  {
    // make sure to subscribe to camera_pose without loop,
    // return;
    Node * n = new Node(msg->header.stamp, msg->pose.pose);
    nNodes.push_back( n );
    ROS_DEBUG( "Recvd msg - camera_pose_callback");


    // ALSO add odometry edges to 1 previous.
    int N = nNodes.size();
    int prev_k = 1;
    if( N <= prev_k )
      return;

    //add conenction from `current` to `current-1`.
    // Edge * e = new Edge( nNodes[N-1], N-1, nNodes[N-2], N-2 );
    // odometryEdges.push_back( e );

    for( int i=0 ; i<prev_k ; i++ )
    {
      Node * a_node = nNodes[N-1];
      Node * b_node = nNodes[N-2-i];
      Edge * e = new Edge( a_node, N-1, b_node, N-2-i, EDGE_TYPE_ODOMETRY );
      e->setEdgeTimeStamps(nNodes[N-1]->time_stamp, nNodes[N-2-i]->time_stamp);

      // TODO add relative transform as edge-inferred (using poses from corresponding edges)
      // ^{w}T_a; a:= N-1
      Matrix4d w_T_a;
      a_node->getCurrTransform( w_T_a );


      // ^{w}T_b; b:= N-2-i
      Matrix4d w_T_b;
      b_node->getCurrTransform( w_T_b );


      // ^{b}T_a = inv[ ^{w}T_b ] * ^{w}T_a
      Matrix4d b_T_a = w_T_b.inverse() * w_T_a;

      // Set
      e->setEdgeRelPose(b_T_a);

      odometryEdges.push_back( e );
    }


  }


  // std::atomic<bool> bool_publish_all;//(true);
  // atomic_bool bool_publish_all = true;
  void publish_all()
  {
    while( true )
    {
      ROS_INFO( "PUBLISH" );
      publish_pose_graph_nodes();
      publish_pose_graph_edges(this->odometryEdges);
      publish_pose_graph_edges(this->loopClosureEdges);
      this_thread::sleep_for(chrono::milliseconds(100) );
    }
  }

  void publish_once()
  {
    publish_pose_graph_nodes();
    publish_pose_graph_edges( this->odometryEdges );
    publish_pose_graph_edges( this->loopClosureEdges );
  }

  void publish_pose_graph_nodes()
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "spheres";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = .6; // Don't forget to set the alpha!

    int nSze = nNodes.size();
    // for( int i=0; i<nNodes.size() ; i+=1 )
    for( int i=max(0,nSze-10); i<nNodes.size() ; i++ ) //optimization trick: only publish last 10. assuming others are already on rviz
    {
      marker.color.r = 0.0;marker.color.g = 0.0;marker.color.b = 0.0; //default color of node

      Node * n = nNodes[i];

      // Publish Sphere
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.id = i;
      marker.ns = "spheres";
      marker.pose.position.x = n->e_p[0];
      marker.pose.position.y = n->e_p[1];
      marker.pose.position.z = n->e_p[2];
      marker.pose.orientation.x = 0.;
      marker.pose.orientation.y = 0.;
      marker.pose.orientation.z = 0.;
      marker.pose.orientation.w = 1.;
      marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 0.0;
      marker.scale.x = .05;marker.scale.y = .05;marker.scale.z = .05;
      pub_pgraph.publish( marker );

      // Publish Text
      marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      marker.id = i;
      marker.ns = "text_label";
      marker.scale.z = .03;

      // pink color text if node doesnt contain images
      if( n->getImageRef().data == NULL )
      { marker.color.r = 1.0;  marker.color.g = .4;  marker.color.b = .4; }
      else
      { marker.color.r = 1.0;  marker.color.g = 1.0;  marker.color.b = 1.0; } //text in white color
      // marker.text = std::to_string(i)+std::string(":")+std::to_string(n->ptCld.cols())+std::string(":")+((n->getImageRef().data)?"I":"~I");

      std::stringstream buffer;
      buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp;
      // buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp << ":" << n->time_image- nNodes[0]->time_stamp  ;
      marker.text = buffer.str();
      // marker.text = std::to_string(i)+std::string(":")+std::to_string( n->time_stamp );
      pub_pgraph.publish( marker );
    }
  }

  void publish_pose_graph_edges( const std::vector<Edge*>& x_edges )
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.018; //0.02
    marker.scale.y = 0.05;
    marker.scale.z = 0.06;
    marker.color.a = .6; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    // cout << "There are "<< odometryEdges.size() << " edges\n";

    int nSze = x_edges.size();
    // for( int i=0 ; i<x_edges.size() ; i++ )
    for( int i=max(0,nSze-10) ; i<x_edges.size() ; i++ ) //optimization trick,
    {
      Edge * e = x_edges[i];
      marker.id = i;
      geometry_msgs::Point start;
      start.x = e->a->e_p[0];
      start.y = e->a->e_p[1];
      start.z = e->a->e_p[2];

      geometry_msgs::Point end;
      end.x = e->b->e_p[0];
      end.y = e->b->e_p[1];
      end.z = e->b->e_p[2];
      marker.points.clear();
      marker.points.push_back(start);
      marker.points.push_back(end);

      if( e->type == EDGE_TYPE_ODOMETRY ) //green - odometry edge
      {    marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;    marker.ns = "odom_edges";}
      else if( e->type == EDGE_TYPE_LOOP_CLOSURE )
      {
        if( e->sub_type == EDGE_TYPE_LOOP_SUBTYPE_BASIC ) // basic loop-edge in red
        { marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.ns = "loop_edges"; }
        else {
          if( e->sub_type == EDGE_TYPE_LOOP_SUBTYPE_3WAY ) // 3way matched loop-edge in pink
          { marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.ns = "loop_edges"; }
          else //other edge subtype in white
          { marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "loop_edges"; }
        }


      }
      else
      {    marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "x_edges";}

      pub_pgraph.publish( marker );
    }
  }
  vector<Node*> nNodes; //list of notes
  vector<Edge*> odometryEdges; //list of odometry edges
  vector<Edge*> loopClosureEdges; //List of closure edges
private:

  // Loop over each node and return the index of the node which is clossest to the specified stamp
  int find_indexof_node( ros::Time stamp )
  {
    ros::Duration diff;
    for( int i=0 ; i<nNodes.size() ; i++ )
    {
      diff = nNodes[i]->time_stamp - stamp;

      // cout << i << " "<< diff.sec << " " << diff.nsec << endl;

      // if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) < int32_t(1000000) ) {
      // if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) == int32_t(0) ) {
      if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
        return i;
      }
    }
    // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
    return -1;
  }



  std::queue<cv::Mat> unclaimed_im;
  std::queue<ros::Time> unclaimed_im_time;
  void flush_unclaimed_im()
  {
    ROS_WARN( "IM:%d, T:%d", (int)unclaimed_im.size(), (int)unclaimed_im_time.size() );

    // std::queue<cv::Mat> X_im;
    // std::queue<ros::Time> X_tm;

    int N = max(20,(int)unclaimed_im.size() );
    // while( !unclaimed_im.empty() )
    for( int i=0 ; i<N ; i++)
    {
      cv::Mat image = cv::Mat(unclaimed_im.front());
      ros::Time stamp = ros::Time(unclaimed_im_time.front());
      unclaimed_im.pop();
      unclaimed_im_time.pop();
      int i_ = find_indexof_node(stamp);
      if( i_ < 0 )
      {
        unclaimed_im.push( image.clone() );
        unclaimed_im_time.push( ros::Time(stamp) );
      }
      else
      {
        nNodes[i_]->setImage( stamp, image );
      }
    }


    // // Put back unfound ones
    // while( !X_tm.empty() )
    // {
    //   unclaimed_im.push( cv::Mat(X_im.front()) );
    //   unclaimed_im_time.push( ros::Time(X_tm.front()) );
    //   X_im.pop();
    //   X_tm.pop();
    // }
  }


  std::queue<Matrix<double,3,Dynamic>> unclaimed_pt_cld;
  std::queue<ros::Time> unclaimed_pt_cld_time;
  void flush_unclaimed_pt_cld()
  {
    ROS_WARN( "PtCld %d, %d", (int)unclaimed_pt_cld.size(), (int)unclaimed_pt_cld_time.size() );
    int M = max(20,(int)unclaimed_pt_cld.size());
    for( int i=0 ; i<M ; i++ )
    {
      Matrix<double,3,Dynamic> e = unclaimed_pt_cld.front();
      ros::Time t = ros::Time( unclaimed_pt_cld_time.front() );
      unclaimed_pt_cld.pop();
      unclaimed_pt_cld_time.pop();
      int i_ = find_indexof_node(t);
      if( i_ < 0 )
      {
        //still not found, push back again
        unclaimed_pt_cld.push( e );
        unclaimed_pt_cld_time.push( t );
      }
      else
      {
        nNodes[i_]->setPointCloud(t, e);
      }
    }

  }


// image, 2xN, image, 2xN, image 2xN, out_image
// or
// image, 1xN 2 channel, image 1xN 2 channel, image 1xN 2 channel
  void plot_3way_match( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                        const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                        const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                        cv::Mat& dst)
  {
    cv::Mat zre = cv::Mat(curr_im.rows, curr_im.cols, CV_8UC3, cv::Scalar(128,128,128) );

    cv::Mat dst_row1, dst_row2;
    cv::hconcat(curr_im, prev_im, dst_row1);
    cv::hconcat(curr_m_im, zre, dst_row2);
    cv::vconcat(dst_row1, dst_row2, dst);



    // Draw Matches
    cv::Point2d p_curr, p_prev, p_curr_m;
    for( int kl=0 ; kl<mat_pts_curr.cols ; kl++ )
    {
      if( mat_pts_curr.channels() == 2 ){
        p_curr = cv::Point2d(mat_pts_curr.at<cv::Vec2f>(0,kl)[0], mat_pts_curr.at<cv::Vec2f>(0,kl)[1] );
        p_prev = cv::Point2d(mat_pts_prev.at<cv::Vec2f>(0,kl)[0], mat_pts_prev.at<cv::Vec2f>(0,kl)[1] );
        p_curr_m = cv::Point2d(mat_pts_curr_m.at<cv::Vec2f>(0,kl)[0], mat_pts_curr_m.at<cv::Vec2f>(0,kl)[1] );
      }
      else {
        p_curr = cv::Point2d(mat_pts_curr.at<float>(0,kl),mat_pts_curr.at<float>(1,kl) );
        p_prev = cv::Point2d(mat_pts_prev.at<float>(0,kl),mat_pts_prev.at<float>(1,kl) );
        p_curr_m = cv::Point2d(mat_pts_curr_m.at<float>(0,kl),mat_pts_curr_m.at<float>(1,kl) );
      }

      cv::circle( dst, p_curr, 4, cv::Scalar(255,0,0) );
      cv::circle( dst, p_prev+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(0,255,0) );
      cv::circle( dst, p_curr_m+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(0,0,255) );
      cv::line( dst,  p_curr, p_prev+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
      cv::line( dst,  p_curr, p_curr_m+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );

      // cv::circle( dst, cv::Point2d(pts_curr[kl]), 4, cv::Scalar(255,0,0) );
      // cv::circle( dst, cv::Point2d(pts_prev[kl])+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(0,255,0) );
      // cv::circle( dst, cv::Point2d(pts_curr_m[kl])+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(0,0,255) );
      // cv::line( dst,  cv::Point2d(pts_curr[kl]), cv::Point2d(pts_prev[kl])+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
      // cv::line( dst,  cv::Point2d(pts_curr[kl]), cv::Point2d(pts_curr_m[kl])+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );
    }
  }


  // msg --> Received with callback
  // mat_ptr_curr, mat_pts_prev, mat_pts_curr_m --> 2xN outputs
  void extract_3way_matches_from_napmsg( const nap::NapMsg::ConstPtr& msg,
        cv::Mat&mat_pts_curr, cv::Mat& mat_pts_prev, cv::Mat& mat_pts_curr_m )
  {
    mat_pts_curr = cv::Mat(2,msg->curr.size(),CV_32F); //a redundancy. may be in the future consider removing one of them
    mat_pts_prev = cv::Mat(2,msg->prev.size(),CV_32F);
    mat_pts_curr_m = cv::Mat(2,msg->curr_m.size(),CV_32F);
    for( int kl=0 ; kl<msg->curr.size() ; kl++ )
    {
      mat_pts_curr.at<float>(0,kl) = (float)msg->curr[kl].x;
      mat_pts_curr.at<float>(1,kl) = (float)msg->curr[kl].y;
      mat_pts_prev.at<float>(0,kl) = (float)msg->prev[kl].x;
      mat_pts_prev.at<float>(1,kl) = (float)msg->prev[kl].y;
      mat_pts_curr_m.at<float>(0,kl) = (float)msg->curr_m[kl].x;
      mat_pts_curr_m.at<float>(1,kl) = (float)msg->curr_m[kl].y;
    }
  }

  // msg --> Received with callback
  // mat_ptr_curr, mat_pts_prev, mat_pts_curr_m --> 1xN  2 channel outputs
  void extract_3way_matches_from_napmsg_as2channel( const nap::NapMsg::ConstPtr& msg,
        cv::Mat&mat_pts_curr, cv::Mat& mat_pts_prev, cv::Mat& mat_pts_curr_m )
  {
    mat_pts_curr = cv::Mat(1,msg->curr.size(),CV_32FC2); //a redundancy. may be in the future consider removing one of them
    mat_pts_prev = cv::Mat(1,msg->prev.size(),CV_32FC2);
    mat_pts_curr_m = cv::Mat(1,msg->curr_m.size(),CV_32FC2);
    for( int kl=0 ; kl<msg->curr.size() ; kl++ )
    {
      mat_pts_curr.at<cv::Vec2f>(0,kl)[0] = (float)msg->curr[kl].x;
      mat_pts_curr.at<cv::Vec2f>(0,kl)[1] = (float)msg->curr[kl].y;
      mat_pts_prev.at<cv::Vec2f>(0,kl)[0] = (float)msg->prev[kl].x;
      mat_pts_prev.at<cv::Vec2f>(0,kl)[1] = (float)msg->prev[kl].y;
      mat_pts_curr_m.at<cv::Vec2f>(0,kl)[0] = (float)msg->curr_m[kl].x;
      mat_pts_curr_m.at<cv::Vec2f>(0,kl)[1] = (float)msg->curr_m[kl].y;
    }
  }



  }

  // Publisher
  ros::NodeHandle nh;
  ros::Publisher pub_pgraph;


  // Camera
  PinholeCamera camera;
};
*/

/*
void undistort( const PinholeCamera& cam, const MatrixXd& Xd, MatrixXd& Xdd )
{
  Xdd = MatrixXd( Xd.rows(), Xd.cols() );
  for( int i=0 ; i<Xd.cols() ; i++)
  {
    double r2 = Xd(0,i)*Xd(0,i) + Xd(1,i)*Xd(1,i);
    double c = 1.0 + cam._k1*r2 + cam._k2*r2*r2;
    Xdd(0,i) = Xd(0,i) * c + 2.0*cam._p1*Xd(0,i)*Xd(1,i) + cam._p2*(r2 + 2.0*Xd(0,i)*Xd(0,i));
    Xdd(1,i) = Xd(1,i) * c + 2.0*cam._p2*Xd(0,i)*Xd(1,i) + cam._p1*(r2 + 2.0*Xd(1,i)*Xd(1,i));
  }
}
*/

void print_matrix( string msg, const Eigen::Ref<const MatrixXd>& M, const Eigen::IOFormat& fmt )
{
  cout << msg<< M.rows() << "_" << M.cols() << "=\n" << M.format(fmt) << endl;

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
  string camera_pose_topic = string("/vins_estimator/camera_pose_no_loop");
  ROS_INFO( "Subscribe to %s", camera_pose_topic.c_str() );
  ros::Subscriber sub_odometry = nh.subscribe( camera_pose_topic, 1000, &DataManager::camera_pose_callback, &dataManager );


  string place_recognition_topic = string("/raw_graph_edge");
  // string place_recognition_topic = string("/colocation");
  ROS_INFO( "Subscribed to %s", place_recognition_topic.c_str() );
  ros::Subscriber sub_place_recognition = nh.subscribe( place_recognition_topic, 1000, &DataManager::place_recog_callback, &dataManager );

  //
  // string point_cloud_topic = string( "/vins_estimator/point_cloud_no_loop" );
  // ROS_INFO( "Subscribed to %s", point_cloud_topic.c_str() );
  // ros::Subscriber sub_pcl_topic = nh.subscribe( point_cloud_topic, 1000, &DataManager::point_cloud_callback, &dataManager );

  //
  //   This is not a requirement for core computation. But is subscribed for debug reasons. Especially to verify correctness of 3way matches
  string image_topic = string( "/vins_estimator/keyframe_image");
  ROS_INFO( "Subscribed to %s", image_topic.c_str() );
  ros::Subscriber sub_image = nh.subscribe( image_topic, 1000, &DataManager::image_callback, &dataManager );


  //--- END Subscribes ---//
  std::cout<< Color::green <<  "Pose Graph Optimization Node by mpkuse!" << Color::def << endl;


  // Setup publisher thread
  // std::thread th( &DataManager::publish_all, &dataManager );



  ros::Rate loop_rate(40);
  while( ros::ok() )
  {
    dataManager.publish_once();
    // ROS_INFO( "spinOnce");

    ros::spinOnce();
    loop_rate.sleep();
  }
  // dataManager.bool_publish_all = false;

  // th.join();



  //---------DONE
  return 0;
}

// NOTE: Mark for removal
/*
  //
  // Loop through all nodes
  //
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(500,1.2f, 8, 20);
  const Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ",", ",\n", "[", "]", "np.array([", "])");
  bool flag = true;
  for( int i=0 ; i< dataManager.nNodes.size() ; i++ )
  {
    flag = true;
    if( dataManager.nNodes[i]->getImageRef().data == NULL  ) {
      ROS_ERROR( "No Image");
      flag = false;
    }

    if( dataManager.nNodes[i]->ptCld.cols() == 0 ) {
      ROS_ERROR( "No PtCld");
      flag = false;
    }
    if( flag == false )
      continue;




    // Detect Key points and plot them as well
    cv::Mat im = dataManager.nNodes[i]->getImageRef();
    cv::Mat im_gray;
    cv::cvtColor( im, im_gray, CV_BGR2GRAY );
    vector<cv::Point2f> n_pts;
    cv::goodFeaturesToTrack(im_gray, n_pts, 200 , 0.1, 20 );
    // for( int q=0 ; q<n_pts.size() ; q++ )
    // {
    //   cv::circle( im, n_pts[q], 5, cv::Scalar(0,15,102), 1 ); //maroon
    // }

    // ORB points
    vector<cv::KeyPoint> orb_keypts;
    cv::Mat orb_descriptors;
    fdetector->detectAndCompute(im, cv::Mat(), orb_keypts, orb_descriptors );
    cout << "orb_descriptors.shape " << orb_descriptors.rows<< orb_descriptors.cols << endl;
    for( int q=0 ; q<orb_keypts.size() ; q++ )
    {
      cv::circle( im, orb_keypts[q].pt, 4, cv::Scalar(0,102,204), 1 ); //green
    }




    Matrix4d M;
    dataManager.nNodes[i]->getCurrTransform(M); // This is ^wM_i
    cout << Color::red << "^wM_i\n" << Color::red << M.format(fmt) << Color::def << endl;

    cout << Color::red <<i  << " of " << dataManager.nNodes.size()<<  " time_pcl   : " << dataManager.nNodes[i]->time_pcl - dataManager.nNodes[0]->time_pcl << endl;
    cout << Color::red <<i << " of " << dataManager.nNodes.size() <<  " time_pose  : " << dataManager.nNodes[i]->time_pose - dataManager.nNodes[0]->time_pcl << endl;
    cout << Color::red <<i  << " of " << dataManager.nNodes.size()<<  " time_image : " << dataManager.nNodes[i]->time_image - dataManager.nNodes[0]->time_pcl << endl;
    cout << Color::def << endl;


    // Acquire point cloud in homogeneous format
    MatrixXd X_h;
    dataManager.nNodes[i]->getPointCloudHomogeneous(X_h);
    // cout << "X_h_"<< X_h.rows() << "_" << X_h.cols() << "=\n" << X_h.format(fmt) << endl;
    // print_matrix("X_h", X_h, fmt );


    // Transform from world co-ordinate to frame co-ordinate
    MatrixXd im_pts;
    im_pts = M.inverse().topLeftCorner<3,4>() * X_h; // 3xN := 3x4 . 4xN
    // print_matrix("im_pts", im_pts, fmt );

    // z-division
    im_pts.row(0).array() /= im_pts.row(2).array();
    im_pts.row(1).array() /= im_pts.row(2).array();
    im_pts.row(2).array() /= im_pts.row(2).array();
    // print_matrix("im_pts", im_pts, fmt );

    // undistort
    MatrixXd im_pts_undistorted;
    undistort( camera, im_pts, im_pts_undistorted );

    // Scale with camera matrix
    // print_matrix( "K_", camera.e_K, fmt );
    MatrixXd u, ud;
    u = camera.e_K * im_pts; //.topLeftCorner(3,im_pts.cols());
    ud = camera.e_K * im_pts_undistorted; //.topLeftCorner(3,im_pts.cols());
    // print_matrix("u", u, fmt );
    // print_matrix("ud", ud, fmt );

    string str1 = string("pcl  ")+to_string(dataManager.nNodes[i]->time_pcl.sec)  + string(":") +   to_string(dataManager.nNodes[i]->time_pcl.nsec );
    string str2 = string("pose ")+to_string(dataManager.nNodes[i]->time_pose.sec ) + string(":") +  to_string(dataManager.nNodes[i]->time_pose.nsec );
    string str3 = string("image")+to_string(dataManager.nNodes[i]->time_image.sec)  + string(":") + to_string(dataManager.nNodes[i]->time_image.nsec );
    cv::putText( im, str1, cvPoint(10,30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,128,255), 1, CV_AA );
    cv::putText( im, str2, cvPoint(10,50), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,128,255), 1, CV_AA );
    cv::putText( im, str3, cvPoint(10,70), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,128,255), 1, CV_AA );
    for( int q=0 ; q<u.cols() ; q++ )
    {
      // cv::circle( im, cv::Point2f( u.at<double>(0,q),u.at<double>(1,q) ), 3, cv::Scalar(0,255,0), -1 );
      // cv::circle( im, cv::Point2f( u(0,q),u(1,q) ), 3, cv::Scalar(0,255,0), -1 );
      cv::circle( im, cv::Point2f( ud(0,q),ud(1,q) ), 3, cv::Scalar(255,0,0), -1 );
    }





    // cv::imshow( "win", dataManager.nNodes[i]->image );
    cv::imshow( "win", im );
    cv::imwrite( ros::package::getPath("nap")+string("/DUMP/tony_images/pgraph/")+to_string(i)+".jpg", im );
    cout << "i=" << i << endl;
    cv::waitKey(30);
  }



  // Loop over all loop-closure edges
  for( int i=0 ; i<dataManager.loopClosureEdges.size() ; i++ )
  {
    int a_id = dataManager.loopClosureEdges[i]->a_id;
    int b_id = dataManager.loopClosureEdges[i]->b_id ;
    // cout << "loop_closure : " << a_id << "<--->" << b_id << endl;

    cv::Mat wrim;
    cv::hconcat( dataManager.loopClosureEdges[i]->a->getImageRef(), dataManager.loopClosureEdges[i]->b->getImageRef(), wrim );

    string fname = ros::package::getPath("nap")+string("/DUMP/tony_images/pgraph/")+to_string(a_id)+ "_" + to_string(b_id)+".jpg";
    cv::imwrite(fname, wrim );
  }
  cout << "END\n";
*/
