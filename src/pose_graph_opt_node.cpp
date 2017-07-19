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
    Modifier def(FG_DEFAULT);
}

class Node
{
public:
  Node( ros::Time time_stamp, geometry_msgs::Pose pose )
  {
    this->time_stamp = ros::Time(time_stamp);
    this->time_pose = ros::Time(time_stamp);
    this->pose = geometry_msgs::Pose(pose);

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


#define EDGE_TYPE_ODOMETRY 0
#define EDGE_TYPE_LOOP_CLOSURE 1
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

const Node *a, *b; //nodes
int type;
int a_id, b_id;
Vector3d e_p;
Quaterniond e_q;
};


class DataManager
{
public:
  DataManager(ros::NodeHandle &nh )
  {
      this->nh = nh;
      pub_pgraph = nh.advertise<visualization_msgs::Marker>( "/mish/pose_nodes", 0 );

  }

  ~DataManager()
  {
    string file_name = ros::package::getPath( "nap" ) + "/DUMP/pose_graph.g2o.csv";
    ofstream fp_pose_graph;
    fp_pose_graph.open( file_name );
    cout << "Write file with " << nNodes.size() << " entries\n";

    for( int i=0 ; i<nNodes.size() ; i++ )
    {
      Node * n = nNodes[i];
      fp_pose_graph << "NODE, "<<i << ", " << n->time_stamp.sec <<":" << n->time_stamp.nsec << ", "
          << n->e_p[0] << ", " << n->e_p[1] << ", " << n->e_p[2] << ", "
          << n->e_q.x() << ", " << n->e_q.y() << ", "<< n->e_q.z() << ", "<< n->e_q.w() << ", "
          << endl;
          // << n->getX()<< ", " << n->getY()<< ", " << n->getZ()<< ", "
          // << n->getQX()<< ", " << n->getQY()<< ", " << n->getQZ()<< ", " << n->getQW() <<endl;
    }


    for( int i=0 ; i<odometryEdges.size() ; i++ )
    {
      Edge * e = odometryEdges[i];
      fp_pose_graph << "EDGE " << e->a_id << "  " << e->b_id << " ::: ";
      fp_pose_graph << "(";
      fp_pose_graph <<  e->a->e_p[0] << ", ";
      fp_pose_graph <<  e->a->e_p[1] << ", ";
      fp_pose_graph <<  e->a->e_p[2] << ", ";
      fp_pose_graph << ");";


      fp_pose_graph << "(";
      fp_pose_graph << e->b->e_p[0] << ", ";
      fp_pose_graph << e->b->e_p[1] << ", ";
      fp_pose_graph << e->b->e_p[2] << ", ";
      fp_pose_graph << ");\n";
    }

    fp_pose_graph.close();

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
    ROS_INFO( "Received - Image - %d", i_ );

    cv::Mat image;
    try{
      image = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image;
    }
    catch( cv_bridge::Exception& e)
    {
      ROS_ERROR( "cv_bridge exception: %s", e.what() );
    }

    if( i_ < 0 )
    {
      unclaimed_im.push( image.clone() );
      unclaimed_im_time.push( ros::Time(msg->header.stamp) );
      flush_unclaimed_im();
    }
    else
    {
      nNodes[i_]->setImage( msg->header.stamp, image );

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
    // TODO
    // read current and previous timestamps.
    // cout << msg->c_timestamp << " " << msg->prev_timestamp << endl;


    // Look it up in nodes list (iterate over nodelist)
    int i_curr = find_indexof_node(msg->c_timestamp);
    int i_prev = find_indexof_node(msg->prev_timestamp);
    ROS_INFO( "%d <--> %d", i_curr, i_prev );
    if( i_curr < 0 || i_prev < 0 )
      return;


    // make vector of loop closure edges
    Edge * e = new Edge( nNodes[i_curr], i_curr, nNodes[i_prev], i_prev, EDGE_TYPE_LOOP_CLOSURE );
    loopClosureEdges.push_back( e );
    odometryEdges.push_back(e) ;
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


  // std::atomic<bool> bool_publish_all;//(true);
  // atomic_bool bool_publish_all = true;
  void publish_all()
  {
    while( true )
    {
      ROS_INFO( "PUBLISH" );
      publish_pose_graph_nodes();
      publish_pose_graph_edges();
      this_thread::sleep_for(chrono::milliseconds(100) );
    }
  }

  void publish_once()
  {
    publish_pose_graph_nodes();
    publish_pose_graph_edges();
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
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = .6; // Don't forget to set the alpha!

    for( int i=0; i<nNodes.size() ; i+=1 )
    {
      marker.color.r = 0.0;marker.color.g = 0.0;marker.color.b = 0.0; //default color of node

      Node * n = nNodes[i];
      if( n->getImageRef().data == NULL ) //no image, then dark green color
      { marker.color.g = 0.8; }

      if( n->ptCld.cols() == 0 ) //no point cloud then
      { marker.color.r = 0.8; }

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
      pub_pgraph.publish( marker );

      // Publish Text
      marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      marker.id = i;
      marker.ns = "text_label";
      { marker.color.r = 1.0;  marker.color.g = 1.0;  marker.color.b = 1.0; } //text in white color
      marker.text = std::to_string(i)+std::string(":")+std::to_string(n->ptCld.cols())+std::string(":")+((n->getImageRef().data)?"I":"~I");
      pub_pgraph.publish( marker );
    }
  }

  void publish_pose_graph_edges()
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "odometry_edges";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.02;
    marker.scale.y = 0.05;
    marker.scale.z = 0.06;
    marker.color.a = .6; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    // cout << "There are "<< odometryEdges.size() << " edges\n";
    for( int i=0 ; i<odometryEdges.size() ; i++ )
    {
      Edge * e = odometryEdges[i];
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

      if( e->type == EDGE_TYPE_ODOMETRY )
      {    marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;}
      else if( e->type == EDGE_TYPE_LOOP_CLOSURE && e->a->ptCld.cols() != 0 && e->b->ptCld.cols() != 0 /*e->a->image.data != NULL && e->b->image.data != NULL*/ )
      {    marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0;}
      else if( e->type == EDGE_TYPE_LOOP_CLOSURE )
      {    marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;}
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
      if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) < int32_t(50000000) ) {
      // if( abs(diff.sec) <= int32_t(0) ) {
        // ROS_INFO( "Found at i=%d", i );
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

  // Publisher
  ros::NodeHandle nh;
  ros::Publisher pub_pgraph;
};

class PinholeCamera
{
public:
  PinholeCamera( string config_file )
  {

    cv::FileStorage fs( config_file, cv::FileStorage::READ );
    if( !fs.isOpened() )
    {
      ROS_ERROR( "Cannot open config file : %s", config_file.c_str() );
      ROS_ERROR( "Quit");
      exit(1);
    }

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

    // TODO: Define the 3x3 Projection matrix eigen and/or cv::Mat.
    m_K = cv::Mat::zeros( 3,3,CV_64F );
    m_K.at<double>(0,0) = _fx;
    m_K.at<double>(1,1) = _fy;
    m_K.at<double>(0,2) = _cx;
    m_K.at<double>(1,2) = _cy;
    m_K.at<double>(2,2) = 1.0;

    m_D = cv::Mat::zeros( 4, 1, CV_64F );
    m_D.at<double>(0,0) = _k1;
    m_D.at<double>(1,0) = _k2;
    m_D.at<double>(2,0) = _p1;
    m_D.at<double>(3,0) = _p2;
    cout << "m_K" << m_K << endl;
    cout << "m_D" << m_D << endl;

    // TODO : Define 4x1 vector of distortion params eigen and/or cv::Mat.
    e_K << _fx, 0.0, _cx,
          0.0,  _fy, _cy,
          0.0, 0.0, 1.0;

    e_D << _k1 , _k2, _p1 , _p2;
    cout << "e_K" << m_K << endl;
    cout << "e_D" << m_D << endl;

  }

  string config_model_type, config_camera_name;
  int config_image_width, config_image_height;

  double _fx, _fy, _cx, _cy;
  double _k1, _k2, _p1, _p2;

  cv::Mat m_K; //3x3
  cv::Mat m_D; //4x1

  Matrix3d e_K;
  Vector4d e_D;

};

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
  string point_cloud_topic = string( "/vins_estimator/point_cloud_no_loop" );
  ROS_INFO( "Subscribed to %s", point_cloud_topic.c_str() );
  ros::Subscriber sub_pcl_topic = nh.subscribe( point_cloud_topic, 1000, &DataManager::point_cloud_callback, &dataManager );

  //
  string image_topic = string( "/vins_estimator/keyframe_image");
  ROS_INFO( "Subscribed to %s", image_topic.c_str() );
  ros::Subscriber sub_image = nh.subscribe( image_topic, 1000, &DataManager::image_callback, &dataManager );

  //
  // string noloop_path_topic = string( "/vins_estimator/path_no_loop");
  // ROS_INFO( "Subscribed to %s", noloop_path_topic.c_str() );
  // ros::Subscriber sub_noloop_path = nh.subscribe( noloop_path_topic, 1000, &DataManager::noloop_path_callback, &dataManager );



  //--- END Subscribes ---//
  std::cout<< "Hello world\n";


  // Setup publisher thread
  // std::thread th( &DataManager::publish_all, &dataManager );



  ros::Rate loop_rate(40);
  while( ros::ok() )
  {
    // dataManager.publish_pose_graph_nodes();
    // dataManager.publish_pose_graph_edges();
    dataManager.publish_once();
    // ROS_INFO( "spinOnce");

    ros::spinOnce();
    loop_rate.sleep();
  }
  // dataManager.bool_publish_all = false;

  // th.join();



  //---------DONE
  return 0;


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

}
