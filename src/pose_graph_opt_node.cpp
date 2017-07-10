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

#include <ros/ros.h>
#include <ros/package.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;

using namespace std;

class Node
{
public:
  Node( ros::Time time_stamp, geometry_msgs::PoseWithCovariance pose )
  {
    this->time_stamp = ros::Time(time_stamp);
    this->pose = geometry_msgs::PoseWithCovariance(pose);

    // opt_position = new double[3];
    // opt_position[0] = pose.pose.position.x;
    // opt_position[1] = pose.pose.position.y;
    // opt_position[2] = pose.pose.position.z;
    // e_p << pose.pose.position.x, pose.pose.position.y,pose.pose.position.z;
    e_p = Vector3d(pose.pose.position.x, pose.pose.position.y,pose.pose.position.z );

    // opt_quat = new double[4];
    // opt_quat[0] = pose.pose.orientation.x;
    // opt_quat[1] = pose.pose.orientation.y;
    // opt_quat[2] = pose.pose.orientation.z;
    // opt_quat[3] = pose.pose.orientation.w;
    e_q = Quaterniond( pose.pose.orientation.w, pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z );
    // TODO extract covariance
  }



  ros::Time time_stamp;
  geometry_msgs::PoseWithCovariance pose;

  //optimization variables
  // double *opt_position; //translation component ^wT_1
  // double *opt_quat;     //rotation component ^wR_1

  Vector3d e_p;
  Quaterniond e_q;

};


class Edge {
public:
  Edge( const Node *a, const Node * b )
  {
    this->a = a;
    this->b = b;

    // edge_rel_position = new double[3];
    // edge_rel_quat = new double[3];


  }

const Node *a, *b; //nodes
double *edge_rel_position; //this be ^aR_b
double *edge_rel_quat;
};

class DataManager
{
public:
  DataManager()
  {
      ;
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

    fp_pose_graph.close();

  }

  void odometry_callback( const nav_msgs::Odometry::ConstPtr& msg )
  {
    Node * n = new Node(msg->header.stamp, msg->pose);
    nNodes.push_back( n );
    ROS_INFO( "Recvd msg - ");


    // ALSO add odometry edges to 1 previous. TODO. later add to n previous
    int N = nNodes.size();
    if( N <= 1 )
      return;

    //add conenction from `current` to `current-1`. TODO. later add n previous nodes
    Edge * e = new Edge( nNodes[N-1], nNodes[N-2] );

    // TODO add relative transform as edge-observation

  }

private:
  vector<Node*> nNodes; //list of notes
  vector<Edge*> odometryEdges; //list of odometry edges
};



int main(int argc, char ** argv )
{
  //--- ROS INIT ---//
  ros::init( argc, argv, "pose_graph_opt_node" );
  ros::NodeHandle nh;


  //--- DataManager ---//
  DataManager dataManager = DataManager();



  //--- Subscribers ---//
  //
  string odometry_topic = string("/vins_estimator/odometry");
  ROS_INFO( "Subscribe to %s", odometry_topic.c_str() );
  ros::Subscriber sub_odometry = nh.subscribe( odometry_topic, 1000, &DataManager::odometry_callback, &dataManager );


  std::cout<< "Hello world\n";

  ros::Rate loop_rate(40);
  while( ros::ok() )
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

}
