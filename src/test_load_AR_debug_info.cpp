#include <iostream>
#include <thread>
using namespace std;

#include <ros/ros.h>
#include <std_msgs/String.h>

#include "DataManager.h"

#define _DEBUG_FOLDER "/home/mpkuse/Desktop/a/drag_posecompute_node/"

int main(int argc, char ** argv )
{
  ros::init( argc, argv, "talker" );
  ros::NodeHandle n;

  string base = string( _DEBUG_FOLDER );

  // Load Camera Info
  PinholeCamera camera = PinholeCamera( base+"/pinhole_camera.yaml" );


  // Load Nodes
  Node ni = Node();
  ni.load_debug_xml( base+string("/node_000020") );


  // Load Meshes
  MeshObject m = MeshObject();
  m.load_debug_xml( base+string("/mesh_000000") );
  cout << "isMeshLoaded: " << m.isMeshLoaded() << endl;
  cout << "isWorldPoseAvailable: "<< m.isWorldPoseAvailable() << endl;



  ros::Publisher pub_chatter = n.advertise< std_msgs::String>( "chatter", 1000 );

  cout << "Hello World\n";

  ros::Rate rate(10);
  int count = 0;
  while( ros::ok() )
  {
    std_msgs::String msg;
    msg.data = string( "Hello") + to_string( count );
    pub_chatter.publish( msg );
    ros::spinOnce();
    rate.sleep();
    count++;
  }


  return 0;
}
