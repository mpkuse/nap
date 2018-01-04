#include <iostream>
#include <thread>
using namespace std;

#include <ros/ros.h>
#include <std_msgs/String.h>

#include "DataManager.h"
#include "SceneRenderer.h"

#define _DEBUG_FOLDER "/home/mpkuse/Desktop/a/drag_posecompute_node/"

int main(int argc, char ** argv )
{
  ros::init( argc, argv, "talker" );
  ros::NodeHandle n;

  string base = string( _DEBUG_FOLDER );

  // Renderer
  SceneRenderer renderer = SceneRenderer();

  // Load Camera Info
  PinholeCamera camera = PinholeCamera( base+"/pinhole_camera.yaml" );
  renderer.setCamera( &camera );

  // Load Meshes
  MeshObject m = MeshObject();
  m.load_debug_xml( base+string("/mesh_000000") );
  cout << "isMeshLoaded: " << m.isMeshLoaded() << endl;
  cout << "isWorldPoseAvailable: "<< m.isWorldPoseAvailable() << endl;
  cout << "nVertices: " << m.getVertices().cols() << endl;
  cout << "nFaces   : " << m.getFaces().size() << endl;
  renderer.addMesh( &m );

  MatrixXd V = m.getVertices();
  MatrixXd w_T_ob = m.getObjectWorldPose();
  MatrixXd w_V = w_T_ob * V;
  cout << "V:\n" << V << endl;
  cout << "w_T_ob:\n" << w_T_ob << endl;
  cout << "w_V:\n" << w_V << endl;


  // Load Nodes
  vector<Node*> all_nodes;
  for( int i=0 ; i<1015 ; i++)
  {
    Node* ni = new Node();
    char node_fname[1000];
    sprintf( node_fname, "%s/node_%06d", base.c_str(), i);
    ni->load_debug_xml( node_fname );
    all_nodes.push_back( ni );
  }

  for( int i=0 ; i<1015; i++ )
  {
    cout << i << endl;
    Matrix4d w_T_c;
    if( all_nodes[i]->getPathPose(w_T_c,0) )
    {
      renderer.render( all_nodes[i]->getImageRef(), w_T_c );
      cv::imshow( "x", all_nodes[i]->getImageRef() );

      cout << (char) cv::waitKey(0) << endl;
    }
  }





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


  // deallocate nodes
  for( int i=0 ; i<all_nodes.size() ; i++ )
  {
    delete all_nodes[i];
  }


  return 0;
}
