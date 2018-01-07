#include <iostream>
#include <thread>
using namespace std;

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Path.h>

#include "DataManager.h"
#include "SceneRenderer.h"

#define _DEBUG_FOLDER "/home/mpkuse/Desktop/a/drag_posecompute_node/"

vector<Node*> all_nodes;
SceneRenderer renderer;

int _NOW_IM_IDX = 25;


void render_current_image()
{
  // Render 25th Frame
  Matrix4d w_T_c;
  if( all_nodes[_NOW_IM_IDX]->getPathPose(w_T_c,0) )
  {
    renderer.render( all_nodes[_NOW_IM_IDX]->getImageRef(), w_T_c );
    cv::imshow( "x", all_nodes[_NOW_IM_IDX]->getImageRef() );

  }
}

void rosmsg_2_matrix4d( const geometry_msgs::Pose& pose, Matrix4d& frame_pose )
{
  Vector3d pose_p;
  Quaterniond pose_q;
  pose_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
  pose_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );


  // pose_p, pose_q --> Matrix4d
  frame_pose = Matrix4d::Zero();
  frame_pose.col(3) << pose_p, 1.0;
  // Matrix3d R = e_q.toRotationMatrix();
  frame_pose.topLeftCorner<3,3>() = pose_q.toRotationMatrix();

}


void matrix4d_2_rosmsg( const Matrix4d& frame_pose, geometry_msgs::Pose& pose )
{
  Matrix3d R = frame_pose.topLeftCorner<3,3>();
  Quaterniond qua = Quaterniond(R);
  pose.position.x = frame_pose(0,3);
  pose.position.y = frame_pose(1,3);
  pose.position.z = frame_pose(2,3);

  pose.orientation.w = qua.w();
  pose.orientation.x = qua.x();
  pose.orientation.y = qua.y();
  pose.orientation.z = qua.z();

}


/// This will update the mesh-pose upon receiving the message from `interactive_marker_server`
void mesh_pose_callback( const geometry_msgs::PoseStamped& msg )
{
  ROS_INFO_STREAM( "+        XXXXX mesh_pose_callback() for mesh "<< msg.header.frame_id  );


  string frame_id = msg.header.frame_id ;
  if( frame_id == "control_marker" )
  {

    return;
  }

  // msg->pose --> pose_p, pose_q
  Matrix4d w_T_ob;
  rosmsg_2_matrix4d( msg.pose, w_T_ob );
  cout << "Recvd Pose (w_T_{" << frame_id << "}):\n" << w_T_ob << endl;



  // search this mesh
  for( int i=0 ; i<renderer.getMeshCount() ; i++ )
  {
    if( frame_id ==  renderer.getMeshName(i) )
    {
      // ROS_INFO_STREAM( "            Found :}" << "set w_T_obj=" << frame_pose );
      cout << "Found "<< frame_id << "; Setting w_T_obj" << endl;
      (renderer.getMeshObject(i))->setObjectWorldPose( w_T_ob );
      return;
    }
  }
  ROS_INFO( "mesh not Found :{");
  return;


}


void publish_path( ros::Publisher pub, const vector<Node*> p )
{
  nav_msgs::Path path;
  path.header.frame_id = "world";

  for( int i=0 ; i<p.size() ; i++ )
  {
    Matrix4d w_T_c;
    if( p[i]->getPathPose( w_T_c, 0 ) ) {
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = "world";
      pose_stamped.header.stamp = p[i]->getPathPoseTimeStamp(0);
      matrix4d_2_rosmsg( w_T_c, pose_stamped.pose );

      path.poses.push_back( pose_stamped );
    }

  }
  pub.publish( path );
}

int main(int argc, char ** argv )
{
  ros::init( argc, argv, "talker" );
  ros::NodeHandle n;

  string base = string( _DEBUG_FOLDER );

  // Renderer
  renderer = SceneRenderer();

  // Load Camera Info
  PinholeCamera camera = PinholeCamera( base+"/pinhole_camera.yaml" );
  renderer.setCamera( &camera );

  // Load Meshes
  MeshObject m1 = MeshObject( "1.obj");
  renderer.addMesh( &m1 );

  MeshObject m2 = MeshObject( "chair.obj" );
  renderer.addMesh( &m2 );

  MeshObject m3 = MeshObject( "simple_car.obj" );
  renderer.addMesh( &m3 );


  // Load All Nodes
  cout << "--- Loading Nodes ---" << endl;
  for( int i=0 ; i<5000 ; i++)
  {
    Node* ni = new Node();
    char node_fname[1000];
    sprintf( node_fname, "%s/node_%06d", base.c_str(), i);
    if( i<10 )
      cout << node_fname << endl;

    if( ! ni->load_debug_xml( node_fname ) )
    {
      cout << "..\n..\n..\n"<< node_fname << endl;
      break;
    }
    all_nodes.push_back( ni );
  }
  cout << "Loaded "<< all_nodes.size() << " nodes\n";





  ros::Publisher pub_path = n.advertise<nav_msgs::Path>("/test_ar/path1", 1000 );
  ros::Subscriber sub_mesh_pose = n.subscribe( "/object_mesh_pose", 1000, mesh_pose_callback );

  cout << "Hello World\n";

  ros::Rate rate(10);
  int count = 0;
  bool _firsttime = true;
  while( ros::ok() )
  {
    render_current_image();
    if( _firsttime )
    {
      publish_path( pub_path, all_nodes );
      if( count > 10 )
        _firsttime = false;
    }
    ros::spinOnce();

    char waitKeyInputChar = (char) cv::waitKey(5);
    // cout << waitKeyInputChar << endl;
    if( waitKeyInputChar == 'a' )
      _NOW_IM_IDX++;
    if( waitKeyInputChar == 'z' )
      _NOW_IM_IDX--;




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
