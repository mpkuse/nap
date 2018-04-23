#include <iostream>
#include <thread>
using namespace std;

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Path.h>

#include "DataManager.h"
#include "SceneRenderer.h"



vector<Node*> all_nodes;
vector<Node*> all_nodesA;
vector<Node*> all_nodesB;
vector<Node*> all_nodesC;
vector<Node*> all_nodesD;
SceneRenderer renderer;

int _NOW_IM_IDX = 25;

void load_node_set( vector<Node*>& node_set, const string& node_set_base )
{
  node_set.clear();
  cout << "--- Loading Nodes ---" << endl;
  for( int i=0 ; i<5000 ; i++)
  {
    Node* ni = new Node();
    char node_fname[1000];
    sprintf( node_fname, "%s/node_%06d", node_set_base.c_str(), i);
    if( i<5 )
      cout << node_fname << endl;

    if( ! ni->load_debug_xml( node_fname ) )
    {
      cout << "..\n..\n..\n"<< node_fname << endl;
      break;
    }
    node_set.push_back( ni );
  }
  cout << "Loaded "<< node_set.size() << " nodes\n";
}

void deallocate_node_set( vector<Node*>& node_set )
{
  cout << "Deallocate node_set of size: "<< node_set.size() << endl;
  for( int i=0 ; i<node_set.size() ; i++ )
  {
    delete node_set[i];
  }
  node_set.clear();
}

int serial_input_2_int( const vector<int>& serial_input  )
{
  int len = serial_input.size();
  int sum = 0;
  for( int i = 0 ; i< len ; i++ )
  {
    sum = 10*sum + serial_input[i];
  }
  return sum;
  // sum += serial_input[len-1];

}

void render_current_image( Node * this_node, const string& window_name, int id )
{
  // Render 25th Frame
  Matrix4d w_T_c;
  // if( all_nodes[_NOW_IM_IDX]->getPathPose(w_T_c,0) )
  // {
  //   renderer.render( all_nodes[_NOW_IM_IDX]->getImageRef(), w_T_c );
  //   cv::imshow( "x", all_nodes[_NOW_IM_IDX]->getImageRef() );
  // }

  if( this_node->getPathPose(w_T_c,id) )
  {
    cv::Mat out;
    renderer.renderIn( this_node->getImageRef(), w_T_c, out );
    cv::imshow( (window_name+string("frame")).c_str(), out );
  }

  // if( this_node->getPathPose(w_T_c,1) && )
  // {
  //   cv::Mat out;
  //   renderer.renderIn( this_node->getImageRef(), w_T_c, out );
  //   cv::imshow( (window_name+string("vio")).c_str(), out );
  // }


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


void publish_path( ros::Publisher pub, const vector<Node*> p, int start, int end, int id )
{
  if( id == 0 || id == 1 )
    ;
  else
    {
      ROS_ERROR_STREAM( "id has to be 0 or 1");
      return;
    }
  if( start < 0 )
    start = 0;
  if( end < 0 )
    end = p.size();

  nav_msgs::Path path;
  path.header.frame_id = "world";

  // for( int i=0 ; i<p.size() ; i++ )
  for( int i=start ; i<end ; i++ )
  {
    Matrix4d w_T_c;
    if( p[i]->getPathPose( w_T_c, id ) ) {
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = "world";
      pose_stamped.header.stamp = p[i]->getPathPoseTimeStamp(id);
      matrix4d_2_rosmsg( w_T_c, pose_stamped.pose );

      path.poses.push_back( pose_stamped );
    }

  }
  pub.publish( path );
}


// #define _DEBUG_FOLDER "/home/mpkuse/Desktop/a/drag_posecompute_node/"
#define _DEBUG_FOLDER "/home/mpkuse/Desktop/a/dump/"
int main(int argc, char ** argv )
{
  ros::init( argc, argv, "talker" );
  ros::NodeHandle n;

  string base = string( _DEBUG_FOLDER );

  // Renderer
  renderer = SceneRenderer();

  // Load Camera Info
  // PinholeCamera camera = PinholeCamera( base+"/pinhole_camera.yaml" );
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
  // load_node_set( all_nodes, base  );
  load_node_set( all_nodesA, base+string("A")  );
  load_node_set( all_nodesB, base+string("B")  );
  load_node_set( all_nodesC, base+string("C")  );
  load_node_set( all_nodesD, base+string("D")  );

/*
  for( int i=0 ; i<all_nodesA.size() ; i++ )
  {
    cout << "---\n";
    cout << i << " " << all_nodesA[i]->getImageTimeStamp() << endl;
    cout << i << " " << all_nodesB[i]->getImageTimeStamp() << endl;
    cout << i << " " << all_nodesC[i]->getImageTimeStamp() << endl;
    cout << i << " " << all_nodesD[i]->getImageTimeStamp() << endl;
    cv::imshow( "A", all_nodesA[i]->getImageRef() );
    cv::imshow( "B", all_nodesB[i]->getImageRef() );
    cv::imshow( "C", all_nodesC[i]->getImageRef() );
    cv::imshow( "D", all_nodesD[i]->getImageRef() );
    cv::waitKey(0);
  //   cout << i << " " << all_nodes[i]->valid_image() << all_nodes[i]->valid_pathpose(0) << all_nodes[i]->valid_pathpose(1) << endl;
  }
  return 0;
*/


  ros::Publisher pub_path = n.advertise<nav_msgs::Path>("/test_ar/path1", 1000 );
  ros::Publisher pub_path_VIO = n.advertise<nav_msgs::Path>("/test_ar/vio", 1000 );
  ros::Publisher pub_pathA = n.advertise<nav_msgs::Path>("/test_ar/pathA", 1000 );
  ros::Publisher pub_pathB = n.advertise<nav_msgs::Path>("/test_ar/pathB", 1000 );
  ros::Publisher pub_pathC = n.advertise<nav_msgs::Path>("/test_ar/pathC", 1000 );
  ros::Publisher pub_pathD = n.advertise<nav_msgs::Path>("/test_ar/pathD", 1000 );
  ros::Subscriber sub_mesh_pose = n.subscribe( "/object_mesh_pose", 1000, mesh_pose_callback );

  cout << "Start AR Demo\n";

  ros::Rate rate(10);
  int count = 0;
  bool _firsttime = true;
  vector<int> serial_input;
  while( ros::ok() )
  {
    // render_current_image();
    // set_current_image_position_marker();
    render_current_image( all_nodesA[_NOW_IM_IDX], string("VIO"), 1 );
    render_current_image( all_nodesA[_NOW_IM_IDX], string("A"), 0 );
    render_current_image( all_nodesB[_NOW_IM_IDX], string("B"), 0 );
    render_current_image( all_nodesC[_NOW_IM_IDX], string("C"), 0 );
    render_current_image( all_nodesD[_NOW_IM_IDX], string("D"), 0 );
    // cout << count << endl;

    if( _firsttime )
    {
      // publish_path( pub_path, all_nodes );
      publish_path( pub_path_VIO, all_nodesA, 0, _NOW_IM_IDX, 1 );
      publish_path( pub_pathA, all_nodesA, 0, _NOW_IM_IDX, 0 );
      publish_path( pub_pathB, all_nodesB, 0, _NOW_IM_IDX, 0 );
      publish_path( pub_pathC, all_nodesC, 0, _NOW_IM_IDX, 0 );
      publish_path( pub_pathD, all_nodesD, 0, _NOW_IM_IDX, 0 );

      // if( count > 10 )
        // _firsttime = false;
    }
    ros::spinOnce();


    cv::Mat zzy = cv::Mat( 10, 10, CV_8UC1, cv::Scalar(0) );
    cv::imshow( "zzy", zzy );
    // waitKey
    {
    char waitKeyInputChar = (char) cv::waitKey(5);
    // cout << waitKeyInputChar << endl;
    if( waitKeyInputChar == 'a' )
      _NOW_IM_IDX++;
    if( waitKeyInputChar == 'z' )
      _NOW_IM_IDX--;
    if( waitKeyInputChar == 'q' )
      break;

    if( waitKeyInputChar >= '0' && waitKeyInputChar <= '9' ) {
      serial_input.push_back( (int) (waitKeyInputChar - '0') );
      ROS_WARN_STREAM( "Detected number: " << (char)waitKeyInputChar << "|array|="<<serial_input.size() );
    }
    if( waitKeyInputChar == 'g' ) {
      int s = serial_input_2_int( serial_input );
      ROS_WARN_STREAM( "GOTO " << s );
      if( s>=0 && s<all_nodesA.size() )
        _NOW_IM_IDX = s;
      serial_input.clear();
    }
    }




    rate.sleep();
    count++;
  }


  // deallocate nodes
  deallocate_node_set( all_nodes );
  deallocate_node_set( all_nodesA );
  deallocate_node_set( all_nodesB );
  deallocate_node_set( all_nodesC );
  deallocate_node_set( all_nodesD );



  return 0;
}
