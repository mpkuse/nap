// Functions and callbacks for AR rendering.

#include "DataManager.h"


///////////////////////////// AR Callbacks //////////////////////////////////
void DataManager::path_vio_callback( const nav_msgs::Path::ConstPtr& msg )
{
  cout << "+    path_vio_callback()     " <<  msg->header.stamp << "size=" << msg->poses.size() << endl;
  //path from VIO (before incorporation of loopclosure)


  // loop thru all poses in path
  for( int i=0 ; i<msg->poses.size() ; i++ )
  {
    // for a pose, find associated node with timestamp
    int i_ = find_indexof_node(msg->poses[i].header.stamp);
    // cout << "+    [vio] " << "  n=" << i_ << " "<< msg->poses[i].header.stamp << endl;

      //    for this node do node->setPathPose(id=1)
      if( i_ >= 0 )
      {
        // nNodes[i_]->setPathPose( msg->poses[i].pose, 1 );
        nNodes[i_]->setPathPose( msg->poses[i].pose, 1, msg->poses[i].header.stamp );
      }
  }


}

void DataManager::path_posegraph_callback( const nav_msgs::Path::ConstPtr& msg ) //path after incorporation of loopclosure
{
  //path from VIO (before incorporation of loopclosure)
  //collect path and associate it with nodes if possible
  cout << "+    path_posegraph_callback() " <<  msg->header.stamp << "size=" << msg->poses.size() << endl;

  // loop thru all poses in path
  for( int i=0 ; i<msg->poses.size() ; i++ )
  {
    // for a pose, find associated node with timestamp
    int i_ = find_indexof_node(msg->poses[i].header.stamp);
    // cout << "+    [pose-graph-optimization] " << "  n=" << i_ << " "<< msg->poses[i].header.stamp << endl;

      //    for this node do node->setPathPose(id=1)
      if( i_ >= 0 )
      {
        // nNodes[i_]->setPathPose( msg->poses[i].pose, 0 );
        nNodes[i_]->setPathPose( msg->poses[i].pose, 0, msg->poses[i].header.stamp );
      }
  }
}

/*
void DataManager::mesh_pose_callback( const geometry_msgs::PoseStamped& msg )
{
  ROS_INFO_STREAM( "+        XXXXX mesh_pose_callback() for mesh "<< msg.header.frame_id  );

  string frame_id = msg.header.frame_id ;
  if( frame_id == "control_marker" )
  {
    make_movie();
    return;
  }

  // msg->pose --> pose_p, pose_q
  Vector3d pose_p;
  Quaterniond pose_q;
  pose_p = Vector3d(msg.pose.position.x, msg.pose.position.y,msg.pose.position.z );
  pose_q = Quaterniond( msg.pose.orientation.w, msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z );


  // pose_p, pose_q --> Matrix4d
  Matrix4d frame_pose;
  frame_pose = Matrix4d::Zero();
  frame_pose.col(3) << pose_p, 1.0;
  // Matrix3d R = e_q.toRotationMatrix();
  frame_pose.topLeftCorner<3,3>() = pose_q.toRotationMatrix();



  // search this mesh
  for( int i=0 ; i<nMeshes.size() ; i++ )
  {
    ROS_INFO_STREAM( "+       "<< i << ": " << nMeshes[i]->getMeshObjectName() );
    if( frame_id ==  nMeshes[i]->getMeshObjectName() )
    {
      ROS_INFO_STREAM( "            Found :}" << "set w_T_obj=" << frame_pose );
      nMeshes[i]->setObjectWorldPose( frame_pose );
      return;
    }
  }
  ROS_INFO( "mesh not Found :{");
  return;


}
*/
///////////////////////////// END AR Callbacks //////////////////////////////////

/*
void DataManager::add_new_meshobject( string obj_name )
{
  MeshObject * obj = new MeshObject(obj_name);
  nMeshes.push_back( obj );
}


void DataManager::threaded_keyboard_listener()
{
  while(1)
  {
      // cout << "Waiting to be pressed\n";
      char c = getchar();
      cout << "You pressed " << uint(c) << endl;
      if( c == 'a' )
      {
        cout << "Start Making an AR Movie\n";



        make_movie();


        break;
      }
  }

  cout << "Finished Making the Movie\n";
}

#define EIGEN_MAT_INFO( message, X ) cout << message << " rows:" << X.rows() << ", cols:" << X.cols() << endl;
void DataManager::make_movie()
{
  // Loop through each node.
  //    get cam position
  //    project object_i using w_T_{obj_i} and w_T_c. \forall i.
  for( int i=0 ; i<nNodes.size() ; i++ )
  {
    cout << "-- node="<< i << endl;
    // Camera Pose
    Matrix4d w_T_c;
    Matrix4d w_T_tilde_c; ///< Corrected
    bool status_1 = nNodes[i]->getPathPose( w_T_c, 1 );
    bool status_2 = nNodes[i]->getPathPose( w_T_tilde_c, 0 );
    EIGEN_MAT_INFO( "w_T_c", w_T_c );
    EIGEN_MAT_INFO( "w_T_tilde_c", w_T_tilde_c );

    if( status_1 == false or status_2 == false )
      continue;

    // Object Pose
    int j=0;
    Matrix4d w_T_ob;
    nMeshes[j]->getObjectWorldPose( w_T_ob );
    EIGEN_MAT_INFO( "w_T_ob", w_T_ob );

    // Object Pose in Camera frame
    Matrix4d c_T_ob;
    c_T_ob = w_T_c.inverse() * w_T_ob;
    EIGEN_MAT_INFO( "c_T_ob", c_T_ob );


    // Project On camera
    MatrixXd out_pts;
    camera.perspectiveProject3DPoints( nMeshes[j]->getVertices(), out_pts );
    cout<< out_pts << endl;

  }
}
*/
