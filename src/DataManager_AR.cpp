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
        nNodes[i_]->setPathPose( msg->poses[i].pose, 1 );
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
        nNodes[i_]->setPathPose( msg->poses[i].pose, 0 );
      }
  }
}


void DataManager::mesh_pose_callback( const geometry_msgs::PoseStamped& msg )
{
  ROS_INFO_STREAM( "+        XXXXX mesh_pose_callback() " );


}
///////////////////////////// END AR Callbacks //////////////////////////////////
