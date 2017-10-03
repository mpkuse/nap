#include "DataManager.h"

DataManager::DataManager(ros::NodeHandle &nh )
{
    this->nh = nh;
}



void DataManager::setCamera( PinholeCamera& camera )
{
  this->camera = camera;

  cout << "--- Camera Params from DataManager ---\n";
  cout << "K\n" << this->camera.e_K << endl;
  cout << "D\n" << this->camera.e_D << endl;
  cout << "--- END\n";
}

void DataManager::setVisualizationTopic( string rviz_topic )
{
  // usually was "/mish/pose_nodes"
  pub_pgraph = nh.advertise<visualization_msgs::Marker>( rviz_topic.c_str(), 0 );
}


DataManager::~DataManager()
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




void DataManager::image_callback( const sensor_msgs::ImageConstPtr& msg )
{
  // Search for the timestamp in pose-graph
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

  // if the timestamp was not found in pose-graph,
  // buffer this image in queue
  if( i_ < 0 )
  {
    // unclaimed_im.push( image.clone() );
    unclaimed_im.push( image_resized.clone() );
    unclaimed_im_time.push( ros::Time(msg->header.stamp) );
    flush_unclaimed_im();
  }
  else //if found than associated the node with this image
  {
    // nNodes[i_]->setImage( msg->header.stamp, image );
    nNodes[i_]->setImage( msg->header.stamp, image_resized );
  }

}


void DataManager::point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg )
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


void DataManager::camera_pose_callback( const nav_msgs::Odometry::ConstPtr msg )
{
  Node * n = new Node(msg->header.stamp, msg->pose.pose);
  nNodes.push_back( n );
  ROS_DEBUG( "Recvd msg - camera_pose_callback");


  // ALSO add odometry edges to 1 previous.
  int N = nNodes.size();
  int prev_k = 1; //TODO: This could be a parameter.
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

    // add relative transform as edge-inferred (using poses from corresponding edges)
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


void DataManager::place_recog_callback( const nap::NapMsg::ConstPtr& msg  )
{
  ROS_INFO( "Received - NapMsg");
  // cout << msg->c_timestamp << " " << msg->prev_timestamp << endl;

  assert( this->camera.isValid() );

  //
  // Look it up in nodes list (iterate over nodelist)
  int i_curr = find_indexof_node(msg->c_timestamp);
  int i_prev = find_indexof_node(msg->prev_timestamp);

  cout << i_curr << "<-->" << i_prev << endl;
  cout <<  msg->c_timestamp-nNodes[0]->time_stamp << "<-->" << msg->prev_timestamp-nNodes[0]->time_stamp << endl;
  cout << "Last Node timestamp : "<< nNodes.back()->time_stamp - nNodes[0]->time_stamp << endl;
  if( i_curr < 0 || i_prev < 0 )
    return;

  //
  // make a loop closure edges
  Edge * e = new Edge( nNodes[i_curr], i_curr, nNodes[i_prev], i_prev, EDGE_TYPE_LOOP_CLOSURE );
  e->setEdgeTimeStamps(msg->c_timestamp, msg->prev_timestamp);

}





// Loop over each node and return the index of the node which is clossest to the specified stamp
int DataManager::find_indexof_node( ros::Time stamp )
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
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
  return -1;
}


void DataManager::flush_unclaimed_im()
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


void DataManager::flush_unclaimed_pt_cld()
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
