#include "DataManager.h"
#include "EdgeManager.h"
// Core functions to subscribe to messages and build the pose graph

DataManager::DataManager(ros::NodeHandle &nh )
{
    this->nh = nh;

    // init republish colocation topic
    pub_chatter_colocation = this->nh.advertise<nap::NapMsg>( "/colocation_chatter", 1000 );



    tfidf = new Feature3dInvertedIndex();



}



DataManager::DataManager(const DataManager &obj) {
   cout << "Copy constructor allocating ptr." << endl;

}

void DataManager::setCamera( const PinholeCamera& camera )
{
  this->camera = camera;

  cout << "--- Camera Params from DataManager ---\n";
  cout << "K\n" << this->camera.e_K << endl;
  cout << "D\n" << this->camera.e_D << endl;
  cout << "--- END\n";
}

void DataManager::setVisualizationTopics( string rviz_topic )
{
  // usually was "/mish/pose_nodes"
    pub_pgraph = nh.advertise<visualization_msgs::Marker>( rviz_topic.c_str(), 0 );
    pub_pgraph_org = nh.advertise<visualization_msgs::Marker>( (rviz_topic+string("_original")).c_str(), 0 );

    pub_bundle = nh.advertise<visualization_msgs::Marker>( (rviz_topic+string("_bundle_opmode28")).c_str(), 0 );

    pub_3dpoints = nh.advertise<visualization_msgs::Marker>( (rviz_topic+string("_3dpoints_analysis")).c_str(), 0 );

    pub_place_recog_status_image = nh.advertise<sensor_msgs::Image>("place_recog_callback_status_image",1000);

}

void DataManager::setOpmodesToProcess( const vector<int>& _enabled_opmode ) {
    enabled_opmode = _enabled_opmode;
}


DataManager::~DataManager()
{
  cout << "In ~DataManager\n";

// This is one way to write the data. But a better way is to use public methods,
// DataManager::getNodesRef() and DataManager::getTFIDFRef() in main() and write-out
// whatever data you need. See example in the main().

/*
// make sure this folder exisits.!
#define _DEBUG_SAVE_BASE_PATH "/home/mpkuse/Desktop/a/drag_posecompute_node/"

#ifdef _DEBUG_POSEGRAPH_2_FILE
  // string base_path = string( "/home/mpkuse/Desktop/a/drag/" );
  string base_path = string( _DEBUG_SAVE_BASE_PATH );
  // string base_path = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/";

  // string file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.nodes.csv";
  string file_name = base_path + "/pose_graph.nodes.csv";
  ofstream fp_nodes;
  fp_nodes.open( file_name );
  cout << "Write file (" <<  file_name << ") with " << nNodes.size() << " entries\n";


  fp_nodes << "#i, t, x, y, z, q.x, q.y, q.z, q.w\n";
  for( int i=0 ; i<nNodes.size() ; i++ )
  {
    Node * n = nNodes[i];

    fp_nodes <<  i << ", " << n->time_stamp  << endl;
              // << e_p[0] << ", " << e_p[1] << ", "<< e_p[2] << ", "
              // << e_q.x() << ", "<< e_q.y() << ", "<< e_q.z() << ", "<< e_q.w() << endl;
  }
  fp_nodes.close();


  // Write Odometry Edges
  // file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.odomedges.csv";
  file_name = base_path + "/pose_graph.odomedges.csv";
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
  // file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.loopedges.csv";
  file_name = base_path + "/pose_graph.loopedges.csv";
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
#endif
*/

}

void DataManager::raw_nap_cluster_assgn_callback( const sensor_msgs::ImageConstPtr& msg )
{
  cout << "clu_assgn rcvd : " << msg->header.stamp << endl;
  int i_ = find_indexof_node(msg->header.stamp);
  cv::Mat clustermap;

  try {
    clustermap = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::MONO8 )->image;
  }
  catch( cv_bridge::Exception& e)
  {
    ROS_ERROR( "cv_bridge exception in raw_nap_cluster_assgn_callback(): %s", e.what() );
  }

  cout << "nap_clusters i_=" << i_ << "     napmap_buffer="<< unclaimed_napmap.size() << endl;
  // if this i_ is found in the pose-graph set()
  if( i_ < 0 )
  {
    unclaimed_napmap.push( clustermap.clone() );
    unclaimed_napmap_time.push( ros::Time(msg->header.stamp) );
    flush_unclaimed_napmap();

    //TODO: Code up the buffering part of nap clusters. For now, you don't need to as
    // this is garunteed to be a bit delayed due to needed computation time
  }
  else //if found than associated the node with this image
  {
    nNodes[i_]->setNapClusterMap( msg->header.stamp, clustermap );
  }


}


void DataManager::image_callback( const sensor_msgs::ImageConstPtr& msg )
{
  // Search for the timestamp in pose-graph
  int i_ = find_indexof_node(msg->header.stamp, true);
  ROS_DEBUG_STREAM( "\t\timage_callback: " << msg->header.stamp << " find_indexof_node: " << i_ );

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

// TODO: Consider complete removal of this subscription. the 3d point clouds from vins_estimator now
// contains everything including global id.
void DataManager::tracked_features_callback( const sensor_msgs::PointCloudConstPtr& msg )
{
  // ROS_INFO( 'Received2d:: Features: %d', (int)msg->points.size() );
  // ROS_INFO( "Received2d");
  int i_ = find_indexof_node(msg->header.stamp);
  cout << "stamp2d : " << msg->header.stamp << endl;
  cout << "Received2d:: Node:"<< i_ <<  " size=" << msg->points.size() << endl;

  // if i_ < 0 : Node not found for this timestamp. Buffer points
  if( i_ < 0 )
  {
    Matrix<double,3,Dynamic> tracked_2d_features;
    tracked_2d_features = Matrix<double,3,Dynamic>(3,msg->points.size()); //in homogeneous co-ords. Qin Tong publishes features points in homogeneous cords
    for( int i=0 ; i<msg->points.size() ; i++ )
    {
      tracked_2d_features(0,i) = msg->points[i].x; //x
      tracked_2d_features(1,i) = msg->points[i].y; //y
      tracked_2d_features(2,i) = msg->points[i].z; //1.0
    }

    //push to buffer
    unclaimed_2d_feat.push( tracked_2d_features );
    unclaimed_2d_feat_time.push( msg->header.stamp );
    flush_unclaimed_2d_feat();
  }
  else // if i_> 0 : Found node for this. Associate these points with a node
  {
    nNodes[i_]->setFeatures2dHomogeneous( msg->header.stamp, msg->points  );
  }


}


void DataManager::point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg )
{
  int i_ = find_indexof_node(msg->header.stamp);
  // cout << "stamp3d : " << msg->header.stamp << endl;
  ROS_DEBUG_STREAM( "\tpoint_cloud_callback: " <<  msg->header.stamp << " find_indexof_node: " << i_ );
  // assert( msg->channels.size() == msg->points.size() && msg->points.size() > 0  );

  if( i_ < 0 )
  {
    // 1.1 msg->points to eigen matrix
    // Matrix<double,3,Dynamic> ptCld;
    MatrixXd _ptCld;
    assert( msg->points.size() > 0  );
    _ptCld = MatrixXd::Zero(4,msg->points.size());
    for( int i=0 ; i<msg->points.size() ; i++ )
    {
      _ptCld(0,i) = msg->points[i].x;
      _ptCld(1,i) = msg->points[i].y;
      _ptCld(2,i) = msg->points[i].z;
      _ptCld(3,i) = 1.0;
    }


    {// In the commit `6d1bb531d02fc37187b966da8245a4f47b1d6ba3` of vins_testbed.
    // IN previous versions there were only 4 channels.
    // there will be 5 channels. ch[0]: un, ch[1]: vn,  ch[2]: u, ch[3]: v.  ch[4]: globalid of the feature.
    // cout << "\tpoints.size() : "<< msg->points.size(); // this will be N (say 92)
    // cout << "\tchannels.size() : "<< msg->channels.size(); //this will be N (say 92)
    // cout << "\tchannels[0].size() : "<< msg->channels[0].values.size(); //this will be 5.
    // cout << "\n";

    // An Example Keypoint msg
    {
        // ---
        // header:
        //   seq: 40
        //   stamp:
        //     secs: 1523613562
        //     nsecs: 530859947
        //   frame_id: world
        // points:
        //   -
        //     x: -7.59081602097
        //     y: 7.11367511749
        //     z: 2.85602664948
        //   .
        //   .
        //   .
        //   -
        //     x: -2.64935922623
        //     y: 0.853760659695
        //     z: 0.796766400337
        // channels:
        //   -
        //     name: ''
        //     values: [-0.06108921766281128, 0.02294199913740158, 310.8721618652344, 260.105712890625, 2.0]
        //     .
        //     .
        //     .
        //   -
        //     name: ''
        //     values: [-0.47983112931251526, 0.8081198334693909, 218.95481872558594, 435.47357177734375, 654.0]
        //   -
        //     name: ''
        //     values: [0.07728647440671921, 1.0073764324188232, 344.2176208496094, 473.7791442871094, 660.0]
        //   -
        //     name: ''
        //     values: [-0.6801641583442688, 0.10506453365087509, 159.75746154785156, 279.6077575683594, 663.0]
    }
    }

    // 1.2 msg->channels[4] --> gid of every pt
    assert( msg->channels.size() == msg->points.size() && msg->channels[0].values.size() == 5 );
    VectorXi _ptCld_id = VectorXi::Constant( msg->points.size(), -1 );
    for( int i=0 ; i<msg->channels.size() ; i++ )
    {
        _ptCld_id(i) = (int)msg->channels[i].values[4];
    }

    // 1.3 msg->channels[0,1] and msg->channels[2,3] --> unvn and uv of every pt (imaged points)
    MatrixXd _ptCld_unvn = MatrixXd::Zero( 3, msg->points.size() ); // 3xN
    MatrixXd _ptCld_uv   = MatrixXd::Zero( 3, msg->points.size() ); // 3xN
    for( int i=0 ; i<msg->points.size() ; i++ )
    {
        _ptCld_unvn(0,i) = msg->channels[i].values[0];
        _ptCld_unvn(1,i) = msg->channels[i].values[1];
        _ptCld_unvn(2,i) = 1.0;
        _ptCld_uv(0,i) = msg->channels[i].values[2];
        _ptCld_uv(1,i) = msg->channels[i].values[3];
        _ptCld_uv(2,i) = 1.0;
    }


    // 2. Put this eigen matrix to queue
    unclaimed_pt_cld.push( _ptCld );
    unclaimed_pt_cld_unvn.push( _ptCld_unvn );
    unclaimed_pt_cld_uv.push( _ptCld_uv );
    unclaimed_pt_cld_time.push( msg->header.stamp );
    unclaimed_pt_cld_globalid.push( _ptCld_id );
    flush_unclaimed_pt_cld();
  }
  else
  {
    // Corresponding node exist
    // nNodes[i_]->setPointCloud( msg->header.stamp, msg->points );
    nNodes[i_]->setPointCloud( msg->header.stamp, msg->points, msg->channels ); //also stores the global id of each of the 3d points


    // Add all the 3d points to inverted index
    assert( msg->channels.size() == msg->points.size() && msg->channels[0].values.size() == 5 );
    for( int i=0 ; i<msg->points.size() ; i++ )
    {
        int _gid = (int)msg->channels[i].values[4];
        Vector4d _3dpt;
        _3dpt << msg->points[i].x, msg->points[i].y, msg->points[i].z, 1.0;

        Vector3d _unvn, _uv;
        _unvn << msg->channels[i].values[0], msg->channels[i].values[1], 1.0;
        _uv   << msg->channels[i].values[2], msg->channels[i].values[3], 1.0;

        // also add _unvn, _uv to tfidf
        tfidf->add( _gid, _3dpt, _unvn, _uv, i_ );
        // tfidf->add( _gid, _3dpt, i_ );
    }


    {
    // cout << "\tOKpoints.size() : "<< msg->points.size();
    // cout << "\tOKchannels.size() : "<< msg->channels.size();
    // cout << "\tOKchannels[0].size() : "<< msg->channels[0].values.size();
    // cout << "\n";
    }

  }

}


void DataManager::camera_pose_callback( const nav_msgs::Odometry::ConstPtr msg )
{
  Node * n = new Node(msg->header.stamp, msg->pose.pose);
  getNodesLock();
  nNodes.push_back( n );
  getNodesUnlock();
  // ROS_DEBUG( "camera_pose_callback msg - camera_pose_callback");
  // cout << "add-node : " << msg->header.stamp << endl;
  ROS_DEBUG_STREAM( "camera_pose_callback add-node : " << msg->header.stamp );


  // ALSO add odometry edges to 1 previous.
  int N = getNodesSize(); //nNodes.size();
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

  // ROS_INFO_STREAM( "Received - NapMsg - " << msg->op_mode  );
  // cout << msg->c_timestamp << " " << msg->prev_timestamp << endl;

  assert( this->camera.isValid() );

  //
  // Look it up in nodes list (iterate over nodelist)
  int i_curr = find_indexof_node(msg->c_timestamp);
  int i_prev = find_indexof_node(msg->prev_timestamp);

  // cout << i_curr << "<-->" << i_prev << endl;
  if( i_curr < 0 || i_prev < 0 )
  {
      ROS_ERROR( "DataManager::place_recog_callback cannot find nodes pointed by napmsg, ignore this nap-msg");
      ROS_INFO( "DataManager::place_recog_callback cannot find nodes pointed by napmsg, ignore this nap-msg");
      return;
  }


  /*
  vector<int> enabled_opmode;
  // enabled_opmode.push_back(10);
  // enabled_opmode.push_back(29);
  // enabled_opmode.push_back(20);
  // enabled_opmode.push_back(18);
  // enabled_opmode.push_back(28);
  enabled_opmode.push_back(17);
  */
  assert( enabled_opmode.size() > 0 );
  // ROS_INFO( "nNodes (in global_nodes vector) = %d", nNodes.size() );


  if( std::find(enabled_opmode.begin(), enabled_opmode.end(),  (int)msg->op_mode  ) != enabled_opmode.end() )
  {
      // found the item
      // OK! let this be processed.
      ROS_DEBUG( "Process napmsg (op_mode=%d)", msg->op_mode );
    //   ROS_INFO( "Process napmsg (op_mode=%d)", msg->op_mode );
  }
  else
  {
      ROS_INFO( "Ignore napmsg (op_mode=%d) as commanded by flags", msg->op_mode );
      return;
  }

  string __msg = to_string(i_curr)+"<-->"+to_string(i_prev)+"  opmode="+to_string(msg->op_mode);
  publish_text_as_image( pub_place_recog_status_image,  __msg );


  //
  // make a loop closure edge
  Edge * e = new Edge( nNodes[i_curr], i_curr, nNodes[i_prev], i_prev, EDGE_TYPE_LOOP_CLOSURE );
  e->setEdgeTimeStamps(msg->c_timestamp, msg->prev_timestamp);

  ///////////////////////////////////
  // Relative Pose Computation     //
  ///////////////////////////////////
  // cout << "n_sparse_matches : " << msg->n_sparse_matches << endl;
  // cout << "co-ordinate match sizes : " << msg->curr.size() << " " << msg->prev.size() << " " << msg->curr_m.size() << endl;

  // //////////////////
  //---- case-a : If 3way matching is empty : do ordinary way to compute relative pose. Borrow code from Qin. Basically using 3d points from odometry (wrt curr) and having known same points in prev do pnp
  // /////////////////
  // Use the point cloud (wrt curr) and do PnP using prev
  // if( msg->n_sparse_matches >= 200 )
  if( msg->op_mode == 10 )
  {
    ROS_INFO( "Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_BASIC");
    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_BASIC);

    // Don't do anything, just forward the match.
    // This contains just 2 timestamps. No other useful info.


    // Set the computed pose into edge
    // e->setEdgeRelPose( p_T_c );

    loopClosureEdges.push_back( e );

    // Re-publish op_mode:= 10 (as is)
    Matrix4d __h;
    int32_t mode = 10;
    // republish_nap( msg->c_timestamp, msg->prev_timestamp, __h, mode );
    republish_nap( msg );

    publish_text_as_image( pub_place_recog_status_image,  __msg +":republish_nap");


    return;
  }


  // //////////////////
  //---- case-b : If 3way matching is not empty : i) Triangulate curr-1 and curr. ii) pnp( 3d pts from (i) ,  prev )
  // //////////////////
  // Pose computation with 3way matching
  // if( msg->n_sparse_matches < 200 && msg->curr.size() > 0 && msg->curr.size() == msg->prev.size() && msg->curr.size() == msg->curr_m.size()  )
  /*
  if( msg->op_mode == 29 ) // this is now out of use. op_mode28 superseeds this.
  {
    ROS_INFO( "Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_3WAY");
    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_3WAY);

    // TODO: Relative pose from 3way matching
    Matrix4d p_T_c = Matrix4d::Identity();
    this->pose_from_3way_matching(msg, p_T_c );

    // Set the computed pose into edge
    e->setEdgeRelPose( p_T_c );

    loopClosureEdges.push_back( e );
    // doOptimization();

    // Re-publish with pose, op_mode:=30
    int32_t mode = 30;
    republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c, mode );


    return;
  }
  */

  if( msg->op_mode == 20 )
  {
    // This is when the expanded matches are present. basically need to just forward this. No geometry computation here.
    ROS_INFO( "Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_GUIDED");

    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_GUIDED);
    loopClosureEdges.push_back( e );

    // Re-publish op_mode:= 10 (as is)
    // Matrix4d __h;
    // int32_t mode = 10;
    // republish_nap( msg->c_timestamp, msg->prev_timestamp, __h, mode );
    republish_nap( msg );
    publish_text_as_image( pub_place_recog_status_image,  __msg +":republish_nap");


    return;

  }

  // Uses class Corvus.
  // this is like opmode18. Also uses the Corvus class but the constructor is different. opmode 17 contains global_idx of the point features. This was not present in opmode18
  // Idea is to use 3d points from tfidf (inverted index). Note that inverted index (along with mean and variance
  // of each 3d point) is also maintained in DataManager which can be a source of robust 3d points.
  if( msg->op_mode == 17 )
  {
        ROS_INFO( "opmode17. Guided match has 3d points and 2d points for this loopmsg. But will use robust 3d points from inverted index maintained internally" );
        cout << "+++++++++++++++++++++++++++++++++\n";
        Corvus cor( tfidf, msg, this->nNodes, this->camera );
        cor.setDebugOutputFolder( BASE__DUMP );

        if( !cor.isValid() )
        {
            publish_text_as_image( pub_place_recog_status_image,  __msg +":constructor_failed");
            return;
        }

        Matrix4d p_T_c_3dprev_2dcurr, p_T_c_2dprev_3dcurr, p_T_c_3dprev_3dcurr;
        bool status_3dprev_2dcurr, status_2dprev_3dcurr, status_3dprev_3dcurr;
        double weight_3dprev_2dcurr, weight_2dprev_3dcurr, weight_3dprev_3dcurr;

        // Using 3dpoints of prev and 2d points of curr
        if( true )
        {
            Matrix4d p_T_c;
            ceres::Solver::Summary summary;
            bool status = cor.computeRelPose_3dprev_2dcurr(p_T_c, summary);
            double weight = min( 1.0, log( summary.initial_cost / summary.final_cost ) / log(10.) );

            // Write in semi-global variables
            p_T_c_3dprev_2dcurr = p_T_c;
            status_3dprev_2dcurr = status;
            weight_3dprev_2dcurr = weight;
            if( status == false )
            {
                cout << "computeRelPose_3dprev_2dcurr Status : Reject\n";
            }


            e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_GUIDED);
            loopClosureEdges.push_back( e );


        }


        // Using 2d points of prev 3dpoints of curr
        if( true )
        {
            Matrix4d p_T_c;
            ceres::Solver::Summary summary;
            bool status = cor.computeRelPose_2dprev_3dcurr(p_T_c, summary);
            double weight = min( 1.0, log( summary.initial_cost / summary.final_cost ) / log(10.) );

            // Write Semi-global
            p_T_c_2dprev_3dcurr = p_T_c;
            status_2dprev_3dcurr = status;
            weight_2dprev_3dcurr = weight;

            if( status == false )
            {
                cout << "computeRelPose_2dprev_3dcurr Status : Reject\n";
            }


            e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_GUIDED);
            loopClosureEdges.push_back( e );


        }


        // 3d3d Align. use both sets of 3d points and align those.
        if( true )
        {
            Matrix4d p_T_c;
            ceres::Solver::Summary summary;
            bool status = cor.computeRelPose_3dprev_3dcurr(p_T_c, summary);
            double weight = min( 1.0, log( summary.initial_cost / summary.final_cost ) / log(10.) );

            // Write Semi-global
            p_T_c_3dprev_3dcurr = p_T_c;
            status_3dprev_3dcurr = status;
            weight_3dprev_3dcurr = weight;

            if( status == false )
            {
                cout << "computeRelPose_3dprev_3dcurr Status : Reject\n";
            }


            e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_GUIDED);
            loopClosureEdges.push_back( e );

        }


        // See if all the 3 estimates are consistent
        cout << endl;
        Vector4d F;
        F << 3.0, 3.0, 3.0, 1.0;
        cout << "[status="<< (status_3dprev_2dcurr?"True  ":"False ") << "]  ";
        cout << "[weight=" << std::fixed << std::setprecision(2) << weight_3dprev_2dcurr << "]\t";
        cout << "p_T_c_3dprev_2dcurr: " <<  (p_T_c_3dprev_2dcurr * F).transpose() << endl;

        cout << "[status="<< (status_2dprev_3dcurr?"True  ":"False ") << "]  ";
        cout << "[weight=" << std::fixed << std::setprecision(2) << weight_2dprev_3dcurr << "]\t";
        cout << "p_T_c_2dprev_3dcurr: " <<  (p_T_c_2dprev_3dcurr * F).transpose() << endl;

        cout << "[status="<< (status_3dprev_3dcurr?"True  ":"False ") << "]  ";
        cout << "[weight=" << std::fixed << std::setprecision(2) << weight_3dprev_3dcurr << "]\t";
        cout << "p_T_c_3dprev_3dcurr: " <<  (p_T_c_3dprev_3dcurr * F).transpose() << endl;

        // if only 1 was true => make weight 1/3
        // if only 2 was true => make weight 2/3
        double weight_multiplier = 0.0;
        weight_multiplier = ( status_3dprev_2dcurr )?weight_multiplier+0.33:weight_multiplier;
        weight_multiplier = ( status_2dprev_3dcurr )?weight_multiplier+0.33:weight_multiplier;
        weight_multiplier = ( status_3dprev_3dcurr )?weight_multiplier+0.33:weight_multiplier;


        // Publish pose as opmode30.
        if( status_3dprev_2dcurr && false ) {
            republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c_3dprev_2dcurr, 30, weight_multiplier* weight_3dprev_2dcurr );
        }

        if( status_2dprev_3dcurr && false ) {
            republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c_2dprev_3dcurr, 30, weight_multiplier *weight_2dprev_3dcurr );
        }

        if( status_3dprev_3dcurr ) {
            republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c_3dprev_3dcurr, 30, weight_multiplier * weight_3dprev_3dcurr );
        }


        char __ghgh__sf[100];
        snprintf( __ghgh__sf, 100, ":3dprev_2dcurr %d %2.2f:2dprev_3dcurr %d %2.2f:3dprev_3dcurr %d %2.2f", status_3dprev_2dcurr, weight_3dprev_2dcurr,
                                                                                                            status_2dprev_3dcurr, weight_3dprev_3dcurr,
                                                                                                            status_3dprev_3dcurr, weight_3dprev_3dcurr
                );
        publish_text_as_image( pub_place_recog_status_image,  __msg +string(__ghgh__sf) );

        cout << "Done...\n";

        return ;

  }


  // Uses class Corvus.
  //this is like opmode20, but doing my own pnp.
  if( msg->op_mode == 18  )
  {
    //   return;
      ROS_INFO( "opmode18. Guided match has 3d points and 2d points for this loopmsg" );

      // my own processing here to compute pose.

      // Compute pose from 3d, 2d.
      //TODO
      cout << "+++++++++++++++++++++++++++++++++++++++++\n";
      TicToc timing;

      timing.tic();
      Corvus cor( msg, this->nNodes, this->camera );
      cor.setDebugOutputFolder( BASE__DUMP );



      cout << "Constructor done in (ms): "<< timing.toc() << endl;
      if( !cor.isValid() )
        return;



      #if CORVUS_DEBUG_LVL >= 1
      cor.publishPoints3d( pub_bundle );
      #endif


      //
      // relative Pose computation
      timing.tic();
      Matrix4d p_T_c;
      ceres::Solver::Summary summary;
      bool status = cor.computeRelPose_3dprev_2dcurr(p_T_c, summary);
      if( status == false )
      {
          cout << "Status : Reject\n";
          cout << "computeRelPose_3dprev_2dcurr() done in (ms): "<< timing.toc() << endl;
          return;
      }

      cout << "computeRelPose_3dprev_2dcurr() done in (ms): "<< timing.toc() << endl;



      #if CORVUS_DEBUG_LVL >= 2
      cor.saveReprojectedImagesFromCeresCallbacks();
      cor.publishCameraPoseFromCeresCallbacks(pub_bundle);
      #endif





      e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_GUIDED);
      loopClosureEdges.push_back( e );



      // Publish pose as opmode30.
      // Re-publish with pose, op_mode:=30
      int32_t mode = 30;
      republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c, mode, 0.18 );
      //TODO: Also add to msg goodness. This can denote the weight

      return ;
  }


  // Uses class LocalBundle..
  // Indicates this napmsg contains bundle
  if( msg->op_mode == 28  )
  {


    // Contains dense features tracked for the bundle. See also nap/script_multiproc/nap_multiproc.py:worker_qdd_processor().
    // The worker_qdd_processor() function
    ROS_INFO( "[Not Error]geometry_node OK! set edge as EDGE_TYPE_LOOP_SUBTYPE_BUNDLE" );



    // Process this nap msg to produce pose from locally tracked dense features

    // this->camera.printCameraInfo(1);
    TicToc timing;

    //
    // Constructor
    //
    timing.tic();
    LocalBundle localBundle = LocalBundle( msg, this->nNodes, this->camera );
    localBundle.setDebugOutputFolder( BASE__DUMP );


    if( localBundle.isValid_incoming_msg == false ) {
        ROS_ERROR( "[Not Error]Ignore message because constructor failed" );
        publish_text_as_image( pub_place_recog_status_image,  __msg +string(":construtor_failed") );
        return;
    }
    ROS_INFO_STREAM( "[Not Error]Done setting bundle data in (ms):" << timing.toc() );




    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_BUNDLE);
    loopClosureEdges.push_back( e );





    //
    // Triangulation
    //
    timing.tic();
    localBundle.randomViewTriangulate( 50, 0 ); // 3d points from iprev-j to iprev+j
    // localBundle.randomViewTriangulate( 50, 1 ); // 3d points from icurr-j to icurr
    ROS_INFO_STREAM( "[Not Error]Done triangulating iprev-j to iprev+j in (ms):" << timing.toc() );
    __msg += ":triangulating iprev-j to iprev+j";



    // Debug computed info .
    #if LOCALBUNDLE_DEBUG_LVL >= 2
    localBundle.saveTriangulatedPoints(); //saves the triangulated points to .txt files.
    #endif

    #if LOCALBUNDLE_DEBUG_LVL >= 1
    localBundle.publishTriangulatedPoints( pub_bundle  );
    localBundle.publishCameras( pub_bundle );
    #endif

    ROS_INFO( "Done triangulating icurr and iprev");


    //
    // Pose COmputation (Ceres)
    //
    timing.tic();
    // localBundle.sayHi();
    // localBundle.crossPoseComputation();
    // localBundle.ceresDummy();

    Matrix4d p_T_c;
    ceres::Solver::Summary summary;
    bool status = localBundle.crossRelPoseComputation3d2d(summary, p_T_c); // OK!
    // p_T_c = localBundle.crossRelPoseJointOptimization3d2d(); // OK! but it needs more work, currently quite slow ~5 sec
    double weight = min( 1.0, log( summary.initial_cost / summary.final_cost ) );


    // Look at what happened at each iteration
    #if LOCALBUNDLE_DEBUG_LVL >= 2
    localBundle.publishCameras_cerescallbacks( pub_bundle, true ); //use `false` with crossRelPoseJointOptimization3d2d()
    #endif

    ROS_INFO_STREAM( "[Not Error]crossRelPoseComputation3d2d() OK. Done in (ms):" << timing.toc() );


    char __ghgh__sf[100];
    snprintf( __ghgh__sf, 100, ":crossRelPoseComputation3d2d %d %2.2f", status, weight );
    publish_text_as_image( pub_place_recog_status_image, __msg+string(__ghgh__sf) );


    // Re-publish with pose, op_mode:=30
    int32_t mode = 30;
    republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c, mode, weight );

    return;
  }

  if( msg->op_mode == 12 )
  {
    // ROS_INFO( "msg->op_mode12 : Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_BASIC");
    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_BASIC);

    // This contains just 2 timestamps. No other useful info.
    // edgemanager->sayHi();
    // assert( is_edge_manager_available );
    // edge_manager->addEdge( i_curr, i_prev, msg->goodness );
    // cout << "[DataManager::place_recog_callback] Adding Edge (opmode12): " << i_curr << "<--->" << i_prev << "\tgoodness=" << msg->goodness << endl;
    // cout << "[DataManager::place_recog_callback] Contains global_id correspondences len = " << msg->visibility_table_idx.size() << endl;

    assert( msg->visibility_table_idx.size() > 0 && msg->visibility_table_idx.size()%2 == 0 && "for opmode12 visibility_table_idx must contain even number of values");
    REdge * re = new REdge( i_curr, i_prev, msg->goodness );

    for( int _y=0 ; _y<msg->visibility_table_idx.size()/2 ; _y++ ) {
        re->add_global_correspondence( msg->visibility_table_idx[2*_y], msg->visibility_table_idx[2*_y+1]  );
    }


    m_r_edges.lock();
    r_edges.push_back( re );
    m_r_edges.unlock();


    // cout << "place recog call back added loopClosureEdges.push_back()\n";
    loopClosureEdges.push_back( e );



    return;
  }


  ROS_ERROR( "in place_recog_callback: Error computing rel pose. Edge added without pose. This might be fatal!");


}

void DataManager::republish_nap( const nap::NapMsg::ConstPtr& msg )
{
  pub_chatter_colocation.publish( *msg );
}

void DataManager::republish_nap( const ros::Time& t_c, const ros::Time& t_p, const Matrix4d& p_T_c, int32_t op_mode, float goodness )
{
  nap::NapMsg msg;

  msg.c_timestamp = t_c;
  msg.prev_timestamp = t_p;
  msg.op_mode = op_mode;
  msg.goodness = goodness;

  // if op_mode is 30 means that pose p_T_c was computed from 3-way matching
  if( op_mode == 30 )
  {
    Matrix3d p_R_c;
    Vector4d p_t_c;

    p_R_c = p_T_c.topLeftCorner<3,3>();
    p_t_c = p_T_c.col(3);

    Quaterniond q = Quaterniond( p_R_c );
    msg.p_T_c.position.x = p_t_c[0];
    msg.p_T_c.position.y = p_t_c[1];
    msg.p_T_c.position.z = p_t_c[2];

    msg.p_T_c.orientation.x = q.x();
    msg.p_T_c.orientation.y = q.y();
    msg.p_T_c.orientation.z = q.z();
    msg.p_T_c.orientation.w = q.w();
  }
  else if( op_mode == 10 ) // contains no pose info.
  {
    ;
  }
  else
  {
    ROS_ERROR( "Cannot re-publish nap. Invalid op_mode" );
  }

  pub_chatter_colocation.publish( msg );
}



// ////////////////////////////////

// Loop over each node and return the index of the node which is clossest to the specified stamp
// #define __DataManager_find_indexof_node__debug( msg ) msg
#define __DataManager_find_indexof_node__debug( msg )

int DataManager::find_indexof_node( ros::Time stamp, bool print_info )
{
  ros::Duration diff;
  // assert( nNodes.size() > 0 && "DataManager::find_indexof_node");
  __DataManager_find_indexof_node__debug( cout << "\tSearch: "<< stamp << endl );
  int nNodes_size = getNodesSize();
  for( int i=0 ; i<nNodes_size ; i++ )
  {
    diff = nNodes[i]->time_stamp - stamp;

    if( print_info ) {
    __DataManager_find_indexof_node__debug( cout << "\t\t"<< i << " of "<< nNodes_size << " " << nNodes[i]->time_stamp <<  " node[i]->time_stamp - stamp(sec,nsec)="<< diff.sec << "," << diff.nsec << endl );
    }

    // if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) < int32_t(1000000) ) {
    // if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) == int32_t(0) ) {
    // if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
    //   return i;
    // }
    // Safer
    if( (diff.sec == 0  &&  diff.nsec < 1000000) || (diff.sec == -1  &&  diff.nsec > (1000000000-1000000) )  ) { // is within 1-mili-sec
        if( print_info) {
            __DataManager_find_indexof_node__debug( cout << "\t\tFound\n" );
        }
        return i;
    }
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_ERROR( "DataManager::find_indexof_node nNodes[last]->time_stamp - stamp=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);

  // for( int i=0 ; i<nNodes.size() ; i++ ) {
    //   __DataManager_find_indexof_node__debug( cout << "\t\t"<< i << " of "<< nNodes.size() << " " << nNodes[i]->time_stamp <<  " diff(sec,nsec)="<< diff.sec << "," << diff.nsec << endl );
  // }

  return -1;
}


void DataManager::flush_unclaimed_napmap()
{
  ROS_WARN( "flush_unclaimed_napmapIM queue_sizes: %d, T:%d", (int)unclaimed_napmap.size(), (int)unclaimed_napmap_time.size() );


  // int N = max(20,(int)unclaimed_im.size() );
  int N = unclaimed_napmap.size() ;
  for( int i=0 ; i<N ; i++)
  {
    cv::Mat image = cv::Mat(unclaimed_napmap.front());
    ros::Time stamp = ros::Time(unclaimed_napmap_time.front());
    unclaimed_napmap.pop();
    unclaimed_napmap_time.pop();
    int i_ = find_indexof_node(stamp);
    if( i_ < 0 )
    {
      unclaimed_napmap.push( image.clone() );
      unclaimed_napmap_time.push( ros::Time(stamp) );
    }
    else
    {
      nNodes[i_]->setNapClusterMap( stamp, image );
    }
  }

}

void DataManager::flush_unclaimed_im()
{
  ROS_WARN( "flush_unclaimed_im queue_sizes: IM:%d, T:%d", (int)unclaimed_im.size(), (int)unclaimed_im_time.size() );

  // std::queue<cv::Mat> X_im;
  // std::queue<ros::Time> X_tm;

  // int N = max(20,(int)unclaimed_im.size() );
  int N = (int)unclaimed_im.size();
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
  ROS_WARN( "flush_unclaimed_pt_cld(): PtCld queue_sizes:  %d, %d, %d", (int)unclaimed_pt_cld.size(), (int)unclaimed_pt_cld_time.size(), (int)unclaimed_pt_cld_globalid.size() );
  assert( unclaimed_pt_cld.size() == unclaimed_pt_cld_globalid.size() );
  assert( unclaimed_pt_cld_unvn.size() == unclaimed_pt_cld_uv.size() );
  assert( unclaimed_pt_cld_time.size() == unclaimed_pt_cld.size() );
  assert( unclaimed_pt_cld_time.size() == unclaimed_pt_cld_unvn.size() );
  int M = max(20,(int)unclaimed_pt_cld.size()); // Potential BUG. If not found, the ptcld is pushed at the end, where you will never get to as you see only first 20 elements!
  for( int i=0 ; i<M ; i++ )
  {
    // Matrix<double,3,Dynamic> e;
    MatrixXd e, e_unvn, e_uv;
    e = unclaimed_pt_cld.front();
    e_unvn = unclaimed_pt_cld_unvn.front();
    e_uv = unclaimed_pt_cld_uv.front();
    ros::Time t = ros::Time( unclaimed_pt_cld_time.front() );
    VectorXi e_globalid;
    e_globalid = unclaimed_pt_cld_globalid.front();
    unclaimed_pt_cld.pop();
    unclaimed_pt_cld_unvn.pop();
    unclaimed_pt_cld_uv.pop();
    unclaimed_pt_cld_time.pop();
    unclaimed_pt_cld_globalid.pop();

    int i_ = find_indexof_node(t);
    if( i_ < 0 )
    {
      //still not found, push back again
      unclaimed_pt_cld.push( e );
      unclaimed_pt_cld_unvn.push( e_unvn );
      unclaimed_pt_cld_uv.push( e_uv );
      unclaimed_pt_cld_time.push( t );
      unclaimed_pt_cld_globalid.push( e_globalid );
    }
    else
    {
    //   nNodes[i_]->setPointCloud(t, e );
      nNodes[i_]->setPointCloud(t, e, e_globalid );


      // Add all the 3d points to inverted index
      assert( e.cols() == e_globalid.size() );
      assert( e.rows() == 4 || e.rows() == 3 );
      for( int i=0 ; i< e.cols()  ; i++ )
      {
          int _gid = (int)e_globalid(i);
          Vector4d _3dpt;
          _3dpt << e(0,i), e(1,i), e(2,i), 1.0;

          Vector3d _2dpt_unvn, _2dpt_uv;
          _2dpt_unvn << e_unvn(0,i), e_unvn(1,i), 1.0;
          _2dpt_uv   << e_uv(0,i), e_uv(1,i), 1.0;

          // Add e_unvn.col(i) to tfidf and add e_uv.col(i) to tfidf
          tfidf->add( _gid, _3dpt, _2dpt_unvn, _2dpt_uv, i_ );
        //   tfidf->add( _gid, _3dpt, i_ );
      }

    }
  }
}
void DataManager::flush_unclaimed_2d_feat()
{
  ROS_WARN( "flush2dfeat queues_sizes : %d, %d", (int)unclaimed_2d_feat.size(), (int)unclaimed_2d_feat_time.size() );
  // int M = max(20,(int)unclaimed_2d_feat.size());
  int M = unclaimed_2d_feat.size();
  cout << "flush_feat2d()\n";
  for( int i=0 ; i<M ; i++ )
  {
    Matrix<double,3,Dynamic> e;
    e = unclaimed_2d_feat.front();
    ros::Time t = ros::Time( unclaimed_2d_feat_time.front() );
    unclaimed_2d_feat.pop();
    unclaimed_2d_feat_time.pop();
    int i_ = find_indexof_node(t);
    if( i_ < 0 )
    {
      //still not found, push back again
      unclaimed_2d_feat.push( e );
      unclaimed_2d_feat_time.push( t );
    }
    else
    {
      cout << "found "<< t << "--> " << i_ << endl;
      nNodes[i_]->setFeatures2dHomogeneous(t, e); //this will be set2dFeatures()
      return;
    }
  }

}

void DataManager::getNodesLock() { m_nNodes.lock(); }
void DataManager::getNodesUnlock() { m_nNodes.unlock(); }
int DataManager::getNodesSize() {
    getNodesLock();
    int n = nNodes.size();
    getNodesUnlock();
    return n;
}

void DataManager::getREdgesLock() { m_r_edges.lock(); }
void DataManager::getREdgesUnlock() { m_r_edges.unlock(); }
int DataManager::getREdgesSize() {
    getREdgesLock();
    int n = r_edges.size();
    getREdgesUnlock();
    return n;
}

// /// Debug file - Mark for removal. The debug txt file is now handled inside
// void DataManager::open_debug_xml( const string& fname)
// {
//   ROS_INFO( "Open DEBUG XML : %s", fname.c_str() );
//   (this->debug_fp).open( fname, cv::FileStorage::WRITE );
// }
//
// const cv::FileStorage& DataManager::get_debug_file_ptr()
// {
//   if( debug_fp.isOpened() == false ) {
//     ROS_ERROR( "get_debug_file_ptr(): debug xml file is not open. Call open_debug_xml() before this function" );
//     return NULL;
//   }
//
//   return debug_fp;
// }
//
// void DataManager::close_debug_xml()
// {
//   if( debug_fp.isOpened() == false )
//   {
//     ROS_ERROR( "close_debug_xml() : Attempting to close a file that is not open. COnsider calling open_debug_xml() before this function");
//   }
//   this->debug_fp.release();
// }
