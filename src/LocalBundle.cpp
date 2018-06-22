#include "LocalBundle.h"

void LocalBundle::sayHi()
{
  cout << "LocalBundle::Hello\n";
  camera.printCameraInfo(1);

  cout << "#nodes in global_nodes = " << global_nodes.size() << endl;

  int count = 0;
  /*
  for( int i=0 ; i< global_nodes.size() ; i++ ) {
    Matrix4d M;
    global_nodes[i]->getOriginalTransform(M);

    cout << "poses_" << i << "\n" << M << endl;
    count++;

    if( count > 3 )
      break;

  }
  */


  cout << "n_pairs : " << n_pairs << endl;
  cout << "n_ptClds: " << uv.size() << endl;


  assert( this->isValid_w_X_iprev_triangulated );
  assert( this->isValid_w_X_icurr_triangulated );
  // Plot the triangulated 3d points on all the available views.
  for( int v=0 ; v<n_ptClds ; v++ )
  {
    //plot observed points
    cv::Mat outImg;
    int node_gid = global_idx_of_nodes[v];
    int node_napid = nap_idx_of_nodes[v];
    string msg;
    msg = "lid="+to_string( v) + "; gid="+to_string(node_gid) + "; nap_id="+to_string(node_napid);
    plot_dense_point_sets( global_nodes[node_gid]->getImageRef(), uv[v], visibility_mask_nodes.row(v),
                       true, true, msg, outImg );


    //save
    assert( _m1set.size() == 2 );
    write_image( to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"___"+to_string(node_napid)+"_observed.png" , outImg );


    //plot reprojected points (3d points being generated from iprev-5 to iprev+5)
    MatrixXd v_X, reproj_pts;
    Matrix4d w_T_gid;
    global_nodes[node_gid]->getOriginalTransform(w_T_gid);//4x4

    v_X = w_T_gid.inverse() * w_X_iprev_triangulated;

    camera.perspectiveProject3DPoints( v_X, reproj_pts);
    plot_dense_point_sets( global_nodes[node_gid]->getImageRef(), reproj_pts, visibility_mask_nodes.row(v),
                       true, true, msg, outImg );

    //save
    write_image( to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"___"+to_string(node_napid)+"_reproj_3diprev.png" , outImg );


    // reproject 3d points generated from icurr-5 to icurr
    v_X = w_T_gid.inverse() * w_X_icurr_triangulated;

    camera.perspectiveProject3DPoints( v_X, reproj_pts);
    plot_dense_point_sets( global_nodes[node_gid]->getImageRef(), reproj_pts, visibility_mask_nodes.row(v),
                       true, true, msg, outImg );

    //save
    write_image( to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"___"+to_string(node_napid)+"_reproj_3dicurr.png" , outImg );



  }
}


void LocalBundle::triangulate_points( int global_idx_i, const MatrixXd& _uv_i,
                         int global_idx_j, const MatrixXd& _uv_j,
                         MatrixXd& _3d
                       )
{

  //
  // Step-1 : Relative pose, Projection Matrix of ix_curr and ix_curr_m

  // K [ I | 0 ]
  MatrixXd I_0;
  I_0 = Matrix4d::Identity().topLeftCorner<3,4>();
  MatrixXd P1 = camera.e_K * I_0; //3x4

  // K [ ^{c-1}T_c ] ==> K [ inv( ^wT_{c-1} ) * ^wT_c ]
  Matrix4d w_T_i;
  global_nodes[global_idx_i]->getOriginalTransform(w_T_i);//4x4
  Matrix4d w_T_j;
  global_nodes[global_idx_j]->getOriginalTransform(w_T_j);//4x4

  MatrixXd Tr; // i_T_j
  Tr = w_T_i.inverse() * w_T_j; //relative transform
  MatrixXd P2 = camera.e_K * Tr.topLeftCorner<3,4>(); //3x4

  cv::Mat xP1(3,4,CV_64F );
  cv::Mat xP2(3,4,CV_64F );
  cv::eigen2cv( P1, xP1 );
  cv::eigen2cv( P2, xP2 );

  //
  // Step-2 : OpenCV Triangulate
  cv::Mat mat_pts_i, mat_pts_j, c_3dpts;
  MatrixXd _uv_i_2, _uv_j_2;
  _uv_i_2 = _uv_i.block(0,0,2,_uv_i.cols());
  _uv_j_2 = _uv_j.block(0,0,2,_uv_j.cols());
  cv::eigen2cv( _uv_i_2, mat_pts_i );
  cv::eigen2cv( _uv_j_2, mat_pts_j );
  // printMatrixInfo( "xP1", xP1 );
  // printMatrixInfo( "xP2", xP2 );
  // printMatrixInfo( "mat_pts_i", mat_pts_i );
  // printMatrixInfo( "mat_pts_j", mat_pts_j );
  // printMatrixInfo( "_uv_i_2", _uv_i_2 );
  // printMatrixInfo( "_uv_j_2", _uv_j_2 );
  // printMatrixInfo( "_uv_i", _uv_i_2 );
  // printMatrixInfo( "_uv_j", _uv_j_2 );

  cv::triangulatePoints( xP1, xP2,  mat_pts_i, mat_pts_j,   c_3dpts );

  // cv::triangulatePoints output is 4d (x,y,z,w). Here w != 1.
  // This is on purpose, and an indicator for far off points.

  MatrixXd _i_3d;
  cv::cv2eigen( c_3dpts, _i_3d );
  // _3d = _i_3d ;
  _3d = w_T_i * _i_3d;


}
int LocalBundle::edge_type_from_node( int nx )
{
  // assert( adj_mat.rows() > 0 && adj_mat.cols() > 0 && adj_mat.rows() == adj_mat.cols() )
  // assert( nx < adj_mat.rows() );

  for( int i=0 ; i< adj_mat.rows() ; i++ )
  {
    if( adj_mat(nx,i) != 0 )
      return adj_mat(nx,i);
  }

  // this is impossible case. This indicates the node nx is not connected to the graph.
  // Which is an impossible case
  assert( false );
}


int LocalBundle::pair_0idx_of_node( int nx )
{
  // Input a node index. Returns a pair index
  for( int i=0 ; i<n_pairs ; i++ )
  {
    if( local_idx_of_pairs[2*i] == nx )
      return i;
  }
  return -1;
}

int LocalBundle::pair_1idx_of_node( int nx )
{
  // Input a node index. Returns a pair index
  for( int i=0 ; i<n_pairs ; i++ )
  {
    if( local_idx_of_pairs[2*i+1] == nx )
      return i;
  }
  return -1;
}


#if LOCALBUNDLE_DEBUG_LVL >= 3
#define randomViewTriangulate_debug( msg )  msg ;
#else
#define randomViewTriangulate_debug( msg )  ;
#endif


#if LOCALBUNDLE_DEBUG_LVL >= 2
#define randomViewTriangulate_debug1(msg ) msg;
#else
#define randomViewTriangulate_debug1(msg ) ;
#endif

void LocalBundle::randomViewTriangulate(int max_itr, int flag )
{
  // Pseudo code :
  // 1. a,b = randomly pick 2 nodes suchthat (edgefrom-a.type==1 or edgefrom-a.type==2 ) and (edgefrom-b.type==1 or edgefrom-b.type==2)
  // 2. p = find_path( a,b )
  // 3. _3dpts = triangulate( a, b )
  // 4. mask_a_b = AND( p1,  pk ) # to get list of points which were tracked correctly in each of a and b
  // 5. store( mask, _3dpt )
  //    repeat

  assert( flag == 0 || flag == 1);

  // printMatrixInfo( "adj_mat", adj_mat );
  assert( n_pairs > 0 );
  assert( n_ptClds > 0 );
  assert( adj_mat.rows() == n_ptClds );
  assert( adj_mat.rows() == adj_mat.cols() );
  assert( adj_mat_dirn.rows() == adj_mat_dirn.cols() );
  assert( adj_mat_dirn.rows() == adj_mat.rows() );

  assert( _1set.size() * _2set.size() * _3set.size() * _m1set.size() > 0 ); //everyone should be +ve

  int n_success = 0;

  // Collection from each random try. This is collected so that DLT can be used for triangulation.
  vector<Matrix4d> w_T_c1, w_T_c2;
  vector<MatrixXd> _1_unvn_undistorted, _2_unvn_undistorted;
  vector<VectorXd> mmask;

  // Random Pairs
  vector< pair<int,int> > vector_of_pairs;
  for( int itr=0 ; itr<max_itr ; itr++ )
  {
    randomViewTriangulate_debug( cout << "---itr="<<itr << "---\n" );

    //////////////////
    ///// Step-1 /////
    //////////////////

    int _1, _2;
    // pick a rrandom node --A
    // _1 = rand() % n_ptClds;
    // _2 = rand() % n_ptClds;

    // pick a random node from _1set, _2set, _3set. --B
    if( flag == 0 )
    {
      int c = rand() % 4;
      switch (c) {
        case 0:
          _1 = _1set[ rand()%_1set.size() ];
          _2 = _1set[ rand()%_1set.size() ];
          break;
        case 1:
          _1 = _2set[ rand()%_2set.size() ];
          _2 = _2set[ rand()%_2set.size() ];
          break;
        case 2:
          _1 = _1set[ rand()%_1set.size() ];
          _2 = _2set[ rand()%_2set.size() ];
          break;
        case 3:
          _1 = _2set[ rand()%_2set.size() ];
          _2 = _1set[ rand()%_1set.size() ];
          break;
        default:
          assert( false && "Impossible");
      }
    }

    if( flag == 1 )
    {
      //pick a random from  _3set
      _1 = _3set[ rand()%_3set.size() ];
      _2 = _3set[ rand()%_3set.size() ];
    }


    // _1 should contain smaller of the 2, _2 should contain larger of the 2.
    // int _tmp_1 = min( _1, _2 );
    // int _tmp_2 = max( _1, _2 );
    // _1 = _tmp_1;
    // _2 = _tmp_2;


    ////// Done picking


    randomViewTriangulate_debug(cout << "picked nodes with local index "<< _1 << " " << _2 << endl);

    if( _1 == _2 )
    {
      randomViewTriangulate_debug(cout << "same, ignore it\n");
      continue;
    }





    // find edge from _1
    int _1_type = edge_type_from_node( _1 );
    int _2_type = edge_type_from_node( _2 );
    randomViewTriangulate_debug(cout << "_1 is of type: "<< _1_type << endl);
    randomViewTriangulate_debug(cout << "_2 is of type: "<< _2_type << endl);

    if( flag == 0 )
    {
    if(  !( ( _1_type==1 || _1_type == 2 ) && ( _2_type==1 || _2_type == 2 )  ) )
    {
      randomViewTriangulate_debug(cout << "reject, since type is something other than 1 or 2\n");
      continue;
    }
    }


    if( flag == 1 )
    {
    if(  !( ( _1_type==3 ) && ( _2_type==3 )  ) )
    {
      randomViewTriangulate_debug(cout << "reject, since type is something other than 1 or 2\n");
      continue;
    }
    }



    // Record pair for debugging
    pair<int,int> a_pair;
    a_pair.first = _1;
    a_pair.second = _2;


    if( find( vector_of_pairs.begin(), vector_of_pairs.end(), a_pair ) != vector_of_pairs.end() )
    {
        randomViewTriangulate_debug( cout << "picked nodes with local index "<< _1 << " " << _2  );
        randomViewTriangulate_debug( cout << "  Already Exist. Dont add.\n" );
        continue;
    }
    else
    {
        randomViewTriangulate_debug( cout << "picked nodes with local index "<< _1 << " " << _2  );
        randomViewTriangulate_debug( cout << "  New pair\n" );
        vector_of_pairs.push_back( a_pair );
    }


    randomViewTriangulate_debug(cout << "process\n");

    //////////////////
    ///// Step-2 ///// You exactly need to find the entire path. Just get an edge going out of _a and an edge going out of _2. But this is an acceptable approximation
    //////////////////

    n_success++;


    // Get global ids of _1, and _2.
    int _1_pairid, _1_globalid, _2_pairid, _2_globalid;
    int _1_localid, _2_localid, _1_napid, _2_napid; // not required. Just for debugging. TODO: Remove once debugging is done.

    _1_pairid = pair_0idx_of_node( _1 ); // if cannot find in 1st position, look at 2nd position
    if( _1_pairid >= 0 ) {
      _1_globalid = global_idx_of_pairs[2*_1_pairid];
      _1_localid = local_idx_of_pairs[2*_1_pairid];
      _1_napid = nap_idx_of_pairs[2*_1_pairid];
      randomViewTriangulate_debug(cout << _1 << " Was found in pair#" << _1_pairid << " in 1st postion\n");
    }
    else {
      _1_pairid = pair_1idx_of_node( _1 );
      assert( _1_pairid >=0 );
      _1_globalid = global_idx_of_pairs[2*_1_pairid+1];
      _1_localid = local_idx_of_pairs[2*_1_pairid+1];
      _1_napid = nap_idx_of_pairs[2*_1_pairid+1];
      randomViewTriangulate_debug(cout << _1 << " Was found in pair#" << _1_pairid << " in 2nd postion\n");
    }



    _2_pairid = pair_0idx_of_node( _2 ); // if cannot find in 1st position, look at 2nd position
    if( _2_pairid >= 0 ) {
      _2_globalid = global_idx_of_pairs[2*_2_pairid];
      _2_localid = local_idx_of_pairs[2*_2_pairid];
      _2_napid = nap_idx_of_pairs[2*_2_pairid];
      randomViewTriangulate_debug(cout << _2 << " Was found in pair#" << _2_pairid << " in 1st postion\n");
    }
    else {
      _2_pairid = pair_1idx_of_node( _2 );
      assert( _2_pairid >=0 );
      _2_globalid = global_idx_of_pairs[2*_2_pairid+1];
      _2_localid = local_idx_of_pairs[2*_2_pairid+1];
      _2_napid = nap_idx_of_pairs[2*_2_pairid+1];
      randomViewTriangulate_debug(cout << _2 << " Was found in pair#" << _2_pairid << " in 2nd postion\n");

    }



    //////////////////
    ///// Step-3 ///// Triangulate
    //////////////////

    randomViewTriangulate_debug(cout << "==>Triangulate globalid "<< _1_globalid << " and " << _2_globalid << endl);
    randomViewTriangulate_debug(cout << "==>Triangulate localid  "<< _1_localid << " and " << _2_localid << endl);
    randomViewTriangulate_debug(cout << "==>Triangulate napid    "<< _1_napid << " and " << _2_napid << endl);


    randomViewTriangulate_debug1( cout << itr << "==>Triangulate globalid=("<< _1_globalid << "," << _2_globalid << ") ; " );
    randomViewTriangulate_debug1( cout << "localid=("<< _1_localid << "," << _2_localid << ") ; " );
    randomViewTriangulate_debug1( cout << "napid=("<< _1_napid << "," << _2_napid << ") ; " << endl );
    assert( _1_localid == _1 );
    assert( _2_localid == _2 );


    Matrix4d __w_T_c1, __w_T_c2;
    global_nodes[_1_globalid]->getOriginalTransform(__w_T_c1);//4x4
    global_nodes[_2_globalid]->getOriginalTransform(__w_T_c2);//4x4
    w_T_c1.push_back( __w_T_c1 );
    w_T_c2.push_back( __w_T_c2 );
    _1_unvn_undistorted.push_back( unvn_undistorted[_1] );
    _2_unvn_undistorted.push_back( unvn_undistorted[_2] );



    //////////////////
    ///// Step-4 ///// Compose Mask
    //////////////////
    VectorXd composed_mask;
    composed_mask = visibility_mask.row(_1_pairid).cwiseProduct(  visibility_mask.row(_2_pairid)   );
    randomViewTriangulate_debug(cout << "#verified pts in " << _1 << " = "<< visibility_mask.row(_1_pairid).sum() << endl);
    randomViewTriangulate_debug(cout << "#verified pts in " << _2 << " = "<< visibility_mask.row(_2_pairid).sum() << endl);
    randomViewTriangulate_debug1( cout << "#verified pts in composed_mask = "<< composed_mask.sum() << endl );


    mmask.push_back( composed_mask );


    // Visualize this _1, _2
    // cv::Mat dst;
    // plot_point_sets( global_nodes[_1_globalid]->getImageRef(), uv[_1],  _1_napid,
    //                  global_nodes[_2_globalid]->getImageRef(), uv[_2],  _2_napid,
    //                  composed_mask, cv::Scalar(0,0,255), true, "", dst
    //               );
    // write_image( to_string(_1_napid) + "_" + to_string(_2_napid)+"_observed.png" , dst );


  }
  cout << "n_success="<< n_success << " max_itr="<< max_itr ;
  cout << "  |vector_of_pairs|=" << vector_of_pairs.size() << endl;
  assert( n_success > 2 );
  assert( n_success == w_T_c1.size() );
  assert( n_success == w_T_c2.size() );
  assert( n_success == _1_unvn_undistorted.size() );
  assert( n_success == _2_unvn_undistorted.size() );
  assert( n_success == mmask.size() );
  assert( n_success == vector_of_pairs.size() );



  // DLT-SVD to get better estimates of 3d points.
  MatrixXd w_X_triang; //triangulated 3d points in world co-prdinates from multiple views
  robust_triangulation( vector_of_pairs, w_T_c1, w_T_c2, _1_unvn_undistorted, _2_unvn_undistorted, mmask, w_X_triang );
  // THIS is unnecassary to w divide. As some points can be at infinity (ie w=0.). However removing this can have some unintented consequences somewhere.
  w_X_triang.row(0).array() /= w_X_triang.row(3).array();
  w_X_triang.row(1).array() /= w_X_triang.row(3).array();
  w_X_triang.row(2).array() /= w_X_triang.row(3).array();
  w_X_triang.row(3).array() /= w_X_triang.row(3).array();
  // NOTE: Now the w_X_triang will be in homogeneous co-ordinates in general. ie. it will have a w component.

  if( flag == 0 )
  {
    this->w_X_iprev_triangulated = MatrixXd(w_X_triang);
    isValid_w_X_iprev_triangulated = true;

    // Also store these 3d points in frame of ref of iprev
    Matrix4d p_T_w;
    gi_T_w( localidx_of_iprev, p_T_w  );
    this->iprev_X_iprev_triangulated = p_T_w * w_X_triang;
    isValid_iprev_X_iprev_triangulated = true;
  }

  if( flag == 1 )
  {
    this->w_X_icurr_triangulated = MatrixXd(w_X_triang);
    isValid_w_X_icurr_triangulated = true;


    // Also store these 3d points in frame of ref of icurr
    Matrix4d c_T_w;
    gi_T_w( localidx_of_icurr, c_T_w );
    this->icurr_X_icurr_triangulated = c_T_w * w_X_triang;
    isValid_icurr_X_icurr_triangulated = true;
  }




}

void LocalBundle::init_camera_marker( visualization_msgs::Marker& marker )
{
     marker.header.frame_id = "world";
     marker.header.stamp = ros::Time::now();
     marker.action = visualization_msgs::Marker::ADD;
     marker.color.a = 1.0; // Don't forget to set the alpha!
     marker.type = visualization_msgs::Marker::LINE_LIST;
    //  marker.id = i;
    //  marker.ns = "camerapose_visual";

     marker.scale.x = 0.005; //width of line-segments
     float __vcam_width = 0.07*2.;
     float __vcam_height = 0.04*2.;
     float __z = 0.05;

     marker.points.clear();
     geometry_msgs::Point pt;
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );

     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );


     // TOSET
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;
}

void LocalBundle::setpose_to_cameravisual( const Matrix4d& w_T_c, visualization_msgs::Marker& marker )
{
    Quaterniond quat( w_T_c.topLeftCorner<3,3>() );
    marker.pose.position.x = w_T_c(0,3);
    marker.pose.position.y = w_T_c(1,3);
    marker.pose.position.z = w_T_c(2,3);
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
}

void LocalBundle::setcolor_to_cameravisual( float r, float g, float b, visualization_msgs::Marker& marker  )
{
    marker.color.r = r;
    marker.color.b = g;
    marker.color.g = b;
}


void LocalBundle::publishCameras_cerescallbacks( const ros::Publisher& pub, bool write_images )
{
    //uses vector_of_callbacks;
    if( vector_of_callbacks.size() == 0 ) {
        ROS_ERROR( " You called LocalBundle::publishCameras_cerescallbacks() however, `vector_of_callbacks.size()` is 0. ");
        return;
    }
    cout << " LocalBundle::publishCameras_cerescallbacks. uses vector_of_callbacks" << endl;
    visualization_msgs::Marker base_camera_visual;
    init_camera_marker( base_camera_visual ); //to use it, you need to set marker.pose, marker.ns, marker.id, marker.color
    base_camera_visual.ns = "cams_ceres_iterations"+to_string( nap_idx_of_nodes[localidx_of_icurr] ) + "_" + to_string( nap_idx_of_nodes[localidx_of_iprev] );


    cout << "vector_of_callbacks.size() : "<< vector_of_callbacks.size() << endl;
    cout << "vector_of_callbacks[0].pose_at_each_iteration.size() : " << vector_of_callbacks[0].pose_at_each_iteration.size() << endl;



    Matrix4d w_T_p = w_T_gi( localidx_of_iprev );
    int id = 0;


    visualization_msgs::Marker iteration_line;
    iteration_line.header = base_camera_visual.header;
    iteration_line.ns = base_camera_visual.ns;
    iteration_line.type = visualization_msgs::Marker::LINE_STRIP;
    iteration_line.action = visualization_msgs::Marker::ADD;
    iteration_line.scale.x = 0.01;
    iteration_line.color.r = 1.0; iteration_line.color.g = 1.0; iteration_line.color.b = 1.0; iteration_line.color.a = 1.0;

    for( int i=0 ; i<vector_of_callbacks.size() ; i++ )
    {
        Align3d2d__4DOFCallback Q = vector_of_callbacks[i];
        cout << "---Info on callback" << i << "\n";
        cout << "Q.gid() : " << Q.getGid() << endl;
        cout << "Q.n_iterations : "<< Q.pose_at_each_iteration.size() << endl;

        iteration_line.points.clear();
        for( int k=0 ; k<Q.pose_at_each_iteration.size() ; k++ )
        {

            Matrix4d ci_T_p = Q.pose_at_each_iteration[k]; //pose at kth iteration
            #if LOCALBUNDLE_DEBUG_LVL >= 3
            cout << k << "("<< Q.loss_at_each_iteration[k] << ") ";
            prettyprintPoseMatrix( ci_T_p );
            #endif

            Matrix4d w_T_ci = Matrix4d::Identity();
            w_T_ci = w_T_p * ci_T_p.inverse();

            setpose_to_cameravisual( w_T_ci, base_camera_visual );
            float __r = (float)k / (float)Q.pose_at_each_iteration.size();
            setcolor_to_cameravisual( __r, .0, .0, base_camera_visual );

            base_camera_visual.id = id++;
            pub.publish( base_camera_visual );

            if( write_images )
            {
                // TODO: Ingeneral, use gid to write an image overlay correctly.
                string ____prettypose_string;
                prettyprintPoseMatrix( ci_T_p, ____prettypose_string  );
                mark3dPointsOnCurrIm( ci_T_p * p_T_w(), "proj3dPointsOnCurr_itr"+to_string(k), "c_T_p "+____prettypose_string );
            }



            geometry_msgs::Point pt_t;
            pt_t.x = w_T_ci(0,3);
            pt_t.y = w_T_ci(1,3);
            pt_t.z = w_T_ci(2,3);
            iteration_line.points.push_back( pt_t );

        }
        iteration_line.id = id++;
        pub.publish( iteration_line );
    }

    cout << " END LocalBundle::publishCameras_cerescallbacks. uses vector_of_callbacks" << endl;
}


void LocalBundle::publishCameras( const ros::Publisher& pub )
{
    cout << " LocalBundle::publishCameras" << endl;

    visualization_msgs::Marker base_camera_visual;
    init_camera_marker( base_camera_visual ); //to use it, you need to set marker.pose, marker.ns, marker.id, marker.color
    base_camera_visual.ns = "cams_"+to_string( nap_idx_of_nodes[localidx_of_icurr] ) + "_" + to_string( nap_idx_of_nodes[localidx_of_iprev] );
    int id =0;

    // mark cameras in _1set
    for( int i=0 ; i<_1set.size() ; i++ )
    {
        setpose_to_cameravisual( w_T_gi( _1set[i] ), base_camera_visual );
        setcolor_to_cameravisual( 1., .0, 1., base_camera_visual ); //pink
        base_camera_visual.id = id++;
        pub.publish( base_camera_visual );
    }


    // mark cameras in _2set
    for( int i=0 ; i<_2set.size() ; i++ )
    {
        setpose_to_cameravisual( w_T_gi( _2set[i] ), base_camera_visual );
        setcolor_to_cameravisual( 1.0, 1.0, 0., base_camera_visual ); //yellow
        base_camera_visual.id = id++;
        pub.publish( base_camera_visual );
    }

    // mark cameras in _3set
    for( int i=0 ; i<_3set.size() ; i++ )
    {
        setpose_to_cameravisual( w_T_gi( _3set[i] ), base_camera_visual );
        setcolor_to_cameravisual( 0., 1., 0., base_camera_visual ); //green
        base_camera_visual.id = id++;
        pub.publish( base_camera_visual );
    }


    // mark icurr, iprev
    setpose_to_cameravisual( w_T_gi( localidx_of_iprev ), base_camera_visual );
    setcolor_to_cameravisual( 0.6, .3, 0., base_camera_visual ); //brownish color
    base_camera_visual.id = id++;
    pub.publish( base_camera_visual );


    setpose_to_cameravisual( w_T_gi( localidx_of_icurr ), base_camera_visual );
    setcolor_to_cameravisual( 0., .4, .4, base_camera_visual );//bluish-green
    base_camera_visual.id = id++;
    pub.publish( base_camera_visual );



    cout << " END LocalBundle::publishCameras" << endl;

}



void LocalBundle::publishTriangulatedPoints(  const ros::Publisher& pub )
{
    cout << " LocalBundle::publishTriangulatedPoints" << endl;

    assert( isValid_w_X_iprev_triangulated );
    assert( isValid_iprev_X_iprev_triangulated );


    // Compute reprojection residue

    printMatrixInfo( "unvn_undistorted[localidx_of_iprev]", unvn_undistorted[localidx_of_iprev] );
    MatrixXd reprojection_residue = MatrixXd::Zero( 2, iprev_X_iprev_triangulated.cols() );
    reprojection_residue.row(0) =  unvn_undistorted[localidx_of_iprev].row(0) -  ( iprev_X_iprev_triangulated.row(0).array() / iprev_X_iprev_triangulated.row(2).array() ).matrix();
    reprojection_residue.row(1) =  unvn_undistorted[localidx_of_iprev].row(1) -  ( iprev_X_iprev_triangulated.row(1).array() / iprev_X_iprev_triangulated.row(2).array() ).matrix();
    write_EigenMatrix( to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"_triangulated3d_obs_residue.txt", reprojection_residue );



    // Publish just 1 marker, type=points
    #if 1
    visualization_msgs::Marker marker;
    string ns = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"__w_X_iprev_triangulated";
    eigenpointcloud_2_ros_markermsg( w_X_iprev_triangulated, marker, ns );
    // set per point color
    for( int i=0 ; i<iprev_X_iprev_triangulated.cols() ; i++ )
    {
        VectorXd _d = iprev_X_iprev_triangulated.col(i);
        double d = sqrt( _d(0)*_d(0) + _d(1)*_d(1) + _d(2)*_d(2) );

        bool is_behind = ( _d(2)/_d(3) < 0 )?true:false;
        bool good_ = ( reprojection_residue.col(i).norm() < 0.1 )?true:false;


        std_msgs::ColorRGBA Kolor;
        Kolor.a = 1.0;
        if( is_behind && good_  )  { Kolor.r=1.; Kolor.g=1.; Kolor.b=0.;} //yellow
        if( is_behind && !good_  ) { Kolor.r=0.; Kolor.g=0.; Kolor.b=1.; } //blue
        if( !is_behind && good_  ) { Kolor.r=0.; Kolor.g=1.; Kolor.b=0.; } //green
        if( !is_behind && !good_ ) { Kolor.r=1.; Kolor.g=1.; Kolor.b=1.;} //white

        marker.colors.push_back( Kolor );

    }
    pub.publish( marker );
    cout << "MarkerInfo: " << marker.type << "  " << marker.ns << " " << marker.id << endl;
    #endif


    #if 0
    // publish 1 marker for each 3d point. good for analysis.
    vector<visualization_msgs::Marker> pointcloud_text;
    string ns2 = "text_"+to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"__w_X_iprev_triangulated";
    eigenpointcloud_2_ros_markertextmsg( w_X_iprev_triangulated, pointcloud_text, ns2 );
    cout << "pointcloud_text.size: "<< pointcloud_text.size() << endl;

    int less_than_2m=0, less_than_5m=0, less_than_10m=0, less_than_20m=0, more_than_20=0, n_behind=0;
    for( int i=0 ; i<pointcloud_text.size() ; i++ )
    {
        // Color behind points differently
        VectorXd _d = iprev_X_iprev_triangulated.col(i);
        double d = sqrt( _d(0)*_d(0) + _d(1)*_d(1) + _d(2)*_d(2) );
        if( d < 2. ) { less_than_2m++; }
        if( d < 5.  && d >= 2 ) { less_than_5m++; }
        if( d < 10.  && d >= 5) { less_than_10m++;}
        if( d < 20. && d >= 10 ) { less_than_20m++;}
        if( d>= 20. ) { more_than_20++; }

        if( _d(2) < 0 )
        {
            pointcloud_text[i].color.r = 0;
            pointcloud_text[i].color.g = 0;
            pointcloud_text[i].color.b = 0;
            n_behind++;
        }



        cout << pointcloud_text[i].ns << " " << pointcloud_text[i].id << " " << pointcloud_text[i].color.r << pointcloud_text[i].color.g << pointcloud_text[i].color.b <<  endl;
        pub.publish( pointcloud_text[i] );
    }

    cout << "less_than_2m: "<< less_than_2m << "; ";
    cout << "between_2m_5m: "<< less_than_5m << "; ";
    cout << "between_5m_10m: "<< less_than_10m << "; ";
    cout << "between_10m_20m: "<< less_than_20m << "; " ;
    cout << "more_than_20: "<< more_than_20 << "; " << endl;
    cout << "n_behind: " << n_behind<< endl;
    #endif







    cout << "Done... LocalBundle::publishTriangulatedPoints" << endl;

}

void LocalBundle::saveTriangulatedPoints()
{
  if( isValid_w_X_icurr_triangulated ) {
    string fname = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]]) + "___w_X_icurr_triangulated.txt";
    write_EigenMatrix( fname, w_X_icurr_triangulated.transpose() );
  }

  if( isValid_w_X_iprev_triangulated ) {
    string fname = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]]) + "___w_X_iprev_triangulated.txt";
    write_EigenMatrix( fname, w_X_iprev_triangulated.transpose() );
  }




  if( isValid_iprev_X_iprev_triangulated ) {
    string fname = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]]) + "___iprev_X_iprev_triangulated.txt";
    write_EigenMatrix( fname, iprev_X_iprev_triangulated.transpose() );
  }

  if( isValid_icurr_X_icurr_triangulated ) {
    string fname = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]]) + "___icurr_X_icurr_triangulated.txt";
    write_EigenMatrix( fname, icurr_X_icurr_triangulated.transpose() );
  }


  // Also write poses of icurr and iprev, ie. c_(R|t)_w and p_(R|t)_w
  string fname;
  fname = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]]) + "____c_Rt_w.txt";
  write_EigenMatrix( fname, gi_T_w( localidx_of_icurr ) );
  fname = to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]]) + "____p_Rt_w.txt";
  write_EigenMatrix( fname, gi_T_w( localidx_of_iprev ) );
}



LocalBundle::LocalBundle( const nap::NapMsg::ConstPtr& msg,
              const vector<Node*>& global_nodes, const PinholeCamera& camera  )
{


  cout << "----\nLocalBundle\n";
  cout << "#PointClouds: " << msg->bundle.size() << endl;

  // Image msg
  sensor_msgs::Image img_msg = msg->visibility_table;
  cout << "Image : "<< img_msg.height << " " << img_msg.width << " " << img_msg.encoding <<  endl;
  int N_pairs = img_msg.height;

  // Image msg ---> cv::Mat
  cv::Mat visibility_table = cv_bridge::toCvCopy( msg->visibility_table, sensor_msgs::image_encodings::MONO8 )->image;
  cout << "cv::Mat : "<< visibility_table.rows << " " << visibility_table.cols << " " << visibility_table.channels() << endl;

  // cv::Mat --> MatrixXd
  MatrixXd e_visibility_table;
  cv2eigen( visibility_table, e_visibility_table  );
  visibility_mask = MatrixXd( e_visibility_table );



  // cout << "visibility_table_image_dim: " <<  table_image.rows << table_image.cols << endl;
  cout << "visibility_table_idx.size() " << msg->visibility_table_idx.size() << endl;
  cout << "visibility_table_stamp.size() " << msg->visibility_table_stamp.size() << endl;


  /////////////////////////////// Assert above data /////////////////////////////
  assert( img_msg.encoding == string("mono8") );
  assert( N_pairs > 0 );
  assert( 3*N_pairs == msg->visibility_table_idx.size()    );
  assert( 2*N_pairs == msg->visibility_table_stamp.size()  );
  assert( visibility_table.rows == N_pairs );




  ////////////////////// Setup other stuff /////////////////
  this->camera = PinholeCamera(camera);
  this->camera.printCameraInfo(0);

  this->global_nodes = global_nodes;

  assert( this->camera.isValid() );
  assert( this->global_nodes.size() > 0 );

  /////////////////////////// loop on tracked points /////////////////////////////
  this->n_ptClds = msg->bundle.size();
  this->n_features = msg->bundle[0].points.size();
  for( int i=0 ; i<msg->bundle.size() ; i++ )
  {
    // cout << "---\nPointbundle "<< i << endl;
    int seq = find_indexof_node( global_nodes, msg->bundle[i].header.stamp );
    int seq_debug = msg->bundle[i].header.seq;

    MatrixXd e_ptsA;
    pointcloud_2_matrix(msg->bundle[i].points, e_ptsA  );
    uv.push_back( e_ptsA );


    MatrixXd e_ptsA_undistored;
    this->camera.undistortPointSet( e_ptsA, e_ptsA_undistored, false );
    // printMatrixInfo( "e_ptsA_undistored", e_ptsA_undistored); //3x878 (N=878)
    uv_undistorted.push_back( e_ptsA_undistored );

    // use the camera to get normalized co-ordinates and undistorded co-ordinates
    MatrixXd e_ptsA_undistored_normalized;
    this->camera.undistortPointSet( e_ptsA, e_ptsA_undistored_normalized, true );
    // printMatrixInfo( "e_ptsA_undistored_normalized", e_ptsA_undistored_normalized);
    unvn_undistorted.push_back( e_ptsA_undistored_normalized );


    global_idx_of_nodes.push_back( seq );
    nap_idx_of_nodes.push_back(seq_debug);


    cout << "pointcloud# " << i << " : global_idx=" << seq <<  "\t#pts=" <<  msg->bundle[i].points.size()  << "\tnap_idx=" << seq_debug <<  "\tvalid_image: " << global_nodes[seq]->valid_image()  << endl;

  }


  ///////////////////////////////////////// loop on pairs ///////////////////////////////
  this->n_pairs = N_pairs;
  cout << "Setting adj_matrix dimensions\n";
  this->adj_mat = MatrixXd::Zero(this->n_ptClds, this->n_ptClds);
  this->adj_mat_dirn = MatrixXd::Zero(this->n_ptClds, this->n_ptClds);
  visibility_mask_nodes = MatrixXd::Ones( n_ptClds, n_features  );
  printMatrixInfo( "adj_mat", adj_mat );
  printMatrixInfo( "adj_mat_dirn", adj_mat_dirn );
  cout << "global_idx\t\t\tnap_idx[]\t\tlocal_idx{}\n";
  for( int i=0 ; i<N_pairs ; i++ )
  {
    //
    // prepare data for ith pair
    ros::Time a_stamp = msg->visibility_table_stamp[2*i];
    ros::Time b_stamp = msg->visibility_table_stamp[2*i+1];


    int a = msg->visibility_table_idx[3*i];
    int b = msg->visibility_table_idx[3*i+1];
    int ttype = msg->visibility_table_idx[3*i+2];


    int a_stamp_global_idx = find_indexof_node( global_nodes, a_stamp ); ///< global index
    int b_stamp_global_idx = find_indexof_node( global_nodes, b_stamp );
    int _i = find_indexof_node( msg->bundle, a_stamp ); ///< local bundle index
    int _j = find_indexof_node( msg->bundle, b_stamp );

    local_idx_of_pairs.push_back(_i);
    local_idx_of_pairs.push_back(_j);
    global_idx_of_pairs.push_back( a_stamp_global_idx );
    global_idx_of_pairs.push_back( b_stamp_global_idx );
    nap_idx_of_pairs.push_back( a );
    nap_idx_of_pairs.push_back( b );
    pair_type.push_back( ttype );

    adj_mat(_i,_j) = (double)ttype;
    adj_mat(_j,_i) = (double)ttype;
    adj_mat_dirn(_i,_j) = (double)i; //index of the pair associated with this edge
    switch(ttype)
    {
      case 1:
        // _1set.push_back( _i );
        // _1set.push_back( _j );
        push_back_uniq( _1set, _i );
        push_back_uniq( _1set, _j );
        break;
      case 2:
        // _2set.push_back( _i );
        // _2set.push_back( _j );
        push_back_uniq( _2set, _i );
        push_back_uniq( _2set, _j );
        break;
      case 3:
        // _3set.push_back( _i );
        // _3set.push_back( _j );
        push_back_uniq( _3set, _i );
        push_back_uniq( _3set, _j );
        break;
      case -1:
        // _m1set.push_back( _i );
        // _m1set.push_back( _j );
        push_back_uniq( _m1set, _i );
        push_back_uniq( _m1set, _j );
        localidx_of_icurr = _i;
        localidx_of_iprev = _j;
        break;
      default:
        assert( false );

    }
    // ttype:
    // -1 : Dense Match
    // 1  : iprev+j
    // 2  : iprev-j
    // 3  : icurr-j


    // Set pair visibility mask into node visibility mask
    visibility_mask_nodes.row(_i) = visibility_mask_nodes.row(_i).cwiseProduct(   visibility_mask.row(i)   );
    visibility_mask_nodes.row(_j) = visibility_mask_nodes.row(_j).cwiseProduct(   visibility_mask.row(i)   );



    //
    // print data (to screen) for this pair

    cout << a_stamp_global_idx << "<--(type="<< ttype << ")-->" << b_stamp_global_idx  << "\t\t";
    cout << "[" << a << "   " << b << "]" << "\t\t";
    cout << "{" << _i << "   " << _j << "}" << endl;




  }

  //
  // Print Sets
  cout << "array lengths of : ";
  cout << "_m1set=" << _m1set.size()  << ";";
  cout << "_1set=" << _1set.size() << ";";
  cout << "_2set=" << _2set.size() << ";";
  cout << "_3set=" << _3set.size() << ";";
  cout << endl;

  cout << "_m1set: ";
  for( int i=0 ; i<_m1set.size() ; i++ ) {
      cout << _m1set[i] << ", ";
  }  cout << endl;

  cout << "_1set: ";
  for( int i=0 ; i<_1set.size() ; i++ ) {
      cout << _1set[i] << ", ";
  }  cout << endl;
  cout << "_2set: ";
  for( int i=0 ; i<_2set.size() ; i++ ) {
      cout << _2set[i] << ", ";
  }  cout << endl;
  cout << "_3set: ";
  for( int i=0 ; i<_3set.size() ; i++ ) {
      cout << _3set[i] << ", ";
  }  cout << endl;




  if( ( _m1set.size() == 2 ) && ( _1set.size() > 2 ) && ( _2set.size() > 2 ) && ( _3set.size() > 2 ) )
  {
      isValid_incoming_msg = true;
  }
  else
  {
      isValid_incoming_msg = false;
  }





  #if 0
  // Analysis on Visibility mask
  cout << "\033[1;31m";
  // Q) In the original pairs, count of in how many view-pair each of the features were visible

  VectorXd _tmp1 = VectorXd::Zero(visibility_mask.cols());
  VectorXd _tmp2 = VectorXd::Zero(visibility_mask.cols());
  VectorXd _tmp3 = VectorXd::Zero(visibility_mask.cols());
  VectorXd _tmp123 = VectorXd::Zero(visibility_mask.cols());
  for( int i=0 ; i<visibility_mask.rows() ; i++ ) //loop over each pair
  {
      VectorXd ___f = visibility_mask.row(i);
      _tmp123 = _tmp123 + ___f;
      switch( pair_type[i] )
      {
          case 1:
            _tmp1 = _tmp1 + ___f ; break;
          case 2:
            _tmp2 = _tmp2 + ___f ; break;
          case 3:
            _tmp3 = _tmp3 +___f ; break;
      }
  }

  cout << "In the original pairs, count of in how many view-pair each of the features were visible\n";
  cout << "featureiD, visible in n views, visibile in n views pairtype=1, pairtype=2, pairtype=3\n";
  for( int i=0 ; i<_tmp123.size() ; i+=5 )
  {
      for(int j=0; j<5 ; j++)
      {
          if ( (i+j) < _tmp123.size() )
              cout << "feat#" << i+j << ": " << _tmp123(i+j) << " " <<  _tmp1(i+j) << " " <<  _tmp2(i+j) << " " <<  _tmp3(i+j) << "\t";
      }
      cout << endl;
  }

  cout << "\033[0m\n";
  #endif
}

// #define write_image_debug( msg ) msg;
#define write_image_debug( msg ) ;
void LocalBundle::write_image( string fname, const cv::Mat& img)
{
    string base = string("/home/mpkuse/Desktop/bundle_adj/dump/localbundle_");
    write_image_debug( cout << "Writing file: "<< base << fname << endl );
    cv::imwrite( (base+fname).c_str(), img );
}


template <typename Derived>
void LocalBundle::write_EigenMatrix(const string& filename, const MatrixBase<Derived>& a)
{
  string base = string("/home/mpkuse/Desktop/bundle_adj/dump/mateigen_");
  std::ofstream file(base+filename);
  if( file.is_open() )
  {
    // file << a.format(CSVFormat) << endl;
    file << a << endl;
    write_image_debug(cout << "\033[1;32m" <<"Written to file: "<< filename  << "\033[0m\n" );
  }
  else
  {
    cout << "\033[1;31m" << "FAIL TO OPEN FILE for writing: "<< filename << "\033[0m\n";

  }
}


void LocalBundle::write_Matrix2d( const string& filename, const double * D, int nRows, int nCols )
{
  string base = string("/home/mpkuse/Desktop/bundle_adj/dump/mat2d_");
  std::ofstream file(base+filename);
  if( file.is_open() )
  {
    int c = 0 ;
    for( int i=0; i<nRows ; i++ )
    {
      file << D[c];
      c++;
      for( int j=1 ; j<nCols ; j++ )
      {
        file << ", " << D[c] ;
        c++;
      }
      file << "\n";
    }
    write_image_debug( cout << "\033[1;32m" <<"Written to file: "<< filename  << "\033[0m\n" );
  }
  else
  {
    cout << "\033[1;31m" << "FAIL TO OPEN FILE for writing: "<< filename << "\033[0m\n";

  }

}

void LocalBundle::write_Matrix1d( const string& filename, const double * D, int n  )
{
  string base = string("/home/mpkuse/Desktop/bundle_adj/dump/mat1d_");
  std::ofstream file(base+filename);
  if( file.is_open() )
  {
    file << D[0];
    for( int i=1 ; i<n ; i++ )
      file << ", " << D[i] ;
    file << "\n";
    write_image_debug(cout << "\033[1;32m" <<"Written to file: "<< filename  << "\033[0m\n");
  }
  else
  {
    cout << "\033[1;31m" << "FAIL TO OPEN FILE for writing: "<< filename << "\033[0m\n";

  }

}


//////////////////////////////////////// Plottting /////////////////////////////////

void LocalBundle::plot_dense_point_sets( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
            bool enable_text, bool enable_status_image, const string& msg ,
            cv::Mat& dst )
{
  assert( pts.rows() == 2 || pts.rows() == 3 );
  assert( mask.size() == pts.cols() );

  cv::Mat outImg = im.clone();
  ColorLUT lut = ColorLUT();

  int count = 0 ;
  int n_every = pts.cols()/75; //10
  for( int kl=0 ; kl<pts.cols() ; kl++ )
  {
    if( mask(kl) == 0 )
      continue;

      count++;

      cv::Point2d A( pts(0,kl), pts(1,kl) );

      //// >>>c = lut.get_color( int( pt1[xi][0] / 10 ) ) + lut.get_color( int( pt1[xi][1] / 10 ) )
      //// >>>c = c / 2
      // cv::Scalar color1 = lut.get_color( A.x / 10 ) ; // cv::Scalar( 255,255,0);
      // cv::Scalar color2 = lut.get_color( A.y / 10 ) ;
      // cv::Scalar color = cv::Scalar( (color1[2]+color2[2])/2 , (color1[1]+color2[1])/2, (color1[0]+color2[0])/2 );

      cv::Scalar color = lut.get_color( kl % 64 );


      cv::circle( outImg, A, 1, color, -1 );

      if( enable_text && kl%n_every == 0 ) // if write text only on 10% of the points to avoid clutter.
        cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
  }


  // Make status image
  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  string s = "Plotted "+to_string(count)+" of "+to_string(mask.size());


  if( !enable_status_image ) {
    // cv::putText( outImg, s.c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2 );
    dst = outImg;
    return;
  }

  cv::putText( status, s.c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );


  // if( msg.length() > 0 )
    // cv::putText( status, msg.c_str(), cv::Point(10,50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2 );

  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
        std::vector<std::string> msg_tokens = split(msg, ':');
        for( int h=0 ; h<msg_tokens.size() ; h++ )
            cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,50+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1.5 );
  }

  cv::vconcat( outImg, status, dst );


}


void LocalBundle::plot_point_sets( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
            const cv::Scalar& color, bool annotate, bool enable_status_image,
            const string& msg ,
            cv::Mat& dst )
{
  assert( pts.rows() == 2 || pts.rows() == 3 );
  assert( mask.size() == pts.cols() );

  cv::Mat outImg = im.clone();

  int count = 0 ;
  for( int kl=0 ; kl<pts.cols() ; kl++ )
  {
    if( mask(kl) == 0 )
      continue;

      count++;

      cv::Point2d A( pts(0,kl), pts(1,kl) );
      cv::circle( outImg, A, 2, color, -1 );

      if( annotate )
        cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
  }


  // Make status image
  cv::Mat status = cv::Mat(100, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  string s = "Plotted "+to_string(count)+" of "+to_string(mask.size());


  if( !enable_status_image ) {
    cv::putText( outImg, s.c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2 );
    return;
  }

  cv::putText( status, s.c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );


  if( msg.length() > 0 )
    cv::putText( status, msg.c_str(), cv::Point(10,70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2 );

  cv::vconcat( outImg, status, dst );


}

void LocalBundle::plot_point_sets( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                      const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                      const VectorXd& mask, const cv::Scalar& color, bool annotate_pts,
                      /*const vector<string>& msg,*/
                      const string& msg,
                    cv::Mat& dst )
{
  // ptsA : ptsB : 2xN or 3xN

  assert( imA.rows == imB.rows );
  assert( imA.cols == imB.cols );
  assert( ptsA.cols() == ptsB.cols() );
  assert( mask.size() == ptsA.cols() );

  cv::Mat outImg;
  cv::hconcat(imA, imB, outImg);

  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    if( mask(kl) == 0 )
      continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    cv::circle( outImg, A, 2,color, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color, -1 );

    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), cv::Scalar(255,0,0) );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
    }
  }



  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, (to_string(count)+" of "+to_string(ptsA.cols())).c_str(), cv::Point(10,70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );


  // put msg in status image
  if( msg.length() > 0 )
    cv::putText( status, msg.c_str(), cv::Point(110,70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );

  cv::vconcat( outImg, status, dst );

}

//////////////////////////////////////// END Plottting /////////////////////////////////
void LocalBundle::pointcloud_2_matrix( const vector<geometry_msgs::Point32>& ptCld, MatrixXd& G )
{
  int N = ptCld.size() ;
  G = MatrixXd( 3, N );
  for( int i=0 ; i<N ; i++ )
  {
    G(0,i) = ptCld[i].x;
    G(1,i) = ptCld[i].y;
    G(2,i) = 1.0;
    assert( ptCld[i].z == -7 );
  }
}


// Loop over each node and return the index of the node which is clossest to the specified stamp
int LocalBundle::find_indexof_node( const vector<Node*>& global_nodes, ros::Time stamp )
{
  ros::Duration diff;
  for( int i=0 ; i<global_nodes.size() ; i++ )
  {
    diff = global_nodes[i]->time_stamp - stamp;

    // cout << i << " "<< diff.sec << " " << diff.nsec << endl;

    if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
      return i;
    }
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
  return -1;
}


int LocalBundle::find_indexof_node( const vector<sensor_msgs::PointCloud>& global_nodes, ros::Time stamp )
{
  ros::Duration diff;
  for( int i=0 ; i<global_nodes.size() ; i++ )
  {
    diff = global_nodes[i].header.stamp - stamp;

    // cout << i << " "<< diff.sec << " " << diff.nsec << endl;

    if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
      return i;
    }
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
  return -1;
}



void LocalBundle::printMatrixInfo( const string& msg, const cv::Mat& M ) {
  cout << msg << ":" << "rows=" << M.rows << ", cols=" << M.cols << ", ch=" << M.channels() << ", type=" << type2str( M.type() ) << endl;

}


void LocalBundle::printMatrixInfo( const string& msg, const MatrixXd& M ) {
  cout << msg << ":" << "rows=" << M.rows() << ", cols=" << M.cols() << endl;
}


string LocalBundle::type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void LocalBundle::printMatrix2d( const string& msg, const double * D, int nRows, int nCols )
{
  cout << msg << endl;
  int c = 0 ;
  cout << "[\n";
  for( int i=0; i<nRows ; i++ )
  {
    cout << "\t[";
    for( int j=0 ; j<nCols ; j++ )
    {
      cout << D[c] << ", ";
      c++;
    }
    cout << "]\n";
  }
  cout << "]\n";

}

void LocalBundle::printMatrix1d( const string& msg, const double * D, int n  )
{
  cout << msg << endl;
  cout << "\t[";
  for( int i=0 ; i<n ; i++ )
    cout << D[i] << ", " ;
  cout << "]\n";
}

void LocalBundle::prettyprintPoseMatrix( const Matrix4d& M )
{
  cout << "YPR      : " << R2ypr(  M.topLeftCorner<3,3>() ).transpose() << "; ";
  cout << "Tx,Ty,Tz : " << M(0,3) << ", " << M(1,3) << ", " << M(2,3) << endl;
}

void LocalBundle::prettyprintPoseMatrix( const Matrix4d& M, string& return_string )
{
   Vector3d ypr;
   ypr = R2ypr(  M.topLeftCorner<3,3>()  );
  // cout << "YPR      : " << R2ypr(  M.topLeftCorner<3,3>() ).transpose() << "; ";
  // cout << "Tx,Ty,Tz : " << M(0,3) << ", " << M(1,3) << ", " << M(2,3) << endl;

  // return_string = "YPR=("+to_string(ypr(0))+","+to_string(ypr(1))+","+to_string(ypr(2))+")";
  // return_string += "  TxTyTz=("+ to_string(M(0,3))+","+ to_string(M(1,3))+","+ to_string(M(2,3))+")";

  char __tmp[200];
  snprintf( __tmp, 200, ":YPR=(%4.2f,%4.2f,%4.2f)  :TxTyTz=(%4.2f,%4.2f,%4.2f)",  ypr(0), ypr(1), ypr(2), M(0,3), M(1,3), M(2,3) );
  return_string = string( __tmp );
}

void LocalBundle::robust_triangulation(  const vector<pair<int,int> >& vector_of_pairs, /* local indices pair */
                           const vector<Matrix4d>& w_T_c1,
                           const vector<Matrix4d>& w_T_c2,
                           const vector<MatrixXd>& _1_unvn_undistorted,
                           const vector<MatrixXd>& _2_unvn_undistorted,
                           const vector<VectorXd>& mask,
                           MatrixXd& result_3dpts_in_world_cords
         )
{

  cout << "<><><><><>><><><><><><><><><><>><><>\nIn function  LocalBundle::robust_triangulation\n";
  // Loop over each point
  int n_pts = _1_unvn_undistorted[0].cols();
  int n_samples = mask.size();

  assert( n_samples == vector_of_pairs.size() );
  assert( n_samples == w_T_c1.size() );
  assert( n_samples == w_T_c2.size() );
  assert( n_samples == _1_unvn_undistorted.size() );
  assert( n_samples == _2_unvn_undistorted.size() );

  vector<double> baseline_of_viewpairs;
  vector<int> howmany_visible_in_viewpairs;
  for( int i=0  ; i<n_samples ; i++ ) {
      Matrix4d baseline = (w_T_c1[i]).inverse()  *  w_T_c2[i];
      double baseline_distance = baseline.col(3).head(3).norm();

      baseline_of_viewpairs.push_back(  baseline_distance  );
      howmany_visible_in_viewpairs.push_back(  mask[i].sum()  );
  }


  /////////////// Analysis of the visibility of each of the features ///////////////////
  // set #if 1 to print info on triangulation, ie. baseline etc.
  #if LOCALBUNDLE_DEBUG_LVL >= 1
  cout << "\033[1;35m";

  // uses mask vector.
  cout << "baselines between " << n_samples << " viewpairs\n";
  cout << "id, globalid, napid[], localid{}, baseline, n_visible_pts\n";

  VectorXd __xtnmih = VectorXd::Zero(mask[0].size());
  VectorXd __gdkeny = VectorXd::Zero(mask[0].size());

  for( int i=0  ; i<n_samples ; i++ ) {
    //   printMatrixInfo( "mask["+to_string(i)+"]", mask[i] );
      __xtnmih = __xtnmih + mask[i];


      Matrix4d baseline = (w_T_c1[i]).inverse()  *  w_T_c2[i];
      double baseline_distance = baseline.col(3).head(3).norm();
      cout << "viewpair#" << i <<  "  ";
      cout << " " << global_idx_of_nodes[ vector_of_pairs[i].first ]<< "," << global_idx_of_nodes[ vector_of_pairs[i].second ] << "   ";
      cout << "[" << nap_idx_of_nodes[ vector_of_pairs[i].first ]<< "," << nap_idx_of_nodes[ vector_of_pairs[i].second ]<< "]   ";
      cout << "{" << vector_of_pairs[i].first << "," << vector_of_pairs[i].second << "}\t\t";

      cout << "baseline_distance=" << baseline_distance << "\t" ;
      cout << "n_visible_pts=" << mask[i].sum() << endl;

      __gdkeny = __gdkeny + baseline_distance * mask[i];

  }
  cout << "\033[0m\n";

  #endif

  #if LOCALBUNDLE_DEBUG_LVL >= 2
  cout << "\033[1;35m";

  for( int i=0; i<__xtnmih.size() ; i+=5  )
  {
      for( int j=0 ; j<5 ; j++ )
      {
          if ( (i+j) < __xtnmih.size() )
          {
              cout << "feat#" << i+j << ":"<< __xtnmih[i+j] << ":"<< __gdkeny[i+j] <<  "\t";
            //   printf( "feat#%d:%d:%4.2f\t", i+j, __xtnmih[i+j], __gdkeny[i+j] );
          }
      }
      cout << endl;
  }


  cout << "\033[0m\n";


  #endif
  ////////////////////////// End of Analysis  //////////////////////////////////////////

  result_3dpts_in_world_cords = MatrixXd::Zero( 4, n_pts );
  assert( mask.size() == n_samples );
  assert( _2_unvn_undistorted.size() == n_samples );
  assert( _1_unvn_undistorted.size() == n_samples );
  assert( w_T_c1.size() == n_samples );
  assert( w_T_c2.size() == n_samples );
  assert( _1_unvn_undistorted[0].rows() == 2 || _1_unvn_undistorted[0].rows() == 3 );
  assert( _2_unvn_undistorted[0].rows() == 2 || _2_unvn_undistorted[0].rows() == 3 );


  cout << "There are total of "<< n_pts << " points to triangulate from " << n_samples << " view-pairs.\n";
  cout << "foreach of the "<< n_pts << " points svd() can yield 3d point having known where it was imaged. See chp12 on Triangulation in Hartley-Zizzerman book.\n";

  int pt_id = 0; //later loop for every point. Be careful with mask of this pair at this point.
  for( int pt_id=0 ; pt_id < n_pts ; pt_id++)
  {
    MatrixXd A = MatrixXd::Zero(4*n_samples, 4 );
    for( int s=0 ; s<n_samples ; s++ ) //loop over pairs
    {

      VectorXd this_mask = mask[s];
      if( mask[s](pt_id) == 0  )
        continue;

      double u, v, ud, vd;
      u = (_1_unvn_undistorted[s])(0,pt_id);
      v = (_1_unvn_undistorted[s])(1,pt_id);
      ud = (_2_unvn_undistorted[s])(0,pt_id);
      vd = (_2_unvn_undistorted[s])(1,pt_id);

      // Do this calculation only when mask of this pair at this pt_id is non-zero.

      Matrix4d P, Pd;
      P = (w_T_c1[s]).inverse();
      Pd = (w_T_c2[s]).inverse();

    //   double weight = 1.0;
      double weight = baseline_of_viewpairs[s]; // higher the baseline, higher should be the weight of this row of least squares.

      A.row( 4*s )   = (-P.row(1)  + v*P.row(2) ) * weight;
      A.row( 4*s+1 ) = ( P.row(0)  - u*P.row(2) ) * weight;
      A.row( 4*s+2 ) = (-Pd.row(1) + vd*Pd.row(2) ) * weight;
      A.row( 4*s+3 ) = ( Pd.row(0) - ud*Pd.row(2) ) * weight;


    }

    // From Chap. 12 (Triangulation) from Hartley-Zizzerman.
    JacobiSVD<MatrixXd> svd( A, ComputeThinU | ComputeThinV );
    MatrixXd Cp = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
    MatrixXd diff = Cp - A;
    // cout << "diff          : " << diff.array().abs().sum() << endl;
    // cout << "singularValues: " << svd.singularValues() << endl;
    // printMatrixInfo( "A", A );
    // printMatrixInfo( "U", svd.matrixU() );
    // printMatrixInfo( "S", svd.singularValues().asDiagonal() );
    // printMatrixInfo( "V", svd.matrixV() );
    MatrixXd V = svd.matrixV();


    // Last col of V. This is quite standard way to solve a homogeneous equation. Good explaination in Appendix 5.4 of Hartley-Zizzerman book
    // Vector4d X_3d = V.col(3);
    result_3dpts_in_world_cords.col( pt_id ) = V.col(3);


  }
  cout << "<><><><><>><><><><><><><><><><>><><>\nDone with function  LocalBundle::robust_triangulation\n";
}


///////////////// Pose compitation related helpers /////////////////

void LocalBundle::raw_to_eigenmat( const double * quat, const double * t, Matrix4d& dstT )
{
  Quaterniond q = Quaterniond( quat[0], quat[1], quat[2], quat[3] );

  dstT = Matrix4d::Zero();
  dstT.topLeftCorner<3,3>() = q.toRotationMatrix();

  dstT(0,3) = t[0];
  dstT(1,3) = t[1];
  dstT(2,3) = t[2];
  dstT(3,3) = 1.0;
}

void LocalBundle::eigenmat_to_raw( const Matrix4d& T, double * quat, double * t)
{
  assert( T(3,3) == 1 );
  Quaterniond q( T.topLeftCorner<3,3>() );
  quat[0] = q.w();
  quat[1] = q.x();
  quat[2] = q.y();
  quat[3] = q.z();
  t[0] = T(0,3);
  t[1] = T(1,3);
  t[2] = T(2,3);
}

void LocalBundle::rawyprt_to_eigenmat( const double * ypr, const double * t, Matrix4d& dstT )
{
  dstT = Matrix4d::Identity();
  Vector3d eigen_ypr;
  eigen_ypr << ypr[0], ypr[1], ypr[2];
  dstT.topLeftCorner<3,3>() = ypr2R( eigen_ypr );
  dstT(0,3) = t[0];
  dstT(1,3) = t[1];
  dstT(2,3) = t[2];
}

void LocalBundle::eigenmat_to_rawyprt( const Matrix4d& T, double * ypr, double * t)
{
  assert( T(3,3) == 1 );
  Vector3d T_cap_ypr = R2ypr( T.topLeftCorner<3,3>() );
  ypr[0] = T_cap_ypr(0);
  ypr[1] = T_cap_ypr(1);
  ypr[2] = T_cap_ypr(2);

  t[0] = T(0,3);
  t[1] = T(1,3);
  t[2] = T(2,3);
}


void LocalBundle::gi_T_gj( int locali, int localj, Matrix4d& M )
{
  Matrix4d w_T_i, w_T_j;
  global_nodes[ global_idx_of_nodes[locali] ]->getOriginalTransform(w_T_i);
  global_nodes[ global_idx_of_nodes[localj] ]->getOriginalTransform(w_T_j);

  M = w_T_i.inverse() * w_T_j;

}

Matrix4d LocalBundle::gi_T_gj( int locali, int localj )
{
  Matrix4d w_T_i, w_T_j;
  global_nodes[ global_idx_of_nodes[locali] ]->getOriginalTransform(w_T_i);
  global_nodes[ global_idx_of_nodes[localj] ]->getOriginalTransform(w_T_j);

  Matrix4d M;
  M = w_T_i.inverse() * w_T_j;
  return M;

}

void LocalBundle::w_T_gi( int locali, Matrix4d& M )
{
  global_nodes[ global_idx_of_nodes[locali] ]->getOriginalTransform(M);
}

Matrix4d LocalBundle::w_T_gi( int locali )
{
  Matrix4d M;
  global_nodes[ global_idx_of_nodes[locali] ]->getOriginalTransform(M);
  return M;
}

void LocalBundle::gi_T_w( int locali, Matrix4d& M )
{
  Matrix4d L;
  global_nodes[ global_idx_of_nodes[locali] ]->getOriginalTransform(L);
  M = L.inverse();
}

Matrix4d LocalBundle::gi_T_w( int locali)
{
  Matrix4d M;
  Matrix4d L;
  global_nodes[ global_idx_of_nodes[locali] ]->getOriginalTransform(L);
  M = L.inverse();
  return M;
}

Vector3d LocalBundle::R2ypr( const Matrix3d& R)
{
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}


Matrix3d LocalBundle::ypr2R( const Vector3d& ypr)
{
  double y = ypr(0) / 180.0 * M_PI;
  double p = ypr(1) / 180.0 * M_PI;
  double r = ypr(2) / 180.0 * M_PI;

  // Eigen::Matrix<double, 3, 3> Rz;
  Matrix3d Rz;
  Rz << cos(y), -sin(y), 0,
      sin(y), cos(y), 0,
      0, 0, 1;

  // Eigen::Matrix<double, 3, 3> Ry;
  Matrix3d Ry;
  Ry << cos(p), 0., sin(p),
      0., 1., 0.,
      -sin(p), 0., cos(p);

  // Eigen::Matrix<double, 3, 3> Rx;
  Matrix3d Rx;
  Rx << 1., 0., 0.,
      0., cos(r), -sin(r),
      0., sin(r), cos(r);

  return Rz * Ry * Rx;
}

///////////////////////////////// Pose Computation ///////////////////////////
void LocalBundle::ceresDummy()
{
    // Make Problem - Sample
    ceres::Problem problem;
    double x[4] = {3, -1, 0, 1};
    printMatrix1d( "x_init", x, 4 );
    ceres::CostFunction * cost_function = PowellResidue::Create();
    problem.AddResidualBlock( cost_function, NULL, x);




    // Solve
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    PowellResidueCallback callback(x);
    options.callbacks.push_back(&callback);
    options.update_state_every_iteration = true;

    ceres::Solve( options, &problem, &summary );

    cout << summary.FullReport() << endl;
    printMatrix1d( "x_final", x, 4 );

}

Matrix4d LocalBundle::crossRelPoseComputation3d2d( ceres::Solver::Summary& summary )
{
    cout << "   In function LocalBundle::crossRelPoseComputation3d2d()\n";
  assert( isValid_iprev_X_iprev_triangulated && iprev_X_iprev_triangulated.cols() > 0 );

  #if LOCALBUNDLE_DEBUG_LVL >= 1
  markObservedPointsOnCurrIm();
  markObservedPointsOnPrevIm();
  markObservedPoints__CP();
  mark3dPointsOnPrevIm( gi_T_w(localidx_of_iprev), "proj3dPointsOnPrev" );
  cout << "Done writing image files init\n";
  #endif


  // will use 3d points from iprev-5 to iprev+5 in ref-frame of iprev, ie. iprev_X_iprev_triang.
  // ceres-solver will estimate c_T_p.

  //
  // Reprojection residues. Only use good 3d points which are in the front.

  MatrixXd reprojection_residue = MatrixXd::Zero( 2, iprev_X_iprev_triangulated.cols() );
  reprojection_residue.row(0) =  unvn_undistorted[localidx_of_iprev].row(0) -  ( iprev_X_iprev_triangulated.row(0).array() / iprev_X_iprev_triangulated.row(2).array() ).matrix();
  reprojection_residue.row(1) =  unvn_undistorted[localidx_of_iprev].row(1) -  ( iprev_X_iprev_triangulated.row(1).array() / iprev_X_iprev_triangulated.row(2).array() ).matrix();


  //
  // Initial Guess
  cout << "~~~~~ Initial Guess c_T_p ~~~~~\n";
  Matrix4d T_cap;
  T_cap = gi_T_gj( localidx_of_icurr, localidx_of_iprev );
  prettyprintPoseMatrix( T_cap );

  // nullout yaw, tx,ty,tz
  cout << "nullout y,tx,ty,tz\n";
  double T_cap_ypr[10];
  double T_cap_quaternion[10], T_cap_t[10];
  eigenmat_to_rawyprt( T_cap, T_cap_ypr, T_cap_t );
  T_cap_ypr[0] = 0;
  T_cap_ypr[1] = 0;T_cap_ypr[2] = 0; //start with identity.
  T_cap_t[0] = 0;T_cap_t[1] = 0;T_cap_t[2] = 0;
  rawyprt_to_eigenmat(  T_cap_ypr, T_cap_t, T_cap );


  eigenmat_to_raw( T_cap, T_cap_quaternion, T_cap_t );
  cout << "T_cap:\n"<< T_cap << endl;
  printMatrix1d( "T_cap_quaternion",T_cap_quaternion, 3 );
  printMatrix1d( "T_cap_t", T_cap_t, 3 );
  prettyprintPoseMatrix( T_cap );
  #if LOCALBUNDLE_DEBUG_LVL >= 1
  string ___prettypose_string;
  prettyprintPoseMatrix( T_cap, ___prettypose_string  );
  mark3dPointsOnCurrIm( T_cap * p_T_w(), "proj3dPointsOnCurr_itr.initial", "c_T_p "+___prettypose_string );
  #endif




  // printMatrix1d( "T_cap_ypr",T_cap_ypr, 3 );
  // printMatrix1d( "T_cap_t", T_cap_t, 3 );
  cout << " ~~~~~ ~~~~~ ~~~~~ ~~~~~\n";



  //
  // Setup Problem
  ceres::Problem problem;
  VectorXd curr_mask = visibility_mask_nodes.row( localidx_of_icurr );

  int nresidual_terms = 0;
  for( int i=0 ; i<this->iprev_X_iprev_triangulated.cols() ; i++ )
  {
    if( curr_mask(i) == 0 ) { // this 3dpoint is not visible in this view
      continue;
    }


    // Only use good 3d points which are in the front.
    bool is_behind = ( (iprev_X_iprev_triangulated(2,i)/iprev_X_iprev_triangulated(3,i) ) < 0 )?true:false;
    bool good_ = ( reprojection_residue.col(i).norm() < 0.4 )?true:false;
    if( !is_behind && good_ ) {} else{ continue ;}

    nresidual_terms++;

    // 4DOF loss
    // ceres::CostFunction * cost_function = Align3d2d__4DOF::Create( this->iprev_X_iprev_triangulated.col(i),
                                                        //   unvn_undistorted[localidx_of_icurr].col(i),
                                                        //   _0_pitch_roll
                                                      /*T_cap_ypr[1], T_cap_ypr[2]*/ //);

    // 6DOF loss
    // ceres::CostFunction * cost_function = Align3d2d::Create( this->iprev_X_iprev_triangulated.col(i),
                                                        // unvn_undistorted[localidx_of_icurr].col(i) );

    ceres::CostFunction * cost_function = Align3dHomogeneous2d::Create( this->iprev_X_iprev_triangulated.col(i),
                                                        unvn_undistorted[localidx_of_icurr].col(i) );


    ceres::LossFunction *loss_function = NULL;
    loss_function = new ceres::HuberLoss(.01);
    // loss_function = new ceres::CauchyLoss(.05);

    problem.AddResidualBlock( cost_function, loss_function, T_cap_quaternion, T_cap_t  );

  }
  cout << "Total 3d points : "<< this->iprev_X_iprev_triangulated.cols() << endl;
  cout << "nresidual_terms : "<< nresidual_terms << endl;
  cout << "curr_mask.sum() : "<< curr_mask.sum() << endl;
  assert( nresidual_terms > 10 ); //should have atleast 10 good points for pnp

  //
  // 4DOF needs normalized step for yaw (not a euclidean step)
  // ceres::LocalParameterization* angle_local_parameterization = AngleLocalParameterization::Create();
  // problem.SetParameterization( &T_cap_ypr[0], angle_local_parameterization );

  // Quaternion parameterization
  ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
  problem.SetParameterization( T_cap_quaternion, quaternion_parameterization );


  //
  // Solve
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  // options.use_nonmonotonic_steps = true;
  // options.minimizer_type = ceres::LINE_SEARCH;
  // ceres::Solver::Summary summary;

  //
  // Callback
  #if LOCALBUNDLE_DEBUG_LVL >= 2
  Align3d2d__4DOFCallback callback;
  callback.setOptimizationVariables_quatertion_t( T_cap_quaternion, T_cap_t );
  // callback.setData( this );
  callback.setGid( global_idx_of_nodes[ localidx_of_icurr ] );
  options.callbacks.push_back(&callback);
  options.update_state_every_iteration = true;
  #endif

  ceres::Solve( options, &problem, &summary );

  cout << summary.BriefReport() << endl;

  #if LOCALBUNDLE_DEBUG_LVL >= 2
  vector_of_callbacks.push_back( callback );
  #endif



  //
  // Retrive optimized pose. This will be c_Tcap_p
  // rawyprt_to_eigenmat( T_cap_ypr, T_cap_t, T_cap );
  raw_to_eigenmat( T_cap_quaternion, T_cap_t, T_cap );

  cout << "Final pose for c_T_p:\n";
  prettyprintPoseMatrix( T_cap );

  #if LOCALBUNDLE_DEBUG_LVL >= 1
  string ___jks_prettypose_string;
  prettyprintPoseMatrix( T_cap, ___jks_prettypose_string  );

  char  __raw_cost_string[100];
  snprintf( __raw_cost_string, 100, "time_ms=%4.2f, e_0=%4.2f, e_%d=%4.2f, #e=%d", 1000.*summary.total_time_in_seconds, summary.initial_cost, summary.num_successful_steps, summary.final_cost, nresidual_terms );
  string __cost_string = string( __raw_cost_string );
  // string __cost_string = "e_0="+to_string(summary.initial_cost)+"   e_" + to_string(summary.num_successful_steps) + "="+to_string(summary.final_cost)+"  #etrms="+to_string(nresidual_terms);
  mark3dPointsOnCurrIm( T_cap * p_T_w(), "proj3dPointsOnCurr_itr.final", "c_T_p "+___jks_prettypose_string+":"+__cost_string );
  #endif

  Matrix4d to_return = T_cap.inverse();
  return to_return;


  // TODO - Decide if this pose is acceptable.
  // Instead of returning pose, return status where, this pose is acceptable on not.
  // The pose can be written to input argument.


}



//< similar to crossRelPoseJointOptimization3d2d() but has joint optimization to compute p_T_c, p_T_{c-1}, p_T_{c-2}, p_T_{c-3}, ...
Matrix4d LocalBundle::crossRelPoseJointOptimization3d2d()
{
    cout << "In function LocalBundle::crossRelPoseJointOptimization3d2d\n";
    // use the 3d points iprev_X_iprev_triangulated.
    assert( isValid_iprev_X_iprev_triangulated );

    // print info
    #if 1
        // Print Sets
        cout << "array lengths of : ";
        cout << "_m1set=" << _m1set.size()  << ";";
        cout << "_1set=" << _1set.size() << ";";
        cout << "_2set=" << _2set.size() << ";";
        cout << "_3set=" << _3set.size() << ";";
        cout << endl;

        cout << "_m1set: ";
        for( int i=0 ; i<_m1set.size() ; i++ ) {
            cout << _m1set[i] ;
            cout << "{" << nap_idx_of_nodes[_m1set[i]] <<  "}";
            cout << "[" << global_idx_of_nodes[_m1set[i]] <<  "]";
            cout << ", ";
        }  cout << endl;
        cout << "_1set: ";
        for( int i=0 ; i<_1set.size() ; i++ ) {
            cout << _1set[i] ;
            cout << "{" << nap_idx_of_nodes[_1set[i]] <<  "}";
            cout << "[" << global_idx_of_nodes[_1set[i]] <<  "]";
            cout << ", ";
        }  cout << endl;
        cout << "_2set: ";
        for( int i=0 ; i<_2set.size() ; i++ ) {
            cout << _2set[i];
            cout << "{" << nap_idx_of_nodes[_2set[i]] <<  "}";
            cout << "[" << global_idx_of_nodes[_2set[i]] <<  "]";
            cout << ", ";
        }  cout << endl;
        cout << "_3set: ";
        for( int i=0 ; i<_3set.size() ; i++ ) {
            cout << _3set[i] ;
            cout << "{" << nap_idx_of_nodes[_3set[i]] <<  "}";
            cout << "[" << global_idx_of_nodes[_3set[i]] <<  "]";
            cout << ", ";
        }  cout << endl;


        cout << "localidx_of_iprev: " << localidx_of_iprev << endl;
        cout << "localidx_of_icurr: " << localidx_of_icurr << endl;


        printMatrixInfo( "iprev_X_iprev_triangulated", iprev_X_iprev_triangulated);

    #endif


    markObservedPointsOnCurrIm();
    markObservedPointsOnPrevIm();
    mark3dPointsOnPrevIm( gi_T_w(localidx_of_iprev), "proj3dPointsOnPrev" ); // here 3d points are the `w_X_iprev_triangulated`


    //
    // Reprojection residues. Only use good 3d points which are in the front.
    MatrixXd reprojection_residue = MatrixXd::Zero( 2, iprev_X_iprev_triangulated.cols() );
    reprojection_residue.row(0) =  unvn_undistorted[localidx_of_iprev].row(0) -  ( iprev_X_iprev_triangulated.row(0).array() / iprev_X_iprev_triangulated.row(2).array() ).matrix();
    reprojection_residue.row(1) =  unvn_undistorted[localidx_of_iprev].row(1) -  ( iprev_X_iprev_triangulated.row(1).array() / iprev_X_iprev_triangulated.row(2).array() ).matrix();



    // use _3set to know all the curr-frames.
    vector<Matrix4d> c_Tcap_p;
    // vector< double* > c_Tcap_ypr_p;
    vector< double* > c_Tcap_quaternion_p;
    vector< double* > c_Tcap_t_p;
    // vector<Matrix3d> _0_pitch_roll;
    // vector<Align3d2d__4DOFCallback> vector_of_callbacks; //< this is now a class private variable
    vector_of_callbacks.clear(); //this is a class variable, which logs data from ceres.
    //setup initial guess.
    for( int i=0 ; i< _3set.size() ; i++ ) //these are all the localids of curr-frames.
    {
        Matrix4d _tmp;
        cout << "{gid=" << global_idx_of_nodes[ _3set[i] ] << "}_T_{gid="<< global_idx_of_nodes[ localidx_of_iprev ] << "}" << endl;
        _tmp = gi_T_gj( _3set[i], localidx_of_iprev );

        double * _tmp_quaternion = new double[5];
        double * _tmp_t = new double[5];
        eigenmat_to_raw( _tmp, _tmp_quaternion, _tmp_t );

        //nullout yaw, tx, ty, tz\n
        // _tmp_ypr[0] = 0;
        // _tmp_t[0] = 0; _tmp_t[1] = 0; _tmp_t[2] = 0;
        // Vector3d _0_p_r; _0_p_r << 0.0, _tmp_ypr[1], _tmp_ypr[2];
        // Matrix3d __tmp__0_pitch_roll = ypr2R( _0_p_r );



        c_Tcap_p.push_back( _tmp );
        // c_Tcap_ypr_p.push_back( _tmp_ypr );
        c_Tcap_t_p.push_back( _tmp_t );
        c_Tcap_quaternion_p.push_back( _tmp_quaternion );
        // _0_pitch_roll.push_back( __tmp__0_pitch_roll );

        // printMatrix1d( "_tmp_quaternion", _tmp_quaternion, 4 );
        // printMatrix1d( "_tmp_ypr", _tmp_ypr, 3 );
        // printMatrix1d( "_tmp_t", _tmp_t, 3 );

        // Make callbacks for each pose
        Align3d2d__4DOFCallback callback;
        callback.setOptimizationVariables_quatertion_t( _tmp_quaternion, _tmp_t );
        callback.setGid( global_idx_of_nodes[ _3set[i] ] );
        vector_of_callbacks.push_back( callback );

    }
    cout << "Done setting initial guess for joint pnp\n";

    //
    // formulate a joint optimization with relative pose constraint (use bigM trick)
    ceres::Problem problem;


    for( int i=0 ; i<_3set.size() ; i++ ) //loop over all elements in _3set (this set contains elements in visinity of icurr)
    {
        cout << "---\n";
        int localidx_of_curr_m_j = _3set[i]; // local id of curr-j \for j=0,1,2,...
        VectorXd curr_mask = visibility_mask_nodes.row(  localidx_of_curr_m_j  );
        cout << "Processing localidx_of_curr_-j: "<< localidx_of_curr_m_j ;
        cout << "  , its mask has "<< curr_mask.size() << " items";
        cout << "  , sizeof(iprev_X_iprev_triangulated): "<< iprev_X_iprev_triangulated.rows() << "X" << iprev_X_iprev_triangulated.cols() << endl;

        int nresidual_terms = 0;
        for( int f=0 ; f<this->iprev_X_iprev_triangulated.cols() ; f++ ) //loop over all 3d points
        {
            // cout << "f=" << f << "  ";
            if( curr_mask(f) == 0 ) { // this 3dpoint is not visible in this view
              continue;
            }

            // Only use good 3d points which are in the front.
            bool is_behind = ( (iprev_X_iprev_triangulated(2,f) ) < 0 )?true:false;
            bool good_ = ( reprojection_residue.col(f).norm() < 0.4 )?true:false;
            if( !is_behind && good_ ) { ; } else{ continue ;}

            nresidual_terms++;

            // 4DOF loss
            // ceres::CostFunction * cost_function = Align3d2d__4DOF::Create(
                                        // this->iprev_X_iprev_triangulated.col(i),
                                        // unvn_undistorted[localidx_of_curr_m_j].col(i),
                                        // _0_pitch_roll[0] );

            // 6DOF loss
            ceres::CostFunction * cost_function = Align3d2d::Create( this->iprev_X_iprev_triangulated.col(f),
                                                                unvn_undistorted[localidx_of_curr_m_j].col(f) );



            ceres::LossFunction *loss_function = NULL;
            loss_function = new ceres::HuberLoss(.01);
            // loss_function = new ceres::CauchyLoss(.05);


            problem.AddResidualBlock( cost_function, loss_function, c_Tcap_quaternion_p[i], c_Tcap_t_p[i]  );
            // cout << "AddResidualBlock()" << endl;
        }
        cout << "Added #residual terms: "<< nresidual_terms << endl;
        assert( nresidual_terms > 0 );

        cout << "---";
        cout << "idx_of_3set_ele="<< i << "; ";
        cout << "localidx_of_curr_m_j="<<localidx_of_curr_m_j <<"; ";
        if( localidx_of_curr_m_j == localidx_of_icurr ) cout << "*";
        cout << "n_residues=" << nresidual_terms << endl;

    }


    // Additional Residue terms to enforce the relative pose between the optimization variables do not change much as it is essentially accurate (because they are nearby poses)
    // TODO


    //
    // Local Parameterization
    for( int i=0 ; i<_3set.size() ; i++ ) //loop over all elements in _3set (this set contains elements in visinity of icurr)
    {
        // ceres::LocalParameterization* angle_local_parameterization = AngleLocalParameterization::Create();
        // problem.SetParameterization( c_Tcap_ypr_p[i], angle_local_parameterization );

        // Quaternion parameterization
        ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
        problem.SetParameterization( c_Tcap_quaternion_p[i], quaternion_parameterization );
    }

    //
    // solve(). Will get p_T_c, p_T_{c-1}, p_T_{c-2}, p_T_{c-3}, ...
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;

    // setup callbacks
    options.update_state_every_iteration = true;
    for( int k=0 ; k<vector_of_callbacks.size() ; k++ )
    {
        options.callbacks.push_back( &(vector_of_callbacks[k]) );
    }

    cout << "Solve()\n";
    ceres::Solve( options, &problem, &summary );
    cout << summary.BriefReport() << endl;


    //
    // retrive final outcome
    Matrix4d c_T_p_result;
    for( int idx_of_3set_ele=0 ; idx_of_3set_ele<_3set.size() ; idx_of_3set_ele++ ) //loop over all elements in _3set (this set contains elements in visinity of icurr)
    {
        // rawyprt_to_eigenmat( c_Tcap_ypr_p[idx_of_3set_ele], c_Tcap_t_p[idx_of_3set_ele], c_Tcap_p[idx_of_3set_ele] );
        raw_to_eigenmat( c_Tcap_quaternion_p[idx_of_3set_ele], c_Tcap_t_p[idx_of_3set_ele], c_Tcap_p[idx_of_3set_ele] );


        int localidx_of_curr_m_j = _3set[idx_of_3set_ele]; // local id of curr-j \for j=0,1,2,...
        if( localidx_of_curr_m_j == localidx_of_icurr ) { c_T_p_result = c_Tcap_p[idx_of_3set_ele]; }

    }


    // deallocate
    for( int i=0 ; i< _3set.size() ; i++ )
    {
        // delete [] c_Tcap_ypr_p[i];
        delete [] c_Tcap_quaternion_p[i];
        delete [] c_Tcap_t_p[i];
    }

    mark3dPointsOnCurrIm( c_T_p_result * p_T_w(), "proj3dPointsOnCurr_itr.final" );


    Matrix4d to_return__p_T_c = c_T_p_result.inverse();
    return to_return__p_T_c;



}

/////////////////////// Image Marking //////////////////////////
void LocalBundle::mark3dPointsOnCurrIm( const Matrix4d& cx_T_w, const string& fname_prefix, const string & msg  )
{
  assert( isValid_w_X_iprev_triangulated );
  int this_local_id = localidx_of_icurr;
  int this_global_id = global_idx_of_nodes[localidx_of_icurr];
  int this_nap_id    = nap_idx_of_nodes[localidx_of_icurr];


  MatrixXd v_X;
  v_X = cx_T_w * w_X_iprev_triangulated;

  MatrixXd reproj_pts;
  camera.perspectiveProject3DPoints( v_X, reproj_pts );

  cv::Mat outImg;
  string fullmsg = "lid="+to_string( this_local_id) + "; gid="+to_string(this_global_id) + "; nap_id="+to_string(this_nap_id);
  fullmsg += ":" + msg;
  plot_dense_point_sets( global_nodes[this_global_id]->getImageRef(), reproj_pts, visibility_mask_nodes.row(localidx_of_icurr),
                     true, true, fullmsg, outImg );


  write_image( to_string(nap_idx_of_nodes[ localidx_of_icurr ])+"_"+to_string(nap_idx_of_nodes[localidx_of_iprev])+"___"+to_string(this_nap_id)+"_"+fname_prefix+".png" , outImg );

}

void LocalBundle::mark3dPointsOnPrevIm( const Matrix4d& px_T_w, const string& fname_prefix, const string& msg )
{
  assert( isValid_w_X_iprev_triangulated );
  int this_local_id = localidx_of_iprev;
  int this_global_id = global_idx_of_nodes[this_local_id];
  int this_nap_id    = nap_idx_of_nodes[this_local_id];


  MatrixXd v_X;
  v_X = px_T_w * w_X_iprev_triangulated;

  MatrixXd reproj_pts;
  camera.perspectiveProject3DPoints( v_X, reproj_pts );

  cv::Mat outImg;
  string fullmsg = "lid="+to_string( this_local_id) + "; gid="+to_string(this_global_id) + "; nap_id="+to_string(this_nap_id);
  fullmsg += ":" + msg;
  plot_dense_point_sets( global_nodes[this_global_id]->getImageRef(), reproj_pts, visibility_mask_nodes.row(this_local_id),
                     true, true, fullmsg, outImg );


  write_image( to_string(nap_idx_of_nodes[ localidx_of_icurr ])+"_"+to_string(nap_idx_of_nodes[localidx_of_iprev])+"___"+to_string(this_nap_id)+"_"+fname_prefix+".png" , outImg );
}

void LocalBundle::markObservedPoints__CP()
{
    cv::Mat outImg_curr, outImg_prev;
    {
        assert( uv.size() == n_ptClds );

        int this_local_id = localidx_of_icurr;
        int this_global_id = global_idx_of_nodes[this_local_id];
        int this_nap_id    = nap_idx_of_nodes[this_local_id];

        cv::Mat outImg;
        string msg = "lid="+to_string( this_local_id) + "; gid="+to_string(this_global_id) + "; nap_id="+to_string(this_nap_id);
        plot_dense_point_sets( global_nodes[this_global_id]->getImageRef(), uv[this_local_id], visibility_mask_nodes.row(this_local_id),
                           true, true, msg, outImg_curr );

    }

    {
        assert( uv.size() == n_ptClds );

        int this_local_id = localidx_of_iprev;
        int this_global_id = global_idx_of_nodes[this_local_id];
        int this_nap_id    = nap_idx_of_nodes[this_local_id];

        cv::Mat outImg;
        string msg = "lid="+to_string( this_local_id) + "; gid="+to_string(this_global_id) + "; nap_id="+to_string(this_nap_id);
        plot_dense_point_sets( global_nodes[this_global_id]->getImageRef(), uv[this_local_id], visibility_mask_nodes.row(this_local_id),
                           true, true, msg, outImg_prev );

    }

    cv::Mat outImg;
    cv::hconcat( outImg_curr, outImg_prev, outImg );
    write_image( to_string(nap_idx_of_nodes[ localidx_of_icurr ])+"_"+to_string(nap_idx_of_nodes[localidx_of_iprev])+"_ObservedPoints.png" , outImg );

}

void LocalBundle::markObservedPointsOnCurrIm()
{
  assert( uv.size() == n_ptClds );

  int this_local_id = localidx_of_icurr;
  int this_global_id = global_idx_of_nodes[this_local_id];
  int this_nap_id    = nap_idx_of_nodes[this_local_id];

  cv::Mat outImg;
  string msg = "lid="+to_string( this_local_id) + "; gid="+to_string(this_global_id) + "; nap_id="+to_string(this_nap_id);
  plot_dense_point_sets( global_nodes[this_global_id]->getImageRef(), uv[this_local_id], visibility_mask_nodes.row(this_local_id),
                     true, true, msg, outImg );


  write_image( to_string(nap_idx_of_nodes[ localidx_of_icurr ])+"_"+to_string(nap_idx_of_nodes[localidx_of_iprev])+"___"+to_string(this_nap_id)+"_ObservedPointsOnCurrIm.png" , outImg );

}


void LocalBundle::markObservedPointsOnPrevIm()
{
  assert( uv.size() == n_ptClds );

  int this_local_id = localidx_of_iprev;
  int this_global_id = global_idx_of_nodes[this_local_id];
  int this_nap_id    = nap_idx_of_nodes[this_local_id];

  cv::Mat outImg;
  string msg = "lid="+to_string( this_local_id) + "; gid="+to_string(this_global_id) + "; nap_id="+to_string(this_nap_id);
  plot_dense_point_sets( global_nodes[this_global_id]->getImageRef(), uv[this_local_id], visibility_mask_nodes.row(this_local_id),
                     true, true, msg, outImg );

  write_image( to_string(nap_idx_of_nodes[ localidx_of_icurr ])+"_"+to_string(nap_idx_of_nodes[localidx_of_iprev])+"___"+to_string(this_nap_id)+"_ObservedPointsOnPrevIm.png" , outImg );
}






/////////////////////////////ROS Publishing helpers ///////////////////////////////
void LocalBundle::eigenpointcloud_2_ros_markermsg( const MatrixXd& M, visualization_msgs::Marker& marker, const string& ns )
{
    assert( M.rows()==3 || M.rows() == 4 );
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.header.seq = 0;
    marker.ns = ns; //"spheres";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = .05;
    marker.scale.y = .05;
    marker.scale.z = 1.05;

    marker.color.r = .8;
    marker.color.g = .8;
    marker.color.b = 0;
    marker.color.a = .9; // Don't forget to set the alpha!

    for( int i=0 ; i<M.cols() ; i++ )
    {
        geometry_msgs::Point pt;
        if( M.rows() == 3 )
        {
            pt.x = M(0,i);
            pt.y = M(1,i);
            pt.z = M(2,i);
        }
        else
        {
            pt.x = M(0,i) / M(3,i);
            pt.y = M(1,i) / M(3,i);
            pt.z = M(2,i) / M(3,i);
        }

        marker.points.push_back( pt );
    }

}

void LocalBundle::eigenpointcloud_2_ros_markertextmsg( const MatrixXd& M,
    vector<visualization_msgs::Marker>& marker_ary, const string& ns )
{
    marker_ary.clear();
    visualization_msgs::Marker marker;
    assert( M.rows()==3 || M.rows() == 4 );
    for( int i=0 ; i<M.cols() ; i++ )
    {
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.header.seq = 0;
        marker.ns = ns; //"spheres";
        marker.id = i;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = .1;
        marker.scale.y = .1;
        marker.scale.z = .1;

        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = .9; // Don't forget to set the alpha!

        marker.text = to_string( i );
        marker.pose.position.x = M(0,i);
        marker.pose.position.y = M(1,i);
        marker.pose.position.z = M(2,i);
        marker.pose.orientation.x = 0.;
        marker.pose.orientation.y = 0.;
        marker.pose.orientation.z = 0.;
        marker.pose.orientation.w = 1.;

        marker_ary.push_back( visualization_msgs::Marker(marker) );
    }

}
