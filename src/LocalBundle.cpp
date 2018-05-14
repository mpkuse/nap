#include "LocalBundle.h"

void LocalBundle::sayHi()
{
  cout << "LocalBundle::Hello\n";
  camera.printCameraInfo(1);

  cout << "#nodes in global_nodes = " << global_nodes.size() << endl;

  int count = 0;
  for( int i=0 ; i< global_nodes.size() ; i++ ) {
    Matrix4d M;
    global_nodes[i]->getOriginalTransform(M);

    cout << "poses_" << i << "\n" << M << endl;
    count++;

    if( count > 3 )
      break;

  }


  cout << "n_pairs : " << n_pairs << endl;
  cout << "n_ptClds: " << uv.size() << endl;
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


void LocalBundle::randomViewTriangulate(int max_itr)
{
  // Pseudo code :
  // 1. a,b = randomly pick 2 nodes suchthat (edgefrom-a.type==1 or edgefrom-a.type==2 ) and (edgefrom-b.type==1 or edgefrom-b.type==2)
  // 2. p = find_path( a,b )
  // 3. _3dpts = triangulate( a, b )
  // 4. mask_a_b = AND( p1,  pk ) # to get list of points which were tracked correctly in each of a and b
  // 5. store( mask, _3dpt )
  //    repeat


  printMatrixInfo( "adj_mat", adj_mat );
  assert( n_pairs > 0 );
  assert( n_ptClds > 0 );
  assert( adj_mat.rows() == n_ptClds );
  assert( adj_mat.rows() == adj_mat.cols() );
  assert( adj_mat_dirn.rows() == adj_mat_dirn.cols() );
  assert( adj_mat_dirn.rows() == adj_mat.rows() );

  int n_success = 0;

  // Collection from each random try. This is collected so that DLT can be used for triangulation.
  vector<Matrix4d> w_T_c1, w_T_c2;
  vector<MatrixXd> _1_unvn_undistorted, _2_unvn_undistorted;
  vector<VectorXd> mmask;

  for( int itr=0 ; itr<max_itr ; itr++ )
  {
    cout << "---itr="<<itr << "---\n";

    //////////////////
    ///// Step-1 /////
    //////////////////

    // pick a rrandom node
    int _1 = rand() % n_ptClds;
    int _2 = rand() % n_ptClds;
    cout << "picked nodes with local index "<< _1 << " " << _2 << endl;

    if( _1 == _2 )
    {
      cout << "same, ignore it\n";
      continue;
    }

    // find edge from _1
    int _1_type = edge_type_from_node( _1 );
    int _2_type = edge_type_from_node( _2 );
    cout << "_1 is of type: "<< _1_type << endl;
    cout << "_2 is of type: "<< _2_type << endl;

    if(  !( ( _1_type==1 || _1_type == 2 ) && ( _2_type==1 || _2_type == 2 )  ) )
    {
      cout << "reject, since type is something other than 1 or 2\n";
      continue;
    }

    cout << "process\n";

    //////////////////
    ///// Step-2 ///// You exactly need to find the entire path. Just get an edge going out of _a and an edge going out of _2
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
      cout << _1 << " Was found in pair#" << _1_pairid << " in 1st postion\n";
    }
    else {
      _1_pairid = pair_1idx_of_node( _1 );
      assert( _1_pairid >=0 );
      _1_globalid = global_idx_of_pairs[2*_1_pairid+1];
      _1_localid = local_idx_of_pairs[2*_1_pairid+1];
      _1_napid = nap_idx_of_pairs[2*_1_pairid+1];
      cout << _1 << " Was found in pair#" << _1_pairid << " in 2nd postion\n";
    }



    _2_pairid = pair_0idx_of_node( _2 ); // if cannot find in 1st position, look at 2nd position
    if( _2_pairid >= 0 ) {
      _2_globalid = global_idx_of_pairs[2*_2_pairid];
      _2_localid = local_idx_of_pairs[2*_2_pairid];
      _2_napid = nap_idx_of_pairs[2*_2_pairid];
      cout << _2 << " Was found in pair#" << _2_pairid << " in 1st postion\n";
    }
    else {
      _2_pairid = pair_1idx_of_node( _2 );
      assert( _2_pairid >=0 );
      _2_globalid = global_idx_of_pairs[2*_2_pairid+1];
      _2_localid = local_idx_of_pairs[2*_2_pairid+1];
      _2_napid = nap_idx_of_pairs[2*_2_pairid+1];
      cout << _2 << " Was found in pair#" << _2_pairid << " in 2nd postion\n";

    }



    //////////////////
    ///// Step-3 ///// Triangulate
    //////////////////

    cout << "==>Triangulate globalid "<< _1_globalid << " and " << _2_globalid << endl;
    cout << "==>Triangulate localid  "<< _1_localid << " and " << _2_localid << endl;
    cout << "==>Triangulate napid    "<< _1_napid << " and " << _2_napid << endl;
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
    cout << "#verified pts in " << _1 << " = "<< visibility_mask.row(_1_pairid).sum() << endl;
    cout << "#verified pts in " << _2 << " = "<< visibility_mask.row(_2_pairid).sum() << endl;
    cout << "#verified pts in composed_mask = "<< composed_mask.sum() << endl;


    mmask.push_back( composed_mask );


    // Visualize this _1, _2
    // cv::Mat dst;
    // plot_point_sets( global_nodes[_1_globalid]->getImageRef(), uv[_1],  _1_napid,
    //                  global_nodes[_2_globalid]->getImageRef(), uv[_2],  _2_napid,
    //                  composed_mask, cv::Scalar(0,0,255), true, "", dst
    //               );
    // write_image( to_string(_1_napid) + "_" + to_string(_2_napid)+"_observed.png" , dst );


  }
  cout << "n_success="<< n_success << " max_itr="<< max_itr << endl;
  assert( n_success == w_T_c1.size() );
  assert( n_success == w_T_c2.size() );
  assert( n_success == _1_unvn_undistorted.size() );
  assert( n_success == _2_unvn_undistorted.size() );
  assert( n_success == mmask.size() );



  // Kalman filter to get better estimates of 3d points.
  MatrixXd w_X_triang; //triangulated 3d points in world co-prdinates from multiple views
  robust_triangulation( w_T_c1, w_T_c2, _1_unvn_undistorted, _2_unvn_undistorted, mmask, w_X_triang );


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

    //plot reprojected points


    //save
    assert( _m1set.size() == 2 );
    write_image( to_string(nap_idx_of_nodes[_m1set[0]])+"_"+to_string(nap_idx_of_nodes[_m1set[1]])+"___"+to_string(node_napid)+"_observed.png" , outImg );

  }


}


void LocalBundle::multiviewTriangulate()
{
  cout << "LocalBundle::multiviewTriangulate()\n";

  // in this function we assume that all the data is ready.
  // will make use of uv, uv_undistorted, global_idx_of_pairs, local_idx_of_pairs etc
  assert( n_pairs > 0 );
  assert( uv.size() == uv_undistorted.size() );
  assert( 2*n_pairs == global_idx_of_pairs.size() );
  assert( 2*n_pairs == local_idx_of_pairs.size() );
  assert( n_pairs   == pair_type.size() );
  assert( n_pairs   == visibility_mask.rows() );


  // lets work with pairs, type=1
  int __i = 0; //ith pair
  vector<MatrixXd> _w_3d_pts;
  for( int __i=0 ; __i < n_pairs ; __i++) // you can also use other items for triangulation. YOu just need to AND the masks
  {
  cout << " Pair#" << __i << ";" <<
          "type=" << pair_type[__i] << ";" <<
          "global_idx(" << global_idx_of_pairs[2*__i] << "," << global_idx_of_pairs[2*__i+1] << ");" <<
          "local__idx(" << local_idx_of_pairs[2*__i] << "," << local_idx_of_pairs[2*__i+1] << ");" <<
          "nap_idx_of_pairs(" << nap_idx_of_pairs[2*__i] << "," << nap_idx_of_pairs[2*__i+1] << ");" <<
          endl;

  int g0 = global_idx_of_pairs[2*__i];
  int g1 = global_idx_of_pairs[2*__i+1];
  int l0 = local_idx_of_pairs[2*__i];
  int l1 = local_idx_of_pairs[2*__i+1];
  int n0 = nap_idx_of_pairs[2*__i];
  int n1 = nap_idx_of_pairs[2*__i+1];

  Matrix4d w_T_g0;
  global_nodes[g0]->getOriginalTransform(w_T_g0);
  Matrix4d w_T_g1;
  global_nodes[g1]->getOriginalTransform(w_T_g1);



  MatrixXd _w_3d; // a 4xN matrix will be return. Think about exactly which
                // co-ordinate system you want this to be in. World co-ordinate makes most sense.
  triangulate_points( g0, uv_undistorted[ l0 ],   g1, uv_undistorted[ l1 ],    _w_3d);
  printMatrixInfo( "_w_3d", _w_3d );
  cout << "_w_3d\n" << _w_3d.block( 0,0, _w_3d.rows(), 5)  << endl;

  _w_3d_pts.push_back( _w_3d );

  }

/*
  assert( global_nodes[global_idx_of_pairs[2*__i]]->valid_image() );
  cv::Mat _tmp = global_nodes[g0]->getImageRef();
  MatrixXd _tmp_uv = uv[l0];
  cv::Mat outImg;
  plot_point_sets( _tmp, _tmp_uv, visibility_mask.row(l0), cv::Scalar(0,69,255), true, "MSG", outImg );
  write_image( to_string(g0)+".png", outImg );


  //
  //
  MatrixXd _g0_3d;
  _g0_3d = w_T_g0.inverse() * _w_3d;
  MatrixXd _g0_2d;
  camera.perspectiveProject3DPoints( _g0_3d, _g0_2d );
  printMatrixInfo( "_g0_2d", _g0_2d );
  cout << "_g0_2d\n" << _g0_2d.block( 0,0, _g0_2d.rows(), 5)  << endl;
  plot_point_sets( _tmp, _g0_2d, visibility_mask.row(l0), cv::Scalar(0,169,0), true, "MSG", outImg );
  write_image( to_string(g0)+"_reproj.png", outImg );





  _tmp = global_nodes[g1]->getImageRef();
  _tmp_uv = uv[l1];
  plot_point_sets( _tmp, _tmp_uv, visibility_mask.row(l1), cv::Scalar(0,69,255), true, "MSG", outImg );
  write_image( to_string(g1)+".png", outImg );


  //
  //
  MatrixXd _g1_3d;
  _g1_3d = w_T_g1.inverse() * _w_3d;
  MatrixXd _g1_2d;
  camera.perspectiveProject3DPoints( _g1_3d, _g1_2d );
  printMatrixInfo( "_g1_2d", _g1_2d );
  cout << "_g1_2d\n" << _g1_2d.block( 0,0, _g1_2d.rows(), 5)  << endl;
  plot_point_sets( _tmp, _g1_2d, visibility_mask.row(l1), cv::Scalar(0,255,0), true, "MSG", outImg );
  write_image( to_string(g1)+"_reproj.png", outImg );
*/

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
  this->camera.printCameraInfo(1);

  this->global_nodes = global_nodes;

  assert( this->camera.isValid() );
  assert( this->global_nodes.size() > 0 );

  /////////////////////////// loop on tracked points /////////////////////////////
  this->n_ptClds = msg->bundle.size();
  this->n_features = msg->bundle[0].points.size();
  for( int i=0 ; i<msg->bundle.size() ; i++ )
  {
    cout << "---\nPointbundle "<< i << endl;
    int seq = find_indexof_node( global_nodes, msg->bundle[i].header.stamp );
    int seq_debug = msg->bundle[i].header.seq;

    MatrixXd e_ptsA;
    pointcloud_2_matrix(msg->bundle[i].points, e_ptsA  );
    uv.push_back( e_ptsA );


    MatrixXd e_ptsA_undistored;
    this->camera.undistortPointSet( e_ptsA, e_ptsA_undistored, false );
    printMatrixInfo( "e_ptsA_undistored", e_ptsA_undistored);
    uv_undistorted.push_back( e_ptsA_undistored );

    // use the camera to get normalized co-ordinates and undistorded co-ordinates
    MatrixXd e_ptsA_undistored_normalized;
    this->camera.undistortPointSet( e_ptsA, e_ptsA_undistored_normalized, true );
    printMatrixInfo( "e_ptsA_undistored_normalized", e_ptsA_undistored_normalized);
    unvn_undistorted.push_back( e_ptsA_undistored_normalized );


    global_idx_of_nodes.push_back( seq );
    nap_idx_of_nodes.push_back(seq_debug);


    cout << "pointcloud : idx=" << seq <<  "\t#pts=" <<  msg->bundle[i].points.size()  << "\tdebug_idx=" << seq_debug <<  "\tvalid_image: " << global_nodes[seq]->valid_image()  << endl;

  }


  ///////////////////////////////////////// loop on pairs ///////////////////////////////
  this->n_pairs = N_pairs;
  cout << "Setting adj_matrix dimensions\n";
  this->adj_mat = MatrixXd::Zero(this->n_ptClds, this->n_ptClds);
  this->adj_mat_dirn = MatrixXd::Zero(this->n_ptClds, this->n_ptClds);
  visibility_mask_nodes = MatrixXd::Ones( n_ptClds, n_features  );
  printMatrixInfo( "adj_mat", adj_mat );
  printMatrixInfo( "adj_mat_dirn", adj_mat_dirn );
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
        _1set.push_back( _i );
        _1set.push_back( _j );
        break;
      case 2:
        _2set.push_back( _i );
        _2set.push_back( _j );
        break;
      case 3:
        _3set.push_back( _i );
        _3set.push_back( _j );
        break;
      case -1:
        _m1set.push_back( _i );
        _m1set.push_back( _j );
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

    /*
    cv::Mat outImg;
    vector<string> __tmp;
    MatrixXd e_ptsA, e_ptsB;
    assert( _i >= 0 );
    assert( _j >= 0 );
    pointcloud_2_matrix(msg->bundle[_i].points, e_ptsA  );
    pointcloud_2_matrix(msg->bundle[_j].points, e_ptsB  );

    VectorXd mask = e_visibility_table.row(i);
    // VectorXd mask = VectorXd::Ones( e_ptsA.cols(), 1 );
    plot_point_sets( global_nodes[a_stamp_global_idx]->getImageRef(), e_ptsA,  a_stamp_global_idx,
                      global_nodes[b_stamp_global_idx]->getImageRef(), e_ptsB, b_stamp_global_idx,
                      mask,
                     __tmp, outImg );
    // cv::imshow( "huhuX", outImg);
    write_image( to_string(a)+"_"+to_string(b)+".png", outImg );
    */

  }




}

void LocalBundle::write_image( string fname, const cv::Mat& img)
{
    cout << "Writing file "<< fname << endl;
    string base = string("/home/mpkuse/Desktop/bundle_adj/dump/org_");
    cv::imwrite( (base+fname).c_str(), img );
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

      cv::Scalar color1 = lut.get_color( A.x / 10 ) ; // cv::Scalar( 255,255,0);
      cv::Scalar color2 = lut.get_color( A.y / 10 ) ;
      cv::Scalar color = cv::Scalar( (color1[2]+color2[2])/2 , (color1[1]+color2[1])/2, (color1[0]+color2[0])/2 );

      //// >>>c = lut.get_color( int( pt1[xi][0] / 10 ) ) + lut.get_color( int( pt1[xi][1] / 10 ) )
      //// >>>c = c / 2
      cv::circle( outImg, A, 1, color, -1 );

      if( enable_text && kl%n_every == 0 ) // if write text only on 10% of the points to avoid clutter.
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



void LocalBundle::robust_triangulation( const vector<Matrix4d>& w_T_c1,
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

  result_3dpts_in_world_cords = MatrixXd::Zero( 4, n_pts );
  assert( mask.size() == n_samples );
  assert( _2_unvn_undistorted.size() == n_samples );
  assert( _1_unvn_undistorted.size() == n_samples );
  assert( w_T_c1.size() == n_samples );
  assert( w_T_c2.size() == n_samples );
  assert( _1_unvn_undistorted[0].rows() == 2 || _1_unvn_undistorted[0].rows() == 3 );
  assert( _2_unvn_undistorted[0].rows() == 2 || _2_unvn_undistorted[0].rows() == 3 );


  cout << "There are total of "<< n_pts << " points to triangulate from " << n_samples << " view-pairs\n";

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

      A.row( 4*s )   = -P.row(1)  + v*P.row(2);
      A.row( 4*s+1 ) =  P.row(0)  - u*P.row(2);
      A.row( 4*s+2 ) = -Pd.row(1) + vd*Pd.row(2);
      A.row( 4*s+3 ) =  Pd.row(0) - ud*Pd.row(2);

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
