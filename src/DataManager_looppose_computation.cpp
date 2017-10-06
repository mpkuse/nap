#include "DataManager.h"

// Functions for computation of relative pose

void DataManager::pose_from_2way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& cm_T_c )
{
  cout << "Do Qin Tong's Code here, for computation of relative pose\nNot Yet implemented";

}

#define _DEBUG_3WAY

void DataManager::pose_from_3way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& cm_T_c )
{
  cout << "Code here for 3way match pose computation\n";



  // Get nodes from the timestamps
  int ix_curr = find_indexof_node(msg->t_curr);
  int ix_prev = find_indexof_node(msg->t_prev);
  int ix_curr_m = find_indexof_node(msg->t_curr_m);
  assert( ix_curr > 0 && ix_prev>0 && ix_curr_m>0 );

  #ifdef _DEBUG_3WAY
    // Open DEBUG file
    char fname[200];
    sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d.opencv", ix_curr, ix_prev );
    cout << "Writing file : " << fname << endl;
    cv::FileStorage debug_fp( fname, cv::FileStorage::WRITE );
  #endif

  //
  // Step-1 : Get 3way matches from msg as matrix
  //
  cv::Mat mat_pts_curr, mat_pts_prev, mat_pts_curr_m; //2xN 1-channel
  extract_3way_matches_from_napmsg(msg, mat_pts_curr, mat_pts_prev, mat_pts_curr_m);


  // The above matches are in observed_image_space. For geometry, need to
  // convert these into undistorted_image_space. Qin Tong had told me it
  // publishes the keyframe original, however the pointcloud is published in
  // undistorted_image_space.
  // TODO: To speedup this conversion, build a lookup table for every point. Non-priority
  cv::Mat undistorted_pts_curr, undistorted_pts_prev, undistorted_pts_curr_m; //2xN 1-channel
  camera.undistortPointSet( mat_pts_curr, undistorted_pts_curr);
  camera.undistortPointSet( mat_pts_prev, undistorted_pts_prev);
  camera.undistortPointSet( mat_pts_curr_m, undistorted_pts_curr_m);


#ifdef _DEBUG_3WAY

  debug_fp << "K" << camera.m_K;
  debug_fp << "D" << camera.m_D;
  debug_fp << "mat_pts_curr" << mat_pts_curr ;
  debug_fp << "undistorted_pts_curr" << undistorted_pts_curr ;
  debug_fp << "mat_pts_prev" << mat_pts_prev ;
  debug_fp << "undistorted_pts_prev" << undistorted_pts_prev ;
  debug_fp << "mat_pts_curr_m" << mat_pts_curr_m ;
  debug_fp << "undistorted_pts_curr_m" << undistorted_pts_curr_m ;


  // Collect Images - for debug
  cv::Mat curr_im, prev_im, curr_m_im;
  // Remember to disable the subscriber for images when not using this debug. Will save up lot of memory here!
  curr_im = this->nNodes[ix_curr]->getImageRef();
  prev_im = this->nNodes[ix_prev]->getImageRef();
  curr_m_im = this->nNodes[ix_curr_m]->getImageRef();

  // Plot 3way match
  cv::Mat dst_plot_3way;
  this->plot_3way_match( curr_im, mat_pts_curr, prev_im, mat_pts_prev, curr_m_im, mat_pts_curr_m, dst_plot_3way );

  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d.jpg", ix_curr, ix_prev );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst_plot_3way );
#endif

  //
  // Step-2 : Triangulation using matched points from curr and curr-1
  //
  cv::Mat c_3dpts_4N;//4xN
  triangulate_points( ix_curr, mat_pts_curr, ix_curr_m, mat_pts_curr_m, c_3dpts_4N );


  _perspective_divide_inplace( c_3dpts_4N );
  // cv::Mat c_3dpts_3N;
  // _from_homogeneous( c_3dpts_4N, c_3dpts_3N );
  //
  // cv::Mat c_3dpts_4N_a; //TODO: Remove this and the next line
  // _to_homogeneous( c_3dpts_3N, c_3dpts_4N_a );


#ifdef _DEBUG_3WAY
  // // write 3d pts to file
  print_cvmat_info( "c_3dpts_4N", c_3dpts_4N );
  debug_fp << "c_3dpts_4N" << c_3dpts_4N ;

  // // Reproject 3d pts.
  cv::Mat _reprojected_pts_into_c;
  camera.perspectiveProject3DPoints( c_3dpts_4N, _reprojected_pts_into_c );
  print_cvmat_info( "reprojected_pts_into_c" , _reprojected_pts_into_c );
  debug_fp << "reprojected_pts_into_c" << _reprojected_pts_into_c;

  //


#endif

  //
  // Step-3 : PnP using a) 3d pts from step2 and b) corresponding matches from prev
  //


  //
  // Step-4 : Put the relative pose from step3 into eigen matrix in required convention
  //



#ifdef _DEBUG_3WAY
  debug_fp.release();
#endif


}





// //////////////// UTILS ////////////////////////// //

void DataManager::extract_3way_matches_from_napmsg( const nap::NapMsg::ConstPtr& msg,
      cv::Mat&mat_pts_curr, cv::Mat& mat_pts_prev, cv::Mat& mat_pts_curr_m )
{
  int N = msg->curr.size();
  assert( N>0 && msg->curr.size()==N && msg->prev.size()==N && msg->curr_m.size()==N );

  mat_pts_curr = cv::Mat(2,N,CV_32F);
  mat_pts_prev = cv::Mat(2,N,CV_32F);
  mat_pts_curr_m = cv::Mat(2,N,CV_32F);
  for( int kl=0 ; kl<N ; kl++ )
  {
    mat_pts_curr.at<float>(0,kl) = (float)msg->curr[kl].x;
    mat_pts_curr.at<float>(1,kl) = (float)msg->curr[kl].y;
    mat_pts_prev.at<float>(0,kl) = (float)msg->prev[kl].x;
    mat_pts_prev.at<float>(1,kl) = (float)msg->prev[kl].y;
    mat_pts_curr_m.at<float>(0,kl) = (float)msg->curr_m[kl].x;
    mat_pts_curr_m.at<float>(1,kl) = (float)msg->curr_m[kl].y;
  }
}




void DataManager::plot_3way_match( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                      const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                      const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                      cv::Mat& dst)
{
  cv::Mat zre = cv::Mat(curr_im.rows, curr_im.cols, CV_8UC3, cv::Scalar(128,128,128) );

  cv::Mat dst_row1, dst_row2;
  cv::hconcat(curr_im, prev_im, dst_row1);
  cv::hconcat(curr_m_im, zre, dst_row2);
  cv::vconcat(dst_row1, dst_row2, dst);



  // Draw Matches
  cv::Point2d p_curr, p_prev, p_curr_m;
  for( int kl=0 ; kl<mat_pts_curr.cols ; kl++ )
  {
    if( mat_pts_curr.channels() == 2 ){
      p_curr = cv::Point2d(mat_pts_curr.at<cv::Vec2f>(0,kl)[0], mat_pts_curr.at<cv::Vec2f>(0,kl)[1] );
      p_prev = cv::Point2d(mat_pts_prev.at<cv::Vec2f>(0,kl)[0], mat_pts_prev.at<cv::Vec2f>(0,kl)[1] );
      p_curr_m = cv::Point2d(mat_pts_curr_m.at<cv::Vec2f>(0,kl)[0], mat_pts_curr_m.at<cv::Vec2f>(0,kl)[1] );
    }
    else {
      p_curr = cv::Point2d(mat_pts_curr.at<float>(0,kl),mat_pts_curr.at<float>(1,kl) );
      p_prev = cv::Point2d(mat_pts_prev.at<float>(0,kl),mat_pts_prev.at<float>(1,kl) );
      p_curr_m = cv::Point2d(mat_pts_curr_m.at<float>(0,kl),mat_pts_curr_m.at<float>(1,kl) );
    }

    cv::circle( dst, p_curr, 4, cv::Scalar(255,0,0) );
    cv::circle( dst, p_prev+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(0,255,0) );
    cv::circle( dst, p_curr_m+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(0,0,255) );
    cv::line( dst,  p_curr, p_prev+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
    cv::line( dst,  p_curr, p_curr_m+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );

    // cv::circle( dst, cv::Point2d(pts_curr[kl]), 4, cv::Scalar(255,0,0) );
    // cv::circle( dst, cv::Point2d(pts_prev[kl])+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(0,255,0) );
    // cv::circle( dst, cv::Point2d(pts_curr_m[kl])+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(0,0,255) );
    // cv::line( dst,  cv::Point2d(pts_curr[kl]), cv::Point2d(pts_prev[kl])+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
    // cv::line( dst,  cv::Point2d(pts_curr[kl]), cv::Point2d(pts_curr_m[kl])+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );
  }
}



void DataManager::triangulate_points( int ix_curr, const cv::Mat& mat_pts_curr,
                         int ix_curr_m, const cv::Mat& mat_pts_curr_m,
                         cv::Mat& c_3dpts )
  {
  //
  // Step-1 : Relative pose, Projection Matrix of ix_curr and ix_curr_m

  // K [ I | 0 ]
  MatrixXd I_0;
  I_0 = Matrix4d::Identity().topLeftCorner<3,4>();
  MatrixXd P1 = camera.e_K * I_0; //3x4

  // K [ ^{c-1}T_c ] ==> K [ inv( ^wT_{c-1} ) * ^wT_c ]
  Matrix4d w_T_cm;
  nNodes[ix_curr_m]->getCurrTransform(w_T_cm);//4x4
  Matrix4d w_T_c;
  nNodes[ix_curr]->getCurrTransform(w_T_c);//4x4

  MatrixXd Tr;
  Tr = w_T_cm.inverse() * w_T_c; //relative transform
  MatrixXd P2 = camera.e_K * Tr.topLeftCorner<3,4>(); //3x4

  cv::Mat xP1(3,4,CV_64F );
  cv::Mat xP2(3,4,CV_64F );
  cv::eigen2cv( P1, xP1 );
  cv::eigen2cv( P2, xP2 );


  //
  // Step-2 : OpenCV Triangulate
  cv::triangulatePoints( xP1, xP2,  mat_pts_curr, mat_pts_curr_m,   c_3dpts );


}



// Given an input matrix of size nXN returns (n+1)xN. Increase the dimension by 1
// Add an extra dimension with 1.0 as the entry. This is NOT an inplace operation.
void DataManager::_to_homogeneous( const cv::Mat& in, cv::Mat& out )
{
  cout << "Not implemented _to_homogeneous\n";
  out = cv::Mat( in.rows+1, in.cols, in.type() );
  int lastindex_of_out = out.rows-1;
  for( int iu=0 ; iu<in.cols ; iu++ ) // loop across all points
  {
    for( int iv=0 ; iv<in.rows ; iv++ ) //x,y,z, (dimensions)
    {
      out.at<float>(iv,iu) = (float)in.at<float>(iv,iu) ;
    }
    out.at<float>(lastindex_of_out, iu) = 1.0;
  }
}


// Given input of size nxN. return (n-1)xN. Decrease the dimension by 1
// Y = X[0:3,:] / X[3,:]. This is not an inplace operation
void DataManager::_from_homogeneous( const cv::Mat& in, cv::Mat& out )
{
  //TODO : Take this part into a function, say fromHomogeneous and toHomogeneous. Probably use these functions from opencv. Verify before though, points are thought to be row-wise. Best is to use my own method
  out = cv::Mat( in.rows-1, in.cols, in.type() );
  int lastindex_of_in = in.rows-1;
  for( int iu=0 ; iu<in.cols ; iu++ )
  {
    for( int iv=0 ; iv<out.rows ; iv++ ) //x,y,z,
    {
      out.at<float>(iv,iu) = (float)in.at<float>(iv,iu) / (float)in.at<float>(lastindex_of_in,iu);
    }
  }
}


// Input of nxN returns nxN. After this call the last row will be 1s.
// X = X / X[n-1,:]
void _perspective_divide_inplace( cv::Mat& in )
{
  for( int iu = 0 ; iu<in.cols ; iu++ )
  {
    float to_divide = in.at<float>( in.row-1, iu );
    for( int j=0 ; j<in.rows ; j++ )
    {
      in.at<float>( j, iu ) /= to_divide;
    }
  }
}




void DataManager::print_cvmat_info( string msg, const cv::Mat& A )
{
  cout << msg << ":" << "rows=" << A.rows << ", cols=" << A.cols << ", ch=" << A.channels() << ", type=" << type2str( A.type() ) << endl;
}

string DataManager::type2str(int type) {
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
