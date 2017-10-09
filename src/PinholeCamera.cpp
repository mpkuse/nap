#include "PinholeCamera.h"


PinholeCamera::PinholeCamera( string config_file )
{

  cv::FileStorage fs( config_file, cv::FileStorage::READ );
  if( !fs.isOpened() )
  {
    ROS_ERROR( "Cannot open config file : %s", config_file.c_str() );
    ROS_ERROR( "Quit");
    mValid = false;
    exit(1);
  }
  this->config_file_name = string(config_file);

  cout << "---Camera Config---\n";
  fs["model_type"] >> config_model_type;     cout << "config_model_type:"<< config_model_type << endl;
  fs["camera_name"] >> config_camera_name;   cout << "config_camera_name:" << config_camera_name << endl;
  fs["image_width"] >> config_image_width;   cout << "config_image_width:" << config_image_width << endl;
  fs["image_height"] >> config_image_height; cout << "config_image_height:" << config_image_height << endl;


  fs["projection_parameters"]["fx"] >> _fx;
  fs["projection_parameters"]["fy"] >> _fy;
  fs["projection_parameters"]["cx"] >> _cx;
  fs["projection_parameters"]["cy"] >> _cy;
  cout << "projection_parameters :: " << _fx << " " << _fy << " " << _cx << " " << _cy << " " << endl;

  fs["distortion_parameters"]["k1"] >> _k1;
  fs["distortion_parameters"]["k2"] >> _k2;
  fs["distortion_parameters"]["p1"] >> _p1;
  fs["distortion_parameters"]["p2"] >> _p2;
  cout << "distortion_parameters :: " << _k1 << " " << _k2 << " " << _p1 << " " << _p2 << " " << endl;
  cout << "---    ---\n";

  // Define the 3x3 Projection matrix eigen and/or cv::Mat.
  m_K = cv::Mat::zeros( 3,3,CV_32F );
  m_K.at<float>(0,0) = _fx;
  m_K.at<float>(1,1) = _fy;
  m_K.at<float>(0,2) = _cx;
  m_K.at<float>(1,2) = _cy;
  m_K.at<float>(2,2) = 1.0;

  m_D = cv::Mat::zeros( 4, 1, CV_32F );
  m_D.at<float>(0,0) = _k1;
  m_D.at<float>(1,0) = _k2;
  m_D.at<float>(2,0) = _p1;
  m_D.at<float>(3,0) = _p2;
  cout << "m_K" << m_K << endl;
  cout << "m_D" << m_D << endl;

  // Define 4x1 vector of distortion params eigen and/or cv::Mat.
  e_K << _fx, 0.0, _cx,
        0.0,  _fy, _cy,
        0.0, 0.0, 1.0;

  e_D << _k1 , _k2, _p1 , _p2;
  cout << "e_K" << m_K << endl;
  cout << "e_D" << m_D << endl;
  mValid = true;

}



void PinholeCamera::_1channel_to_2channel( const cv::Mat& input, cv::Mat& output )
{
  assert( input.rows == 2 && input.channels()==1 );
  output = cv::Mat( 1, input.cols, CV_32FC2 );
  for( int l=0 ; l<input.cols ; l++ )
  {
    output.at<cv::Vec2f>(0,l)[0] = input.at<float>(0,l);
    output.at<cv::Vec2f>(0,l)[1] = input.at<float>(1,l);
  }

}

void PinholeCamera::_2channel_to_1channel( const cv::Mat& input, cv::Mat& output )
{
  assert( input.rows == 1 && input.channels()==2 );
  output = cv::Mat( 2, input.cols, CV_32F );
  for( int l=0 ; l<input.cols ; l++ )
  {
    output.at<float>(0,l) = input.at<cv::Vec2f>(0,l)[0];
    output.at<float>(1,l) = input.at<cv::Vec2f>(0,l)[1];
  }
}



//////////////////////////////////////////
void PinholeCamera::perspectiveProject3DPoints( cv::Mat& _3dpts, Matrix4f& T,
                                  cv::Mat& out_pts  )
{
  MatrixXf c_X;
  cv::cv2eigen( _3dpts, c_X ); //3xN

  MatrixXf cm_X;
  cm_X = T * c_X;

  cv::Mat _3dpts_cm;
  cv::eigen2cv( cm_X, _3dpts_cm );

  perspectiveProject3DPoints( _3dpts_cm, out_pts );
}


// Input 3d points in homogeneous co-ordinates 4xN matrix.
void PinholeCamera::perspectiveProject3DPoints( cv::Mat& _3dpts,
                                  cv::Mat& out_pts )
{
    // cout << "Not Implemented perspectiveProject3DPoints\n";

    // //call opencv's projectPoints()
    // cv::Mat rVec = cv::Mat::zeros( 3,1, CV_32F );
    // cv::Mat tVec = cv::Mat::zeros( 3,1, CV_32F );
    // out_pts = cv::Mat( 2, _3dpts.cols, CV_32FC1 );
    //
    //
    // print_cvmat_info( "_3dpts", _3dpts );
    // print_cvmat_info( "rVec", rVec );
    // print_cvmat_info( "tVec", tVec );
    // print_cvmat_info( "m_K", m_K );
    // print_cvmat_info( "m_D", m_D );
    // print_cvmat_info( "out_pts", out_pts );
    // cv::projectPoints( _3dpts, rVec, tVec, m_K, m_D, out_pts );
    // return;

    // DIY - Do It Yourself Projection
    MatrixXf c_X;
    cv::cv2eigen( _3dpts, c_X ); //4xN
    // c_X.row(0).array() /= c_X.row(3).array();
    // c_X.row(1).array() /= c_X.row(3).array();
    // c_X.row(2).array() /= c_X.row(3).array();
    // c_X.row(3).array() /= c_X.row(3).array();

    Matrix3f cam_intrin;
    cv::cv2eigen( m_K, cam_intrin );

    Vector4f cam_dist;
    cv::cv2eigen( m_D, cam_dist );


    // K [ I | 0 ]
    MatrixXf I_0;
    I_0 = Matrix4f::Identity().topLeftCorner<3,4>();
    // MatrixXf P1;
    // P1 = cam_intrin * I_0; //3x4

    // Project and Perspective Divide
    MatrixXf im_pts;
    im_pts = I_0 * c_X; //in normalized image co-ordinate. Beware that distortion need to be applied in normalized co-ordinates
    im_pts.row(0).array() /= im_pts.row(2).array();
    im_pts.row(1).array() /= im_pts.row(2).array();
    im_pts.row(2).array() /= im_pts.row(2).array();

    // Apply Distortion
    MatrixXf Xdd = MatrixXf( im_pts.rows(), im_pts.cols() );
    for( int i=0 ; i<im_pts.cols() ; i++)
    {
      float r2 = im_pts(0,i)*im_pts(0,i) + im_pts(1,i)*im_pts(1,i);
      float c = 1.0f + (float)k1()*r2 + (float)k2()*r2*r2;
      Xdd(0,i) = im_pts(0,i) * c + 2.0f*(float)p1()*im_pts(0,i)*im_pts(1,i) + (float)p2()*(r2 + 2.0*im_pts(0,i)*im_pts(0,i));
      Xdd(1,i) = im_pts(1,i) * c + 2.0f*(float)p2()*im_pts(0,i)*im_pts(1,i) + (float)p1()*(r2 + 2.0*im_pts(1,i)*im_pts(1,i));
      Xdd(2,i) = 1.0f;
    }

    MatrixXf out = cam_intrin * Xdd;


    // cv::eigen2cv( im_pts, out_pts );
    cv::eigen2cv( out, out_pts );

}



void PinholeCamera::undistortPointSet( const cv::Mat& pts_observed_image_space, cv::Mat& pts_undistorted_image_space )
{
  // cout << "Not Implemented undistort_points\n";

  cv::Mat _in, _out;
  _1channel_to_2channel( pts_observed_image_space, _in );

  // this is because, undistort function takes in a 1xN 2channel input
  //call opencv's undistortPoints()
  cv::undistortPoints( _in, _out, m_K,  m_D, cv::Mat::eye(3,3, CV_32F), m_K );
  // If you do not set m_K the returned points will be in normalized co-ordinate system.


  _2channel_to_1channel( _out, pts_undistorted_image_space );

}



void PinholeCamera::print_cvmat_info( string msg, const cv::Mat& A )
{
  cout << msg << ":" << "rows=" << A.rows << ", cols=" << A.cols << ", ch=" << A.channels() << ", type=" << type2str( A.type() ) << endl;
}

string PinholeCamera::type2str(int type) {
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
