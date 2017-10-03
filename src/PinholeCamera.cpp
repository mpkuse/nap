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
void PinholeCamera::perspectiveProject3DPoints( cv::Mat& _3dpts )
{
    cout << "Not Implemented perspectiveProject3DPoints";
}

void PinholeCamera::triangulatePoints(  )
{
    cout << "Not Implemented triangulatePoints";
}
