#include "Node.h"

Node::Node()
{
  m_e_ = false;
  m_org_ = false;
  m_3dpts = false;
  m_2dfeats = false;
  m_path_pose = false;
  m_path_pose_corrected = false;
}

Node::Node( ros::Time time_stamp, geometry_msgs::Pose pose )
{
  this->time_stamp = ros::Time(time_stamp);
  this->time_pose = ros::Time(time_stamp);
  this->pose = geometry_msgs::Pose(pose);

  // TODO
  // Consider also storing original poses. e_p and e_q can be evolving poses (ie. optimization variables)
  // Basically need to revisit this when getting it to work with ceres

  // opt_position = new double[3];
  // opt_position[0] = pose.pose.position.x;
  // opt_position[1] = pose.pose.position.y;
  // opt_position[2] = pose.pose.position.z;
  // e_p << pose.pose.position.x, pose.pose.position.y,pose.pose.position.z;
  e_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
  org_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
  m_e_ = true;

  // opt_quat = new double[4];
  // opt_quat[0] = pose.pose.orientation.x;
  // opt_quat[1] = pose.pose.orientation.y;
  // opt_quat[2] = pose.pose.orientation.z;
  // opt_quat[3] = pose.pose.orientation.w;
  e_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
  org_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
  m_org_ = true;
  // TODO extract covariance

  m_3dpts = false;
  m_2dfeats = false;


  path_pose_p = Vector3d(-10,-10,-10); //position
  path_pose_q = Quaterniond(1.,0.,0.,0.); //quaternion
  m_path_pose = false;

  path_pose_corrected_p = Vector3d(-10,-10,-10);
  path_pose_corrected_q = Quaterniond(1.,0.,0.,0.);
  m_path_pose_corrected = false;
}

void Node::getCurrTransform(Matrix4d& M)
{
  M = Matrix4d::Zero();
  M.col(3) << e_p, 1.0;
  // Matrix3d R = e_q.toRotationMatrix();
  M.topLeftCorner<3,3>() = e_q.toRotationMatrix();


  // cout << "e_p\n" << e_p << endl;
  // cout << "e_q [w,x,y,z]\n" << e_q.w() << " " << e_q.x() << " " << e_q.y() << " " << e_q.z() << " " << endl;
  // cout << "R\n" << R << endl;
  // cout << "M\n"<< M << endl;
}

void Node::getOriginalTransform(Matrix4d& M)
{
  M = Matrix4d::Zero();
  M.col(3) << org_p, 1.0;
  // Matrix3d R = org_q.toRotationMatrix();
  M.topLeftCorner<3,3>() = org_q.toRotationMatrix();

  // cout << "e_p\n" << e_p << endl;
  // cout << "e_q [w,x,y,z]\n" << e_q.w() << " " << e_q.x() << " " << e_q.y() << " " << e_q.z() << " " << endl;
  // cout << "R\n" << R << endl;
  // cout << "M\n"<< M << endl;
}

////////////// 3d points
void Node::setPointCloud( ros::Time time, const vector<geometry_msgs::Point32> & points )
{
  ptCld = Matrix<double,3,Dynamic>(3,points.size());
  for( int i=0 ; i<points.size() ; i++ )
  {
    ptCld(0,i) = points[i].x;
    ptCld(1,i) = points[i].y;
    ptCld(2,i) = points[i].z;
  }
  this->time_pcl = ros::Time(time);
  m_3dpts = true;
}

void Node::setPointCloud( ros::Time time, const Matrix<double,3,Dynamic>& e )
{
  ptCld = Matrix<double,3,Dynamic>( e );
  this->time_pcl = ros::Time(time);
  m_3dpts = true;
}

const Matrix<double,3,Dynamic>& Node::getPointCloud( )
{
  return ptCld;
}


void Node::getPointCloudHomogeneous( MatrixXd& M )
{
  M = MatrixXd(4, ptCld.cols() );
  for( int i=0 ; i<ptCld.cols() ; i++ )
  {
    M(0,i) = ptCld(0,i);
    M(1,i) = ptCld(1,i);
    M(2,i) = ptCld(2,i);
    M(3,i) = 1.0;
  }
}

////////////// 2d tracked features
void Node::setFeatures2dHomogeneous( ros::Time time, const vector<geometry_msgs::Point32> & points )
{
  feat2d = Matrix<double,3,Dynamic>(3,points.size());
  for( int i=0 ; i<points.size() ; i++ )
  {
    feat2d(0,i) = points[i].x;
    feat2d(1,i) = points[i].y;
    feat2d(2,i) = points[i].z;
  }
  this->time_feat2d = ros::Time(time);
  m_2dfeats = true;
}

void Node::setFeatures2dHomogeneous( ros::Time time, const Matrix<double,3,Dynamic>& e )
{
  feat2d = Matrix<double,3,Dynamic>( e );
  this->time_feat2d = ros::Time(time);
  m_2dfeats = true;
}

void Node::getFeatures2dHomogeneous( MatrixXd& M )
{
  // return feat2d;
  M = MatrixXd(feat2d );
}


/////////////// Image
void Node::setImage( ros::Time time, const cv::Mat& im )
{
  image = cv::Mat(im.clone());
  this->time_image = ros::Time(time);
}


const cv::Mat& Node::getImageRef()
{
  return image;
}


/////////////// Nap Cluster Map
void Node::setNapClusterMap( ros::Time time, const cv::Mat& im )
{
  this->nap_clusters = cv::Mat(im.clone());
  this->time_nap_clustermap = ros::Time(time);
}


const cv::Mat& Node::getNapClusterMap()
{
  return nap_clusters;
}






void Node::write_debug_xml( char * fname )
{
  cv::FileStorage fs( fname, cv::FileStorage::WRITE );

  if( valid_3dpts() ) {
    // 3d pts
    MatrixXd c_M; //4xN
    getPointCloudHomogeneous(c_M);

    cv::Mat c_M_mat;
    cv::eigen2cv( c_M, c_M_mat );

    fs << "c_3dpts" << c_M_mat;
    fs << "c_3dpts_timestamp" << time_pcl.toSec();
  }

  // feat2d
  if( valid_2dfeats() ) {
    MatrixXd c_feat2d;
    getFeatures2dHomogeneous( c_feat2d );

    cv::Mat c_feat2d_mat;
    cv::eigen2cv( c_feat2d, c_feat2d_mat);
    fs << "c_feat2d" << c_feat2d_mat;
    fs << "c_feat2d_timestamp" << time_feat2d.toSec();
  }

  // Transform
  if( valid_currTransform() ) {
    Matrix4d w_T_c;
    getCurrTransform( w_T_c );

    cv::Mat w_T_c_mat;
    cv::eigen2cv( w_T_c, w_T_c_mat );
    fs <<  "w_T_c" << w_T_c_mat; //this is from camera pose msg
    fs << "w_T_c_timestamp" << time_pose.toSec();
  }


  // Save npy file of cluster map
  if( this->valid_clustermap() )
  {
    char newf[200];
    sprintf( newf, "%s_clustermap.png", fname );
    // cnpy::npy_save( newf,  )
    cv::imwrite( newf, this->nap_clusters );
    fs << "clustermap_timestamp" << time_nap_clustermap.toSec();
  }

  if( valid_image() )
  {
    char newf[200];
    sprintf( newf, "%s_image.png", fname );
    cv::imwrite( newf, this->image );
    fs << "image_timestamp" << time_image.toSec();
  }

  // write path pose
  if( valid_pathpose( 1 ) ) //vio
  {
    Matrix4d pathpose__w_T_c;
    getPathPose( pathpose__w_T_c, 1 );

    cv::Mat pathpose__w_T_c_mat;
    cv::eigen2cv( pathpose__w_T_c, pathpose__w_T_c_mat );
    fs <<  "pathpose_nominal__w_T_c_mat" << pathpose__w_T_c_mat;
    fs <<  "pathpose_nominal__timestamp" << path_pose_timestamp.toSec();

  }

  if( valid_pathpose( 0 ) ) //after pose-graph-optimization
  {
    Matrix4d pathpose__w_T_c;
    getPathPose( pathpose__w_T_c, 0 );

    cv::Mat pathpose__w_T_c_mat;
    cv::eigen2cv( pathpose__w_T_c, pathpose__w_T_c_mat );
    fs <<  "pathpose_corrected__w_T_c_mat" << pathpose__w_T_c_mat;
    fs <<  "pathpose_corrected__timestamp" << path_pose_corrected_timestamp.toSec();
  }

  fs.release();
}

bool Node::load_debug_xml( const string& fname  )
{
  cout << "-----Open file : " << fname << "------" << endl;
  cv::FileStorage fs( fname, cv::FileStorage::READ );
  if( fs.isOpened() == false )
  {
    ROS_ERROR_STREAM( "in Node::load_debug_xml, Cannot open file " << fname );
    return false;
  }

  ros::Time timestamp;
  double t;

  //
  // 3dpts + timestamp



  // 2dfeat + timestamp


  // transform (of camera) + timestamp


  // nap clusters. ignore for now


  // image + timestamp
  fs["image_timestamp"] >> t;
  cout << timestamp << endl;
  timestamp.fromSec(t);
  cout << timestamp << endl;
  char newf[200];
  sprintf( newf, "%s_image.png", fname.c_str() );
  cv::Mat image = cv::imread( newf );
  if( image.empty() ) {
    cout << "Cannot read image";
  }
  else {
    setImage( timestamp, image );
  }




  // transform (of camera from vio-path) + timestamp

  // transform (of camera from after pose-graph-optimization-path) + timestamp

  fs.release();
}


void Node::setPathPose( const geometry_msgs::Pose& pose, int id )
{
  // from VIO
  if( id == 1 ) {
    this->path_pose_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
    this->path_pose_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
    m_path_pose = true;
    return;
  }

  // after pose-graph-optimization
  if( id == 0 ) {
    this->path_pose_corrected_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
    this->path_pose_corrected_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
    m_path_pose_corrected = true;
    return;
  }

  ROS_ERROR( "Invalid id in Node::setPathPose()");
}

void Node::setPathPose( const geometry_msgs::Pose& pose, int id, ros::Time timestamp )
{
  // from VIO
  if( id == 1 ) {
    this->path_pose_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
    this->path_pose_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
    path_pose_timestamp = timestamp;
    m_path_pose = true;
    return;
  }

  // after pose-graph-optimization
  if( id == 0 ) {
    this->path_pose_corrected_p = Vector3d(pose.position.x, pose.position.y,pose.position.z );
    this->path_pose_corrected_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
    path_pose_corrected_timestamp = timestamp;
    m_path_pose_corrected = true;
    return;
  }

  ROS_ERROR( "Invalid id in Node::setPathPose()");
}

// returns pose of node in world co-ord
bool Node::getPathPose( Matrix4d& w_T_c, int id )
{
    // from vio
    if( id == 1 ) {
      w_T_c = Matrix4d::Zero();
      w_T_c.col(3) << path_pose_p, 1.0;
      // Matrix3d R = e_q.toRotationMatrix();
      w_T_c.topLeftCorner<3,3>() = path_pose_q.toRotationMatrix();
      return m_path_pose;
    }

    // after pose-graph-optimization
    if( id == 0 ) {
      w_T_c = Matrix4d::Zero();
      w_T_c.col(3) << path_pose_corrected_p, 1.0;
      // Matrix3d R = e_q.toRotationMatrix();
      w_T_c.topLeftCorner<3,3>() = path_pose_corrected_q.toRotationMatrix();
      return m_path_pose_corrected;
    }

}


bool Node::valid_pathpose( int id )
{
  //from vio
  if( id == 1 )
    return m_path_pose;

  // after pose-graph-optimization
  if( id == 0 )
    return m_path_pose_corrected;
}
