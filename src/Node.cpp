#include "Node.h"

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

  // opt_quat = new double[4];
  // opt_quat[0] = pose.pose.orientation.x;
  // opt_quat[1] = pose.pose.orientation.y;
  // opt_quat[2] = pose.pose.orientation.z;
  // opt_quat[3] = pose.pose.orientation.w;
  e_q = Quaterniond( pose.orientation.w, pose.orientation.x,pose.orientation.y,pose.orientation.z );
  // TODO extract covariance
}

void Node::getCurrTransform(Matrix4d& M)
{
  M = Matrix4d::Zero();
  M.col(3) << e_p, 1.0;
  Matrix3d R = e_q.toRotationMatrix();
  M.topLeftCorner<3,3>() = e_q.toRotationMatrix();

  // cout << "e_p\n" << e_p << endl;
  // cout << "e_q [w,x,y,z]\n" << e_q.w() << " " << e_q.x() << " " << e_q.y() << " " << e_q.z() << " " << endl;
  // cout << "R\n" << R << endl;
  // cout << "M\n"<< M << endl;
}

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
}

void Node::setPointCloud( ros::Time time, const Matrix<double,3,Dynamic>& e )
{
  ptCld = Matrix<double,3,Dynamic>( e );
  this->time_pcl = ros::Time(time);
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


void Node::setImage( ros::Time time, const cv::Mat& im )
{
  image = cv::Mat(im.clone());
  this->time_image = ros::Time(time);
}


const cv::Mat& Node::getImageRef()
{
  return image;
}