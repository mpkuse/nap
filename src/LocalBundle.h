#pragma once
/** LocalBundle.h

  This class will form a local-bundle from NapMsg_opmode28. opmode_28
  msg contains:
    a) sensor_msgs/PointCloud[] bundle ==> tracked points in each of the images in question. look at PointCloud.header->stamp and node.timestamp to associate all the images with pose graph nodes
    b) sensor_msgs/Image visibility_table ==> N x F binary-valued-image. N is number of pairs and F is total dense features
    c) int32[] visibility_table_idx ==> timestamps of each image in the pair. size of this array is 2N.

  Author  : Manohar Kuse <mpkuse@connect.ust.hk>
  Created : 8th May, 2018

*/
#include <iostream>
#include <string>
#include <vector>


//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>


#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/Image.h>

#include <nap/NapMsg.h>

#include "Node.h"

using namespace std;
using namespace cv;


class LocalBundle {

public:
  LocalBundle( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes   );

  void sayHi();

private:
  int find_indexof_node(  const vector<Node*>& global_nodes, ros::Time stamp );
  int find_indexof_node( const vector<sensor_msgs::PointCloud>& global_nodes, ros::Time stamp );


  void write_image( string fname, const cv::Mat&);
  void plot_point_sets( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                        const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                        const VectorXd& mask,
                        const vector<string>& msg,
                      cv::Mat& outImg );

  // Given a pointcloud, get a Eigen::MatrixXd
  void pointcloud_2_matrix( const vector<geometry_msgs::Point32>& ptCld, MatrixXd& G );

};
