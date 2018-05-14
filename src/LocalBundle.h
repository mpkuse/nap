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



#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/SVD>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include <nap/NapMsg.h>
#include "PinholeCamera.h"
#include "Node.h"
#include "ColorLUT.h"

using namespace std;
using namespace cv;


class LocalBundle {

public:
  LocalBundle( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera   );

  void multiviewTriangulate();
  void randomViewTriangulate(int max_itr);
  void sayHi();

private:
  int find_indexof_node(  const vector<Node*>& global_nodes, ros::Time stamp );
  int find_indexof_node( const vector<sensor_msgs::PointCloud>& global_nodes, ros::Time stamp );


  void write_image( string fname, const cv::Mat&);

  // Plots [ imA | imaB ] with points correspondences
  // [Input]
  //    imA, imB : Images
  //    ptsA, ptsB : 2xN or 3xN
  //    idxA, idxB : Index of each of the image. This will appear in status part. No other imppact of these
  //    mask : binary mask, 1 ==> plot, 0 ==> ignore point. Must be of length N
  // [Output]
  //    outImg : Output image
  void plot_point_sets( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                        const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                        const VectorXd& mask,const cv::Scalar& color, bool annotate_pts,
                        /*const vector<string>& msg,*/
                        const string& msg,
                      cv::Mat& outImg );

   // Plots im with its points.
   // [Input]
   //     im : Image
   //     pts : 2xN or 3xN
   //     mask : binary mask, 1 ==> plot, 0 ==> ignore point. Must be of length N
   //     color : color of the circles. eg. cv::Scalar(255,128,0)
   //     annotate_pts : true will also overlay putText ie. index of the point on the image
   //     enable_status_image : true will append a status image of height 100px at the bottom of the image.
   void plot_point_sets( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
                  const cv::Scalar& color, bool annotate_pts, bool enable_status_image, const string& msg ,
                  cv::Mat& outImg );



   // Plotting a point set on single image. Tailored to plot large number of points. Colors of each
   // point smartly set
  void plot_dense_point_sets( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
              bool enable_text, bool enable_status_image, const string& msg ,
              cv::Mat& dst );

  // Given a pointcloud, get a Eigen::MatrixXd
  void pointcloud_2_matrix( const vector<geometry_msgs::Point32>& ptCld, MatrixXd& G );

  void printMatrixInfo( const string& msg, const cv::Mat& M );
  void printMatrixInfo( const string& msg, const MatrixXd& M );
  string type2str(int type);


  int n_ptClds;
  vector<int> _1set, _2set, _3set, _m1set; //list of nodes in each of the types. TODO. currently not in use
  vector<MatrixXd> uv; // original points for each frame. Same as that received
  vector<MatrixXd> unvn_undistorted; // normalized image co-ordinates. Undistorted points.
  vector<MatrixXd> uv_undistorted; // undistored points
  vector<int> global_idx_of_nodes; //global index of nodes
  vector<int> nap_idx_of_nodes; //nap index of nodes

  MatrixXd visibility_mask_nodes; // n_ptsClds x 100. 100 is total features.

  MatrixXd adj_mat;
  MatrixXd adj_mat_dirn; ///< Adjacency matrix

  // pairs info
  vector<int> global_idx_of_pairs;
  vector<int> local_idx_of_pairs;
  vector<int> nap_idx_of_pairs; // index from nap_multiproc.py node
  vector<int> pair_type;
  MatrixXd visibility_mask ; //n_pairsx100. 100 is total features
  int n_pairs;
  int n_features;


  // Camera
  const PinholeCamera camera;
  // All the nodes to get relative pose info for whichever pose needed.
  const vector<Node*> global_nodes;



  // Related to Triangulation

  // Given global ids of 2 nodes, this function returns triangulated points
  // [Input]
  //    global_idx_i, global_idx_j : global ids
  //    _uv_i : undistorted uv of node _i 3xN
  //    _uv_j : undistorted uv of node _j 3xN
  // [Output]
  //    _3d : the 3d points 4xN in *** world frame ***
  void triangulate_points( int global_idx_i, const MatrixXd& _uv_i,
                           int global_idx_j, const MatrixXd& _uv_j,
                           MatrixXd& _3d
                         );


   // Given a node index (local), finds the edge type out from this node. Returns the first non-zero value from
   // corresponding row of adj_mat
   int edge_type_from_node( int nx );

   int pair_0idx_of_node( int nx ); // looks at local_idx_of_pairs[2*i]
   int pair_1idx_of_node( int nx ); // looks at local_idx_of_pairs[2*i+1]





   // RObist triangulation (DLT)
   void robust_triangulation( const vector<Matrix4d>& w_T_c1,
                              const vector<Matrix4d>& w_T_c2,
                              const vector<MatrixXd>& _1_unvn_undistorted,
                              const vector<MatrixXd>& _2_unvn_undistorted,
                              const vector<VectorXd>& mask,
                              MatrixXd& result_3dpts
            );


};
