#pragma once
/** Processor class for opmode19

    Give 3d points and 2d points computation of pose.


    This class is named after the star Corvus whose name is derived from arabic for
    "break of the crow".

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 8th May, 2018
*/


#include <iostream>
#include <stdio.h> //for printf and sprintf
#include <string>
#include <vector>
#include <math.h>
#include <sstream>
#include <iterator>


//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>


#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>


#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/SVD>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


#include <ceres/ceres.h>

#include <nap/NapMsg.h>
#include "PinholeCamera.h"
#include "Node.h"
#include "ColorLUT.h"
#include "tic_toc.h"

#include "LocalBundle.h"

using namespace std;
using namespace cv;

// 0 : No debugging or minimal debugging info
// 1 : Some images written
// 2 : Most of images written
// 3 : Lot of text written
#define CORVUS_DEBUG_LVL 2


class Corvus
{
public:
    // Corvus();
    Corvus( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera );
    bool isValid() { return is_data_set; }

    // Publish and Debug
    void publishPoints3d( const ros::Publisher& pub );
    void saveReprojectedImagesFromCeresCallbacks(  ); // writes images with reprojections at every iteration
    void publishCameraPoseFromCeresCallbacks( const ros::Publisher& pub ) ;


    // Pose Computation
    Matrix4d computeRelPose_3dprev_2dcurr(); //< compute pose using 3d points from previous and 2d points from curr.

    void sayHi();
private:

    //
    //  Crate an image : [C | P ]. Mark the observed points. ie. [ uv_prev |  uv_curr ].
    void saveObservedPoints( );

    // Create an image : [ C | P ]. Mark the projected points.
    //         [[  PI( c_T_p * p_T_w * w_P ) ||   PI( p_T_w * w_P)   ]]
    //
    // c_T_p : pose of frame-p as observed from frame c.
    // fname_suffix : suffix in the image file name
    // image_caption_msg : a string that will go in status part of the image.
    void saveReprojectedPoints( const Matrix4d& c_T_p, const string& fname_suffix, const string image_caption_msg = string( "No Caption" ) );



    vector<Align3d2d__4DOFCallback> vector_of_callbacks;


    int find_indexof_node( const vector<Node*>& global_nodes, ros::Time stamp );
    Matrix4d w_T_gid( int gid );
    Matrix4d gid_T_w( int gid );


    // Camera
    const PinholeCamera camera;
    // All the nodes to get relative pose info for whichever pose needed.
    const vector<Node*> global_nodes;


    int globalidx_of_curr, globalidx_of_prev;

    // 3d points
    MatrixXd w_prev, w_curr;

    // 2d points (normalized image co-ordinates)
    MatrixXd unvn_prev, unvn_curr; //< note here that unvn_* is received in the msg and uv_* are infered from unvn_
    MatrixXd uv_prev, uv_curr;

    bool is_data_set = false;


    // publishing helpers
    void eigenpointcloud_2_ros_markermsg( const MatrixXd& M, visualization_msgs::Marker& marker, const string& ns );


    // Writing to file
    void write_image( string fname, const cv::Mat& img );


    // plotting
    /// Plots a 2xN or 3xN(homogeneous) matrix representing the points uv on image on the provided image
    void plot_points( const cv::Mat& im, const MatrixXd& pts,
                bool enable_text, bool enable_status_image, const string& msg ,
                cv::Mat& dst );


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
                        //   const VectorXd& mask,
                          const cv::Scalar& color, bool annotate_pts,
                          const string& msg,
                        cv::Mat& dst );


    void raw_to_eigenmat( const double * quat, const double * t, Matrix4d& dstT );
    void eigenmat_to_raw( const Matrix4d& T, double * quat, double * t);
    void rawyprt_to_eigenmat( const double * ypr, const double * t, Matrix4d& dstT );
    void eigenmat_to_rawyprt( const Matrix4d& T, double * ypr, double * t);
    Vector3d R2ypr( const Matrix3d& R);
    Matrix3d ypr2R( const Vector3d& ypr);
    void prettyprintPoseMatrix( const Matrix4d& M );
    void prettyprintPoseMatrix( const Matrix4d& M, string& return_string );




    template<typename Out>
    void split(const std::string &s, char delim, Out result);
    std::vector<std::string> split(const std::string &s, char delim);




    // camera publishing
    void init_camera_marker( visualization_msgs::Marker& marker );
    void setpose_to_cameravisual( const Matrix4d& w_T_c, visualization_msgs::Marker& marker );
    void setcolor_to_cameravisual( float r, float g, float b, visualization_msgs::Marker& marker  );


};
