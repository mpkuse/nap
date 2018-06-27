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

// ros
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
#include "Feature3dInvertedIndex.h"

using namespace std;
// using namespace cv;  //don't do using namespace std; On some versions of opencv there is cv::Node which conflicts with my class Node.

// 0 : No debugging or minimal debugging info. No images written
// 1 : Some images written
// 2 : Most of images written. Use with caution.
// 3 : Lot of text written in addition to lots of images.
#define CORVUS_DEBUG_LVL 1


// Enabling below will compile 3d2d with switching constraints (recommended)
#define CORVUS__align3d2d_with_switching_constraints__ 1

// 3d3d alignment with switching constraints.
#define CORVUS__align3d3d_with_switching_constraints__ 1



class Corvus
{
public:
    Corvus() {} ;
    Corvus( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera );
    Corvus( const Feature3dInvertedIndex  * tfidf, const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera );
    bool isValid() { return is_data_set; }

    // Publish and Debug
    void publishPoints3d( const ros::Publisher& pub );
    void saveReprojectedImagesFromCeresCallbacks(  ); // writes images with reprojections at every iteration
    void publishCameraPoseFromCeresCallbacks( const ros::Publisher& pub ) ;


    // Pose Computation
    bool computeRelPose_3dprev_2dcurr(Matrix4d& to_return_p_T_c, ceres::Solver::Summary& summary ); //< compute pose using 3d points from previous and 2d points from curr.
    bool computeRelPose_2dprev_3dcurr( Matrix4d& to_return_p_T_c, ceres::Solver::Summary& summary ); //< compute pose using 3d points from current and 2d points from prev.
    bool computeRelPose_3dprev_3dcurr( Matrix4d& to_return_p_T_c, ceres::Solver::Summary& summary );


    void sayHi();

    void setDebugOutputFolder( const string& debug_output_dir ) {
        this->BASE__DUMP=debug_output_dir;
        debug_directory_is_set=true;
        ROS_INFO( "Corvus DEBUG Directory :: %s",  BASE__DUMP.c_str() );
    }
private:
      vector<int> enabled_opmode;
      string BASE__DUMP;
      bool debug_directory_is_set = false;

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
    void saveReprojectedPoints( const Matrix4d& c_T_p, const string& fname_suffix, const string image_caption_msg = string( "No Caption" ), const double * switches= NULL );



    // Create an image [ C| P ]. Mark the 3d points of curr ie. w_C onto both images
    // [[     PI(c_T_w * w_C)  ||    PI(p_T_c * c_T_w * w_C)    ]]
    void saveReprojectedPoints__w_C( const Matrix4d& p_T_c, const string& fname_suffix, const string image_caption_msg = string( "No Caption" ), const double * switches= NULL);



    vector<Align3d2d__4DOFCallback> vector_of_callbacks;


    int find_indexof_node( const vector<Node*>& global_nodes, ros::Time stamp );
    Matrix4d w_T_gid( int gid );
    Matrix4d gid_T_w( int gid );


    // Camera
    const PinholeCamera camera;
    // All the nodes to get relative pose info for whichever pose needed.
    const vector<Node*> global_nodes;


    int globalidx_of_curr, globalidx_of_prev; //< this is global node index

    // 3d points
    MatrixXd w_prev, w_curr;

    // 2d points (normalized image co-ordinates)
    MatrixXd unvn_prev, unvn_curr; //< note here that unvn_* is received in the msg and uv_* are infered from unvn_
    MatrixXd uv_prev, uv_curr;

    // global idx of w_prev and w_curr
    VectorXi gidx_of_curr, gidx_of_prev; //< this is global id of each of the features.
    VectorXd sqrtweight_w_curr, sqrtweight_w_prev; //< For each point the weight is 1/(1+Q) where Q=max( sigmaX, sigmaY, sigmaZ ). 3d points that have less uncertainity will have higher weight.
    bool is_gidx_set = false;

    bool is_data_set = false;
    bool is_data_tfidf = false;


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
    // [Output]
    //    outImg : Output image
    void plot_point_sets( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                          const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                          const cv::Scalar& color, bool annotate_pts,
                          const string& msg,
                        cv::Mat& dst );


    // Plots [ imA | imaB ] with points correspondences
    // [Input]
    //    imA, imB : Images
    //    ptsA, ptsB : 2xN or 3xN
    //    idxA, idxB : Index of each of the image. This will appear in status part. No other imppact of these
    //    switches   : double array of switches (from the switching constraint optimization to align 3d2d or 3d3d points)
    // [Output]
    //    outImg : Output image
    void plot_point_sets_with_switches( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                          const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                        //   const VectorXd& mask,
                        const double * switches,
                          const cv::Scalar& color, bool annotate_pts,
                          /*const vector<string>& msg,*/
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


    int _switches_stats( const double * switches, int len, double thresh=0.8 )
    {
        assert( switches != NULL );
        int count_inliers=0;
        for( int i=0 ; i<len ; i++ )
        {
            if( switches[i] > thresh )
                count_inliers++;
        }
        return count_inliers;
    }
};
