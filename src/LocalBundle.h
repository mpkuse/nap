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
#include <math.h>


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

using namespace std;
using namespace cv;


class LocalBundle {

public:
  LocalBundle( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera   );

  // this was for unit test. Remove this if not needed.
  void multiviewTriangulate();


  // Picks 2 random views from the set of images. Triangulates those points and stores the
  // 3d points in a class variable.
  // [Input]
  //    max_itr: Maximum number of tries
  //    flag   : if flag is 0, will use images around iprev set. In this case 3d points stored in `this->w_X_iprev_triangulated`
  //             if flag is 1, will use images from icurr set.
  void randomViewTriangulate(int max_itr, int flag);


  void ceresDummy();
  void crossPoseComputation();
  void crossPoseComputation3d2d();    //< Estimate global pose c_T_w, this gives optimization difficulties.
  void crossRelPoseComputation3d2d(); //< Essentially like PNP. (will expand to multiple frames)

  void sayHi();

  // Given a pose c_T_w, project the 3d points on curr and write image to disk.
  void mark3dPointsOnCurrIm( const Matrix4d& c_T_w, const string& fname_prefix  );
  void markObservedPointsOnCurrIm();

  void mark3dPointsOnPrevIm( const Matrix4d& p_T_w, const string& fname_prefix );
  void markObservedPointsOnPrevIm();

  Matrix4d p_T_w() { return gi_T_w(localidx_of_iprev); }


  // Write out data for debugging
  void saveTriangulatedPoints();
  void publishTriangulatedPoints( const ros::Publisher& pub ); // will publish marker. pub is init in DataManager::setVisualizationTopic()

private:
  void LocalBundle::eigenpointcloud_2_ros_markermsg( const MatrixXd& M, visualization_msgs::Marker& marker, const string& ns );
  void LocalBundle::eigenpointcloud_2_ros_markertextmsg( const MatrixXd& M, vector<visualization_msgs::Marker>& marker, const string& ns );

private:
  int find_indexof_node(  const vector<Node*>& global_nodes, ros::Time stamp );
  int find_indexof_node( const vector<sensor_msgs::PointCloud>& global_nodes, ros::Time stamp );


  void write_image( string fname, const cv::Mat&);
  template <typename Derived>
  void write_EigenMatrix(const string& filename, const MatrixBase<Derived>& a);
  void write_Matrix2d( const string& filename, const double * D, int nRows, int nCols );
  void write_Matrix1d( const string& filename, const double * D, int n  );


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
  void printMatrix2d( const string& msg, const double * D, int nRows, int nCols );
  void printMatrix1d( const string& msg, const double * D, int n  );


  int n_ptClds;
  vector<int> _1set, _2set, _3set, _m1set; //list of nodes in each of the types (list of localids).
  int localidx_of_icurr, localidx_of_iprev;

  // node info (dense point matches)
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
   void robust_triangulation( const vector<pair<int,int> >& vector_of_pairs, /* local indices pair */
                              const vector<Matrix4d>& w_T_c1,
                              const vector<Matrix4d>& w_T_c2,
                              const vector<MatrixXd>& _1_unvn_undistorted,
                              const vector<MatrixXd>& _2_unvn_undistorted,
                              const vector<VectorXd>& mask,
                              MatrixXd& result_3dpts
            );

    bool isValid_w_X_iprev_triangulated=false;
    MatrixXd w_X_iprev_triangulated ; //< Triangulated points (DLT-SVD) from iprev-5 to iprev+5.

    bool isValid_w_X_icurr_triangulated=false;
    MatrixXd w_X_icurr_triangulated ; //< Triangulated points (DLT-SVD) from icurr-5 to icurr.


    // same as above, but in respective local co-ordinates, to eliminate the effect of drift in world poses
    bool isValid_iprev_X_iprev_triangulated=false;
    MatrixXd iprev_X_iprev_triangulated ; //< Triangulated points (DLT-SVD) from iprev-5 to iprev+5.

    bool isValid_icurr_X_icurr_triangulated=false;
    MatrixXd icurr_X_icurr_triangulated ; //< Triangulated points (DLT-SVD) from icurr-5 to icurr.



    // conversion of pose from Matrix4d and double *. (Quaternion and Translation)
    // We assume that doible * quat is organized as (w,x,y,z)
    void raw_to_eigenmat( const double * quat, const double * t, Matrix4d& dstT );
    void eigenmat_to_raw( const Matrix4d& T, double * quat, double * t);

    // Conversion of pose from Matrix4d and double *. (YPR and translation)
    void rawyprt_to_eigenmat( const double * ypr, const double * t, Matrix4d& dstT );
    void eigenmat_to_rawyprt( const Matrix4d& T, double * ypr, double * t);

    void gi_T_gj(int locali, int localj, Matrix4d& M );
    void w_T_gi( int locali, Matrix4d& M );
    void gi_T_w( int locali, Matrix4d& M );

    Matrix4d gi_T_gj(int locali, int localj );
    Matrix4d w_T_gi( int locali );
    Matrix4d gi_T_w( int locali );

    // idea roughly borrowed from pkg `pose_graph/utility/utility.h`
    Vector3d LocalBundle::R2ypr( const Matrix3d& R);
    Matrix3d LocalBundle::ypr2R( const Vector3d& ypr);


};

class PowellResidueCallback: public ceres::IterationCallback {
public:
  PowellResidueCallback( double * _x)   {
    x = _x;
  }
  virtual ceres::CallbackReturnType operator()( const ceres::IterationSummary& summary )
  {
    cout << summary.iteration << "   cost=" << summary.cost << endl;
    cout << "x: "<< x[0] << " " << x[1] << " " << x[2] << " " << x[3] << endl;
    return ceres::SOLVER_CONTINUE;
  }
private:
  double * x;
};

class PowellResidue {
public:
  template <typename T>
  bool operator()(const T* const x, T* residual ) const {
    residual[0] = x[0] + T(10.) * x[1];
    residual[1] = T(sqrt(5.0)) * (x[2]-x[3]);
    residual[2] = (x[1] - T(2.0)*x[2])*(x[1] - T(2.0)*x[2]);
    residual[3] = T(sqrt(10.0)) * ( x[0] - x[3] ) * ( x[0] - x[3] );
    return true;
  }


  static ceres::CostFunction* Create()
  {
    return ( new ceres::AutoDiffCostFunction<PowellResidue,4,4>
      (
        new PowellResidue()
      )
    );
  }


};

class DampleResidue {
public:
  template <typename T>
  bool operator()(const T* const x, T* residual ) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }
};


class Align3dPointsResidue {
public:
  Align3dPointsResidue(double * p_X, double * c_Xd )
  {
    _p_X[0] = p_X[0];
    _p_X[1] = p_X[1];
    _p_X[2] = p_X[2];

    _c_Xd[0] = c_Xd[0];
    _c_Xd[1] = c_Xd[1];
    _c_Xd[2] = c_Xd[2];
  }

  // R : p_R_c
  // Tr : p_Tr_c
  template<typename T>
  bool operator()( const T* const R, const T* const Tr, T* residual ) const {
    residual[0] = T(_p_X[0]) -  ( R[0] * T(_c_Xd[0]) + R[1] * T(_c_Xd[1]) + R[2] * T(_c_Xd[2]) ) - Tr[0];
    residual[1] = T(_p_X[1]) -  ( R[3] * T(_c_Xd[0]) + R[4] * T(_c_Xd[1]) + R[5] * T(_c_Xd[2]) ) - Tr[1];
    residual[2] = T(_p_X[2]) -  ( R[6] * T(_c_Xd[0]) + R[7] * T(_c_Xd[1]) + R[8] * T(_c_Xd[2]) ) - Tr[2];
    return true;
  }

private:
  double _p_X[3], _c_Xd[3];
};


class Align3dPointsResidueEigen {
public:
  // p_X : 3d point (triangulated from iprev) in iprev frame of reference.
  // c_Xd : 3d point (triangulated from icurr) in icurr frame of reference .
  Align3dPointsResidueEigen( const VectorXd& _X, const VectorXd& _Xd )
  {
    X << _X(0) , _X(1) , _X(2) ;
    Xd << _Xd(0) , _Xd(1) , _Xd(2) ;
  }

  // R: p_R_c. This is represented as quaternion
  // Tr : p_Tr_c
  template<typename T>
  bool operator()( const T* const quat, const T* const tran, T*e  ) const
  {
    Quaternion<T> q( quat[0], quat[1], quat[2], quat[3] );//w,x,y,z
    Matrix<T,3,1> t;
    t<< tran[0], tran[1], tran[2];

    Matrix<T,3,1> ___X;
    ___X << T(X(0)), T(X(1)), T(X(2));

    Matrix<T,3,1> ___Xd;
    ___Xd << T(Xd(0)), T(Xd(1)), T(Xd(2));


    Matrix<T,3,1> error;
    error = ___X - q.toRotationMatrix() * ___Xd - t;

    e[0] = error(0);
    e[1] = error(1);
    e[2] = error(2);
    return true;

  }

  // minimize_{T}   ||  p_X - T * c_Xd ||_2
  static ceres::CostFunction* Create( const VectorXd& _X, const VectorXd& _Xd )
  {
    return ( new ceres::AutoDiffCostFunction<Align3dPointsResidueEigen,3,4,3>
      (
        new Align3dPointsResidueEigen(_X, _Xd)
      )
    );
  }

private:
  Vector3d X;
  Vector3d Xd;

};


class Align3d2d {
public:
  Align3d2d( const Vector3d& _3d, const Vector2d& _2d )
          :_3d(_3d), _2d(_2d) {}

  // w_T_c
  template <typename T>
  bool operator()( const T* const quat, const T* const tran, T*e  ) const
  {
    Quaternion<T> q( quat[0], quat[1], quat[2], quat[3] );//w,x,y,z
    Matrix<T,3,1> t;
    t<< tran[0], tran[1], tran[2];


    Matrix<T,3,1> w_X; //3d co-ordinates in world ref-frame
    w_X << T(_3d(0)), T(_3d(1)), T(_3d(2));

    Matrix<T,2,1> unvn; //normalized undistorrted observed points
    unvn << T(_2d(0)), T(_2d(1));

    Matrix<T,3,1> c_X; //3d co-ordinates in camera ref-frame
    c_X = q.toRotationMatrix() * w_X + t;

    Matrix<T,2,1> error; // this variable is redundant. consider removal. TODO
    error(0) = c_X(0) / c_X(2) - unvn(0);
    error(1) = c_X(1) / c_X(2) - unvn(1);

    e[0] = error(0);
    e[1] = error(1);
    return true;

  }

  static ceres::CostFunction* Create( const VectorXd& __3d, const VectorXd& __2d )
  {
    Vector3d a;
    a << __3d(0), __3d(1), __3d(2);

    Vector2d b;
    b << __2d(0), __2d(1);

    return (
      new ceres::AutoDiffCostFunction<Align3d2d,2,4,3>( new Align3d2d( a, b) )
    );
  }


private:
  Vector3d _3d; //3d point in world co-ordinate
  Vector2d _2d; //undistorrted normalized observed points

};


/// To watch the intermediate values while optimizing
class Align3d2d__4DOFCallback: public ceres::IterationCallback{
public:
  Align3d2d__4DOFCallback(double * _yaw, double * _t )
  {
    yaw = _yaw; t = _t;
  }
  void setConstants( double* _pitch, double * _roll )
  {
    pitch = _pitch;
    roll  = _roll;
  }
  void setData( LocalBundle* _b )
  {
    b = _b;
  }



  virtual ceres::CallbackReturnType operator()( const ceres::IterationSummary& summary )
  {
    cout << summary.iteration << "  cost=" << summary.cost << endl;
    cout << "yaw=" << yaw[0] << "\tt="<< t[0] << " " << t[1] << " " << t[2] << endl;

    Vector3d ypr;
    ypr << yaw[0], *pitch, *roll;

    Matrix4d T = Matrix4d::Identity();
    T.topLeftCorner<3,3>() = ypr2R( ypr );
    T(0,3) = t[0];
    T(1,3) = t[1];
    T(2,3) = t[2];
    // cout << "T_"<< summary.iteration << "\n" << T << endl;

    // b->mark3dPointsOnCurrIm( T, "itr"+to_string(summary.iteration) );
    b->mark3dPointsOnCurrIm( T * b->p_T_w(), "proj3dPointsOnCurr_itr"+to_string(summary.iteration) );

    return ceres::SOLVER_CONTINUE;
  }

private:
  // optimization variables
  double *yaw; //size=1
  double *t;   //size=3

  // constants
  double * pitch;
  double * roll;

  LocalBundle * b;
  Matrix3d ypr2R( const Vector3d& ypr)
  {
    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    // Eigen::Matrix<double, 3, 3> Rz;
    Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    // Eigen::Matrix<double, 3, 3> Ry;
    Matrix3d Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    // Eigen::Matrix<double, 3, 3> Rx;
    Matrix3d Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
  }
};



class Align3d2d__4DOF {
public:
  Align3d2d__4DOF( const Vector3d& _3d, const Vector2d& _2d, double pitch, double roll )
          :_3d(_3d), _2d(_2d), pitch(pitch), roll(roll) {}

  // w_T_c
  template <typename T>
  bool operator()( const T* const yaw, const T* const tran, T*e  ) const
  {

    ////// function ypr2R ///////
    // use yaw, pitch and roll to form a rotation Matrix<T,3,3>
    Matrix<T,3,3> Rot;

    T yy = yaw[0] / T(180.0) * T(M_PI); //yaw in radians
    T pp = T(pitch) / T(180.0) * T(M_PI); //pitch in radians
    T rr = T(roll) / T(180.0) * T(M_PI); //roll in radians

    Matrix<T,3,3> Rz;
    Rz << cos(yy), -sin(yy), T(0.),
        sin(yy), cos(yy), T(0.),
        T(0.), T(0.), T(1.);

    Matrix<T,3,3> Ry;
    Ry << cos(pp), T(0.), sin(pp),
        T(0.), T(1.), T(0.),
        -sin(pp), T(0.), cos(pp);

    Matrix<T,3,3> Rx;
    Rx << T(1.), T(0.), T(0.),
        T(0.), cos(rr), -sin(rr),
        T(0.), sin(rr), cos(rr);

    Rot = Rz * Ry * Rx;

    //////////// END ///////////
    Matrix<T,3,1> t;
    t<< tran[0], tran[1], tran[2];


    Matrix<T,3,1> w_X; //3d co-ordinates in world ref-frame
    w_X << T(_3d(0)), T(_3d(1)), T(_3d(2));

    Matrix<T,2,1> unvn; //normalized undistorrted observed points
    unvn << T(_2d(0)), T(_2d(1));

    Matrix<T,3,1> c_X; //3d co-ordinates in camera ref-frame
    c_X = Rot * w_X + t;

    Matrix<T,2,1> error; // this variable is redundant. consider removal. TODO
    error(0) = c_X(0) / c_X(2) - unvn(0);
    error(1) = c_X(1) / c_X(2) - unvn(1);

    e[0] = error(0);
    e[1] = error(1);
    return true;

  }

  static ceres::CostFunction* Create( const VectorXd& __3d, const VectorXd& __2d, double __pitch, double __roll )
  {
    Vector3d a;
    a << __3d(0), __3d(1), __3d(2);

    Vector2d b;
    b << __2d(0), __2d(1);

    return (
      new ceres::AutoDiffCostFunction<Align3d2d__4DOF,2,1,3>( new Align3d2d__4DOF( a, b, __pitch, __roll) )
    );
  }


private:
  Vector3d _3d; //3d point in world co-ordinate
  Vector2d _2d; //undistorrted normalized observed points
  double pitch;
  double roll;

};



template <typename T>
T NormalizeAngle(const T& angle_degrees) {
	// if this non-linearity can be removed the angular problem (for constant translation) is a linear problem.
	// the l1-norm iros2014 paper by Luca Corlone suggest to use another latent variable for this and penalize it to be near zero.
  if (angle_degrees > T(180.0))
  	return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
  	return angle_degrees + T(360.0);
  else
  	return angle_degrees;
};

class AngleLocalParameterization {
 public:

  template <typename T>
  bool operator()(const T* theta, const T* delta_theta,
                  T* theta_plus_delta) const {
    *theta_plus_delta =
        NormalizeAngle(*theta + *delta_theta);

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};
