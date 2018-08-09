#pragma once

// Std headers
#include <iostream>
#include <vector>

// ros
#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>



// Threading
#include <thread>
#include <mutex>
#include <atomic>


// Ceres
#include <ceres/ceres.h>
// using namespace ceres;

// Eigen
#include <Eigen/Core>
using namespace Eigen;

// Custom Headers
// #include "Feature3dInvertedIndex.h"
// #include "Node.h"
// #include "PinholeCamera.h"
#include "DataManager.h"
#include "LocalBundle.h"
#include "utils/PoseManipUtils.h"
#include "utils/RosMarkerUtils.h"

using namespace std;

#define __EdgeManager_DEBUG_LVL 1

#define _COLOR_RED_ "\033[1;31m"
#define _COLOR_GREEN_ "\033[1;32m"
#define _COLOR_YELLOW_ "\033[1;33m"
#define _COLOR_BLUE_ "\033[1;34m"
#define _COLOR_DEFAULT_ "\033[0m"


class REdge {
public:
    REdge( int _i_curr, int _i_prev, float _goodness )
        : i_curr(_i_curr), i_prev(_i_prev), goodness(_goodness)
    {}
    int i_curr, i_prev; //< These indices are for nNodes.
    float goodness;

    // eg. 3234 <---> 231
    // gid_curr[i]=3234 <---> gid_prev[i]=231.
    vector<int> gid_curr, gid_prev; //note that length of both these vectors will be 1.

    // Add i^{th} correspondence
    void add_global_correspondence( int gid_curr_i, int gid_prev_i );

};

class EdgeManager
{
public:
    EdgeManager( DataManager* dm);
    void sayHi() { cout << "Hello\n"; }

    void loop(); //expected to be run in a separate thread. This monitors nEdges

private:
    DataManager * manager;
    bool is_datamanager_available = false;


    /// Given a set of 3d points, ie. matrix of size 4xN, convert this to visualization msg.
    /// put points in msg.points[N].
    void _3dpoints_to_markerpoints( const MatrixXd& w_X, visualization_msgs::Marker& m_points );

    /// Given point pair make line-visualization-msg. puts in msg.points[2*N]
    void _3dpointspair_to_markerlines( const MatrixXd& w_X, const MatrixXd& w_Xd,
        visualization_msgs::Marker& m_lines );


    /// For every points, w_X: 4xN, we are also supplied with variance of the 3d estimate of this point.
    /// in visualization msg, for everypoint we add 3 lines : X \plus_minus [delX,0,0] and X \plus_minus [0,delY,0] and X \plus_minus [0,0,delZ].
    void _3dpoints_with_var_to_markerpoints( const MatrixXd& w_X, const MatrixXd& w_X_variance,
        visualization_msgs::Marker& m_points_with_var );

};
