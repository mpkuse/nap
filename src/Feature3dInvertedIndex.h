#pragma once

/** Maintains an inverted index of the 3d points that are collected by `point_cloud_callback`.

Each frame has a pointcloud of itself which is obtained from vins_estimator. This pointcloud
also has global_id of each feature. Give to this class the 3d point and globalid of the feature to store.
This constructs an inverted index. This will let me search for a given global_id faster.

    Author  : Manohar Kuse<mpkuse@connect.ust.hk>
    Created : 20th June, 2018
**/




#include <iostream>
#include <stdio.h> //for printf and sprintf
#include <string>
#include <vector>
#include <map>
#include <tuple>

#include <thread>
#include <mutex>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/SVD>
using namespace Eigen;


using namespace std;

#define __Feature3dInvertedIndex_STORE_3D_POINTS__ 0 // will not store any 3d points. only running average is stored.

#define __Feature3dInvertedIndex_DEBUG_LVL 0

class Feature3dInvertedIndex
{
public:
    Feature3dInvertedIndex();
    void sayHi();


    //TODO make add threadsafe.
    void add( int global_idx_of_feature, const Vector4d& _3dpt, const int in_node );

    // Return false if the idx was not found.
    bool query_feature_n_occurences( int global_idx_of_feature, int &n );
    bool query_feature_mean( int global_idx_of_feature, Vector4d& M );
    bool query_feature_var( int global_idx_of_feature, Vector4d& V );

    bool exists( int gidx_of_feat );

private:
    // TODO no need to save all the 3d points. Just need to save the mean-value and variance. (running sum)
    #if __Feature3dInvertedIndex_STORE_3D_POINTS__ > 0
    std::map< int, vector<Vector4d> > DS;  ///< int is here the global idx of the feature.
    #endif
    std::map< int, vector<int> > D_in; ///< 1st int is the global id. list of nodeids where this feature was seen

    std::map< int, Vector4d > D_running_mean;
    std::map< int, Vector4d > D_running_var;




};
