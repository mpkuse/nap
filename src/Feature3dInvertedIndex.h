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
    // a 3d point with global id as `global_idx_of_feature` and whose coordinates
    // are `_3dpt` seen in frame-`in_node`.
    void add( int global_idx_of_feature, const Vector4d& _3dpt, const int in_node );

    // a 3d point with global id as `global_idx_of_feature` and whose coordinates
    // are `_3dpt` seen in frame-`in_node`. This 3d point is observed at image
    // co-ordinates `_uv`. _unvn is the corresponding normalized image co-ordinates.
    void add( int global_idx_of_feature,
        const Vector4d& _3dpt,
        const Vector3d& _unvn, const Vector3d& _uv,
        const int in_node );

    // Return false if the idx was not found.
    bool query_feature_n_occurences( int global_idx_of_feature, int &n ); //< In how many frames was this feature seen
    bool query_feature_where_occurence( int global_idx_of_feature, std::vector<int>& occured_in ); // return the list of frames where this feature was seen
    bool query_feature_where_unvn_occurence( int global_idx_of_feature, std::vector<Vector3d>& occured_at );
    bool query_feature_where_uv_occurence( int global_idx_of_feature, std::vector<Vector3d>& occured_at );


    bool query_feature_mean( int global_idx_of_feature, Vector4d& M ); //< 3D co-ordinate of this feature. Mean-value
    bool query_feature_var( int global_idx_of_feature, Vector4d& V ); //< 3d co-ordinate of this feature. var-value

    bool exists( int gidx_of_feat );

    int nFeatures() ;
    void lockDataStructure();
    void unlockDataStructure();

private:
    // TODO no need to save all the 3d points. Just need to save the mean-value and variance. (running sum)
    #if __Feature3dInvertedIndex_STORE_3D_POINTS__ > 0
    std::map< int, vector<Vector4d> > DS;  ///< int is here the global idx of the feature. 3D point at each nodeid
    #endif

    std::map< int, vector<int> > D_in; ///< 1st int is the global id. list of nodeids where this feature was seen
    std::map< int, vector<Vector3d> > D_unvn; ///< 1st int is the global id. list of images points normalized cords.
    std::map< int, vector<Vector3d> > D_uv;

    std::map< int, Vector4d > D_running_mean;
    std::map< int, Vector4d > D_running_var;

    std::mutex m_;

    int n_features = 0;

};
