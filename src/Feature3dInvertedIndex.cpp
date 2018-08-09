#include "Feature3dInvertedIndex.h"


Feature3dInvertedIndex::Feature3dInvertedIndex()
{

}

void Feature3dInvertedIndex::add( int global_idx_of_feature, const Vector4d& _3dpt, const int in_node )
{
    m_.lock();
    bool is_exist = exists( global_idx_of_feature );

    #if __Feature3dInvertedIndex_DEBUG_LVL >= 2
    if( !is_exist )
        cout << "NEW  ";
    else
        cout << "SEEN ";
    cout << "Add "<< global_idx_of_feature << " ";
    cout << "XYZ=" << _3dpt(0) << "," << _3dpt(1) << "," <<_3dpt(2) << "," << _3dpt(3) << " ";
    cout << "seen in node " << in_node;
    cout << endl;
    #endif

    if( !is_exist ) {
        n_features++;
    }


    assert( global_idx_of_feature >= 0 );


    int n = D_in[global_idx_of_feature].size();

    #if __Feature3dInvertedIndex_STORE_3D_POINTS__ > 0
    DS[global_idx_of_feature].push_back( _3dpt );
    #endif
    D_in[global_idx_of_feature].push_back( in_node );

    if( !is_exist ) // New so init mean, var
    {
        D_running_mean[ global_idx_of_feature] = Vector4d( _3dpt );
        D_running_var[ global_idx_of_feature ] = Vector4d::Zero();
    }
    else // Already exisit so, update mean, variance
    {
        Vector4d x_mean_m = D_running_mean[ global_idx_of_feature ]; //previous mean
        D_running_mean[ global_idx_of_feature ] += (_3dpt -  D_running_mean[ global_idx_of_feature ]) / ( (double) n + 1);
        Vector4d x_mean = D_running_mean[ global_idx_of_feature ]; //updated mean


        Vector4d delta1 = _3dpt - x_mean_m;
        Vector4d delta2 = _3dpt - x_mean;
        D_running_var[ global_idx_of_feature ] += ( delta1.array() * delta2.array() ).matrix(); // later at query time, need to divide this quantity by n (for biased estimate) or n-1 (unboased estimate)
    }

    m_.unlock();

}



void Feature3dInvertedIndex::add( int global_idx_of_feature,
    const Vector4d& _3dpt,
    const Vector3d& _unvn, const Vector3d& _uv,
    const int in_node )
{
    m_.lock();
    bool is_exist = exists( global_idx_of_feature );

    #if __Feature3dInvertedIndex_DEBUG_LVL >= 2
    if( !is_exist )
        cout << "NEW  ";
    else
        cout << "SEEN ";
    cout << "Add "<< global_idx_of_feature << "\t";
    cout << "XYZ=" << _3dpt(0) << "," << _3dpt(1) << "," <<_3dpt(2) << "," << _3dpt(3) << "\t";
    cout << "unvn=" << _unvn(0) << "," << _unvn(1) << "\t";
    cout << "uv=" << _uv(0) << "," << _uv(1) << "\t";
    cout << "seen in node " << in_node << "\t";
    cout << endl;
    #endif

    if( !is_exist ) {
        n_features++;
    }


    assert( global_idx_of_feature >= 0 );
    assert( _3dpt(3) == 1.0 && _uv(2) == 1.0 && _unvn(2) == 1.0 );


    int n = D_in[global_idx_of_feature].size();

    #if __Feature3dInvertedIndex_STORE_3D_POINTS__ > 0
    DS[global_idx_of_feature].push_back( _3dpt );
    #endif

    D_in[global_idx_of_feature].push_back( in_node );
    D_unvn[global_idx_of_feature].push_back( _unvn );
    D_uv[global_idx_of_feature].push_back( _uv );



    if( !is_exist ) // New so init mean, var
    {
        D_running_mean[ global_idx_of_feature] = Vector4d( _3dpt );
        D_running_var[ global_idx_of_feature ] = Vector4d::Zero();
    }
    else // Already exisit so, update mean, variance
    {
        Vector4d x_mean_m = D_running_mean[ global_idx_of_feature ]; //previous mean
        D_running_mean[ global_idx_of_feature ] += (_3dpt -  D_running_mean[ global_idx_of_feature ]) / ( (double) n + 1);
        Vector4d x_mean = D_running_mean[ global_idx_of_feature ]; //updated mean


        Vector4d delta1 = _3dpt - x_mean_m;
        Vector4d delta2 = _3dpt - x_mean;
        D_running_var[ global_idx_of_feature ] += ( delta1.array() * delta2.array() ).matrix(); // later at query time, need to divide this quantity by n (for biased estimate) or n-1 (unboased estimate)
    }

    m_.unlock();

}

int Feature3dInvertedIndex::nFeatures() {
    m_.lock();
    int result = n_features;
    m_.unlock();
    return result;
}

void Feature3dInvertedIndex::lockDataStructure() { m_.lock(); }
void Feature3dInvertedIndex::unlockDataStructure() { m_.unlock(); }


void Feature3dInvertedIndex::sayHi()
{
    cout << "Feature3dInvertedIndex::sayHi\n. Printing mean and variances of each of the features.\n";

    // for( auto it = D_in.begin() ; it != D_in.end() ; it++  )
    // {
    //     cout << it->first << ":" ;
    //     vector<int> J = it->second;
    //     for( int i=0 ; i<J.size() ; i++ )
    //         cout << J[i] << " ";
    //     cout << endl;
    // }

    #if __Feature3dInvertedIndex_STORE_3D_POINTS__ > 0
    // to verify that the running mean and usual mean is correct. Uses DS which is commented-out privatve variable.

    for( auto it = D_in.begin() ; it != D_in.end() ; it++  )
    {
        cout << it->first << ":" ;

        vector<Vector4d> J = DS[it->first];
        int n = J.size(); //number of occurences
        cout << " ("<< J.size() << ") ";


        // mean
        Vector4d mean = Vector4d::Zero();
        for( int i=0 ; i<J.size() ; i++ )
            mean = mean + J[i];
        mean = mean / (double)J.size();

        // cout << "\tmean=" << mean(0) << ", " << mean(1) << ", " << mean(2) << ", "  << mean(3) << ";";
        printf( "\tmean=%4.4lf,%4.4lf,%4.4lf,%4.4lf", mean(0), mean(1), mean(2), mean(3) );

        //variance
        Vector4d variance = Vector4d::Zero();
        if( J.size() > 1 )
        {
            for( int i=0 ; i<J.size() ; i++ )
            {
                variance += (   (J[i] - mean).array() * (J[i] - mean).array()   ).matrix();
            }
            // variance = variance / (double)( J.size() - 1 );
            variance = variance / (double)( J.size() );
        }
        // cout << "\tvariance=" << variance(0) << ", " << variance(1) << ", " << variance(2) << ", "  << variance(3) << ";";
        printf( "\tvariance=%4.4lf,%4.4lf,%4.4lf,%4.4lf", variance(0), variance(1), variance(2), variance(3) );

        Vector4d runing_mean = D_running_mean[ it->first ];
        printf( "\n\t\tr_mean=%4.4lf,%4.4lf,%4.4lf,%4.4lf", runing_mean(0), runing_mean(1), runing_mean(2), runing_mean(3) );

        Vector4d runing_var = D_running_var[ it->first ] / (double)n;
        printf( "\tr_var=%4.4lf,%4.4lf,%4.4lf,%4.4lf", runing_var(0), runing_var(1), runing_var(2), runing_var(3) );




        cout << endl;

    }

    #endif

    for( auto it = D_in.begin() ; it != D_in.end() ; it++  )
    {
        cout << it->first << ":" ;
        int gid = it->first;

        int n;
        bool status = query_feature_n_occurences(gid, n);
        assert( status );
        cout << "#(" << n << ")"; // was seen n times

        Vector4d M, V;
        bool status_mean = query_feature_mean( gid, M );
        bool status_var  = query_feature_var( gid, V );

        printf( "\tr_mean=%4.4lf,%4.4lf,%4.4lf,%4.4lf", M(0), M(1), M(2), M(3) );
        printf( "\tr_var=%4.4lf,%4.4lf,%4.4lf,%4.4lf", V(0), V(1), V(2), V(3) );
        cout << endl;


        vector<int> where_nodes;
        vector<Vector3d> where_unvn, where_uv;
        status = query_feature_where_occurence(gid, where_nodes); assert( status );
        status = query_feature_where_unvn_occurence(gid, where_unvn); assert( status );
        status = query_feature_where_uv_occurence(gid, where_uv); assert( status );
        assert( where_nodes.size() == where_unvn.size() && where_unvn.size() == where_uv.size() );
        for( int h = 0 ; h<where_nodes.size() ; h++ )
        {
            cout << "\t in node " << where_nodes[h] << "  at uv**" << where_uv[h](0) << "," << where_uv[h](1) << "**  ";
            cout << "  unvn**" << where_unvn[h](0) << "," << where_unvn[h](1) << "**  " << endl;
        }
        cout << endl;

    }

}

bool Feature3dInvertedIndex::exists( int gidx_of_feat )
{
    if( D_in.find(gidx_of_feat) != D_in.end()  )
        return true; //found
    else
        return false; //not found.
}


bool Feature3dInvertedIndex::query_feature_n_occurences( int global_idx_of_feature, int &n )
{
    m_.lock();
    if( exists(global_idx_of_feature) == false ) {
        m_.unlock();
        return false;
    }

    n = D_in[ global_idx_of_feature ].size();
    m_.unlock();
    return true;

}

 //< returns a list of frame-ids where this feature was visible
bool Feature3dInvertedIndex::query_feature_where_occurence( int global_idx_of_feature, std::vector<int>& occured_in )
{
    m_.lock();
    if( exists(global_idx_of_feature) == false ) {
        m_.unlock();
        return false;
    }
    occured_in = D_in[global_idx_of_feature];
    m_.unlock();
    return true;
}

bool Feature3dInvertedIndex::query_feature_where_unvn_occurence( int global_idx_of_feature, std::vector<Vector3d>& occured_at )
{
    m_.lock();
    if( exists(global_idx_of_feature) == false ) {
        m_.unlock();
        return false;
    }
    occured_at = D_unvn[global_idx_of_feature];
    m_.unlock();
    return true;
}

bool Feature3dInvertedIndex::query_feature_where_uv_occurence( int global_idx_of_feature, std::vector<Vector3d>& occured_at )
{
    m_.lock();
    if( exists(global_idx_of_feature) == false ) {
        m_.unlock();
        return false;
    }
    occured_at = D_uv[global_idx_of_feature];
    m_.unlock();
    return true;
}

bool Feature3dInvertedIndex::query_feature_mean( int global_idx_of_feature, Vector4d& M )
{
    m_.lock();
    if( exists(global_idx_of_feature) == false ) {
        m_.unlock();
        return false;
    }

    M = D_running_mean[ global_idx_of_feature ];
    m_.unlock();
    return true;
}

bool Feature3dInvertedIndex::query_feature_var( int global_idx_of_feature, Vector4d& V )
{
    m_.lock();
    if( exists(global_idx_of_feature) == false ) {
        m_.unlock();
        return false;
    }
    m_.unlock();

    int n;
    query_feature_n_occurences( global_idx_of_feature, n );
    m_.lock();
    V = D_running_var[ global_idx_of_feature ] / (double)n;
    m_.unlock();
    return true;

}
