#include "Corvus.h"

// Corvus::Corvus()
// {
//
// }

void Corvus::sayHi()
{
    cout << "Corvus::sayHi()\n";
}

// Constructor crafted for opmode17. In this mode there are also global_idx of the point features. These
// can be used to get better values of 3d points for pnp.
Corvus::Corvus( const Feature3dInvertedIndex  * tfidf, const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera )
{
    cout << "Corvus()  ";
    cout << "Opmode: " <<  msg->op_mode << endl;
    cout << "curr.size, prev.size: " << msg->curr.size() << " " << msg->prev.size() << endl;
    // cout << "curr.timestamp, prev.timestamp: " << msg->c_timestamp << " " << msg->prev_timestamp << endl;


    // msg->curr : u, U, u, U ... (projected points interspaced with its 3d point)
    // msg->prev : same as msg->curr but for points in prev.

    assert( msg->curr.size() % 2 == 0 && msg->prev.size()%2 == 0 );

    // Setup camera and global_nodes
    this->global_nodes = global_nodes;
    this->camera = camera;
    // this->camera.printCameraInfo(0);


    // Lookup c_timestamp and prev_timestamp in global_nodes
    this->globalidx_of_curr = find_indexof_node( this->global_nodes, msg->c_timestamp );
    this->globalidx_of_prev = find_indexof_node( this->global_nodes, msg->prev_timestamp );
    cout << "globalidx_of_curr,globalidx_of_prev: " << globalidx_of_curr << " " << globalidx_of_prev << endl;
    assert( this->globalidx_of_curr >= 0 && this->globalidx_of_prev >= 0 );


    assert( msg->curr.size() == msg->prev.size() );

    // Collect tracked points and 2d points of curr
    w_curr = MatrixXd::Zero( 4, msg->curr.size()/2 );
    unvn_curr = MatrixXd::Zero( 3, msg->curr.size()/2 );
    gidx_of_curr = VectorXi::Zero( msg->curr.size()/2 );
    sqrtweight_w_curr = VectorXd::Zero( msg->curr.size()/2 ); //< For each point the weight is 1/(1+Q) where Q=max( sigmaX, sigmaY, sigmaZ ). 3d points that have less uncertainity will have higher weight.
    for( int i=0 ; i< msg->curr.size()/2 ; i++ )
    {
        geometry_msgs::Point32 u = msg->curr[2*i];
        geometry_msgs::Point32 U = msg->curr[2*i+1];

        w_curr( 0, i ) = U.x; //TODO. Need not read 3d points. Instead query the inverted index to get the good 3d points.
        w_curr( 1, i ) = U.y;
        w_curr( 2, i ) = U.z;
        w_curr( 3, i ) = 1.0;
        unvn_curr( 0, i ) = u.x;
        unvn_curr( 1, i ) = u.y;
        unvn_curr( 2, i ) = 1.0;

        // assert( u.z == -180 );
        gidx_of_curr(i) = u.z;


        // Using TFIDF
        Vector4d M_curr, V_curr;
        bool status_curr = tfidf->query_feature_mean(  gidx_of_curr(i) , M_curr );
        bool status_curr_V = tfidf->query_feature_var(  gidx_of_curr(i) , V_curr );
        assert( status_curr && status_curr_V );
        w_curr.col(i) = M_curr;
        sqrtweight_w_curr(i) = sqrt(1.0 / ( 1.0 + (double) max( max( V_curr(0), V_curr(1) ), V_curr(2) ) ) );

    }


    // Collect tracked points and 2d points of prev
    w_prev = MatrixXd::Zero( 4, msg->prev.size()/2 );
    unvn_prev = MatrixXd::Zero( 3, msg->prev.size()/2 );
    gidx_of_prev = VectorXi::Zero( msg->prev.size()/2 );
    sqrtweight_w_prev = VectorXd::Zero( msg->prev.size()/2 ); //< For each point the weight is 1/(1+Q) where Q=max( sigmaX, sigmaY, sigmaZ ). 3d points that have less uncertainity will have higher weight.
    for( int i=0 ; i< msg->prev.size()/2 ; i++ )
    {
        geometry_msgs::Point32 u = msg->prev[2*i];
        geometry_msgs::Point32 U = msg->prev[2*i+1];

        w_prev( 0, i ) = U.x; //TODO. Need not read 3d points. Instead query the inverted index to get the good 3d points.
        w_prev( 1, i ) = U.y;
        w_prev( 2, i ) = U.z;
        w_prev( 3, i ) = 1.0;
        unvn_prev( 0, i ) = u.x;
        unvn_prev( 1, i ) = u.y;
        unvn_prev( 2, i ) = 1.0;

        // assert( u.z == -180 );
        gidx_of_prev(i) = u.z;


        // Using TFIDF
        Vector4d M_prev, V_prev;
        bool status_prev = tfidf->query_feature_mean(  gidx_of_prev(i) , M_prev );
        bool status_prev_V = tfidf->query_feature_var(  gidx_of_curr(i) , V_prev );
        assert( status_prev & status_prev_V );
        w_prev.col(i) = M_prev;
        sqrtweight_w_prev(i) = sqrt( 1.0 / ( 1.0 + (double) max( max( V_prev(0), V_prev(1) ), V_prev(2) ) ) );

    }

    is_data_set = true;
    is_gidx_set = true;
    is_data_tfidf = true;



    // Print 3d 3d points from curr and prev
    #if CORVUS_DEBUG_LVL >= 3
    {
        int n_matched_pts = gidx_of_prev.size();
        for( int i=0 ; i< n_matched_pts ; i++ )
        {
            cout << "i="<< i << "  "<<  gidx_of_curr(i) << " <---> " << gidx_of_prev(i) << endl;
            cout << "\tw_curr: "<< w_curr(0,i) << "," << w_curr(1,i) << "," << w_curr(2,i) << "," << w_curr(3,i) << ";";
            Vector4d M_curr, V_curr;
            bool status_curr = tfidf->query_feature_mean(  gidx_of_curr(i) , M_curr );
            bool status_curr_V = tfidf->query_feature_var(  gidx_of_curr(i) , V_curr );
            assert( status_curr && status_curr_V );
            cout << "\tM_curr: " << M_curr(0) << "," << M_curr(1) << "," <<M_curr(2) << "," <<M_curr(3) << ";";
            cout << "\tw=" << 1.0 / ( 1.0 + (double) max( max( V_curr(0), V_curr(1) ), V_curr(2) ) );
            cout << "\tV_curr: " << V_curr(0) << "," << V_curr(1) << "," <<V_curr(2) << "," <<V_curr(3) << ";";


            cout << endl;
            cout << "\tw_prev: "<< w_prev(0,i) << "," << w_prev(1,i) << "," << w_prev(2,i) << "," << w_prev(3,i) << ";";
            Vector4d M_prev, V_prev;
            bool status_prev = tfidf->query_feature_mean(  gidx_of_prev(i) , M_prev );
            bool status_prev_V = tfidf->query_feature_var(  gidx_of_curr(i) , V_prev );
            assert( status_prev & status_prev_V );
            cout << "\tM_prev: " << M_prev(0) << "," << M_prev(1) << "," <<M_prev(2) << "," <<M_prev(3) << ";";
            cout << "\tw=" << 1.0 / ( 1.0 + (double) max( max( V_prev(0), V_prev(1) ), V_prev(2) ) );
            cout << "\tV_prev: " << V_prev(0) << "," << V_prev(1) << "," <<V_prev(2) << "," <<V_prev(3) << ";";


            cout << endl;

            // cout << "i="<< i << "   ";
            // cout << "(" << gidx_of_curr(i) <<  ") ";
            // cout << "XYZ: "<< w_curr(0,i) << "," << w_curr(1,i) << "," << w_curr(2,i) << "," << w_curr(3,i) << ";";
            // cout << "<-->";
            // cout << "(" << gidx_of_prev(i) <<  ") ";
            // cout << "XYZ: "<< w_prev(0,i) << "," << w_prev(1,i) << "," << w_prev(2,i) << "," << w_prev(3,i) << ";";
            // cout << endl;
        }

    }
    #endif



    this->camera.normalizedImCords_2_imageCords( unvn_curr, uv_curr );
    this->camera.normalizedImCords_2_imageCords( unvn_prev, uv_prev );
    cout << "Done Corvus() with TFIDF\n";
}



// Original Constructor (opmode 18)
Corvus::Corvus( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes, const PinholeCamera& camera )
{
    cout << "Corvus()  ";
    cout << "Opmode: " <<  msg->op_mode << endl;
    cout << "curr.size, prev.size: " << msg->curr.size() << " " << msg->prev.size() << endl;
    // cout << "curr.timestamp, prev.timestamp: " << msg->c_timestamp << " " << msg->prev_timestamp << endl;


    // msg->curr : u, U, u, U ... (projected points interspaced with its 3d point)
    // msg->prev : same as msg->curr but for points in prev.

    assert( msg->curr.size() % 2 == 0 && msg->prev.size()%2 == 0 );

    // Setup camera and global_nodes
    this->global_nodes = global_nodes;
    this->camera = camera;
    // this->camera.printCameraInfo(0);


    // Lookup c_timestamp and prev_timestamp in global_nodes
    this->globalidx_of_curr = find_indexof_node( this->global_nodes, msg->c_timestamp );
    this->globalidx_of_prev = find_indexof_node( this->global_nodes, msg->prev_timestamp );
    cout << "globalidx_of_curr,globalidx_of_prev: " << globalidx_of_curr << " " << globalidx_of_prev << endl;
    assert( this->globalidx_of_curr >= 0 && this->globalidx_of_prev >= 0 );


    assert( msg->curr.size() == msg->prev.size() );

    // Collect tracked points and 2d points of curr
    w_curr = MatrixXd::Zero( 4, msg->curr.size()/2 );
    unvn_curr = MatrixXd::Zero( 3, msg->curr.size()/2 );
    for( int i=0 ; i< msg->curr.size()/2 ; i++ )
    {
        geometry_msgs::Point32 u = msg->curr[2*i];
        geometry_msgs::Point32 U = msg->curr[2*i+1];

        w_curr( 0, i ) = U.x;
        w_curr( 1, i ) = U.y;
        w_curr( 2, i ) = U.z;
        w_curr( 3, i ) = 1.0;
        unvn_curr( 0, i ) = u.x;
        unvn_curr( 1, i ) = u.y;
        unvn_curr( 2, i ) = 1.0;

        assert( u.z == -180 );
    }


    // Collect tracked points and 2d points of prev
    w_prev = MatrixXd::Zero( 4, msg->curr.size()/2 );
    unvn_prev = MatrixXd::Zero( 3, msg->curr.size()/2 );
    for( int i=0 ; i< msg->prev.size()/2 ; i++ )
    {
        geometry_msgs::Point32 u = msg->prev[2*i];
        geometry_msgs::Point32 U = msg->prev[2*i+1];

        w_prev( 0, i ) = U.x;
        w_prev( 1, i ) = U.y;
        w_prev( 2, i ) = U.z;
        w_prev( 3, i ) = 1.0;
        unvn_prev( 0, i ) = u.x;
        unvn_prev( 1, i ) = u.y;
        unvn_prev( 2, i ) = 1.0;

        assert( u.z == -180 );
    }
    is_data_set = true;


    this->camera.normalizedImCords_2_imageCords( unvn_curr, uv_curr );
    this->camera.normalizedImCords_2_imageCords( unvn_prev, uv_prev );
    cout << "Done Corvus()\n";
}


void Corvus::publishPoints3d( const ros::Publisher& pub )
{
    visualization_msgs::Marker marker;
    string ns = to_string(globalidx_of_curr)+"_"+to_string(globalidx_of_prev)+"__w_prev";
    eigenpointcloud_2_ros_markermsg( w_prev, marker, ns );
    // set per point color
    /*for( int i=0 ; i<iprev_X_iprev_triangulated.cols() ; i++ )
    {
        VectorXd _d = iprev_X_iprev_triangulated.col(i);
        double d = sqrt( _d(0)*_d(0) + _d(1)*_d(1) + _d(2)*_d(2) );

        bool is_behind = ( _d(2) < 0 )?true:false;
        bool good_ = ( reprojection_residue.col(i).norm() < 0.1 )?true:false;


        std_msgs::ColorRGBA Kolor;
        Kolor.a = 1.0;
        if( is_behind && good_  )  { Kolor.r=1.; Kolor.g=1.; Kolor.b=0.;} //yellow
        if( is_behind && !good_  ) { Kolor.r=0.; Kolor.g=0.; Kolor.b=1.; } //blue
        if( !is_behind && good_  ) { Kolor.r=0.; Kolor.g=1.; Kolor.b=0.; } //green
        if( !is_behind && !good_ ) { Kolor.r=1.; Kolor.g=1.; Kolor.b=1.;} //white

        marker.colors.push_back( Kolor );

    }*/
    pub.publish( marker );


    visualization_msgs::Marker marker2;
    string ns2 = to_string(globalidx_of_curr)+"_"+to_string(globalidx_of_prev)+"__w_curr";
    eigenpointcloud_2_ros_markermsg( w_curr, marker2, ns2 );
    marker2.color.r = 0;
    pub.publish( marker2 );

}





////////////////// Real stuff ///////////////////////////////////




//< compute pose using 3d points from previous and 2d points from curr.
bool Corvus::computeRelPose_3dprev_2dcurr( Matrix4d& to_return_p_T_c, ceres::Solver::Summary& summary )
{
    assert( isValid() );
    cout << "\nCorvus::computeRelPose_3dprev_2dcurr()\n";

    MatrixXd p_prev =  w_T_gid(globalidx_of_prev).inverse() *  w_prev;
    // also use unvn_curr



    //
    // Initial Guess
    Matrix4d c_T_p;
    c_T_p = w_T_gid( globalidx_of_curr ).inverse() * w_T_gid( globalidx_of_prev );

    double c_T_p_quat[10], c_T_p_trans[10], c_T_p_ypr[10];
    // convert to yprt for nullout
    eigenmat_to_rawyprt( c_T_p, c_T_p_ypr, c_T_p_trans );
    // nullout translation.
    c_T_p_trans[0] = 0.0; c_T_p_trans[1] = 0.0; c_T_p_trans[2] = 0.0;
    c_T_p_ypr[0] = 0.0;
    rawyprt_to_eigenmat( c_T_p_ypr, c_T_p_trans, c_T_p );

    Matrix4d c_T_p_init = c_T_p;
    eigenmat_to_raw( c_T_p, c_T_p_quat, c_T_p_trans );




    #if CORVUS_DEBUG_LVL > 0
    saveObservedPoints();
    saveReprojectedPoints( c_T_p, string("itr0") );
    #endif


    ceres::Problem problem;
    #ifndef CORVUS__align3d2d_with_switching_constraints__
    //
    // Setup problem - Ordinary 3d2d alignment
    for( int i=0 ; i<p_prev.cols() ; i++ )
    {
        ceres::CostFunction * cost_function = Align3d2d::Create( p_prev.col(i), unvn_curr.col(i),  sqrtweight_w_prev(i) );

        ceres::LossFunction *loss_function = NULL;
        // loss_function = new ceres::HuberLoss(.01);
        loss_function = new ceres::CauchyLoss(.05);


        problem.AddResidualBlock( cost_function, loss_function, c_T_p_quat, c_T_p_trans );
    }
    #endif

    #ifdef CORVUS__align3d2d_with_switching_constraints__
    //
    // Setup Problem - with Switching constraints
    double * switches = new double[p_prev.cols()]; //remeber to deallocate after Solve()

    for( int i=0 ; i<p_prev.cols() ; i++ )
    {
        ceres::CostFunction * cost_function = Align3d2dWithSwitchingConstraints::Create( p_prev.col(i), unvn_curr.col(i),  sqrtweight_w_prev(i) );

        ceres::LossFunction *loss_function = NULL;
        // loss_function = new ceres::HuberLoss(.01);
        // loss_function = new ceres::CauchyLoss(.05);

        switches[i] = 0.999;
        problem.AddResidualBlock( cost_function, loss_function, &switches[i], c_T_p_quat, c_T_p_trans );
    }
    #endif

    // Quaternion parameterization
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( c_T_p_quat, quaternion_parameterization );



    //
    // Solve
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    #if CORVUS_DEBUG_LVL > 4
    options.minimizer_progress_to_stdout = true;
    #endif
    // ceres::Solver::Summary summary; // This was removed because it is better to return the summary. This is done as an referemce argument to this function

    #if CORVUS_DEBUG_LVL >= 2
    //
    // Callback
    Align3d2d__4DOFCallback callback;
    callback.setOptimizationVariables_quatertion_t( c_T_p_quat, c_T_p_trans );
    options.callbacks.push_back(&callback);
    options.update_state_every_iteration = true;

    #endif


    ceres::Solve( options, &problem, &summary );
    cout << summary.BriefReport() << endl;


    //
    // Retrive Optimized Pose
    raw_to_eigenmat( c_T_p_quat, c_T_p_trans, c_T_p );
    to_return_p_T_c = c_T_p.inverse();
    cout << "p_T_c_init (inv of c_T_p):" ; prettyprintPoseMatrix( c_T_p_init );
    cout << "p_T_c_final (inv of c_T_p):" ; prettyprintPoseMatrix( to_return_p_T_c );
    #ifdef CORVUS__align3d2d_with_switching_constraints__
    // info on final states of switches
    int switching_inliers = _switches_stats( switches,  p_prev.cols(), 0.8 );
    cout << "switching_inliers=" << switching_inliers << " of " << p_prev.cols() << endl;
    #endif




    //  - Decide if this pose is acceptable.
    // Instead of returning pose, return status where, this pose is acceptable on not.
    // The pose can be written to input argument.

    bool status = true;
    string status_string = string("OK");
    {
        // Rule : Too much changes in the optimization variable.
        if( c_T_p.col(3).head(3).norm() > 2. ) {
            cout << "[Reject]: c_T_p.col(3).head(3).norm() > 2.\n";
            status_string = "[Reject]: c_T_p.col(3).head(3).norm() > 2";
            status = false;
        }

        // Rule: If less than 15 inliers with switching, suggest to reject
        #ifdef CORVUS__align3d2d_with_switching_constraints__
            // if less than 15 switching inliers, suggest to reject.
            if( switching_inliers < 15 ) {
                cout << "[Reject]: less than 15 switching inliers, suggest to reject.\n";
                status_string = string( "[Reject]: less than 15 switching inliers, suggest to reject." );
                status = false;
            }
        #endif
    }



    //
    // Process callback and write info to disk
    #if CORVUS_DEBUG_LVL > 0
    char __caption_string[500];
    // sprintf( __caption_string, "IsSolutionUsable=%d, cost0=%4.4f, final=%4.4f", summary.IsSolutionUsable(), summary.initial_cost(), summary.final_cost() );
    // sprintf( __caption_string, "total_time_msec=%4.2f, cost0=%4.4f, cost%d=%4.4f. %s", 1000.*summary.total_time_in_seconds, (float)summary.initial_cost,  summary.num_successful_steps, summary.final_cost, (status)?"OK":"Reject" );
    sprintf( __caption_string, "total_time_msec=%4.2f, cost0=%4.4f, cost%d=%4.4f. %s", 1000.*summary.total_time_in_seconds, (float)summary.initial_cost,  summary.num_successful_steps, summary.final_cost, status_string.c_str() );

    string __c_T_p_prettyprint;
    prettyprintPoseMatrix( c_T_p, __c_T_p_prettyprint );
    // saveReprojectedPoints( c_T_p, string("final"), string( __caption_string )+":c_T_p "+__c_T_p_prettyprint );

    #ifdef CORVUS__align3d2d_with_switching_constraints__
    saveReprojectedPoints( c_T_p, string("final"), string( __caption_string )+":c_T_p "+__c_T_p_prettyprint, switches );
    #else
    saveReprojectedPoints( c_T_p, string("final"), string( __caption_string )+":c_T_p "+__c_T_p_prettyprint );
    #endif


    #endif

    #if CORVUS_DEBUG_LVL > 1
    vector_of_callbacks.clear();
    vector_of_callbacks.push_back( callback );
    #endif

    #ifdef CORVUS__align3d2d_with_switching_constraints__
    delete [] switches; // delete the switch variables
    #endif

    return status;

}



// This uses 3d points from curr and 2d points from previous. This was added later (on 21st June, 2018)
bool Corvus::computeRelPose_2dprev_3dcurr( Matrix4d& to_return_p_T_c, ceres::Solver::Summary& summary )
{
    assert( isValid() );
    cout << "\nCorvus::computeRelPose_2dprev_3dcurr()\n";

    MatrixXd c_curr =  w_T_gid(globalidx_of_curr).inverse() *  w_curr; // 3d points of curr in curr's frame-of-ref
    // also use unvn_curr



    //
    // Initial Guess
    Matrix4d p_T_c;
    p_T_c = w_T_gid( globalidx_of_prev ).inverse() * w_T_gid( globalidx_of_curr );

    double p_T_c_quat[10], p_T_c_trans[10], p_T_c_ypr[10];
    // convert to yprt for nullout
    eigenmat_to_rawyprt( p_T_c, p_T_c_ypr, p_T_c_trans );
    // nullout translation.
    p_T_c_trans[0] = 0.0; p_T_c_trans[1] = 0.0; p_T_c_trans[2] = 0.0;
    p_T_c_ypr[0] = 0.0;
    rawyprt_to_eigenmat( p_T_c_ypr, p_T_c_trans, p_T_c );


    Matrix4d p_T_c_init = p_T_c;
    eigenmat_to_raw( p_T_c, p_T_c_quat, p_T_c_trans );



    // Save 3d points of curr projected on both views
    #if CORVUS_DEBUG_LVL > 0
    saveObservedPoints();
    saveReprojectedPoints__w_C( p_T_c, string("itr0") );
    #endif


    ceres::Problem problem;
    #ifndef CORVUS__align3d2d_with_switching_constraints__

    //
    // Setup problem
    for( int i=0 ; i<c_curr.cols() ; i++ )
    {
        ceres::CostFunction * cost_function = Align3d2d::Create( c_curr.col(i), unvn_prev.col(i),  sqrtweight_w_curr(i) );

        ceres::LossFunction *loss_function = NULL;
        // loss_function = new ceres::HuberLoss(.01);
        loss_function = new ceres::CauchyLoss(.05);


        problem.AddResidualBlock( cost_function, loss_function, p_T_c_quat, p_T_c_trans );
    }
    #endif

    #ifdef CORVUS__align3d2d_with_switching_constraints__
    //
    // Setup Problem - with Switching constraints
    double * switches = new double[c_curr.cols()]; //remeber to deallocate after Solve()

    for( int i=0 ; i<c_curr.cols() ; i++ )
    {
        ceres::CostFunction * cost_function = Align3d2dWithSwitchingConstraints::Create( c_curr.col(i), unvn_prev.col(i),  sqrtweight_w_curr(i) );

        ceres::LossFunction *loss_function = NULL;
        // loss_function = new ceres::HuberLoss(.01);
        // loss_function = new ceres::CauchyLoss(.05);

        switches[i] = 0.999;
        problem.AddResidualBlock( cost_function, loss_function, &switches[i], p_T_c_quat, p_T_c_trans );
    }
    #endif

    // Quaternion parameterization
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( p_T_c_quat, quaternion_parameterization );



    //
    // Solve
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    #if CORVUS_DEBUG_LVL > 4
    options.minimizer_progress_to_stdout = true;
    #endif
    // ceres::Solver::Summary summary; // This was removed because it is better to return the summary. This is done as an referemce argument to this function

    #if CORVUS_DEBUG_LVL >= 2
    //
    // Callback
    Align3d2d__4DOFCallback callback;
    callback.setOptimizationVariables_quatertion_t( p_T_c_quat, p_T_c_trans );
    options.callbacks.push_back(&callback);
    options.update_state_every_iteration = true;

    #endif


    ceres::Solve( options, &problem, &summary );
    cout << summary.BriefReport() << endl;


    //
    // Retrive Optimized Pose
    raw_to_eigenmat( p_T_c_quat, p_T_c_trans, p_T_c );
    cout << "p_T_c_init:" ; prettyprintPoseMatrix( p_T_c_init );
    cout << "p_T_c_final:" ; prettyprintPoseMatrix( p_T_c );
    to_return_p_T_c = p_T_c;
    #ifdef CORVUS__align3d2d_with_switching_constraints__
    // info on final states of switches
    int switching_inliers = _switches_stats( switches,  c_curr.cols(), 0.8 );
    cout << "switching_inliers=" << switching_inliers << " of " << c_curr.cols() << endl;
    #endif




    //  - Decide if this pose is acceptable.
    // Instead of returning pose, return status where, this pose is acceptable on not.
    // The pose can be written to input argument.

    bool status = true;
    string status_string = string("OK");
    {
        // Rule: If too much change, then reject
        if( p_T_c.col(3).head(3).norm() > 2. ) {
            cout << "[Reject]: p_T_c.col(3).head(3).norm() > 2.\n";
            status_string = "[Reject]: p_T_c.col(3).head(3).norm() > 2.";
            status = false;
        }


        // Rule: If less than 15 inliers with switching, suggest to reject
        #ifdef CORVUS__align3d2d_with_switching_constraints__
            // if less than 15 switching inliers, suggest to reject.
            if( switching_inliers < 15 ) {
                cout << "[Reject]: less than 15 switching inliers, suggest to reject.\n";
                status_string = string("[Reject]: less than 15 switching inliers, suggest to reject.");
                status = false;
            }
        #endif
    }



    //
    // Process callback and write info to disk
    #if CORVUS_DEBUG_LVL > 0
    char __caption_string[500];
    // sprintf( __caption_string, "IsSolutionUsable=%d, cost0=%4.4f, final=%4.4f", summary.IsSolutionUsable(), summary.initial_cost(), summary.final_cost() );
    // sprintf( __caption_string, "total_time_msec=%4.2f, cost0=%4.4f, cost%d=%4.4f. %s", 1000.*summary.total_time_in_seconds, (float)summary.initial_cost,  summary.num_successful_steps, summary.final_cost, (status)?"OK":"Reject" );
    sprintf( __caption_string, "total_time_msec=%4.2f, cost0=%4.4f, cost%d=%4.4f. %s", 1000.*summary.total_time_in_seconds, (float)summary.initial_cost,  summary.num_successful_steps, summary.final_cost, status_string.c_str() );

    string __p_T_c_prettyprint;
    prettyprintPoseMatrix( p_T_c, __p_T_c_prettyprint );
    // saveReprojectedPoints__w_C( p_T_c, string("final"), string( __caption_string )+":p_T_c "+__p_T_c_prettyprint );

    #ifdef CORVUS__align3d2d_with_switching_constraints__
    saveReprojectedPoints__w_C( p_T_c, string("final"), string( __caption_string )+":p_T_c "+__p_T_c_prettyprint, switches );
    #else
    saveReprojectedPoints__w_C( p_T_c, string("final"), string( __caption_string )+":p_T_c "+__p_T_c_prettyprint );
    #endif


    #endif

    // TODO
    #if CORVUS_DEBUG_LVL > 1
    vector_of_callbacks.clear();
    vector_of_callbacks.push_back( callback );
    #endif


    #ifdef CORVUS__align3d2d_with_switching_constraints__
    delete [] switches; // delete the switch variables
    #endif

    return status;

}


// 3d3d alignment.
// Solve  minimize_{p_T_c}   || c_T_p * p_X__{of prev}  -   p_X'__{of curr} ||
bool Corvus::computeRelPose_3dprev_3dcurr( Matrix4d& to_return_p_T_c, ceres::Solver::Summary& summary )
{
    assert( isValid() );
    cout << "\nCorvus::computeRelPose_3dprev_3dcurr()\n";

    //
    // convert w_P to p_P
    MatrixXd p_P = gid_T_w( globalidx_of_prev ) * w_prev;


    //
    // convert w_C to c_C
    MatrixXd c_C = gid_T_w( globalidx_of_curr ) * w_curr;

    // minimize_{ c_T_p \in SE(3) }    || p_P - pTc * c_C ||

    //
    // Initial Guess (Identity)
    Matrix4d p_T_c = Matrix4d::Identity();
    double p_T_c_quat[10], p_T_c_trans[10];

    Matrix4d p_T_c_init = p_T_c;
    eigenmat_to_raw( p_T_c, p_T_c_quat, p_T_c_trans);


    // Save 3d points of curr projected on both views
    #if CORVUS_DEBUG_LVL > 0
    saveObservedPoints();
    saveReprojectedPoints__w_C( p_T_c, string("3d3d_itr0") );
    #endif


    assert( p_P.cols() == c_C.cols() );
    ceres::Problem problem;
    #ifndef CORVUS__align3d3d_with_switching_constraints__
    //
    // Setup Problem - Ordinary
    for( int i=0 ; i<p_P.cols() ; i++ )
    {
        ceres::CostFunction * cost_function = Align3dPointsResidueEigen::Create( p_P.col(i), c_C.col(i), sqrt(sqrtweight_w_prev(i)*sqrtweight_w_curr(i)) );

        ceres::LossFunction *loss_function = NULL;
        // loss_function = new ceres::HuberLoss(.01);
        loss_function = new ceres::CauchyLoss(.1);

        problem.AddResidualBlock( cost_function, loss_function, p_T_c_quat, p_T_c_trans );
    }
    #endif

    #ifdef CORVUS__align3d3d_with_switching_constraints__
    //
    // Setup problem with switch constraints.
    double * switches = new double[p_P.cols()]; //remeber to deallocate after Solve()
    for( int i=0 ; i<p_P.cols() ; i++ )
    {

        ceres::CostFunction * cost_function = Align3dPointsWithSwitchingConstraints::Create( p_P.col(i), c_C.col(i), sqrt(sqrtweight_w_prev(i)*sqrtweight_w_curr(i)) );

        ceres::LossFunction *loss_function = NULL;
        // loss_function = new ceres::HuberLoss(.01);
        // loss_function = new ceres::CauchyLoss(.1);

        switches[i] = 0.999;
        problem.AddResidualBlock( cost_function, loss_function, &switches[i], p_T_c_quat, p_T_c_trans );
    }
    #endif

    //
    // Quaternion parameterization
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( p_T_c_quat, quaternion_parameterization );


    //
    // SOlve
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    ceres::Solve( options, &problem, &summary );
    cout << summary.BriefReport() << endl;
    // cout << summary.FullReport() << endl;


    //
    // Retrive Optimized Pose
    raw_to_eigenmat( p_T_c_quat, p_T_c_trans, p_T_c );
    cout << "p_T_c_init:" ; prettyprintPoseMatrix( p_T_c_init );
    cout << "p_T_c_final:" ; prettyprintPoseMatrix( p_T_c );
    to_return_p_T_c = p_T_c;
    #ifdef CORVUS__align3d3d_with_switching_constraints__
    // info on final states of switches
    int switching_inliers = _switches_stats( switches,  p_P.cols(), 0.8 );
    cout << "switching_inliers=" << switching_inliers << " of " << p_P.cols() << endl;
    #endif


    bool status = true;
    string status_string = string( "OK" );
    {

        // Rule: If less than 15 inliers with switching, suggest to reject
        #ifdef CORVUS__align3d3d_with_switching_constraints__
            // if less than 15 switching inliers, suggest to reject.
            if( switching_inliers < 15 ) {
                cout << "[Reject]: less than 15 switching inliers, suggest to reject.\n";
                status = false;
            }
        #endif

    }


    //
    // Write to disk
    #if CORVUS_DEBUG_LVL > 0
    char __caption_string[500];
    // sprintf( __caption_string, "IsSolutionUsable=%d, cost0=%4.4f, final=%4.4f", summary.IsSolutionUsable(), summary.initial_cost(), summary.final_cost() );
    // sprintf( __caption_string, "total_time_msec=%4.2f, cost0=%4.4f, cost%d=%4.4f. %s", 1000.*summary.total_time_in_seconds, (float)summary.initial_cost,  summary.num_successful_steps, summary.final_cost, (status)?"OK":"Reject" );
    sprintf( __caption_string, "total_time_msec=%4.2f, cost0=%4.4f, cost%d=%4.4f. %s", 1000.*summary.total_time_in_seconds, (float)summary.initial_cost,  summary.num_successful_steps, summary.final_cost, status_string.c_str() );

    string __p_T_c_prettyprint;
    prettyprintPoseMatrix( p_T_c, __p_T_c_prettyprint );
    // saveReprojectedPoints__w_C( p_T_c, string("3d3d_final"), string( __caption_string )+":p_T_c "+__p_T_c_prettyprint );

    #ifdef CORVUS__align3d3d_with_switching_constraints__
    saveReprojectedPoints__w_C( p_T_c, string("3d3d_final"), string( __caption_string )+":p_T_c "+__p_T_c_prettyprint, switches );
    #else
    saveReprojectedPoints__w_C( p_T_c, string("3d3d_final"), string( __caption_string )+":p_T_c "+__p_T_c_prettyprint );
    #endif

    #endif


    #ifdef CORVUS__align3d3d_with_switching_constraints__
    delete [] switches;
    #endif

    return status;

}

/////////////////////////////ROS Publishing helpers ///////////////////////////////
void Corvus::eigenpointcloud_2_ros_markermsg( const MatrixXd& M, visualization_msgs::Marker& marker, const string& ns )
{
    assert( M.rows()==3 || M.rows() == 4 );
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.header.seq = 0;
    marker.ns = ns; //"spheres";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = .05;
    marker.scale.y = .05;
    marker.scale.z = 1.05;

    marker.color.r = .8;
    marker.color.g = .8;
    marker.color.b = 0;
    marker.color.a = .9; // Don't forget to set the alpha!
    for( int i=0 ; i<M.cols() ; i++ )
    {
        geometry_msgs::Point pt;
        pt.x = M(0,i);
        pt.y = M(1,i);
        pt.z = M(2,i);
        marker.points.push_back( pt );
    }

}



////////////////////// Basic Helpers ///////////////////////////////
// Loop over each node and return the index of the node which is clossest to the specified stamp
int Corvus::find_indexof_node( const vector<Node*>& global_nodes, ros::Time stamp )
{
  ros::Duration diff;
  for( int i=0 ; i<global_nodes.size() ; i++ )
  {
    diff = global_nodes[i]->time_stamp - stamp;

    // cout << i << " "<< diff.sec << " " << diff.nsec << endl;

    if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
      return i;
    }
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
  return -1;
}


Matrix4d Corvus::w_T_gid( int gid )
{
    assert( gid >=0  && gid< global_nodes.size() );
    Matrix4d M; //w_T_i
    global_nodes[gid]->getOriginalTransform( M );
    return M;
}

Matrix4d Corvus::gid_T_w( int gid )
{
    assert( gid >=0  && gid< global_nodes.size() );
    Matrix4d M; //w_T_i
    global_nodes[gid]->getOriginalTransform( M );
    return M.inverse();
}

////////////////////// Writing Utils //////////////////////

#define write_image_debug( msg ) msg;
// #define write_image_debug( msg ) ;
void Corvus::write_image( string fname, const cv::Mat& img)
{
    assert( debug_directory_is_set && "Appears that debug directory is not set for class LocalBundle\n");

    // string base = string("/home/mpkuse/Desktop/bundle_adj/dump/corvus_");
    string base = BASE__DUMP+string("/dump/corvus_");


    #if CORVUS_DEBUG_LVL > 2
    write_image_debug( cout << "Writing file: "<< base << fname << endl );
    #endif
    cv::imwrite( (base+fname).c_str(), img );
}


////////////////////// Plotting //////////////////////////////
void Corvus::plot_points( const cv::Mat& im, const MatrixXd& pts,
            bool enable_text, bool enable_status_image, const string& msg ,
            cv::Mat& dst )
{
  assert( pts.rows() == 2 || pts.rows() == 3 );

  cv::Mat outImg = im.clone();
  ColorLUT lut = ColorLUT();

  int count = 0 ;
  for( int kl=0 ; kl<pts.cols() ; kl++ )
  {
      count++;
      cv::Point2d A( pts(0,kl), pts(1,kl) );


      cv::Scalar color = lut.get_color( kl % 64 );


      cv::circle( outImg, A, 1, color, -1 );

      if( enable_text ) // if write text only on 10% of the points to avoid clutter.
        cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
  }
  dst = outImg;
  return;
}



void Corvus::plot_point_sets( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                      const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                    //   const VectorXd& mask,
                      const cv::Scalar& color, bool annotate_pts,
                      /*const vector<string>& msg,*/
                      const string& msg,
                    cv::Mat& dst )
{
  // ptsA : ptsB : 2xN or 3xN

  assert( imA.rows == imB.rows );
  assert( imA.cols == imB.cols );
  assert( ptsA.cols() == ptsB.cols() );
  // assert( mask.size() == ptsA.cols() );

  cv::Mat outImg;
  cv::hconcat(imA, imB, outImg);

  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    // if( mask(kl) == 0 )
    //   continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    cv::circle( outImg, A, 2,color, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color, -1 );

    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), cv::Scalar(255,0,0) );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
    }
  }



  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, "marked # pts: "+to_string(count), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );


  // put msg in status image
  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
      std::vector<std::string> msg_tokens = split(msg, ':');
      for( int h=0 ; h<msg_tokens.size() ; h++ )
          cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1.5 );
  }

  cv::vconcat( outImg, status, dst );

}

// switches come from the switching constraints optimization for alignment.
void Corvus::plot_point_sets_with_switches( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                      const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                    //   const VectorXd& mask,
                    const double * switches,
                      const cv::Scalar& color, bool annotate_pts,
                      /*const vector<string>& msg,*/
                      const string& msg,
                    cv::Mat& dst )
{
  // ptsA : ptsB : 2xN or 3xN

  assert( imA.rows == imB.rows );
  assert( imA.cols == imB.cols );
  assert( ptsA.cols() == ptsB.cols() );
  // assert( mask.size() == ptsA.cols() );

  cv::Mat outImg;
  cv::hconcat(imA, imB, outImg);

  // loop over all points
  int count = 0;
  double sum_of_switches = 0.;
  int switching_inliers = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    // if( mask(kl) == 0 )
    //   continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    cv::circle( outImg, A, 2,color, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color, -1 );

    cv::Scalar line_color = cv::Scalar( 255,0,0 );
    if( switches != NULL )
    {
        char __tmp_string[20];
        snprintf( __tmp_string, 20, "%2.2f", (float)switches[kl] );

        if( switches[kl] > 0.8 )
        {
            cv::putText( outImg, string(__tmp_string), 0.5*(A+B+cv::Point2d(imA.cols,0)), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
            switching_inliers++;
        }
        else {
            cv::putText( outImg, string(__tmp_string), 0.5*(A+B+cv::Point2d(imA.cols,0)), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(10,10,10), 1 );
            line_color = cv::Scalar( 10, 10, 10 );
        }
        sum_of_switches += switches[kl];
    }

    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), line_color );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
    }
  }



  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );

  char marked[100];
  snprintf( marked, 100, "marked # pts: %d of %d; sum_of_switches=%4.2f", switching_inliers,  count, (double)sum_of_switches );
  cv::putText( status, string(marked), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );
  // cv::putText( status, "marked # pts: "+to_string(count)+"; sum_of_switches="+to_string((double)sum_of_switches), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );


  // put msg in status image
  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
      std::vector<std::string> msg_tokens = split(msg, ':');
      for( int h=0 ; h<msg_tokens.size() ; h++ )
          cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1.5 );
  }

  cv::vconcat( outImg, status, dst );

}


void Corvus::saveObservedPoints( )
{
    cv::Mat dst;
    assert( global_nodes[globalidx_of_prev]->valid_image() );
    assert( global_nodes[globalidx_of_curr]->valid_image() );
    cv::Mat prev_im = global_nodes[globalidx_of_prev]->getImageRef();
    cv::Mat curr_im = global_nodes[globalidx_of_curr]->getImageRef();

    /*
    // plot uv_prev on prev_im
    cout << "valid_image: " << global_nodes[globalidx_of_prev]->valid_image() << endl;
    assert( global_nodes[globalidx_of_prev]->valid_image() );
    cv::imshow( "prev_im", prev_im );
    plot_points( prev_im, uv_prev, true, true, string(""), dst  );
    write_image( to_string(globalidx_of_prev)+"_observed.png", dst  );

    // reproject w_prev on prev_im.
    MatrixXd p_prev =  w_T_gid(globalidx_of_prev).inverse() *  w_prev;
    MatrixXd projected_prev;
    camera.perspectiveProject3DPoints( p_prev, projected_prev );
    plot_points( prev_im, projected_prev, true, true, string(""), dst  );
    write_image( to_string(globalidx_of_prev)+"_reproj.png", dst  );
    */

    // plot uv_curr on curr_im



    // plot [ curr | prev ] with uv marked on each
    plot_point_sets( curr_im, uv_curr, globalidx_of_curr,
                     prev_im, uv_prev, globalidx_of_prev,
                     cv::Scalar(0,255,0), true, string("my"), dst );
    write_image( to_string(globalidx_of_curr) + "_" + to_string(globalidx_of_prev)+"_observed_match.png", dst );


}



// Create an image : [ C | P ]. Mark the projected points.
//         [[  PI( c_T_p * p_T_w * w_P ) ||   PI( p_T_w * w_P)   ]]
void Corvus::saveReprojectedPoints( const Matrix4d& c_T_p, const string& fname_suffix, const string image_caption_msg, const double * switches )
{
    Matrix4d p_T_w = gid_T_w( globalidx_of_prev );

    // PI( c_T_p * p_T_w * w_P )
    MatrixXd c_prev = (c_T_p * p_T_w) * w_prev;
    MatrixXd projected_c_prev;
    camera.perspectiveProject3DPoints( c_prev, projected_c_prev );

    // PI( p_T_w * w_P)
    MatrixXd p_prev = p_T_w * w_prev;
    MatrixXd projected_p_prev;
    camera.perspectiveProject3DPoints( p_prev, projected_p_prev );


    //
    assert( global_nodes[globalidx_of_prev]->valid_image() );
    assert( global_nodes[globalidx_of_curr]->valid_image() );
    cv::Mat prev_im = global_nodes[globalidx_of_prev]->getImageRef();
    cv::Mat curr_im = global_nodes[globalidx_of_curr]->getImageRef();
    cv::Mat dst;
    if( switches == NULL ) {
        plot_point_sets( curr_im, projected_c_prev, globalidx_of_curr,
                         prev_im, projected_p_prev, globalidx_of_prev,
                         cv::Scalar( 0,0,255 ), true,
                         string( "project w_prev on both frames.")+fname_suffix+":"+image_caption_msg,
                         dst
                     );
     }

     if( switches != NULL ) {
         plot_point_sets_with_switches( curr_im, projected_c_prev, globalidx_of_curr,
                          prev_im, projected_p_prev, globalidx_of_prev,
                          switches,
                          cv::Scalar( 0,0,255 ), true,
                          string( "project w_prev on both frames.")+fname_suffix+":"+image_caption_msg,
                          dst
                      );
     }


    write_image( to_string(globalidx_of_curr) + "_" + to_string(globalidx_of_prev)+"_reprojected_w_P_"+fname_suffix+".png", dst );

}

// Create an image : [ C | P ]. Mark the projected points. 3D points are w_curr (3d points of current image)
//          [[     PI(c_T_w * w_C)  ||    PI(p_T_c * c_T_w * w_C)    ]]
void Corvus::saveReprojectedPoints__w_C( const Matrix4d& p_T_c, const string& fname_suffix, const string image_caption_msg,  const double * switches )
{
    Matrix4d c_T_w = gid_T_w( globalidx_of_curr );

    //  PI(c_T_w * w_C)
    MatrixXd c_curr = c_T_w * w_curr;
    MatrixXd projected_c_curr;
    camera.perspectiveProject3DPoints( c_curr, projected_c_curr );

    // PI(p_T_c * c_T_w * w_C)
    MatrixXd p_curr = p_T_c * c_T_w * w_curr;
    MatrixXd projected_p_curr;
    camera.perspectiveProject3DPoints( p_curr, projected_p_curr );


    //
    assert( global_nodes[globalidx_of_prev]->valid_image() );
    assert( global_nodes[globalidx_of_curr]->valid_image() );
    cv::Mat prev_im = global_nodes[globalidx_of_prev]->getImageRef();
    cv::Mat curr_im = global_nodes[globalidx_of_curr]->getImageRef();
    cv::Mat dst;

    if( switches == NULL ) {
    plot_point_sets( curr_im, projected_c_curr, globalidx_of_curr,
                     prev_im, projected_p_curr, globalidx_of_prev,
                     cv::Scalar( 0,0,255 ), true,
                     string( "project w_curr on both frames.")+fname_suffix+":"+image_caption_msg,
                     dst
                 );
     }

     if( switches != NULL ) {
         plot_point_sets_with_switches( curr_im, projected_c_curr, globalidx_of_curr,
                          prev_im, projected_p_curr, globalidx_of_prev,
                          switches,
                          cv::Scalar( 0,0,255 ), true,
                          string( "project w_curr on both frames.")+fname_suffix+":"+image_caption_msg,
                          dst
                      );
     }


    write_image( to_string(globalidx_of_curr) + "_" + to_string(globalidx_of_prev)+"_reprojected__w_C_"+fname_suffix+".png", dst );

}

void Corvus::saveReprojectedImagesFromCeresCallbacks( )
{
    if( vector_of_callbacks.size() == 0 )
    {
        cout << "saveReprojectedImagesFromCeresCallbacks:Nothing in vector_of_callbacks\n";
        return;
    }

    Align3d2d__4DOFCallback callback = vector_of_callbacks[0];
    cout << "processCeresCallbacks: # iterations = " << callback.pose_at_each_iteration.size() << endl;

    for( int itr=0 ; itr<callback.pose_at_each_iteration.size() ; itr++ )
    {
        Matrix4d ci_T_p = callback.pose_at_each_iteration[itr]; //pose at kth iteration
        double loss_i = callback.loss_at_each_iteration[itr] ;

        char __caption_string[500];
        sprintf( __caption_string, "loss_%d=%4.4f", itr, loss_i );
        saveReprojectedPoints( ci_T_p, "itr"+to_string(itr), string( __caption_string ) );
    }
}


void Corvus::publishCameraPoseFromCeresCallbacks( const ros::Publisher& pub )
{
    if( vector_of_callbacks.size() == 0 )
    {
        cout << "publishCameraPoseFromCeresCallbacks:Nothing in vector_of_callbacks\n";
        return;
    }

    Align3d2d__4DOFCallback callback = vector_of_callbacks[0];
    cout << "processCeresCallbacks: # iterations = " << callback.pose_at_each_iteration.size() << endl;

    Matrix4d p_T_w = gid_T_w( globalidx_of_prev );

    visualization_msgs::Marker cam_marker;
    init_camera_marker( cam_marker );
    cam_marker.ns = to_string(globalidx_of_curr)+"_"+to_string(globalidx_of_prev)+"__cams";
    int id = 0;

    // publish curr and prev
    setcolor_to_cameravisual( 1.0, 0.8, 0, cam_marker );
    setpose_to_cameravisual( w_T_gid( globalidx_of_prev ), cam_marker );
    cam_marker.id = id++;
    pub.publish( cam_marker );
    setcolor_to_cameravisual( .8, 0., 1.0, cam_marker );
    setpose_to_cameravisual( w_T_gid( globalidx_of_curr ), cam_marker );
    cam_marker.id = id++;
    pub.publish( cam_marker );




    visualization_msgs::Marker iteration_line;
    iteration_line.header = cam_marker.header;
    iteration_line.ns = cam_marker.ns;
    iteration_line.type = visualization_msgs::Marker::LINE_STRIP;
    iteration_line.action = visualization_msgs::Marker::ADD;
    iteration_line.scale.x = 0.01;
    iteration_line.color.r = 1.0; iteration_line.color.g = 1.0; iteration_line.color.b = 1.0; iteration_line.color.a = 1.0;
    for( int itr=0 ; itr<callback.pose_at_each_iteration.size() ; itr++ )
    {
        Matrix4d ci_T_p = callback.pose_at_each_iteration[itr]; //pose at kth iteration
        cout << "itr=" << itr << " ";
        prettyprintPoseMatrix( ci_T_p );

        Matrix4d w_T_ci = (ci_T_p * p_T_w ).inverse();


        float c = (float)itr/(float)callback.pose_at_each_iteration.size();
        setcolor_to_cameravisual( c, 0., 0 , cam_marker );
        setpose_to_cameravisual( w_T_ci, cam_marker );
        cam_marker.id = id++;
        pub.publish( cam_marker );


        geometry_msgs::Point pt_t;
        pt_t.x = w_T_ci(0,3);
        pt_t.y = w_T_ci(1,3);
        pt_t.z = w_T_ci(2,3);
        iteration_line.points.push_back( pt_t );
    }

    iteration_line.id = id++;
    pub.publish(iteration_line );
}


void Corvus::init_camera_marker( visualization_msgs::Marker& marker )
{
     marker.header.frame_id = "world";
     marker.header.stamp = ros::Time::now();
     marker.action = visualization_msgs::Marker::ADD;
     marker.color.a = 1.0; // Don't forget to set the alpha!
     marker.type = visualization_msgs::Marker::LINE_LIST;
    //  marker.id = i;
    //  marker.ns = "camerapose_visual";

     marker.scale.x = 0.005; //width of line-segments
     float __vcam_width = 0.07*1.5;
     float __vcam_height = 0.04*1.5;
     float __z = 0.05;

     marker.points.clear();
     geometry_msgs::Point pt;
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );

     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );


     // TOSET
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;
}

void Corvus::setpose_to_cameravisual( const Matrix4d& w_T_c, visualization_msgs::Marker& marker )
{
    Quaterniond quat( w_T_c.topLeftCorner<3,3>() );
    marker.pose.position.x = w_T_c(0,3);
    marker.pose.position.y = w_T_c(1,3);
    marker.pose.position.z = w_T_c(2,3);
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
}

void Corvus::setcolor_to_cameravisual( float r, float g, float b, visualization_msgs::Marker& marker  )
{
    marker.color.r = r;
    marker.color.b = g;
    marker.color.g = b;
}

///////////////// Pose compitation related helpers /////////////////

void Corvus::raw_to_eigenmat( const double * quat, const double * t, Matrix4d& dstT )
{
  Quaterniond q = Quaterniond( quat[0], quat[1], quat[2], quat[3] );

  dstT = Matrix4d::Zero();
  dstT.topLeftCorner<3,3>() = q.toRotationMatrix();

  dstT(0,3) = t[0];
  dstT(1,3) = t[1];
  dstT(2,3) = t[2];
  dstT(3,3) = 1.0;
}

void Corvus::eigenmat_to_raw( const Matrix4d& T, double * quat, double * t)
{
  assert( T(3,3) == 1 );
  Quaterniond q( T.topLeftCorner<3,3>() );
  quat[0] = q.w();
  quat[1] = q.x();
  quat[2] = q.y();
  quat[3] = q.z();
  t[0] = T(0,3);
  t[1] = T(1,3);
  t[2] = T(2,3);
}

void Corvus::rawyprt_to_eigenmat( const double * ypr, const double * t, Matrix4d& dstT )
{
  dstT = Matrix4d::Identity();
  Vector3d eigen_ypr;
  eigen_ypr << ypr[0], ypr[1], ypr[2];
  dstT.topLeftCorner<3,3>() = ypr2R( eigen_ypr );
  dstT(0,3) = t[0];
  dstT(1,3) = t[1];
  dstT(2,3) = t[2];
}

void Corvus::eigenmat_to_rawyprt( const Matrix4d& T, double * ypr, double * t)
{
  assert( T(3,3) == 1 );
  Vector3d T_cap_ypr = R2ypr( T.topLeftCorner<3,3>() );
  ypr[0] = T_cap_ypr(0);
  ypr[1] = T_cap_ypr(1);
  ypr[2] = T_cap_ypr(2);

  t[0] = T(0,3);
  t[1] = T(1,3);
  t[2] = T(2,3);
}

Vector3d Corvus::R2ypr( const Matrix3d& R)
{
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}


Matrix3d Corvus::ypr2R( const Vector3d& ypr)
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



void Corvus::prettyprintPoseMatrix( const Matrix4d& M )
{
  cout << "YPR      : " << R2ypr(  M.topLeftCorner<3,3>() ).transpose() << "; ";
  cout << "Tx,Ty,Tz : " << M(0,3) << ", " << M(1,3) << ", " << M(2,3) << endl;
}

void Corvus::prettyprintPoseMatrix( const Matrix4d& M, string& return_string )
{
   Vector3d ypr;
   ypr = R2ypr(  M.topLeftCorner<3,3>()  );
  // cout << "YPR      : " << R2ypr(  M.topLeftCorner<3,3>() ).transpose() << "; ";
  // cout << "Tx,Ty,Tz : " << M(0,3) << ", " << M(1,3) << ", " << M(2,3) << endl;

  // return_string = "YPR=("+to_string(ypr(0))+","+to_string(ypr(1))+","+to_string(ypr(2))+")";
  // return_string += "  TxTyTz=("+ to_string(M(0,3))+","+ to_string(M(1,3))+","+ to_string(M(2,3))+")";

  char __tmp[200];
  snprintf( __tmp, 200, "YPR=(%4.2f,%4.2f,%4.2f)  TxTyTz=(%4.2f,%4.2f,%4.2f)",  ypr(0), ypr(1), ypr(2), M(0,3), M(1,3), M(2,3) );
  return_string = string( __tmp );

}




// other helpers
template<typename Out>
void Corvus::split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> Corvus::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}
