#include "EdgeManager.h"

void REdge::add_global_correspondence( int gid_curr_i, int gid_prev_i )
{
    gid_curr.push_back( gid_curr_i );
    gid_prev.push_back( gid_prev_i );
}


EdgeManager::EdgeManager( DataManager* dm)
{
    manager = dm;
    is_datamanager_available = true;
}


#define __EdgeManager_loop_DEBUG_LVL__ 1
void EdgeManager::loop()
{
    vector<bool> seen;

    assert(is_datamanager_available);
    manager->getCameraRef().printCameraInfo(0);
    ros::Rate loop_rate(0.5);

    int prev_len = 0;
    int problem_instance = 0;
    while( ros::ok() )
    {
        loop_rate.sleep();

        //// QUEUE STATUS ///////////////////////////////////////
        cout << _COLOR_RED_ << "[loop] len(nNodes)= " << manager->getNodesSize() << ";\t";
        cout << "len(global_features)=" << manager->getTFIDFRef()->nFeatures() << ";\t";
        // cout << "len(nEdges)=" << nEdges.size() << ";\t";
        cout << "len(nEdges)=" << manager->getREdgesSize() << ";\t";
        cout << _COLOR_DEFAULT_ << endl;

        // // Here is how you need to use PoseUtils
        // Matrix4d __tty = Matrix4d::Identity();
        // cout << "prettyprintMatrix4d: " << PoseManipUtils::prettyprintMatrix4d( __tty ) << endl;

        // // Here is how to generate markers
        // visualization_msgs::Marker m;
        // RosMarkerUtils::init_text_marker( m );
        // m.text = "HEllo from mpkuse";
        // m.scale.z = 3;
        // m.ns = "the_text";
        // manager->getMarkerPublisher().publish( m );

        ///////////////////////////////////////////////////////

        int LEN = manager->getREdgesSize();

        if( LEN == prev_len ) {
            continue;
        }

        cout << "----\n";

        //////// General Usage - Printing /////////
        vector< pair<int,int> > set_L;
        vector< int > set_L_n_occurences;
        cout << _COLOR_RED_ << "Detected new edge from idx=["<< prev_len << " to " << LEN << ")" << _COLOR_DEFAULT_ << endl;
        for( int i=prev_len ; i < LEN ; i++ ) // loop on the newly added REdges
        {
            int i_curr = (manager->getREdgesRef())[i]->i_curr;
            int i_prev = (manager->getREdgesRef())[i]->i_prev;
            cout << _COLOR_RED_<< "\tRedge["<< i << "]: " << i_curr << "<--->" << i_prev;
            cout << " contains "<< (manager->getREdgesRef())[i]->gid_curr.size() << " feature correspondences\n" << _COLOR_DEFAULT_;
            assert( (manager->getREdgesRef())[i]->gid_curr.size() == (manager->getREdgesRef())[i]->gid_prev.size() );
            for( int h=0; h<(manager->getREdgesRef())[i]->gid_curr.size() ; h++ )
            {
                int gid_curr_h = (manager->getREdgesRef())[i]->gid_curr[h];
                int gid_prev_h = (manager->getREdgesRef())[i]->gid_prev[h];

                // printing
                #if __EdgeManager_loop_DEBUG_LVL__ >= 2
                cout << _COLOR_YELLOW_ << "\t" << gid_curr_h << "<-->" << gid_prev_h << _COLOR_DEFAULT_;
                if( h%5 == 0 )
                    cout << endl;
                #endif

                pair<int,int> this_pair( gid_curr_h, gid_prev_h );
                vector<pair<int,int>>::iterator it = std::find( set_L.begin(), set_L.end(), this_pair );
                if( it == set_L.end() )
                {
                    // not found , so add
                    set_L.push_back( this_pair );
                    set_L_n_occurences.push_back( 1 );
                }
                else
                {
                    // found at index : `it - set_L.begin()`
                    set_L_n_occurences[ int(it - set_L.begin()) ]++;
                }
            }
            #if __EdgeManager_loop_DEBUG_LVL__ >= 2
            cout << endl;
            #endif
        }

        cout << _COLOR_YELLOW_ << "Found "<< set_L.size() << " unique pairs from " <<  LEN-prev_len << " views\n" << _COLOR_DEFAULT_;
        #if __EdgeManager_loop_DEBUG_LVL__ >= 2
        for( int i=0 ; i<set_L.size() ; i++ )
        {
            cout << set_L[i].first << "<--#" << set_L_n_occurences[i] << "-->" << set_L[i].second << "\t";
            if( i%5==0 )
                cout << endl;
        }
        cout << endl;
        #endif
        //////////OUTPUT: set_L and set_L_n_occurences ///////////////////

        ///// Collect 3d points in  4xN arrays.
        MatrixXd w_X  = MatrixXd::Zero( 4, set_L.size() ); // 3D points of {curr_i} \forall i
        MatrixXd w_Xd = MatrixXd::Zero( 4, set_L.size() ); // 3D points of {prev_i} \forall i
        MatrixXd w_X_variance  = MatrixXd::Zero( 4, set_L.size() ); // 3D points of {curr_i} \forall i
        MatrixXd w_Xd_variance = MatrixXd::Zero( 4, set_L.size() ); // 3D points of {prev_i} \forall i
        for( int i=0 ; i<set_L.size() ; i++ )
        {
            int gid_1 = set_L[i].first;
            int gid_2 = set_L[i].second;
            int common_in = set_L_n_occurences[i] ;

            // Mean
            Vector4d gid_1_3dpt, gid_2_3dpt;
            manager->getTFIDFRef()->query_feature_mean( gid_1, gid_1_3dpt );
            manager->getTFIDFRef()->query_feature_mean( gid_2, gid_2_3dpt );
            w_X.col( i )  = gid_1_3dpt;
            w_Xd.col( i ) = gid_2_3dpt;

            // Variance
            Vector4d gid_1_3dpt_var, gid_2_3dpt_var;
            manager->getTFIDFRef()->query_feature_var( gid_1, gid_1_3dpt_var );
            manager->getTFIDFRef()->query_feature_var( gid_2, gid_2_3dpt_var );
            w_X_variance.col( i )  = gid_1_3dpt_var;
            w_Xd_variance.col( i ) = gid_2_3dpt_var;
        }
        ////// Output `w_X`(3d points of current set), `w_Xd` (3d points from prev set)



        /////////////////// Visualize Cameras that we are going to use //////////////
        int xid;
        xid = 0;
        for( int i=prev_len ; i < LEN ; i++, xid++ ) // loop on the newly added REdges
        {
            int i_curr = (manager->getREdgesRef())[i]->i_curr;
            int i_prev = (manager->getREdgesRef())[i]->i_prev;

            visualization_msgs::Marker nm, nn;
            RosMarkerUtils::init_camera_marker( nm, 2. );
            // nm.ns = "problem_"+to_string( problem_instance )+"_views_of";
            nm.ns = "_views";
            nm.id = 2*xid;

            RosMarkerUtils::init_camera_marker( nn, 2. );
            // nn.ns = "problem_"+to_string( problem_instance )+"views_of_";
            nn.ns = "_views";
            nn.id = 2*xid+1;

            Matrix4d w_T_curr, w_T_prev;
            manager->getNodesRef()[i_curr]->getCurrTransform(w_T_curr);
            manager->getNodesRef()[i_prev]->getCurrTransform(w_T_prev);

            RosMarkerUtils::setpose_to_marker( w_T_curr, nm );
            RosMarkerUtils::setcolor_to_marker( 1.0, 1.0, 0.0, nm ); // yellow

            RosMarkerUtils::setpose_to_marker( w_T_prev, nn );
            RosMarkerUtils::setcolor_to_marker( 1.0, 0.07, 0.6, nn ); // pink


            manager->getMarkerPublisher().publish( nm );
            manager->getMarkerPublisher().publish( nn );
        }
        xid = 2*(xid);
        for( ; xid < 50 ; xid++ ) // zero out other markers in rviz
        {
            visualization_msgs::Marker _tmp;
            RosMarkerUtils::init_line_marker( _tmp );
            _tmp.ns = "_views";
            _tmp.id = xid;
            manager->getMarkerPublisher().publish( _tmp );
        }


        /////// Visualize the 3D points that are going to be used
        visualization_msgs::Marker m_points__w_X;
        _3dpoints_to_markerpoints( w_X, m_points__w_X ); //< Just plot 3D point
        m_points__w_X.ns = "correspondences_points"; m_points__w_X.id = 0;
        RosMarkerUtils::setcolor_to_marker( 1.0, 1.0, 0.0, 1.0, m_points__w_X ); // yellow
        manager->getMarkerPublisher().publish( m_points__w_X );

        #if 0
        _3dpoints_with_var_to_markerpoints( w_X, w_X_variance, m_points__w_X ); //< Plot 3d point with variance
        m_points__w_X.ns = "correspondences_points"; m_points__w_X.id = 1;
        RosMarkerUtils::setcolor_to_marker( 1.0, 1.0, 0.0, 1.0, m_points__w_X );
        manager->getMarkerPublisher().publish( m_points__w_X );
        #endif


        visualization_msgs::Marker m_points__w_Xd;
        _3dpoints_to_markerpoints( w_Xd, m_points__w_Xd );//< Just plot 3D point
        m_points__w_Xd.ns = "correspondences_points"; m_points__w_Xd.id = 2;
        RosMarkerUtils::setcolor_to_marker( 1.0, .07, 0.6, 1.0, m_points__w_Xd ); //pink
        manager->getMarkerPublisher().publish( m_points__w_Xd );


        #if 0
        _3dpoints_with_var_to_markerpoints( w_Xd, w_Xd_variance, m_points__w_Xd ); //< Plot 3d point with variance
        m_points__w_Xd.ns = "correspondences_points"; m_points__w_Xd.id = 3;
        RosMarkerUtils::setcolor_to_marker( 1.0, .07, 0.6, .0, m_points__w_Xd );
        manager->getMarkerPublisher().publish( m_points__w_Xd );
        #endif


        #if 1
        /////// Visualize correspondences as lines
        visualization_msgs::Marker m_points_correspondences;
        _3dpointspair_to_markerlines( w_X, w_Xd, m_points_correspondences );
        m_points_correspondences.ns = "correspondences_points"; m_points_correspondences.id = 4;
        RosMarkerUtils::setcolor_to_marker( 0.95, 0.65, 0.37, 0.1, m_points_correspondences ); // skin color
        manager->getMarkerPublisher().publish( m_points_correspondences );
        #endif



        ////// END. Output `w_X`(3d points of current set), `w_Xd` (3d points from prev set)



        /////////// Setup Optimization Problem - 3D3D alignment ////////////////////////////
        // minimize || F_i - T . F_j || . F_i in ref-frame of i_curr. F_j in ref-frame of i_prev

        int i_curr = (manager->getREdgesRef())[prev_len]->i_curr; // reference frames
        int i_prev = (manager->getREdgesRef())[prev_len]->i_prev;
        printf( "ceres-solve ::: \\underset{^{%d} T_{%d} }{minimize} \\sum_i || ^{%d} F_i - ^{%d} T_{%d} . ^{%d} F'_i ||_2^2\n", i_curr, i_prev, i_curr, i_curr, i_prev, i_prev );


        Matrix4d w_T_curr, w_T_prev;
        manager->getNodesRef()[i_curr]->getCurrTransform(w_T_curr);
        manager->getNodesRef()[i_prev]->getCurrTransform(w_T_prev);

        MatrixXd c_X  = w_T_curr.inverse() * w_X;
        MatrixXd p_Xd = w_T_prev.inverse() * w_Xd;
        // cout << "w_X:\n" << w_X.transpose()  << endl;
        // cout << "c_X:\n" <<  c_X.transpose()  << endl;
        // cout << endl;
        // cout << "w_Xd:\n" << w_Xd.transpose()  << endl;
        // cout << "p_Xd:\n" << p_Xd.transpose()  << endl;

        //
        // Initial Guess
        Matrix4d c_T_p = Matrix4d::Identity();
        double c_quat_p[5];
        double c_t_p[5];
        PoseManipUtils::eigenmat_to_raw( c_T_p, c_quat_p, c_t_p );
        #if __EdgeManager_DEBUG_LVL >= 2
        cout << "c_T_p (initial):\n" << c_T_p << endl;
        #endif
        cout << "c_T_p (initial): " << PoseManipUtils::prettyprintMatrix4d( c_T_p ) << endl;



        //
        // Setup Residue Terms
        ceres::Problem problem;
        for( int i=0 ; i<set_L.size() ; i++ )
        {
            ceres::CostFunction * cost_function = Align3dPointsResidueEigen::Create( c_X.col(i), p_Xd.col(i), 1.0 );
            problem.AddResidualBlock( cost_function, new ceres::CauchyLoss( 0.1 ), c_quat_p, c_t_p );
        }

        //
        // Run
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        #if __EdgeManager_DEBUG_LVL >= 2
        options.minimizer_progress_to_stdout = true;
        #endif
        ceres::Solver::Summary summary;

        ceres::Solve( options, &problem, &summary );
        cout << summary.BriefReport() << endl;
        cout << "3D-3D Aligned "<< c_X.cols() << "pair of 3d points;   ";
        cout << "took (in sec) : " << summary.total_time_in_seconds << ";   ";
        cout << "nIterations   : " << summary.num_successful_steps;
        cout << endl;

        //
        // Retrive Results (Optimization Variables)
        PoseManipUtils::raw_to_eigenmat( c_quat_p, c_t_p, c_T_p );
        #if __EdgeManager_DEBUG_LVL >= 2
        cout << "c_T_p (optimized):\n" << c_T_p << endl;
        #endif
        cout << "c_T_p (optimized): " << PoseManipUtils::prettyprintMatrix4d( c_T_p ) << endl;





        //
        // Forward p_T_c to the Graph-Pose-Graph-Optimization Engine (separate Node)
        ros::Time t_c = manager->getNodesRef()[i_curr]->time_stamp;
        ros::Time t_p = manager->getNodesRef()[i_prev]->time_stamp;
        Matrix4d p_T_c;
        p_T_c = c_T_p.inverse();
        manager->republish_nap( t_c, t_p, p_T_c, 30, 1.0  );


        ////////////////// Post Optimization Visualization /////////////////////////////
        //
        // Visualize the effect of solving the 3d3d alignment problem
        // plot the 3d points in new co-ordinate system.Note that w_Xd (prev) points remain the same
        MatrixXd w_X_new;
        w_X_new = (w_T_prev * p_T_c) * c_X;



        visualization_msgs::Marker m_points__w_X_new;
        _3dpoints_to_markerpoints( w_X_new, m_points__w_X_new );
        m_points__w_X_new.ns = "correspondences_points"; m_points__w_X_new.id = 5;
        RosMarkerUtils::setcolor_to_marker( 1.0, .08, 0.6, 1.0, m_points__w_X_new );
        manager->getMarkerPublisher().publish( m_points__w_X_new );

        #if 1
        visualization_msgs::Marker new_correspondences;
        _3dpointspair_to_markerlines( w_X_new, w_Xd, new_correspondences );
        new_correspondences.ns = "correspondences_points"; new_correspondences.id = 6;
        RosMarkerUtils::setcolor_to_marker( .6, 0.12, .85, 0.1, new_correspondences );
        manager->getMarkerPublisher().publish( new_correspondences );
        #endif


        // Done

        prev_len =  manager->getREdgesSize();
        problem_instance++;

    }
}



void EdgeManager::_3dpoints_to_markerpoints( const MatrixXd& w_X, visualization_msgs::Marker& m_points )
{
    assert( w_X.cols() > 0 );
    assert( (w_X.rows() == 3) || (w_X.rows() == 4 && w_X(3,0) > .999 && w_X(3,0) < 1.001 ) );

    RosMarkerUtils::init_points_marker( m_points );
    for( int i=0 ; i<w_X.cols() ; i++ )
    {
        // Set point and color for 1st
        geometry_msgs::Point pt1;
        pt1.x = w_X(0,i);
        pt1.y = w_X(1,i);
        pt1.z = w_X(2,i);

        m_points.points.push_back( pt1 );
    }
}


void EdgeManager::_3dpoints_with_var_to_markerpoints( const MatrixXd& w_X, const MatrixXd& w_X_variance,
    visualization_msgs::Marker& m_points_with_var )
{
    assert( w_X.cols() > 0 );
    assert( (w_X.rows() == 3) || (w_X.rows() == 4 && w_X(3,0) > .999 && w_X(3,0) < 1.001 ) );
    assert( w_X_variance.rows() == 3 || w_X_variance.rows() == 4 );
    assert( w_X_variance.cols() == w_X.cols() );

    RosMarkerUtils::init_line_marker( m_points_with_var );
    m_points_with_var.color.a = 0.1;

    std_msgs::ColorRGBA red_color, green_color, blue_color;
    red_color.r = 1.0; red_color.g=0.0; red_color.b = 0.0; red_color.a = .2;
    green_color.r = 0.0; green_color.g=1.0; green_color.b = 0.0; green_color.a = .2;
    blue_color.r = 0.0; blue_color.g=0.0; blue_color.b = 1.0; blue_color.a = .2;

    m_points_with_var.points.clear();
    for( int i=0 ; i<w_X.cols() ; i++ )
    {

        // X-axis
        {
            geometry_msgs::Point pt1, pt2;
            pt1.x = w_X(0,i) + sqrt(w_X_variance( 0, i));
            pt1.y = w_X(1,i) ;
            pt1.z = w_X(2,i) ;

            pt2.x = w_X(0,i) - sqrt(w_X_variance( 0, i));
            pt2.y = w_X(1,i) ;
            pt2.z = w_X(2,i) ;

            m_points_with_var.points.push_back( pt1 );
            m_points_with_var.points.push_back( pt2 );
            m_points_with_var.colors.push_back( red_color );
            m_points_with_var.colors.push_back( red_color );
        }


        // Y-axis
        {
            geometry_msgs::Point pt1, pt2;

            pt1.x = w_X(0,i) ;
            pt1.y = w_X(1,i) + sqrt(w_X_variance( 1, i));
            pt1.z = w_X(2,i) ;

            pt2.x = w_X(0,i) ;
            pt2.y = w_X(1,i) - sqrt(w_X_variance( 1, i));
            pt2.z = w_X(2,i) ;

            m_points_with_var.points.push_back( pt1 );
            m_points_with_var.points.push_back( pt2 );
            m_points_with_var.colors.push_back( green_color );
            m_points_with_var.colors.push_back( green_color );
        }

        // Z-axis
        {
            geometry_msgs::Point pt1, pt2;

            pt1.x = w_X(0,i) ;
            pt1.y = w_X(1,i) ;
            pt1.z = w_X(2,i) + sqrt(w_X_variance( 2, i));

            pt2.x = w_X(0,i) ;
            pt2.y = w_X(1,i) ;
            pt2.z = w_X(2,i) - sqrt(w_X_variance( 2, i));

            m_points_with_var.points.push_back( pt1 );
            m_points_with_var.points.push_back( pt2 );
            m_points_with_var.colors.push_back( blue_color );
            m_points_with_var.colors.push_back( blue_color );
        }
    }
}

void EdgeManager::_3dpointspair_to_markerlines( const MatrixXd& w_X, const MatrixXd& w_Xd,
    visualization_msgs::Marker& m_lines )
{
    assert( w_X.cols() > 0 && w_Xd.cols() > 0  && w_X.cols() == w_Xd.cols() );
    assert( (w_X.rows() == 3) || (w_X.rows() == 4 && w_X(3,0) > .999 && w_X(3,0) < 1.001) );
    assert( (w_Xd.rows() == 3) || (w_Xd.rows() == 4 && w_X(3,0) > .999 && w_X(3,0) < 1.001) );


    RosMarkerUtils::init_line_marker( m_lines );
    for( int i=0 ; i<w_X.cols() ; i++ )
    {
        // Set point and color for 1st
        geometry_msgs::Point pt1;
        pt1.x = w_X(0,i);
        pt1.y = w_X(1,i);
        pt1.z = w_X(2,i);

        geometry_msgs::Point pt2;
        pt2.x = w_Xd(0,i);
        pt2.y = w_Xd(1,i);
        pt2.z = w_Xd(2,i);

        m_lines.points.push_back( pt1 );
        m_lines.points.push_back( pt2 );
    }
}
