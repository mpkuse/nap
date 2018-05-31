#include "DataManager.h"

// This contains functions related to rviz visualization (using marker) of the pose graph

void DataManager::publish_once()
{
  // publish_pose_graph_nodes();
  publish_pose_graph_nodes_original_poses();
  publish_pose_graph_edges( this->odometryEdges );
  publish_pose_graph_edges( this->loopClosureEdges );
  // publish_node_pointcloud();
}


void DataManager::publish_pose_graph_nodes()
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "spheres";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.05;
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;
  marker.color.a = .6; // Don't forget to set the alpha!

  int nSze = nNodes.size();
  // for( int i=0; i<nNodes.size() ; i+=1 )
  for( int i=max(0,nSze-10); i<nNodes.size() ; i++ ) //optimization trick: only publish last 10. assuming others are already on rviz
  {
    marker.color.r = 0.0;marker.color.g = 0.0;marker.color.b = 0.0; //default color of node

    Node * n = nNodes[i];



    // Publish Sphere
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.id = i;
    marker.ns = "spheres";
    Matrix4d pose_mat_curr; //w_T_c
    n->getCurrTransform( pose_mat_curr );
    marker.pose.position.x = pose_mat_curr(0,3);
    marker.pose.position.y = pose_mat_curr(1,3);
    marker.pose.position.z = pose_mat_curr(2,3);
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 0.0;
    marker.scale.x = .05;marker.scale.y = .05;marker.scale.z = .05;
    pub_pgraph.publish( marker );

    // Publish Text
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.id = i;
    marker.ns = "text_label";
    marker.scale.z = .03;

    // pink color text if node doesnt contain images
    if( n->getNapClusterMap().data == NULL )
    { marker.color.r = 1.0;  marker.color.g = .4;  marker.color.b = .4; }
    else
    { marker.color.r = 1.0;  marker.color.g = 1.0;  marker.color.b = 1.0; } //text in white color
    // marker.text = std::to_string(i)+std::string(":")+std::to_string(n->ptCld.cols())+std::string(":")+((n->getImageRef().data)?"I":"~I");

    std::stringstream buffer;
    buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp << ":" << n->getn3dpts() << ":" << n->getn2dfeat();
    // buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp << ":" << n->time_image- nNodes[0]->time_stamp  ;
    marker.text = buffer.str();
    // marker.text = std::to_string(i)+std::string(":")+std::to_string( n->time_stamp );
    pub_pgraph.publish( marker );



    //
    // Write Node Image along with feat2d to file //
    char imfile_name[200];
    // sprintf( imfile_name, "/home/mpkuse/Desktop/a/drag2/kf_%d.png", i );
    sprintf( imfile_name, "%s/kf_%d.png", _DEBUG_SAVE_BASE_PATH, i );

    if( !if_file_exist(imfile_name) )
    {
      if( n->valid_image() && n->valid_3dpts() && n->valid_2dfeats() ) {
        cout << "3d:"<< n->valid_3dpts()   << "(" << n->getn3dpts() << "); ";
        cout << "2d:"<< n->valid_2dfeats() << "(" << n->getn2dfeat() << ")\n";

        // Write original image
        cout << "Writing file "<< imfile_name << endl;
        cv::imwrite( imfile_name, n->getImageRef() );

        // Write Node data to file.
        char debug_fname[100];
        // sprintf( debug_fname, "/home/mpkuse/Desktop/a/drag2/kf_%d.yaml", i );
        sprintf( debug_fname, "%s/kf_%d.yaml", _DEBUG_SAVE_BASE_PATH, i );
        n->write_debug_xml( debug_fname );


        // Get 3dpoints - Probably don't need to bother with 3dpts.
        // MatrixXd c_M; //4xN
        // n->getPointCloudHomogeneous(c_M);
        //
        // // Project 3d points on camera
        // MatrixXd reprojM;
        // camera.perspectiveProject3DPoints( c_M, reprojM );
        //
        // MatrixXf reproj_float;
        // reproj_float = reprojM.cast<float>();
        // cv::Mat reprojM_mat;
        // cv::eigen2cv( reproj_float, reprojM_mat );
        //
        //
        // // plot reproj-3d points on image
        // cv::Mat dst;
        // plot_point_sets( n->getImageRef(), reprojM_mat, dst, cv::Scalar(0,0,244));


        // Get 2d features
        MatrixXd c_feat2d_normed, c_feat2d;
        n->getFeatures2dHomogeneous( c_feat2d_normed );
        camera.normalizedImCords_2_imageCords( c_feat2d_normed, c_feat2d );



        // plot 2dfeats on image
        cv::Mat dst;
        plot_point_sets( n->getImageRef(), c_feat2d, dst, cv::Scalar(0,0,244), string("feat2d in red"));


        // Write annotated image
        // Write Node data to file.
        // sprintf( debug_fname, "/home/mpkuse/Desktop/a/drag2/kf_%d_anno.png", i );
        sprintf( debug_fname, "%s/kf_%d_anno.png", _DEBUG_SAVE_BASE_PATH, i );
        cv::imwrite( debug_fname, dst );



      }
    }
    // END


  }
}


void DataManager::publish_pose_graph_nodes_original_poses()
{
  bool enable_camera_visual_at_each_pose = false;
  bool enable_spheres_at_each_pose = true;
  bool enable_text_at_each_pose = true;
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "spheres";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.05;
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;
  marker.color.a = .6; // Don't forget to set the alpha!

  int nSze = nNodes.size();
  // for( int i=0; i<nNodes.size() ; i+=1 )
  for( int i=max(0,nSze-10); i<nNodes.size() ; i++ ) //optimization trick: only publish last 10. assuming others are already on rviz
  {
    marker.color.r = 0.0;marker.color.g = 0.0;marker.color.b = 0.0; //default color of node

    Node * n = nNodes[i];

    // Publish Sphere
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.id = i;
    marker.ns = "spheres";
    Matrix4d original_pose;
    n->getOriginalTransform(original_pose);
    Quaterniond original_pose_quaternion = Quaterniond( original_pose.topLeftCorner<3,3>() );
    marker.pose.position.x = original_pose(0,3); //-20.;
    marker.pose.position.y = original_pose(1,3);
    marker.pose.position.z = original_pose(2,3);
    marker.pose.orientation.x = original_pose_quaternion.x(); //0.;
    marker.pose.orientation.y = original_pose_quaternion.y(); //0.;
    marker.pose.orientation.z = original_pose_quaternion.z(); //0.;
    marker.pose.orientation.w = original_pose_quaternion.w(); //1.;
    marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0;
    marker.scale.x = .05;marker.scale.y = .05;marker.scale.z = .05;
    if( enable_spheres_at_each_pose) pub_pgraph_org.publish( marker );

    // Publish Text
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.id = i;
    marker.ns = "text_label";
    marker.scale.z = .03;

    if( !n->valid_image() )
    { marker.color.r = 1.0;  marker.color.g = .4;  marker.color.b = .4; } //pink
    else
    { marker.color.r = 1.0;  marker.color.g = 1.0;  marker.color.b = 1.0; } //text in white color
    // marker.text = std::to_string(i)+std::string(":")+std::to_string(n->ptCld.cols())+std::string(":")+((n->getImageRef().data)?"I":"~I");

    std::stringstream buffer;
    // buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp; //long version
    buffer << i; //short version
    // buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp << ":" << n->time_image- nNodes[0]->time_stamp  ;
    marker.text = buffer.str();
    // marker.text = std::to_string(i)+std::string(":")+std::to_string( n->time_stamp );
    if( enable_text_at_each_pose) pub_pgraph_org.publish( marker );


    if( enable_camera_visual_at_each_pose )
    {
        // Visual for Camera
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.id = i;
        marker.ns = "camerapose_visual";

        marker.scale.x = 0.005; //width of line-segments

        float __vcam_width = 0.07;
        float __vcam_height = 0.04;
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
        pub_pgraph_org.publish( marker );
    }




  }
}

void DataManager::publish_pose_graph_edges( const std::vector<Edge*>& x_edges )
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.id = 0;
  marker.type = visualization_msgs::Marker::ARROW;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.018; //0.02
  marker.scale.y = 0.05;
  marker.scale.z = 0.06;
  marker.color.a = .6; // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  // cout << "There are "<< odometryEdges.size() << " edges\n";

  int nSze = x_edges.size();
  // for( int i=0 ; i<x_edges.size() ; i++ )
  for( int i=max(0,nSze-10) ; i<x_edges.size() ; i++ ) //optimization trick,
  {
    Edge * e = x_edges[i];
    marker.id = i;
    geometry_msgs::Point start;
    Matrix4d pose_a; //w_T_a
    e->a->getCurrTransform( pose_a );
    start.x = pose_a(0,3);
    start.y = pose_a(1,3);
    start.z = pose_a(2,3);

    geometry_msgs::Point end;
    Matrix4d pose_b; //w_T_b
    e->b->getCurrTransform( pose_b );
    end.x = pose_b(0,3);
    end.y = pose_b(1,3);
    end.z = pose_b(2,3);
    marker.points.clear();
    marker.points.push_back(start);
    marker.points.push_back(end);

    if( e->type == EDGE_TYPE_ODOMETRY ) //green - odometry edge
    {    marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;    marker.ns = "odom_edges";}
    else if( e->type == EDGE_TYPE_LOOP_CLOSURE )
    {
      switch(e->sub_type)
      {
        case EDGE_TYPE_LOOP_SUBTYPE_BASIC: // basic loop-edge in cyan
          marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "loop_edges_basic";
          break;

        case EDGE_TYPE_LOOP_SUBTYPE_3WAY: // 3way matched loop-edge in red
          marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.ns = "loop_edges_3way";
          marker.scale.y = 0.1; //fat arrow for 3way
          break;

        case EDGE_TYPE_LOOP_SUBTYPE_GUIDED: // Dark blue
          marker.color.r = .2; marker.color.g = 0.0; marker.color.b = 0.8; marker.ns = "loop_edges_2way_guided";
          break;

        case EDGE_TYPE_LOOP_SUBTYPE_BUNDLE: // Orange
          marker.color.r = 1.0; marker.color.g = 0.64; marker.color.b = 0.0; marker.ns = "loop_edges_bundle";
          break;

        default:
          marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "loop_edges_default";
          break;

      }

      /*
      if( e->sub_type == EDGE_TYPE_LOOP_SUBTYPE_BASIC ) // basic loop-edge in red
      { marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.ns = "loop_edges"; }
      else {
        if( e->sub_type == EDGE_TYPE_LOOP_SUBTYPE_3WAY ) // 3way matched loop-edge in pink
        { marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.ns = "loop_edges"; }
        else //other edge subtype in white
        { marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "loop_edges"; }
      }
      */


    }
    else
    {    marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "x_edges";}

    pub_pgraph.publish( marker );
  }
}



void DataManager::publish_node_pointcloud()
{
    int nvalid=0;
    for( int i=nNodes.size()-10 ; i < nNodes.size() ; i++ )
    {
        if( ! nNodes[i]->valid_3dpts() )
            continue;
        nvalid++;

        MatrixXd w_Q;
        w_Q = nNodes[i]->getPointCloud();

        VectorXi globalid_Q;
        globalid_Q = nNodes[i]->getPointCloudGlobalIds();

        // marking for each point. Only for plooting
        vector<string> globalid_Q_str;
        for( int u=0; u<globalid_Q.size() ; u++ )
            globalid_Q_str.push_back( to_string(globalid_Q(u)) );

        assert( w_Q.cols() == globalid_Q.size() && w_Q.cols() > 0  );


        cout << i << " ";
        printMatrixInfo( "w_Q", w_Q );
        cout << "globalid_Q.size() : " << globalid_Q.size() << endl;

        // Make a marker msg.
        visualization_msgs::Marker marker;
        eigenpointcloud_2_ros_markermsg( w_Q,  marker, "raw_3dpoints" );
        marker.id = i;
        pub_3dpoints.publish( marker );



        // Project
        Matrix4d w_T_cam;
        nNodes[i]->getOriginalTransform( w_T_cam );
        MatrixXd cam_Q = w_T_cam.inverse() * w_Q;

        MatrixXd cam_u;
        camera.perspectiveProject3DPoints( cam_Q , cam_u );

        cv::Mat dst;
        assert( nNodes[i]->valid_image() );
        cv::Scalar pt_color(0,255,0);

        plot_point_on_image( nNodes[i]->getImageRef(), cam_u, VectorXd::Ones(cam_u.cols()),
                    pt_color, true, false, "", dst );
        write_image( to_string(i)+".png", dst  );

        plot_point_on_image( nNodes[i]->getImageRef(), cam_u, VectorXd::Ones(cam_u.cols()),
                    pt_color, globalid_Q_str, false, "", dst );
        write_image( to_string(i)+"_globalid.png", dst  );

    }
    cout << "nvalid=" << nvalid << " of " << nNodes.size() << endl;
}

void DataManager::plot_3way_match( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                      const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                      const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                      cv::Mat& dst, const string& msg)
{
  cv::Mat zre = cv::Mat(curr_im.rows, curr_im.cols, CV_8UC3, cv::Scalar(128,128,128) );

  if( msg.length() > 0 ) {
    cv::putText( zre, msg, cv::Point(5,50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,255) );
  }

  cv::Mat dst_row1, dst_row2;
  cv::hconcat(curr_im, prev_im, dst_row1);
  cv::hconcat(curr_m_im, zre, dst_row2);
  cv::vconcat(dst_row1, dst_row2, dst);



  // Draw Matches
  cv::Point2d p_curr, p_prev, p_curr_m;
  for( int kl=0 ; kl<mat_pts_curr.cols ; kl++ )
  {
    if( mat_pts_curr.channels() == 2 ){
      p_curr = cv::Point2d(mat_pts_curr.at<cv::Vec2f>(0,kl)[0], mat_pts_curr.at<cv::Vec2f>(0,kl)[1] );
      p_prev = cv::Point2d(mat_pts_prev.at<cv::Vec2f>(0,kl)[0], mat_pts_prev.at<cv::Vec2f>(0,kl)[1] );
      p_curr_m = cv::Point2d(mat_pts_curr_m.at<cv::Vec2f>(0,kl)[0], mat_pts_curr_m.at<cv::Vec2f>(0,kl)[1] );
    }
    else {
      p_curr = cv::Point2d(mat_pts_curr.at<float>(0,kl),mat_pts_curr.at<float>(1,kl) );
      p_prev = cv::Point2d(mat_pts_prev.at<float>(0,kl),mat_pts_prev.at<float>(1,kl) );
      p_curr_m = cv::Point2d(mat_pts_curr_m.at<float>(0,kl),mat_pts_curr_m.at<float>(1,kl) );
    }

    cv::circle( dst, p_curr, 4, cv::Scalar(255,0,0) );
    cv::circle( dst, p_prev+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(0,255,0) );
    cv::circle( dst, p_curr_m+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(0,0,255) );
    cv::line( dst,  p_curr, p_prev+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
    cv::line( dst,  p_curr, p_curr_m+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );

    // cv::circle( dst, cv::Point2d(pts_curr[kl]), 4, cv::Scalar(255,0,0) );
    // cv::circle( dst, cv::Point2d(pts_prev[kl])+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(0,255,0) );
    // cv::circle( dst, cv::Point2d(pts_curr_m[kl])+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(0,0,255) );
    // cv::line( dst,  cv::Point2d(pts_curr[kl]), cv::Point2d(pts_prev[kl])+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
    // cv::line( dst,  cv::Point2d(pts_curr[kl]), cv::Point2d(pts_curr_m[kl])+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );
  }
}


void DataManager::plot_3way_match_clean( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                      const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                      const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                      cv::Mat& dst, const string& msg)
{
  cv::Mat zre = cv::Mat(curr_im.rows, curr_im.cols, CV_8UC3, cv::Scalar(128,128,128) );

  if( msg.length() > 0 ) {
    cv::putText( zre, msg, cv::Point(5,50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,255) );
  }

  cv::Mat dst_row1, dst_row2;
  cv::putText( zre, "C   P", cv::Point(5,30), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0,0,255) );
  cv::putText( zre, "Cm   ", cv::Point(5,80), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0,0,255) );
  cv::putText( zre, to_string(mat_pts_curr.cols), cv::Point(5,130), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0,0,255) );
  cv::hconcat(curr_im, prev_im, dst_row1);
  cv::hconcat(curr_m_im, zre, dst_row2);
  cv::vconcat(dst_row1, dst_row2, dst);



  // Draw Matches
  cv::Point2d p_curr, p_prev, p_curr_m;
  for( int kl=0 ; kl<mat_pts_curr.cols ; kl++ )
  {
    if( mat_pts_curr.channels() == 2 ){
      p_curr = cv::Point2d(mat_pts_curr.at<cv::Vec2f>(0,kl)[0], mat_pts_curr.at<cv::Vec2f>(0,kl)[1] );
      p_prev = cv::Point2d(mat_pts_prev.at<cv::Vec2f>(0,kl)[0], mat_pts_prev.at<cv::Vec2f>(0,kl)[1] );
      p_curr_m = cv::Point2d(mat_pts_curr_m.at<cv::Vec2f>(0,kl)[0], mat_pts_curr_m.at<cv::Vec2f>(0,kl)[1] );
    }
    else {
      p_curr = cv::Point2d(mat_pts_curr.at<float>(0,kl),mat_pts_curr.at<float>(1,kl) );
      p_prev = cv::Point2d(mat_pts_prev.at<float>(0,kl),mat_pts_prev.at<float>(1,kl) );
      p_curr_m = cv::Point2d(mat_pts_curr_m.at<float>(0,kl),mat_pts_curr_m.at<float>(1,kl) );
    }

    cv::circle( dst, p_curr, 4, cv::Scalar(255,0,0) );
    cv::circle( dst, p_prev+cv::Point2d(curr_im.cols,0), 4, cv::Scalar(255,0,0) );
    cv::circle( dst, p_curr_m+cv::Point2d(0,curr_im.rows), 4, cv::Scalar(255,0,0) );

    char to_s[20];
    sprintf( to_s, "%d", kl);
    cv::putText( dst, to_s, p_curr, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255,0,255) );
    cv::putText( dst, to_s, p_prev+cv::Point2d(curr_im.cols,0), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255,0,255) );
    cv::putText( dst, to_s, p_curr_m+cv::Point2d(0,curr_im.rows), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255,0,255) );


    // cv::line( dst,  p_curr, p_prev+cv::Point2d(curr_im.cols,0), cv::Scalar(255,0,0) );
    // cv::line( dst,  p_curr, p_curr_m+cv::Point2d(0,curr_im.rows), cv::Scalar(255,30,255) );

  }
}

bool DataManager::if_file_exist( char * fname )
{
  ifstream f(fname);
  return f.good();
}

bool DataManager::if_file_exist( string fname ) { if_file_exist( fname.c_str() ); }

std::vector<std::string>
DataManager::split( std::string const& original, char separator )
{
    std::vector<std::string> results;
    std::string::const_iterator start = original.begin();
    std::string::const_iterator end = original.end();
    std::string::const_iterator next = std::find( start, end, separator );
    while ( next != end ) {
        results.push_back( std::string( start, next ) );
        start = next + 1;
        next = std::find( start, end, separator );
    }
    results.push_back( std::string( start, next ) );
    return results;
}

void DataManager::plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                        const cv::Scalar& color, const string& msg )
{
  MatrixXf pts_set_float;
  pts_set_float = pts_set.cast<float>();

  cv::Mat pts_set_mat;
  cv::eigen2cv( pts_set_float, pts_set_mat );

  plot_point_sets( im, pts_set_mat, dst, color, msg );
}

void DataManager::plot_point_sets( const cv::Mat& im, const cv::Mat& pts_set, cv::Mat& dst, const cv::Scalar& color, const string& msg )
{
  // TODO consider addressof(a) == addressof(b)
  // dst = im.clone();
  dst = cv::Mat( im.rows, im.cols, CV_8UC3 );

  if( im.channels() == 1 )
    cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
  else
    im.copyTo(dst);

  // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
  if( msg.length() > 0 ) {
    vector<std::string> msg_split;
    msg_split = this->split( msg, ';' );
    for( int q=0 ; q<msg_split.size() ; q++ )
      cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
  }


  //pts_set is 2xN
  cv::Point2d pt;
  for( int i=0 ; i<pts_set.cols ; i++ )
  {
    pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
    cv::circle( dst, pt, 4,color );

    char to_s[20];
    sprintf( to_s, "%d", i);
    cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255,255,255) - color  );

  }
}



/////////////////////////////ROS Publishing helpers ///////////////////////////////
void DataManager::eigenpointcloud_2_ros_markermsg( const MatrixXd& M, visualization_msgs::Marker& marker, const string& ns )
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
        if( M.rows() == 3 )
        {
            pt.x = M(0,i);
            pt.y = M(1,i);
            pt.z = M(2,i);
        }
        else
        {
            pt.x = M(0,i) / M(3,i);
            pt.y = M(1,i) / M(3,i);
            pt.z = M(2,i) / M(3,i);
        }

        marker.points.push_back( pt );
    }

}

void DataManager::eigenpointcloud_2_ros_markertextmsg( const MatrixXd& M,
    vector<visualization_msgs::Marker>& marker_ary, const string& ns )
{
    marker_ary.clear();
    visualization_msgs::Marker marker;
    assert( M.rows()==3 || M.rows() == 4 );
    for( int i=0 ; i<M.cols() ; i++ )
    {
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.header.seq = 0;
        marker.ns = ns; //"spheres";
        marker.id = i;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = .1;
        marker.scale.y = .1;
        marker.scale.z = .1;

        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = .9; // Don't forget to set the alpha!

        marker.text = to_string( i );
        marker.pose.position.x = M(0,i);
        marker.pose.position.y = M(1,i);
        marker.pose.position.z = M(2,i);
        marker.pose.orientation.x = 0.;
        marker.pose.orientation.y = 0.;
        marker.pose.orientation.z = 0.;
        marker.pose.orientation.w = 1.;

        marker_ary.push_back( visualization_msgs::Marker(marker) );
    }

}


void DataManager::printMatrixInfo( const string& msg, const MatrixXd& M )
{
  cout << msg << ":" << "rows=" << M.rows() << ", cols=" << M.cols() << endl;
}
