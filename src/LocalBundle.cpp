#include "LocalBundle.h"

void LocalBundle::sayHi()
{
  cout << "LocalBundle::Hello\n";
}


LocalBundle::LocalBundle( const nap::NapMsg::ConstPtr& msg, const vector<Node*>& global_nodes )
{
  cout << "----\nLocalBundle\n";
  cout << "#PointClouds: " << msg->bundle.size() << endl;

  sensor_msgs::Image img_msg = msg->visibility_table;
  cout << "Image : "<< img_msg.height << " " << img_msg.width << " " << img_msg.encoding <<  endl;
  int N_pairs = img_msg.height;

  // cv_bridge::CvImagePtr cv_ptr;
  // cv_ptr = cv_bridge::toCvShare( img_msg );
  // cv::Mat table_image = cv_bridge::toCvShare( &(msg->visibility_table), "mono8" )->image ;
  // cout << "visibility_table_image_dim: " <<  table_image.rows << table_image.cols << endl;
  cout << "visibility_table_idx.size() " << msg->visibility_table_idx.size() << endl;
  cout << "visibility_table_stamp.size() " << msg->visibility_table_stamp.size() << endl;


  /////////////////////////////// Assert above data /////////////////////////////
  assert( img_msg.encoding == string("mono8") );
  assert( 3*N_pairs == msg->visibility_table_idx.size()    );
  assert( 2*N_pairs == msg->visibility_table_stamp.size()  );

  /////////////////////////// loop on tracked points /////////////////////////////
  for( int i=0 ; i<msg->bundle.size() ; i++ )
  {
    int seq = find_indexof_node( global_nodes, msg->bundle[i].header.stamp );
    int seq_debug = msg->bundle[i].header.seq;
    cout << "pointcloud : idx=" << seq <<  "\t#pts=" <<  msg->bundle[i].points.size()  << "\tdebug_idx=" << seq_debug <<  "\tvalid_image: " << global_nodes[seq]->valid_image()  << endl;

    // cv::Mat outImg;
    // vector<string> __tmp;
    // MatrixXd e_ptsA, e_ptsB;
    // pointcloud_2_matrix(msg->bundle[i].points, e_ptsA  );
    // plot_point_sets( global_nodes[seq]->getImageRef(), e_ptsA, seq,
    //                   global_nodes[seq]->getImageRef(), e_ptsA, seq,
    //                   VectorXd::Ones( e_ptsA.cols(), 1 ),
    //                  __tmp, outImg );
    // cv::imshow( "huhu", outImg);
    // write_image( to_string(seq_debug)+".png", outImg );
    // cv::waitKey(200);
  }


  ///////////////////////////////////////// loop on pairs ///////////////////////////////
  for( int i=0 ; i<N_pairs ; i++ )
  {
    ros::Time a_stamp = msg->visibility_table_stamp[2*i];
    ros::Time b_stamp = msg->visibility_table_stamp[2*i+1];
    int a_stamp_idx = find_indexof_node( global_nodes, a_stamp );
    int b_stamp_idx = find_indexof_node( global_nodes, b_stamp );

    int a = msg->visibility_table_idx[3*i];
    int b = msg->visibility_table_idx[3*i+1];
    int ttype = msg->visibility_table_idx[3*i+2];

    cout << a_stamp_idx << "<--(type="<< ttype << ")-->" << b_stamp_idx  << "\t\t";
    cout << "[" << a << "   " << b << "]" << "\t\t";

    int _i = find_indexof_node( msg->bundle, a_stamp );
    int _j = find_indexof_node( msg->bundle, b_stamp );
    cout << "{" << _i << "   " << _j << "}" << endl;

    cv::Mat outImg;
    vector<string> __tmp;
    MatrixXd e_ptsA, e_ptsB;
    assert( _i >= 0 );
    assert( _j >= 0 );
    pointcloud_2_matrix(msg->bundle[_i].points, e_ptsA  );
    pointcloud_2_matrix(msg->bundle[_j].points, e_ptsB  );
    plot_point_sets( global_nodes[a_stamp_idx]->getImageRef(), e_ptsA,  a_stamp_idx,
                      global_nodes[b_stamp_idx]->getImageRef(), e_ptsB, b_stamp_idx,
                      VectorXd::Ones( e_ptsA.cols(), 1 ),
                     __tmp, outImg );
    // cv::imshow( "huhuX", outImg);
    write_image( to_string(a_stamp_idx)+"__"+to_string(b_stamp_idx)+".png", outImg );


  }
}

void LocalBundle::write_image( string fname, const cv::Mat& img)
{
    cout << "Writing file "<< fname << endl;
    string base = string("/home/mpkuse/Desktop/bundle_adj/dump/CXX_");
    cv::imwrite( (base+fname).c_str(), img );
}

void LocalBundle::plot_point_sets( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                      const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                      const VectorXd& mask,
                      const vector<string>& msg,
                    cv::Mat& dst )
{
  // ptsA : ptsB : 2xN or 3xN

  assert( imA.rows == imB.rows );
  assert( imA.cols == imB.cols );
  assert( ptsA.cols() == ptsB.cols() );
  assert( mask.size() == ptsA.cols() );

  cv::Mat outImg;
  cv::hconcat(imA, imB, outImg);

  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    if( mask(kl) == 0 )
      continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    cv::circle( outImg, A, 2, cv::Scalar(0,255,0), -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2, cv::Scalar(0,255,0), -1 );

    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), cv::Scalar(255,0,0) );
  }



  cv::Mat status = cv::Mat(100, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, (to_string(count)+" of "+to_string(ptsA.cols())).c_str(), cv::Point(10,70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );

  cv::vconcat( outImg, status, dst );

}


void LocalBundle::pointcloud_2_matrix( const vector<geometry_msgs::Point32>& ptCld, MatrixXd& G )
{
  int N = ptCld.size() ;
  G = MatrixXd( 3, N );
  for( int i=0 ; i<N ; i++ )
  {
    G(0,i) = ptCld[i].x;
    G(1,i) = ptCld[i].y;
    G(2,i) = 1.0;
    assert( ptCld[i].z == -7 );
  }
}


// Loop over each node and return the index of the node which is clossest to the specified stamp
int LocalBundle::find_indexof_node( const vector<Node*>& global_nodes, ros::Time stamp )
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


int LocalBundle::find_indexof_node( const vector<sensor_msgs::PointCloud>& global_nodes, ros::Time stamp )
{
  ros::Duration diff;
  for( int i=0 ; i<global_nodes.size() ; i++ )
  {
    diff = global_nodes[i].header.stamp - stamp;

    // cout << i << " "<< diff.sec << " " << diff.nsec << endl;

    if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
      return i;
    }
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
  return -1;
}
