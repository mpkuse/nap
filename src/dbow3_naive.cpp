#include <iostream>
#include <vector>
#include <string>
#include <ctime>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <nap/NapMsg.h>
#include <nap/NapNodeMsg.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <DBoW3/DBoW3.h>

using namespace DBoW3;
using namespace std;


class BOWPlaces
{
public:
  BOWPlaces( ros::NodeHandle& nh, string vocabulary_file )
  {
    cout << "Load DBOW3 Vocabulary : "<< vocabulary_file << endl;
    voc.load( vocabulary_file );
    db.setVocabulary(voc, false, 0);

    cout << "Creating ORB Detector\n";
    fdetector=cv::ORB::create();

    current_image_index = -1;

    this->nh = nh;
    pub_colocations = nh.advertise<nap::NapMsg>( "/colocation_dbow", 1000 );
    pub_raw_edge = nh.advertise<nap::NapMsg>( "/raw_graph_edge", 10000 );
    pub_raw_node = nh.advertise<nap::NapNodeMsg>( "/raw_graph_node", 10000 );

    // time publishers
    pub_time_desc_computation = nh.advertise<std_msgs::Float32>( "/time_dbow/pub_time_desc_computation", 1000 );
    pub_time_similarity = nh.advertise<std_msgs::Float32>( "/time_dbow/pub_time_similarity", 1000 );
    pub_total_time = nh.advertise<std_msgs::Float32>( "/time_dbow/pub_time_total", 1000 );
  }

  void imageCallback( const sensor_msgs::ImageConstPtr& msg )
  {
    clock_t startTime = clock();
    std::cout << "Rcvd - "<< current_image_index+1 << endl;
    time_vec.push_back(msg->header.stamp);
    cv::Mat im = cv_bridge::toCvShare(msg, "bgr8")->image;
    current_image_index++;
    //TODO : posisbly need to resize image to (320x240). Might have to deep copy the im
    // cv::imshow( "win", im );
    // cv::waitKey(30);

    // Publish NapNodeMsg with timestamp, label
    nap::NapNodeMsg node_msg;
    node_msg.node_timestamp = msg->header.stamp;
    node_msg.node_label = std::to_string(current_image_index);
    pub_raw_node.publish( node_msg );




    // Extract ORB keypoints and descriptors
    clock_t startORBComputation = clock();
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    fdetector->detectAndCompute(im, cv::Mat(), keypoints, descriptors);
    cout << "# of keypoints : "<< keypoints.size() << endl;
    cout << "descriptors shape : "<< descriptors.rows << "x" << descriptors.cols << endl;
    cout << "ORB Keypoints and Descriptors in (sec): "<< double( clock() - startORBComputation) / CLOCKS_PER_SEC << endl;

    std_msgs::Float32 msg_1;
    msg_1.data = 1000. * double( clock() - startORBComputation) / CLOCKS_PER_SEC;
    pub_time_desc_computation.publish( msg_1 );




    // Add descriptors to DB
    clock_t queryStart = clock();
    db.add(descriptors);

    // Query
    QueryResults ret;
    db.query(descriptors, ret, 20, current_image_index-30 );
    cout << "DBOW3 Query in (sec): "<< double( clock() - queryStart) / CLOCKS_PER_SEC << endl;
    // cout << "Searching for Image "  << current_image_index << ". " << ret << endl;
    std_msgs::Float32 msg_2;
    msg_2.data = 1000. * double( clock() - queryStart) / CLOCKS_PER_SEC;
    pub_time_similarity.publish( msg_2 );


    // Publish NapMsg
    for( int i=0 ; i < ret.size() ; i++ )
    {
      if( ret[i].Score > 0.05 )
      {
        nap::NapMsg coloc_msg;
        coloc_msg.c_timestamp = msg->header.stamp;
        coloc_msg.prev_timestamp = time_vec[ ret[i].Id ];
        coloc_msg.goodness = ret[i].Score;
        pub_colocations.publish( coloc_msg );
        pub_raw_edge.publish( coloc_msg );
        cout << ret[i].Id << ":" << ret[i].Score << "(" << time_vec[ ret[i].Id ] << ")  " ;
      }
    }
    cout << endl;
    std_msgs::Float32 msg_3;
    msg_3.data = 1000. * double( clock() - startTime) / CLOCKS_PER_SEC;
    pub_total_time.publish( msg_3 );
  }


private:
  Vocabulary voc;
  Database db;
  cv::Ptr<cv::Feature2D> fdetector;
  int current_image_index;

  vector<ros::Time> time_vec;

  ros::NodeHandle nh;
  ros::Publisher pub_colocations;
  ros::Publisher pub_raw_node;
  ros::Publisher pub_raw_edge;


  ros::Publisher pub_time_desc_computation;
  ros::Publisher pub_time_similarity;
  ros::Publisher pub_total_time;

};


int main( int argc, char ** argv )
{
  ros::init( argc, argv, "dbow3_naive");
  ros::NodeHandle nh;

  BOWPlaces places(nh, "/home/mpkuse/catkin_ws/src/nap/slam_data/dbow3_vocab/orbvoc.dbow3");


  image_transport::ImageTransport it(nh);
  // image_transport::Subscriber sub = it.subscribe( "/mv_29900616/image_raw", 100, &BOWPlaces::imageCallback, &places );
  image_transport::Subscriber sub = it.subscribe( "/color_image_inp", 100, &BOWPlaces::imageCallback, &places );
  std::cout << "Subscribed to /mv_29900616/image_raw\n";
  ros::spin();
}
