#include <iostream>
#include <vector>
#include <string>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <nap/NapMsg.h>

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
  }

  void imageCallback( const sensor_msgs::ImageConstPtr& msg )
  {
    std::cout << "Rcvd - "<< current_image_index+1 << endl;
    time_vec.push_back(msg->header.stamp);
    cv::Mat im = cv_bridge::toCvShare(msg, "bgr8")->image;
    current_image_index++;
    //TODO : posisbly need to resize image to (320x240). Might have to deep copy the im
    // cv::imshow( "win", im );
    // cv::waitKey(30);

    // Extract ORB keypoints and descriptors
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    fdetector->detectAndCompute(im, cv::Mat(), keypoints, descriptors);
    cout << "# of keypoints : "<< keypoints.size() << endl;
    cout << "descriptors shape : "<< descriptors.rows << "x" << descriptors.cols << endl;


    // Add descriptors to DB
    db.add(descriptors);

    // Query
    QueryResults ret;
    db.query(descriptors, ret, 20, current_image_index-30 );
    // cout << "Searching for Image "  << current_image_index << ". " << ret << endl;


    // Publish NapMsg
    for( int i=0 ; i < ret.size() ; i++ )
    {
      if( ret[i].Score > 0.03 )
      {
        nap::NapMsg coloc_msg;
        coloc_msg.c_timestamp = msg->header.stamp;
        coloc_msg.prev_timestamp = time_vec[ ret[i].Id ];
        coloc_msg.goodness = ret[i].Score;
        pub_colocations.publish( coloc_msg );
        cout << ret[i].Id << ":" << ret[i].Score << "(" << time_vec[ ret[i].Id ] << ")  " ;
      }
    }
    cout << endl;
  }


private:
  Vocabulary voc;
  Database db;
  cv::Ptr<cv::Feature2D> fdetector;
  int current_image_index;

  vector<ros::Time> time_vec;

  ros::NodeHandle nh;
  ros::Publisher pub_colocations;

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
