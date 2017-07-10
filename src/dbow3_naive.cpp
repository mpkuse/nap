#include <iostream>
#include <vector>
#include <string>
#include <ctime>

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <nap/NapMsg.h>
#include <nap/NapNodeMsg.h>
#include <nap/NapVisualEdgeMsg.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <DBoW3/DBoW3.h>

#include "cnpy.h"

using namespace DBoW3;
using namespace std;


class BOWPlaces
{
public:
  ~BOWPlaces()
  {
    json_bow << ",\"total_images\": "<< current_image_index+1 << "\n}";
    json_bow.close();

    cout <<  "Written loop_candidates_dbow\n";
    loop_candidates_dbow.close();

    // Write S_thumbnails as npy
    unsigned int shape[] = {0,0,0,0};
    shape[0] = S_thumbnails.size();
    shape[1] = S_thumbnails[0].rows;
    shape[2] = S_thumbnails[0].cols;
    shape[3] = (unsigned int)S_thumbnails[0].channels();
    unsigned char * data_raw = new unsigned char[shape[0]*shape[1]*shape[2]*shape[3]];
    for( int i=0 ; i<S_thumbnails.size() ; i++ )
    {
      memcpy( &data_raw[i*shape[1]*shape[2]*shape[3]], S_thumbnails[i].data, shape[1]*shape[2]*shape[3]*sizeof(char) );
    }
    std::string npy_file_name = ros::package::getPath( "nap" )+std::string("/DUMP/S_thumbnail_dbow.npy");
    cout << "Write FIle : " << npy_file_name << std::endl;
    cnpy::npy_save( npy_file_name, data_raw, (const unsigned int *)shape,4, "w" );
  }
  BOWPlaces( ros::NodeHandle& nh, string vocabulary_file )
  {
    cout << "Load DBOW3 Vocabulary : "<< vocabulary_file << endl;
    voc.load( vocabulary_file );
    db.setVocabulary(voc, false, 0);
    cout << "Vocabulary: "<< voc << endl;

    cout << "Creating ORB Detector\n";
    fdetector=cv::ORB::create();

    current_image_index = -1;

    this->nh = nh;
    pub_colocations = nh.advertise<nap::NapMsg>( "/colocation_dbow", 1000 );
    pub_raw_edge = nh.advertise<nap::NapMsg>( "/raw_graph_edge", 1000 );
    pub_raw_edge_visual = nh.advertise<nap::NapVisualEdgeMsg>( "/raw_graph_visual_edge", 1000 );
    pub_raw_node = nh.advertise<nap::NapNodeMsg>( "/raw_graph_node", 1000 );

    // time publishers
    pub_time_desc_computation = nh.advertise<std_msgs::Float32>( "/time_dbow/pub_time_desc_computation", 1000 );
    pub_time_similarity = nh.advertise<std_msgs::Float32>( "/time_dbow/pub_time_similarity", 1000 );
    pub_total_time = nh.advertise<std_msgs::Float32>( "/time_dbow/pub_time_total", 1000 );

    // json file - BOW vector at every image (sparse)
    string json_file_name = ros::package::getPath( "nap" ) + "/DUMP/dbow_per_image.json";
    json_bow.open( json_file_name);
    cout << "Opening file : " << json_file_name << endl;

    // loop_candidates_dbow
    string loop_candidates_fname = ros::package::getPath( "nap" ) + "/DUMP/loop_candidates_dbow.csv";
    loop_candidates_dbow.open( loop_candidates_fname);
    cout << "Opening file : " << loop_candidates_fname << endl;


  }


  int call_back_count = -1;
  int SKIP = 2;
  void imageCallback( const sensor_msgs::ImageConstPtr& msg )
  {
    // This to skip frames. ie. process every SKIP frame that arrives
    call_back_count++;
    if( call_back_count%SKIP != 0 )
      return;

    clock_t startTime = clock();
    std::cout << "Rcvd - "<< current_image_index+1 << endl;
    time_vec.push_back(msg->header.stamp);
    cv::Mat im = cv_bridge::toCvShare(msg, "bgr8")->image;
    current_image_index++;
    cv::Mat im_thumbnail;
    cv::resize(im, im_thumbnail, cv::Size(80,60) );
    // S_thumbnails.push_back( im_thumbnail );
    S_thumbnails.push_back( im.clone() );
    //TODO : posisbly need to resize image to (320x240). Might have to deep copy the im
    // cv::imshow( "win", im );
    // cv::waitKey(30);

    // Publish NapNodeMsg with timestamp, label
    nap::NapNodeMsg node_msg;
    node_msg.node_timestamp = msg->header.stamp;
    node_msg.node_label = std::to_string(current_image_index);
    node_msg.node_label_str = std::to_string(current_image_index);
    node_msg.color_r = 210./255.;
    node_msg.color_g = 180./255.;
    node_msg.color_b = 140./255.;
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
    BowVector bow_vec;
    db.add(descriptors, &bow_vec);
    // write current image's BOW representation to json file
    // bow_to_file( current_image_index, bow_vec, im_thumbnail  );



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
      if( ret[i].Score > 0.055 )
      {
        nap::NapMsg coloc_msg;
        coloc_msg.c_timestamp = msg->header.stamp;
        coloc_msg.prev_timestamp = time_vec[ ret[i].Id ];
        coloc_msg.goodness = ret[i].Score;
        coloc_msg.color_r = 1.0;
        coloc_msg.color_g = .0;
        coloc_msg.color_b = .0;

        // Visual Message - Same as above but has 2 images
        nap::NapVisualEdgeMsg visual_edge_msg;
        visual_edge_msg.c_timestamp = msg->header.stamp;
        visual_edge_msg.prev_timestamp = time_vec[ ret[i].Id ];
        visual_edge_msg.goodness = ret[i].Score;
        cv::Mat im__1 = S_thumbnails[current_image_index]; //cv::Mat(240,320, CV_8UC3, cv::Scalar(10,100,150) );
        cv::Mat im__2 = S_thumbnails[ ret[i].Id ]; //cv::Mat(240,320, CV_8UC3, cv::Scalar(255,0,0) );
        visual_edge_msg.curr_image = *cv_bridge::CvImage( std_msgs::Header(), "bgr8", im__1).toImageMsg();
        visual_edge_msg.prev_image = *cv_bridge::CvImage( std_msgs::Header(), "bgr8", im__2).toImageMsg();
        visual_edge_msg.curr_label = std::to_string(current_image_index);
        visual_edge_msg.prev_label = std::to_string(ret[i].Id )+string( ";;" )+std::to_string(ret[i].Score);



        if( ( coloc_msg.c_timestamp - coloc_msg.prev_timestamp) > ros::Duration(10) ) {
          pub_raw_edge_visual.publish( visual_edge_msg );
          pub_colocations.publish( coloc_msg );
          pub_raw_edge.publish( coloc_msg );
          cout << ret[i].Id << ":" << ret[i].Score << "(" << time_vec[ ret[i].Id ] << ")  " ;
          loop_candidates_dbow << current_image_index << "," << ret[i].Id << "," << ret[i].Score << ",-1,-1\n";
        }
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
  ros::Publisher pub_raw_edge_visual;


  ros::Publisher pub_time_desc_computation;
  ros::Publisher pub_time_similarity;
  ros::Publisher pub_total_time;

  // Maintains also a thumbnails of all the images for visualization/debuging. This is not required for bag of words Computation
  std::vector<cv::Mat> S_thumbnails;


  // Json file writing of BOW representation at each image
  ofstream json_bow;
  ofstream loop_candidates_dbow;
  void bow_to_file( int current_image_index, const BowVector& bow_vec, const cv::Mat& im_thumbnail )
  {
    if( current_image_index == 0 )
    {
      json_bow << "{\"vocabulary_size\": "<< voc.size() << ",\n";
    }
    else
      json_bow << ",\n";

    BowVector::const_iterator vit;
    int bow_vec_size = bow_vec.size()-1;
    // json_bow << "{\"" << current_image_index << "\": " << bow_vec.size() << "}" << endl;


    //---- FIRST
    json_bow << "\"" << current_image_index << ".id\": [";
    int i=0;
    for( vit=bow_vec.begin(); vit!= bow_vec.end() ; ++vit )
    {
      json_bow << vit->first;

      if( i<bow_vec_size )
        json_bow << " ,";
      else
        break;


      i++;

    }
    json_bow << "],\n";



    //---- SECOND
    json_bow << "\"" << current_image_index << ".wt\": [";
    i=0;
    for( vit=bow_vec.begin(); vit!= bow_vec.end() ; ++vit )
    {
      json_bow << vit->second;

      if( i<bow_vec_size )
        json_bow << " ,";
      else
        break;


      i++;

    }
    json_bow << "],\n";


    //---- Thumbnail
    json_bow << "\"" << current_image_index << ".im\": [";
    for( int ii=0 ; ii<im_thumbnail.rows; ii++ )
    {
      if( ii>0 )
        json_bow << ",";
      json_bow << "[";
      for( int jj=0 ; jj<im_thumbnail.cols ; jj++ )
      {
        uint b = im_thumbnail.at<cv::Vec3b>(ii,jj)[0];
        uint g = im_thumbnail.at<cv::Vec3b>(ii,jj)[1];
        uint r = im_thumbnail.at<cv::Vec3b>(ii,jj)[2];

        if( jj>0 )
          json_bow << ",";
        json_bow << "[" << b << "," << g << "," << r <<  "]";

      }
      json_bow << "]";
    }
    json_bow << "]\n";

  }




};


int main( int argc, char ** argv )
{
  ros::init( argc, argv, "dbow3_naive");
  ros::NodeHandle nh;

  //TODO: replace this by absolute by querying ros for package nap's path
  std::string nap_path = ros::package::getPath( "nap" );
  BOWPlaces places(nh, nap_path+"/slam_data/dbow3_vocab/orbvoc.dbow3");
  // BOWPlaces places(nh, "/home/mpkuse/catkin_ws/src/nap/slam_data/dbow3_vocab/orbvoc.dbow3");


  image_transport::ImageTransport it(nh);
  string color_image_topic_name = string("/color_image_inp");
  // string color_image_topic_name = string("/android/image");

  // image_transport::Subscriber sub = it.subscribe( "/mv_29900616/image_raw", 100, &BOWPlaces::imageCallback, &places );
  image_transport::Subscriber sub = it.subscribe(color_image_topic_name.c_str(), 100, &BOWPlaces::imageCallback, &places );
  std::cout << "Subscribed to "<< color_image_topic_name << "\n";
  ros::spin();
}
