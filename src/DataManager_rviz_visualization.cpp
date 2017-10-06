#include "DataManager.h"

// This contains functions related to rviz visualization (using marker) of the pose graph

void DataManager::publish_once()
{
  publish_pose_graph_nodes();
  publish_pose_graph_edges( this->odometryEdges );
  publish_pose_graph_edges( this->loopClosureEdges );
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
    marker.pose.position.x = n->e_p[0];
    marker.pose.position.y = n->e_p[1];
    marker.pose.position.z = n->e_p[2];
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
    if( n->getImageRef().data == NULL )
    { marker.color.r = 1.0;  marker.color.g = .4;  marker.color.b = .4; }
    else
    { marker.color.r = 1.0;  marker.color.g = 1.0;  marker.color.b = 1.0; } //text in white color
    // marker.text = std::to_string(i)+std::string(":")+std::to_string(n->ptCld.cols())+std::string(":")+((n->getImageRef().data)?"I":"~I");

    std::stringstream buffer;
    buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp;
    // buffer << i << ":" << n->time_stamp - nNodes[0]->time_stamp << ":" << n->time_image- nNodes[0]->time_stamp  ;
    marker.text = buffer.str();
    // marker.text = std::to_string(i)+std::string(":")+std::to_string( n->time_stamp );
    pub_pgraph.publish( marker );
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
    start.x = e->a->e_p[0];
    start.y = e->a->e_p[1];
    start.z = e->a->e_p[2];

    geometry_msgs::Point end;
    end.x = e->b->e_p[0];
    end.y = e->b->e_p[1];
    end.z = e->b->e_p[2];
    marker.points.clear();
    marker.points.push_back(start);
    marker.points.push_back(end);

    if( e->type == EDGE_TYPE_ODOMETRY ) //green - odometry edge
    {    marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;    marker.ns = "odom_edges";}
    else if( e->type == EDGE_TYPE_LOOP_CLOSURE )
    {
      if( e->sub_type == EDGE_TYPE_LOOP_SUBTYPE_BASIC ) // basic loop-edge in red
      { marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.ns = "loop_edges"; }
      else {
        if( e->sub_type == EDGE_TYPE_LOOP_SUBTYPE_3WAY ) // 3way matched loop-edge in pink
        { marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.ns = "loop_edges"; }
        else //other edge subtype in white
        { marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "loop_edges"; }
      }


    }
    else
    {    marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.ns = "x_edges";}

    pub_pgraph.publish( marker );
  }
}
