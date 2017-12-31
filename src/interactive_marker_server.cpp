#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/menu_handler.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <string>
using namespace std;
using namespace visualization_msgs;

boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server;
interactive_markers::MenuHandler menu_handler;
ros::Publisher pub_obj_pose;

void processFeedback(
    const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback )
{
  // ROS_INFO_STREAM( feedback->marker_name << " is now at "
      // << feedback->pose.position.x << ", " << feedback->pose.position.y
      // << ", " << feedback->pose.position.z );

      std::ostringstream s;
      s << "Feedback from marker '" << feedback->marker_name << "' "
          << " / control '" << feedback->control_name << "'";



      std::ostringstream mouse_point_ss;
      if( feedback->mouse_point_valid )
      {
        mouse_point_ss << " at " << feedback->mouse_point.x
                       << ", " << feedback->mouse_point.y
                       << ", " << feedback->mouse_point.z
                       << " in frame " << feedback->header.frame_id;
      }




    switch ( feedback->event_type )
     {
       case visualization_msgs::InteractiveMarkerFeedback::BUTTON_CLICK:
         ROS_INFO_STREAM( s.str() << ": button click" << mouse_point_ss.str() << "." );
         break;

       case visualization_msgs::InteractiveMarkerFeedback::MENU_SELECT:
         ROS_INFO_STREAM( s.str() << ": menu item " << feedback->menu_entry_id << " clicked" << mouse_point_ss.str() << "." );
         break;

       case visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE:
         ROS_INFO_STREAM( s.str() << ": pose changed"
             << "\nposition = "
             << feedback->pose.position.x
             << ", " << feedback->pose.position.y
             << ", " << feedback->pose.position.z
             << "\norientation = "
             << feedback->pose.orientation.w
             << ", " << feedback->pose.orientation.x
             << ", " << feedback->pose.orientation.y
             << ", " << feedback->pose.orientation.z
             << "\nframe: " << feedback->header.frame_id
             << " time: " << feedback->header.stamp.sec << "sec, "
             << feedback->header.stamp.nsec << " nsec" );
         break;

       case visualization_msgs::InteractiveMarkerFeedback::MOUSE_DOWN:
         ROS_INFO_STREAM( s.str() << ": mouse down" << mouse_point_ss.str() << "." );
         break;

       case visualization_msgs::InteractiveMarkerFeedback::MOUSE_UP:
         ROS_INFO_STREAM( s.str() << ": mouse up" << mouse_point_ss.str() << "." );

         // publish updated pose
         geometry_msgs::PoseStamped msg_to_send;
         msg_to_send.header = feedback->header;
         msg_to_send.header.frame_id = feedback->marker_name;
         msg_to_send.pose = feedback->pose;
         pub_obj_pose.publish( msg_to_send );


         break;
     }
}

Marker makeBox( InteractiveMarker &msg )
{
  Marker marker;

  marker.type = Marker::CUBE;
  // msg.scale *= 10.;
  marker.scale.x = msg.scale * 0.45;
  marker.scale.y = msg.scale * 0.45;
  marker.scale.z = msg.scale * 0.45;
  marker.color.r = 0.5;
  marker.color.g = 0.5;
  marker.color.b = 0.5;
  marker.color.a = .5;


  // marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  // marker.mesh_resource = "package://nap/resources/1.obj";

  return marker;
}

Marker makeMeshBox( InteractiveMarker &msg, string mesh_fname )
{
  Marker marker;

  // marker.type = Marker::CUBE;
  msg.scale *= 10.;
  marker.scale.x = 1.0; //msg.scale * 0.45;
  marker.scale.y = 1.0; //msg.scale * 0.45;
  marker.scale.z = 1.0; //msg.scale * 0.45;
  marker.color.r = 0.5;
  marker.color.g = 0.5;
  marker.color.b = 0.5;
  marker.color.a = .5;


  marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  // marker.mesh_resource = "package://nap/resources/1.obj";
  ROS_INFO_STREAM( "Load Mesh : " << string("package://nap/resources/") + mesh_fname );
  marker.mesh_resource = string("package://nap/resources/") + mesh_fname;

  return marker;
}

InteractiveMarkerControl& makeBoxControl( InteractiveMarker &msg )
{
  InteractiveMarkerControl control;
  control.always_visible = true;
  // control.markers.push_back( makeBox(msg) );
  control.markers.push_back( makeMeshBox(msg, msg.name) );
  msg.controls.push_back( control );

  return msg.controls.back();
}

void make6DofMarker( const tf::Vector3& position, string name )
{
  bool fixed = true;
  unsigned int interaction_mode = visualization_msgs::InteractiveMarkerControl::NONE;
  bool show_6dof = true;

  InteractiveMarker int_marker;
  int_marker.header.frame_id = "world";
  tf::pointTFToMsg(position, int_marker.pose.position);
  int_marker.scale = 1;

  int_marker.name = name.c_str(); //"simple_6dof";
  int_marker.description = name.c_str();//"Simple 6-DOF Control";

  // insert a box
  makeBoxControl(int_marker);
  int_marker.controls[0].interaction_mode = interaction_mode;

  InteractiveMarkerControl control;

  if ( fixed )
  {
    // int_marker.name += "_fixed";
    int_marker.description += "\n(fixed orientation)";
    control.orientation_mode = InteractiveMarkerControl::FIXED;
  }

  if (interaction_mode != visualization_msgs::InteractiveMarkerControl::NONE)
  {
      std::string mode_text;
      if( interaction_mode == visualization_msgs::InteractiveMarkerControl::MOVE_3D )         mode_text = "MOVE_3D";
      if( interaction_mode == visualization_msgs::InteractiveMarkerControl::ROTATE_3D )       mode_text = "ROTATE_3D";
      if( interaction_mode == visualization_msgs::InteractiveMarkerControl::MOVE_ROTATE_3D )  mode_text = "MOVE_ROTATE_3D";
      // int_marker.name += "_" + mode_text;
      int_marker.description = std::string("3D Control") + (show_6dof ? " + 6-DOF controls" : "") + "\n" + mode_text;
  }

  if(show_6dof)
  {
    control.orientation.w = 1;
    control.orientation.x = 1;
    control.orientation.y = 0;
    control.orientation.z = 0;
    control.name = "rotate_x";
    control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(control);
    control.name = "move_x";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);

    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 1;
    control.orientation.z = 0;
    control.name = "rotate_z";
    control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(control);
    control.name = "move_z";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);

    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 0;
    control.orientation.z = 1;
    control.name = "rotate_y";
    control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(control);
    control.name = "move_y";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);
  }

  server->insert(int_marker);
  server->setCallback(int_marker.name, &processFeedback);
  if (interaction_mode != visualization_msgs::InteractiveMarkerControl::NONE)
    menu_handler.apply( *server, int_marker.name );
}

int main(int argc, char** argv)
{
  ros::init( argc, argv, "interactive_marker_server" );
  ros::NodeHandle n;

  server.reset( new interactive_markers::InteractiveMarkerServer("interactive_marker_server", "", false) );

  tf::Vector3 position;

  // Start multiple markers here.
  position = tf::Vector3( -3,3, 0 );
  make6DofMarker( position, "chair.obj" );


  // A Publisher
  pub_obj_pose = n.advertise<geometry_msgs::PoseStamped>( "object_mesh_pose", 1000 );


  server->applyChanges();
  ros::spin();

  server.reset();
}
