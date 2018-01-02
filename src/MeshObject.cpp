#include "MeshObject.h"


MeshObject::MeshObject(const string obj_name )
{
  cout << "Constructor MeshObject\n";
  m_loaded = false;
  m_world_pose_available = false;
  this->obj_name = string(obj_name);

  string path = ros::package::getPath("nap") + "/resources/";
  string obj_file_nme = path + obj_name;

  cout << "Resource Path : " << path << endl;
  cout << "Open File     : " << obj_file_nme << endl;

  // Loading mesh here
  load_obj( obj_file_nme );


  m_world_pose_available = false;
  m_loaded = true;

}


void MeshObject::setObjectWorldPose( Matrix4d w_T_ob )
{
  this->w_T_ob = Matrix4d( w_T_ob );
  m_world_pose_available = true;
}


bool MeshObject::getObjectWorldPose( Matrix4d& w_T_ob )
{
  w_T_ob = Matrix4d( this->w_T_ob );
  return m_world_pose_available;
}


bool MeshObject::load_obj( string fname )
{

  cout << "MeshObject::load_obj()\n";

  ifstream myfile( fname.c_str() );

  if( !myfile.is_open() )
  {
    ROS_ERROR_STREAM( "Fail to open file: "<< fname );
    return false;
  }


  // line-by-line reading
  vector<string> vec_of_items;
  int nvertex=0, nfaces = 0;


  for( string line; getline( myfile, line ) ; )
  {
    // cout << "l:" << line << endl;
    split( line, ' ', vec_of_items );
    if( vec_of_items.size() <= 0 )
      continue;

    if( vec_of_items[0] == "v" )
    {
      nvertex++;
      Vector3d vv;
      vv << stod( vec_of_items[1] ), stod( vec_of_items[2] ), stod( vec_of_items[3] ) ;
      vertices.push_back( vv );
    }


    if( vec_of_items[0] == "f" )
    {
      nfaces++;
      Vector3i vv;
      vv << stoi( vec_of_items[1] ), stod( vec_of_items[2] ), stod( vec_of_items[3] ) ;
      faces.push_back( vv );
    }


  }


  cout << "Vertex: "<< nvertex << "  Faces: " << nfaces << endl;
  o_X = MatrixXd( 4, nvertex );
  for( int i=0 ; i<nvertex ; i++ )
  {
    o_X.col(i) << vertices[i], 1.0 ;
  }
  // cout << o_X << endl;

  cout << "end MeshObject::load_obj()\n";

}


void MeshObject::split(const std::string &s, char delim, vector<string>& vec_of_items)
{
    std::stringstream ss(s);
    std::string item;
    // vector<string> vec_of_items;
    vec_of_items.clear();
    while (std::getline(ss, item, delim)) {
        // *(result++) = item;
        vec_of_items.push_back( item );
    }
}


bool MeshObject::writeMeshWorldPose()
{
  string interactive_finalpose_filename = ros::package::getPath("nap") + "/resources/" + obj_name + ".worldpose";
  if( !isWorldPoseAvailable() )
  {
    cout << "WorldPose not available not writing. " << interactive_finalpose_filename << endl;
    return false;
  }


  cout << "Write File " << interactive_finalpose_filename << endl;
  ofstream myfile( interactive_finalpose_filename.c_str() );

  if( !myfile.is_open() )
  {
    ROS_ERROR_STREAM( "MeshObject::writeMeshWorldPose(): Cannot write file :"<< interactive_finalpose_filename );
    return false;
  }

  myfile << "#,x,y,z;q.w,q.x,q.y,q.z\n";
  Vector3d t;
  t << w_T_ob(0,3), w_T_ob(1,3), w_T_ob(2,3);

  Quaterniond qt( w_T_ob.topLeftCorner<3,3>() );
  myfile << t.x() << "," << t.y() << "," << t.z() << "\n";
  myfile << qt.w() << "," << qt.x() << "," << qt.y() << "," << qt.z() ;//<< "\n";
  return true;

}
