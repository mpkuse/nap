#include "SceneRenderer.h"

SceneRenderer::SceneRenderer()
{

}


void SceneRenderer::setCamera( PinholeCamera* cam )
{ this->cam = cam; }

void SceneRenderer::addMesh( MeshObject* mesh )
{
  this->objGraph.push_back( mesh );
}


void SceneRenderer::render( const cv::Mat& canvas, const Matrix4d& w_T_c )
{
  // Get Vertices
  MeshObject *m = objGraph[0];

  assert( m->isMeshLoaded() );
  MatrixXd ob_V = m->getVertices(); //ob_V


  // Prepare c_T_ob and mesh points in camere-frame-of-ref
  assert( m->isWorldPoseAvailable() );
  Matrix4d w_T_ob = m->getObjectWorldPose();

  Matrix4d c_T_ob;
  c_T_ob = w_T_c.inverse() * w_T_ob; // c_T_ob


  MatrixXd c_V = c_T_ob * ob_V; // vertices in camera-frame
  cout << "ob_V:\n" << ob_V << endl;
  // cout << "c_T_ob:\n" << c_T_ob << endl;
  cout << "c_V:\n" << c_V << endl;

  // Perspective Projection
  MatrixXd c_v;
  cam->perspectiveProject3DPoints( c_V, c_v );
  cout << "c_v(Projected):\n" << c_v << endl;


  // plot these points on canvas
  cv::Mat buf = canvas.clone();

  // Plot points
  for( int i=0 ; i<c_v.cols() ; i++ ) //loop on all the points
  {
    cout << i << ": "<< c_v(0,i) << ", " << c_v(1,i) << endl;
    cv::circle( buf, cv::Point(c_v(0,i), c_v(1,i)), 4, cv::Scalar(0,0,255), 1, 8 );
  }

  // Plot triangles
  MatrixXi faces = m->getFaces();
  cout << "nFaces : "<< faces.rows() << "x" << faces.cols() << endl;
  for( int f=0 ; f<faces.cols() ; f++ )
  {
    // Vector3i face = faces[f];
    int f1 = faces(0,f)-1; //face-indexing starts with 1 in .obj
    int f2 = faces(1,f)-1;
    int f3 = faces(2,f)-1;
    cout << f << ":f  " << f1 << ", " << f2 << ", " << f3 << endl;
    cv::Point p1 = cv::Point( c_v(0,f1), c_v(1,f1) );
    cv::Point p2 = cv::Point( c_v(0,f2), c_v(1,f2) );
    cv::Point p3 = cv::Point( c_v(0,f3), c_v(1,f3) );

    //might have issue wth the pose. 

    cv::line( buf, p1, p2, cv::Scalar(0,255,255), 1, 8 );
    cv::line( buf, p1, p3, cv::Scalar(0,255,255), 1, 8 );
    cv::line( buf, p3, p2, cv::Scalar(0,255,255), 1, 8 );
  }

  cv::imshow( "buf", buf);

}
