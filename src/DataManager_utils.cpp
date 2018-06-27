#include "DataManager.h"

// #define write_image_debug_dm( msg ) ;
#define write_image_debug_dm( msg ) msg;
void DataManager::write_image( string fname, const cv::Mat& img)
{
    // string base = string("/home/mpkuse/Desktop/bundle_adj/dump/datamgr_");
    string base = BASE__DUMP+"/dump/datamgr_";
    write_image_debug_dm( cout << "Writing file: "<< base << fname << endl );
    cv::imwrite( (base+fname).c_str(), img );
}


template <typename Derived>
void DataManager::write_EigenMatrix(const string& filename, const MatrixBase<Derived>& a)
{
  // string base = string("/home/mpkuse/Desktop/bundle_adj/dump/datamgr_mateigen_");
  string base = BASE__DUMP+"/dump/datamgr_";
  std::ofstream file(base+filename);
  if( file.is_open() )
  {
    // file << a.format(CSVFormat) << endl;
    file << a << endl;
    write_image_debug_dm(cout << "\033[1;32m" <<"Written to file: "<< filename  << "\033[0m\n" );
  }
  else
  {
    cout << "\033[1;31m" << "FAIL TO OPEN FILE for writing: "<< filename << "\033[0m\n";

  }
}


void DataManager::write_Matrix2d( const string& filename, const double * D, int nRows, int nCols )
{
  // string base = string("/home/mpkuse/Desktop/bundle_adj/dump/datamgr_mat2d_");
  string base = BASE__DUMP+"/dump/datamgr_";
  std::ofstream file(base+filename);
  if( file.is_open() )
  {
    int c = 0 ;
    for( int i=0; i<nRows ; i++ )
    {
      file << D[c];
      c++;
      for( int j=1 ; j<nCols ; j++ )
      {
        file << ", " << D[c] ;
        c++;
      }
      file << "\n";
    }
    write_image_debug_dm( cout << "\033[1;32m" <<"Written to file: "<< filename  << "\033[0m\n" );
  }
  else
  {
    cout << "\033[1;31m" << "FAIL TO OPEN FILE for writing: "<< filename << "\033[0m\n";

  }

}

void DataManager::write_Matrix1d( const string& filename, const double * D, int n  )
{
  // string base = string("/home/mpkuse/Desktop/bundle_adj/dump/datamgr_mat1d_");
  string base = BASE__DUMP+"/dump/datamgr_";
  std::ofstream file(base+filename);
  if( file.is_open() )
  {
    file << D[0];
    for( int i=1 ; i<n ; i++ )
      file << ", " << D[i] ;
    file << "\n";
    write_image_debug_dm(cout << "\033[1;32m" <<"Written to file: "<< filename  << "\033[0m\n");
  }
  else
  {
    cout << "\033[1;31m" << "FAIL TO OPEN FILE for writing: "<< filename << "\033[0m\n";

  }

}



void DataManager::plot_point_on_image( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
            const cv::Scalar& color, bool annotate, bool enable_status_image,
            const string& msg ,
            cv::Mat& dst )
{
  assert( pts.rows() == 2 || pts.rows() == 3 );
  assert( mask.size() == pts.cols() );

  cv::Mat outImg = im.clone();

  int count = 0 ;
  for( int kl=0 ; kl<pts.cols() ; kl++ )
  {
    if( mask(kl) == 0 )
      continue;

      count++;

      cv::Point2d A( pts(0,kl), pts(1,kl) );
      cv::circle( outImg, A, 2, color, -1 );

      if( annotate )
        cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
  }

  if( !enable_status_image ) {
      dst = outImg;
      return;
  }


  // Make status image
  cv::Mat status = cv::Mat(100, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  string s = "Plotted "+to_string(count)+" of "+to_string(mask.size());



  cv::putText( status, s.c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );


  if( msg.length() > 0 )
    cv::putText( status, msg.c_str(), cv::Point(10,70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2 );

  cv::vconcat( outImg, status, dst );


}



void DataManager::plot_point_on_image( const cv::Mat& im, const MatrixXd& pts, const VectorXd& mask,
            const cv::Scalar& color, vector<string> annotate, bool enable_status_image,
            const string& msg ,
            cv::Mat& dst )
{
  assert( pts.rows() == 2 || pts.rows() == 3 );
  assert( mask.size() == pts.cols() );
  assert( pts.cols() == annotate.size() );

  cv::Mat outImg = im.clone();

  int count = 0 ;
  for( int kl=0 ; kl<pts.cols() ; kl++ )
  {
    if( mask(kl) == 0 )
      continue;

      count++;

      cv::Point2d A( pts(0,kl), pts(1,kl) );
      cv::circle( outImg, A, 2, color, -1 );


      cv::putText( outImg, annotate[kl], A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1 );
  }

  if( !enable_status_image ) {
      dst = outImg;
      return;
  }


  // Make status image
  cv::Mat status = cv::Mat(100, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  string s = "Plotted "+to_string(count)+" of "+to_string(mask.size());



  cv::putText( status, s.c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );


  if( msg.length() > 0 )
    cv::putText( status, msg.c_str(), cv::Point(10,70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2 );

  cv::vconcat( outImg, status, dst );


}
