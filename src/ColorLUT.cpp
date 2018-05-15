#include "ColorLUT.h"


ColorLUT::ColorLUT()
{
  //make colors
  // cout << "Make Colors";
  // _color = []
  _color.push_back( cv::Scalar(0, 0, 0) );
  _color.push_back( cv::Scalar(0, 255, 0) );
  _color.push_back( cv::Scalar(0, 0, 255) );
  _color.push_back( cv::Scalar(255, 0, 0) );
  _color.push_back( cv::Scalar(1, 255, 254) );
  _color.push_back( cv::Scalar(255, 166, 254) );
  _color.push_back( cv::Scalar(255, 219, 102) );
  _color.push_back( cv::Scalar(0, 100, 1) );
  _color.push_back( cv::Scalar(1, 0, 103) );
  _color.push_back( cv::Scalar(149, 0, 58) );
  _color.push_back( cv::Scalar(0, 125, 181) );
  _color.push_back( cv::Scalar(255, 0, 246) );
  _color.push_back( cv::Scalar(255, 238, 232) );
  _color.push_back( cv::Scalar(119, 77, 0) );
  _color.push_back( cv::Scalar(144, 251, 146) );
  _color.push_back( cv::Scalar(0, 118, 255) );
  _color.push_back( cv::Scalar(213, 255, 0) );
  _color.push_back( cv::Scalar(255, 147, 126) );
  _color.push_back( cv::Scalar(106, 130, 108) );
  _color.push_back( cv::Scalar(255, 2, 157) );
  _color.push_back( cv::Scalar(254, 137, 0) );
  _color.push_back( cv::Scalar(122, 71, 130) );
  _color.push_back( cv::Scalar(126, 45, 210) );
  _color.push_back( cv::Scalar(133, 169, 0) );
  _color.push_back( cv::Scalar(255, 0, 86) );
  _color.push_back( cv::Scalar(164, 36, 0) );
  _color.push_back( cv::Scalar(0, 174, 126) );
  _color.push_back( cv::Scalar(104, 61, 59) );
  _color.push_back( cv::Scalar(189, 198, 255) );
  _color.push_back( cv::Scalar(38, 52, 0) );
  _color.push_back( cv::Scalar(189, 211, 147) );
  _color.push_back( cv::Scalar(0, 185, 23) );
  _color.push_back( cv::Scalar(158, 0, 142) );
  _color.push_back( cv::Scalar(0, 21, 68) );
  _color.push_back( cv::Scalar(194, 140, 159) );
  _color.push_back( cv::Scalar(255, 116, 163) );
  _color.push_back( cv::Scalar(1, 208, 255) );
  _color.push_back( cv::Scalar(0, 71, 84) );
  _color.push_back( cv::Scalar(229, 111, 254) );
  _color.push_back( cv::Scalar(120, 130, 49) );
  _color.push_back( cv::Scalar(14, 76, 161) );
  _color.push_back( cv::Scalar(145, 208, 203) );
  _color.push_back( cv::Scalar(190, 153, 112) );
  _color.push_back( cv::Scalar(150, 138, 232) );
  _color.push_back( cv::Scalar(187, 136, 0) );
  _color.push_back( cv::Scalar(67, 0, 44) );
  _color.push_back( cv::Scalar(222, 255, 116) );
  _color.push_back( cv::Scalar(0, 255, 198) );
  _color.push_back( cv::Scalar(255, 229, 2) );
  _color.push_back( cv::Scalar(98, 14, 0) );
  _color.push_back( cv::Scalar(0, 143, 156) );
  _color.push_back( cv::Scalar(152, 255, 82) );
  _color.push_back( cv::Scalar(117, 68, 177) );
  _color.push_back( cv::Scalar(181, 0, 255) );
  _color.push_back( cv::Scalar(0, 255, 120) );
  _color.push_back( cv::Scalar(255, 110, 65) );
  _color.push_back( cv::Scalar(0, 95, 57) );
  _color.push_back( cv::Scalar(107, 104, 130) );
  _color.push_back( cv::Scalar(95, 173, 78) );
  _color.push_back( cv::Scalar(167, 87, 64) );
  _color.push_back( cv::Scalar(165, 255, 210) );
  _color.push_back( cv::Scalar(255, 177, 103) );
  _color.push_back( cv::Scalar(0, 155, 255) );
  _color.push_back( cv::Scalar(232, 94, 190) );
}

cv::Scalar ColorLUT::get_color(int i )
{
  assert( i >= 0 && i<_color.size() );
  return _color[i];
  return cv::Scalar(255,23, 100 );
}
