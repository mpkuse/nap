#pragma once
/** ColorLUT.h

    Lookup colors. 64 or so uniq colors.

*/

#include <iostream>
#include <string>
#include <vector>


//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
class ColorLUT {
public:
  ColorLUT();

  cv::Scalar get_color(int i );

private:
  vector<cv::Scalar> _color; 
};
