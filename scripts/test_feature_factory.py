""" unit test for FeatureFactory class. To test the yaml file reading

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 25th Dec, 2017 (I know!)
"""

import cv2


from FeatureFactory import FeatureFactory

fname = '/home/mpkuse/catkin_ws/src/VINS_testbed/config/point_gray/point_gray_config.yaml'
fac = FeatureFactory( fname )
