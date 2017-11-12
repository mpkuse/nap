""" This class provides a callback for feature. These are tracked points
    from the VINS system. They are all internally managed in an array.

    Later from main thread, one can query it with timestamp index to get
    the feature points.
"""
import rospy
import rospkg
import time
import code

import pickle

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)

from sensor_msgs.msg import PointCloud

class FeatureFactory:
    def __init__(self):
        print 'FeatureFactory'
        self.timestamp = []
        self.features = [] #in Normalized co-ordinates or original image.
        self.global_index = [] #2d array

        self.K_org = np.array( [  [530.849368,0.0,476.381888], [0.0,530.859614,300.383281], [0.0,0.0,1.0]  ]  ) #K for 240, 320 image

        self.K = self.K_org
        self.K[0,:] = self.K[0,:] / 3.0
        self.K[1,:] = self.K[1,:] / 2.5



    def tracked_features_callback(self, data ):
        # print 'Received tracked feature', data.header.stamp, len( data.points ), len( data.channels )

        # Store Timestamp
        self.timestamp.append( data.header.stamp )

        # Store normalized co-ordinates
        X_normed = np.zeros( (3, len(data.points)) ) #in homogeneous co-ordinates
        for i,pt in enumerate(data.points):
            X_normed[0,i] = pt.x
            X_normed[1,i] = pt.y
            X_normed[2,i] = 1.0
        # print X_normed.shape
        self.features.append( X_normed )

        # Store Global Index of these pts
        gindex = np.array( data.channels[0].values )
        self.global_index.append( gindex )
        # print 'gindex.shape', gindex.shape
        # print gindex

    def find_index( self, stamp ):
        # print 'find_index'
        del_duration = rospy.Duration.from_sec( 0.001 ) #1ms

        for i in range( len(self.timestamp) ):
            t = self.timestamp[i]
            # print (t - stamp)
            if (t - stamp) < del_duration and (t - stamp) > -del_duration:
                return i
        return -1


    def dump_to_file( self,  fname ):
        """ This function writes the lists to file as pickle"""

        # timestamps
        print 'Writing pickle: ',  fname+'_timestamps.pickle'
        with open( fname+'_timestamps.pickle', 'wb') as fp:
            pickle.dump( self.timestamp, fp )

        # Features in normalized co-ordinates
        print 'Writing pickle: ',  fname+'_features.pickle'
        with open( fname+'_features.pickle', 'wb') as fp:
            pickle.dump( self.features, fp )

        # Gobal feature index
        print 'Writing pickle: ',  fname+'_global_index.pickle'
        with open( fname+'_global_index.pickle', 'wb') as fp:
            pickle.dump( self.global_index, fp )

    def load_from_pickle( self, fname ):
        """  This function loads the pickle saved by dump_to_file() function"""
        # timestamps
        print 'Opening pickle: ',  fname+'_timestamps.pickle'
        with open( fname+'_timestamps.pickle', 'rb') as fp:
            self.timestamp = pickle.load( fp )

        # Features in normalized co-ordinates
        print 'Opening pickle: ',  fname+'_features.pickle'
        with open( fname+'_features.pickle', 'rb') as fp:
            self.features = pickle.load( fp )

        # Gobal feature index
        print 'Opening pickle: ',  fname+'_global_index.pickle'
        with open( fname+'_global_index.pickle', 'rb') as fp:
            self.global_index = pickle.load( fp )
