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
import yaml
import re

from sensor_msgs.msg import PointCloud

from multiprocessing import Process, Queue, Manager


class FeatureFactory:
    def __init__(self, vins_config_yaml=None, manager=None):
        print 'FeatureFactory'

        if manager is None:
            print 'Using Ordinary lists'
            self.timestamp = []
            self.features = [] #in Normalized co-ordinates or original image.
            self.features_obs = [] # observed
            self.global_index = [] #list of 1d array
            self.point3d = []
        else:
            print 'Using Semaphore lists'
            self.timestamp = manager.list()
            self.features = manager.list() #in Normalized co-ordinates or original image.
            self.features_obs = manager.list() # observed image co-rodinates
            self.global_index = manager.list() #list of 1d array
            self.point3d = manager.list()

        # Old static way
        # self.K_org = np.array( [  [530.849368,0.0,476.381888], [0.0,530.859614,300.383281], [0.0,0.0,1.0]  ]  ) #K for 240, 320 image
        #
        # self.K = self.K_org
        # self.K[0,:] = self.K[0,:] / 3.0
        # self.K[1,:] = self.K[1,:] / 2.5

        if vins_config_yaml is None:
            return

        # Load from file
        # Load Camera instrinsics from VINS config file.
        # The file to read is in yaml format. Can be read with opencv.
        print 'Read VINS config YAML: ', vins_config_yaml

        yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", self.__opencv_matrix)
        outDict = self.__readYAMLFile( vins_config_yaml )

        fx = outDict['projection_parameters']['fx']
        fy = outDict['projection_parameters']['fy']
        cx = outDict['projection_parameters']['cx']
        cy = outDict['projection_parameters']['cy']
        # not reading distortion params here as it is not needed. We get the undistorted normalized points.
        # in the future if need be, it can easy be read.

        image_width = float(outDict['image_width'])
        image_height = float(outDict['image_height'])

        self.K_org = np.array( [  [fx,0.0,cx], [0.0,fy,cy], [0.0,0.0,1.0]  ]  ) #K for 240, 320 image
        self.K = self.K_org.copy()
        self.K[0,:] = self.K[0,:] / (image_width/320.)
        self.K[1,:] = self.K[1,:] / (image_height/240.)
        print 'self.K\n', self.K
        print 'self.K_org\n', self.K_org
        # code.interact( local=locals() )


    def __opencv_matrix(self,loader, node):
        """ https://stackoverflow.com/questions/11141336/filestorage-for-opencv-python-api/35573146 """
        mapping = loader.construct_mapping(node, deep=True)
        mat = np.array(mapping["data"])
        mat.resize(mapping["rows"], mapping["cols"])
        return mat

    def __readYAMLFile(self,fileName):
        ret = {}
        skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
        with open(fileName) as fin:
            for i in range(skip_lines):
                fin.readline()
            yamlFileOut = fin.read()
            myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
            yamlFileOut = myRe.sub(r': \1', yamlFileOut)
            ret = yaml.load(yamlFileOut)
        return ret

    # An Example Keypoint msg
        #  ---
        #  header:
        #    seq: 40
        #    stamp:
        #      secs: 1523613562
        #      nsecs: 530859947
        #    frame_id: world
        #  points:
        #    -
        #      x: -7.59081602097
        #      y: 7.11367511749
        #      z: 2.85602664948
        #    .
        #    .
        #    .
        #    -
        #      x: -2.64935922623
        #      y: 0.853760659695
        #      z: 0.796766400337
        #  channels:
        #    -
        #      name: ''
        #      values: [-0.06108921766281128, 0.02294199913740158, 310.8721618652344, 260.105712890625, 2.0]
        #      .
        #      .
        #      .
        #    -
        #      name: ''
        #      values: [-0.47983112931251526, 0.8081198334693909, 218.95481872558594, 435.47357177734375, 654.0]
        #    -
        #      name: ''
        #      values: [0.07728647440671921, 1.0073764324188232, 344.2176208496094, 473.7791442871094, 660.0]
        #    -
        #      name: ''
        #      values: [-0.6801641583442688, 0.10506453365087509, 159.75746154785156, 279.6077575683594, 663.0]
    def tracked_features_callback(self, data ):
        # print 'Received tracked feature', data.header.stamp, len( data.points ), len( data.channels )
        assert len( data.points ) == len( data.channels ) , "in FeatureFactor/tracked_features_callback() data.channels and data.points must have same count"
        nPts = len( data.points )

        # Store Timestamp
        self.timestamp.append( data.header.stamp )

        # msg.points have 3d points.
        # There will be nPts number of channels. Each channel will have 4 numbers denoting [ u_normed, v_normed, u, v]

        # Store 3d points
        X_3d = np.zeros( (4, nPts) )
        for i,pt in enumerate( data.points ):
            X_3d[0,i] = pt.x
            X_3d[1,i] = pt.y
            X_3d[2,i] = pt.z
            X_3d[3,i] = 1.0
        self.point3d.append( X_3d )


        # Store normalized co-ordinates
        assert( len(data.channels[0].values) == 5 )

        X_normed = np.zeros( (3, nPts) ) #in homogeneous co-ordinates
        X_observed = np.zeros( (3, nPts) ) # raw observed
        for i, ch in enumerate( data.channels ):
            X_normed[0,i] = ch.values[0]
            X_normed[1,i] = ch.values[1]
            X_normed[2,i] = 1.0

            X_observed[0,i] = ch.values[2]
            X_observed[1,i] = ch.values[3]
            X_observed[2,i] = 1.0
        self.features.append( X_normed )
        self.features_obs.append( X_observed )

        #TODO: Now there is no concept of global indices. This needs fixing.
        # Also I have disabled subscribing to keyframe image in nap node for debugging. remember to uncomment it. Also deal with global index

        # Store Global Index of these pts
        # gindex = np.array( data.channels[0].values )
        # self.global_index.append( gindex )
        # print 'gindex.shape', gindex.shape
        # print gindex

        # In the commit `6d1bb531d02fc37187b966da8245a4f47b1d6ba3` of vins_testbed.
        # IN previous versions there were only 4 channels.
        # there will be 5 channels. ch[0]: un, ch[1]: vn,  ch[2]: u, ch[3]: v.  ch[4]: globalid of the feature.
        gindex = []
        assert( len(data.channels[0].values) == 5 )
        for i, ch in enumerate( data.channels ):
            gindex.append( int(ch.values[4]) )
        self.global_index.append( np.array(gindex) )


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
        print 'len=', len(self.timestamp)
        with open( fname+'_timestamps.pickle', 'wb') as fp:
            pickle.dump( self.timestamp, fp )

        # Features in normalized co-ordinates
        print 'Writing pickle: ',  fname+'_features.pickle'
        print 'len=', len(self.features)
        with open( fname+'_features.pickle', 'wb') as fp:
            pickle.dump( self.features, fp )

        # Gobal feature index
        print 'Writing pickle: ',  fname+'_global_index.pickle'
        print 'len=', len(self.global_index)
        with open( fname+'_global_index.pickle', 'wb') as fp:
            pickle.dump( self.global_index, fp )

        # Save self.K and self.K_org
        print 'Writing pickle: ', fname+'_cam_intrinsic_K.pickle'
        print 'len=', len(self.K)
        with open( fname+'_cam_intrinsic_K.pickle', 'wb') as fp:
            pickle.dump( self.K, fp )

        print 'Writing pickle: ', fname+'_cam_intrinsic_K_org.pickle'
        print 'len=', len(self.K_org)
        with open( fname+'_cam_intrinsic_K_org.pickle', 'wb') as fp:
            pickle.dump( self.K_org, fp )

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

        # Load camera intrinsics
        print 'Opening pickle: ',  fname+'_cam_intrinsic_K.pickle'
        with open( fname+'_cam_intrinsic_K.pickle', 'rb') as fp:
            self.K = pickle.load( fp )

        print 'Opening pickle: ',  fname+'_cam_intrinsic_K_org.pickle'
        with open( fname+'_cam_intrinsic_K_org.pickle', 'rb') as fp:
            self.K_org = pickle.load( fp )
