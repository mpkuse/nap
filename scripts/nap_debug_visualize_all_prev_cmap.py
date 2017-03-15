#!/usr/bin/python
""" Subscribes to images topic for every key-frame (or semi key frame) images.
    Load GT poses file. Publish a) GT poses b) Poses of Nearest neighbours of every frame processed.
    Note that, poses of neighbours is essentially GTs of frame index

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 9th Mar, 2017
"""


import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import cv2
from cv_bridge import CvBridge, CvBridgeError

import Queue
import numpy as np


import time
from collections import namedtuple
import code
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from CartWheelFlow import VGGDescriptor
import DimRed
from Quaternion import Quat
import VPTree


#
import TerminalColors
tcolor = TerminalColors.bcolors()


# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'


# PARAM_GT_POSES_FILE = PKG_PATH+'/other_seqs/kitti_dataset/poses/00.txt' #kitti
# PARAM_GT_POSES_FILE = PKG_PATH+'/other_seqs/altizure_seq/poses/02.txt' #altizure-seq
PARAM_GT_POSES_FILE = PKG_PATH+'/other_seqs/ust_drone_seq/poses/10.txt' #ust_drone_seq-seq
#----OR----#
# PARAM_MALAGA_ID = 99
# PARAM_GT_POSES_FILE = PKG_PATH+'other_seqs/malaga_dataset/malaga-urban-dataset-extract-%02d/malaga-urban-dataset-extract-%02d_all-sensors_GPS.txt' %(PARAM_MALAGA_ID,PARAM_MALAGA_ID)


PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'



def callback_string( data ):
    rospy.logdebug( rospy.get_caller_id() + 'I heard %s', data.data )


def callback_image( data ):
    rospy.logdebug( 'Received Image : %d,%d' %( data.height, data.width ) )
    cv_image = CvBridge().imgmsg_to_cv2( data, 'bgr8' )

    im_queue.put( cv_image )
    im_indx_queue.put( int(data.header.frame_id) )
    #I am embedding the image-index in DB as frame_id string. This is for knowning the GT pose


## PUB : Publisher which publishes Marker()
## im_indx : Index of the place
## color : 3-tuple. Color of the marker
## marker_id : id of the marker (optional). If None, will use im_indx as the id
def publish_gt( PUB, im_indx, color, marker_id=None ):
    if gt_poses.shape[1] == 12: #each row has 12 elements, as in kitti
        M = gt_poses[im_indx,:].reshape(3,4)
        t = M[:,3]
        R = M[0:3,0:3]
    elif gt_poses.shape[1] == 25: #as in malaga dataset
        t = gt_poses[im_indx,[8,9,10]]
        # R = np.array( [[1.0,0.0,0.0], [0.0,0.0,1.0], [0.0,-1.0,0.0]] )
        R = np.eye(3)
    else:
        rospy.logerror( 'Invalid dimension for GT matrix. should be either Nx25 or Nx12')
        quit()

    quat = Quat( R )
    rospy.logdebug( quat.q )

    p = Marker()
    p.header.stamp = rospy.Time.now()
    p.header.frame_id = 'base_link'
    p.lifetime = rospy.Duration()
    p.id = im_indx if marker_id is None else marker_id
    p.type = Marker.CUBE
    p.action = Marker.ADD
    p.pose.position.x = t[0]
    p.pose.position.y = t[1]
    p.pose.position.z = t[2]
    p.pose.orientation.x = quat.q[0]
    p.pose.orientation.y = quat.q[1]
    p.pose.orientation.z = quat.q[2]
    p.pose.orientation.w = quat.q[3]
    p.scale.x = 1.5 if im_indx > 0 else 0.0001
    p.scale.y = 1.5 if im_indx > 0 else 0.0001
    p.scale.z = 1.5 if im_indx > 0 else 0.0001
    if len(color) == 4:
        p.color.a = color[3]
    else:
        p.color.a = 1.0
    p.color.r = color[0]
    p.color.g = color[1]
    p.color.b = color[2]
    PUB.publish( p )
    rospy.logdebug( 'Published marker')






## Plot all the found neighbours, color coded with dot distances
cmap = np.loadtxt( PKG_PATH+'/scripts/CoolWarmFloat33.csv', comments='#', delimiter=',' )
def dist_to_color( dist ): #dist is in [0,1]
    for i in range(1,cmap.shape[0]):
        if cmap[i-1,0] <= dist and dist <= cmap[i,0]:
            return cmap[i-1,1:], i-1


## Publish all previously scene points, color coded by distance
def publish_likelihoods_all_colorcoded( PUB, im_indx, likelihoods ):
    pt_ary = []
    col_ary = []
    do_not_show_for = 50 if gt_poses.shape[1] == 12 else 6
    for i, nei in enumerate(likelihoods):
        if nei.L +do_not_show_for < im_indx:
            d_0_1 = min( max(0.0,nei.dist-0.05), 1.0 ) #convert neu.dist to a number between 0 and 1

            clr, inx = dist_to_color(  d_0_1 )
            # print '%d(%3d) ' %(nei.L,inx),

            # This is quick-dirty fix, but is very slow. ~60ms with 400 points
            # publish_gt( PUB, nei.L, color=(clr[0],clr[1],clr[2]), marker_id=None)

            # Publish as points
            if gt_poses.shape[1] == 12: #each row has 12 elements, as in kitti
                M = gt_poses[nei.L,:].reshape(3,4)
                t = M[:,3]
                R = M[0:3,0:3]
            elif gt_poses.shape[1] == 25: #as in malaga dataset
                t = gt_poses[nei.L,[8,9,10]]
                # R = np.array( [[1.0,0.0,0.0], [0.0,0.0,1.0], [0.0,-1.0,0.0]] )
                R = np.eye(3)
            else:
                rospy.logerror( 'Invalid dimension for GT matrix. should be either Nx25 or Nx12')
                quit()

            pt_ary.append( Point(x=t[0], y=t[1], z=t[2]) )
            col_ary.append( ColorRGBA(r=clr[0], g=clr[1], b=clr[2], a=1.0) )

    if len(pt_ary) > 0: #only publish if there are any points to publish
        p = Marker()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = 'base_link'
        p.lifetime = rospy.Duration()
        p.id = 0
        p.type = Marker.POINTS
        p.action = Marker.ADD
        quat = Quat( R )
        p.pose.orientation.x = quat.q[0]
        p.pose.orientation.y = quat.q[1]
        p.pose.orientation.z = quat.q[2]
        p.pose.orientation.w = quat.q[3]
        p.points = pt_ary
        p.colors = col_ary
        p.scale.x = 1.5
        p.scale.y = 1.5
        p.scale.z = 1.5
        PUB.publish( p )





    # print ''




## Normalize colors for an image - used in NetVLAD descriptor extraction
def rgbnormalize( im ):
    im_R = im[:,:,0].astype('float32')
    im_G = im[:,:,1].astype('float32')
    im_B = im[:,:,2].astype('float32')
    S = im_R + im_G + im_B
    out_im = np.zeros(im.shape)
    out_im[:,:,0] = im_R / (S+1.0)
    out_im[:,:,1] = im_G / (S+1.0)
    out_im[:,:,2] = im_B / (S+1.0)

    return out_im

## Normalize a batch of images. Calls `rgbnormalize()` - used in netvlad computation
def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized


#
# Load Ground Truth Poses
gt_poses = np.loadtxt( PARAM_GT_POSES_FILE, comments='%' ) #each row is 12 numbers representing 3x4 [R|t] matrix

# malaga_gps_file_name = PARAM_DATASET_PATH+'/malaga-urban-dataset-extract-%02d_all-sensors_GPS.txt' %( PARAM_DATASET_ID )
# GPS_poses = np.loadtxt( gps_file_name, comments='%')

#
# Setup Callbacks and Publishers
im_queue = Queue.Queue()
im_indx_queue = Queue.Queue()

rospy.init_node( 'listener', log_level=rospy.INFO )
rospy.Subscriber( 'chatter', String, callback_string )
rospy.Subscriber( 'semi_keyframes', Image, callback_image )

PUB_gt_path = rospy.Publisher( 'gt_path', Marker, queue_size=1000)
PUB_nn = rospy.Publisher( 'nn', Marker, queue_size=1000)



#
# Init netvlad - def computational graph, load trained model
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=1)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )


#
# Init DimRed Mapping (Dimensionality Reduction by Learning Invariant Mapping)
dm_vlad_word = tf.placeholder( 'float', [None,None], name='vlad_word' )
net = DimRed.DimRed()
dm_vlad_char = net.fc( dm_vlad_word )
tensorflow_saver2 = tf.train.Saver( net.return_vars() )
tensorflow_saver2.restore( tensorflow_session, PARAM_MODEL_DIM_RED )
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL_DIM_RED, tcolor.ENDC




#
# Main Loop
rate = rospy.Rate(15)
treeroot = None
Likelihood = namedtuple( 'Likelihood', 'L dist')
S = np.zeros( (25000,128) )
S_im_index = np.zeros( 25000 )
loop_index = -1
while not rospy.is_shutdown():
    rate.sleep()
    print '---\nQueue Size : ', im_queue.qsize(), im_indx_queue.qsize()
    if im_queue.qsize() < 1 and im_indx_queue.qsize() < 1:
        rospy.loginfo( 'Empty Queue...Waiting' )
        continue


    im_raw = im_queue.get()
    im_indx = im_indx_queue.get()
    loop_index += 1

    # Convert to RGB format and resize
    im_ = cv2.resize( im_raw, (320,240) )
    A = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )


    ############# descriptors compute starts
    # VLAD Computation
    d_compute_time_ms = []
    startTime = time.time()
    im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
    im_batch[0,:,:,:] = A.astype('float32')
    im_batch_normalized = normalize_batch( im_batch )
    d_compute_time_ms.append( (time.time() - startTime)*1000. )

    startTime = time.time()
    feed_dict = {tf_x : im_batch_normalized,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
    d_WORD = tff_vlad_word[0,:]
    d_compute_time_ms.append( (time.time() - startTime)*1000. )


    # Dim Reduction
    startTime = time.time()
    dmm_vlad_char = tensorflow_session.run( dm_vlad_char, feed_dict={dm_vlad_word: tff_vlad_word})
    dmm_vlad_char = dmm_vlad_char
    d_CHAR = dmm_vlad_char[0,:]
    d_compute_time_ms.append( (time.time() - startTime)*1000. )

    ###### END of descriptor compute : d_WORD, d_CHAR, d_compute_time_ms[] ###############
    rospy.loginfo( '[%6.2fms] Descriptor Computation' %(sum(d_compute_time_ms)) )



    ################## Array Insert (in S)
    startSimTime = time.time()
    # S[loop_index,:] = np.multiply( np.sign(d_CHAR), np.sqrt(abs(d_CHAR)) ) #doesnt work like this. get all blues
    S[loop_index,:] = d_CHAR
    S_im_index[loop_index] = int(im_indx)
    # sim_score =  1.0 - np.dot( S[0:loop_index+1,:], d_CHAR )
    sim_score =  np.sqrt( 1.0 - np.dot( S[0:loop_index+1,:], d_CHAR ) )
    # sim_score =  np.dot( S[:loop_index,:], d_CHAR )


    rospy.loginfo( '[%6.2fms] Similarity with all prev in' %( (time.time() - startSimTime)*1000. ) )

    startTimePub = time.time()
    #collect nn
    likelihoods = []
    for inxx in range(loop_index):
        likelihoods.append( Likelihood(L=S_im_index[inxx], dist=sim_score[inxx] ) )


    # #print all nn obtained
    # print 'NN of ', pos_index, ":",
    # for nn in all_nn:
    #     print nn[1].idx, '(%5.3f), ' %(nn[0]), #idx, distance
    #     # if PARAM_DEBUG:
    #         # APPEARENCE_CONFUSION[ pos_index ,nn[1].idx] = 1.0 #np.exp( np.negative(nn[0]) )
    # print ''

    ################# Publish

    publish_gt( PUB_gt_path, im_indx, color=(1.0,1.0,0.5), marker_id=50000 ) #in yellow
    # publish_gt( PUB_gt_path, im_indx, color=(1.0,1.0,0.5,0.5) ) #in yellow (entire trail)
    publish_likelihoods_all_colorcoded( PUB_nn, im_indx, likelihoods ) #all prev color-coded. Blue is small value, red high value

    rospy.loginfo( '[%6.2fms] SPublished in ' %( (time.time() - startTimePub)*1000. ) )

    # cv2.imshow( 'win', im_ )
    # cv2.waitKey(10)
