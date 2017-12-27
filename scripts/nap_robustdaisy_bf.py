#!/usr/bin/python
""" Subscribes to images topic for every key-frame (or semi key frame) images.
    Publish asynchronously time-time message when a loop is detected.
    Images are indexed by time. In the future possibly index with distance
    using an IMU. Note that this script does not know about poses (generated from SLAM system)

    In this edition (2) of this script, there is an attempt to organize this code.
    The core netvlad place recognition system is moved into the class
    `PlaceRecognition`.

    In this edition (3) of this script, implemented a graph based merging method
    for appearence. 2 types of merges. a) seq merge b) loop-merge.
    Each time instant is a graph node represented with union-set data structure.
    The main threads does a seq merge. Another thread runs in bg and does
    merges async. This is currently out of favour for a simple brute-force dot product
    strategy.

    In this edition (4), graph merges have been abandoned. Now the logic is
    this node will subscribe to keyframes, along with detected features
    from the vins system. This node is supposed to identify loop closure
    and repond with timestamps and matching features. It will have 3 modes
    for feature matching
    a) op_mode=10. It is easily matchable by pose-graph-opt node from DB features. (not in use)
    b) op_mode=20. Forcefully matched features given out .
    c) op_mode=30. 3way match given out. This is captured by another node which
                    inturn computes the pnp pose from these 3way matches

    In this edition (5), we are going to do robust daisy. In this we have a
    voting scheme and quantitative metrics to judge quality of match. This
    scoring can essentially eliminate false matches early before going to
    loop-closure module or the pnp computation module.

    The descriptor comparison can be in either i) Brute force or ii) Using FAISS
    (product quantization). The matchings can be done with either daisy or gms-matcher.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 3rd Apr, 2017
        Edition : 2 (of nap_time_node.py)
        Edition : 4 (nap_daisy_bf.py 3rd Nov, 2017)
        Edition : 5 (nap_robustdaisy_bf.py 25th Dec, 2017)

"""


import rospy
import rospkg
import time
import code

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from nap.msg import NapMsg
from nap.msg import NapNodeMsg
from nap.msg import NapVisualEdgeMsg
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud

from PlaceRecognitionNetvlad import PlaceRecognitionNetvlad
from FeatureFactory import FeatureFactory
from FastPlotter import FastPlotter

from GeometricVerification import GeometricVerification
from gms_matcher import GmsRobe

from ColorLUT import ColorLUT

import TerminalColors
tcol = TerminalColors.bcolors()


############# PARAMS #############
PKG_PATH = rospkg.RosPack().get_path('nap')
# PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'
# PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k48/model-13000' #PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
# PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_tokyoTM/model-3500' #trained with tokyo, normalization is simple '
# PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_b20_tokyoTM_mean_aggregation/model-3750' #trained with mean aggregation in place of usual sum aggregation in netvlad_layer
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_b20_tokyoTM_pos_set_dev/model-6500' #trained with rotation without black borders and with pos-set-dev
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_b20_resnet/model-3750' # trained similar to above but with a resnet neural net - the default model

# tf2.logs
# PARAM_MODEL = PKG_PATH+'/tf2.logs/attempt_resnet6_K64_P8_N8/model-5000' # new resnet6_64
# PARAM_MODEL = PKG_PATH+'/tf2.logs/attempt_vgg6_K64_P8_N8/model-9500' #new vgg

PARAM_MODEL = PKG_PATH+'/tf2.logs/attempt_resnet6_K16_P8_N8/model-9250' #K=16
# PARAM_MODEL = PKG_PATH+'/tf2.logs/attempt_resnet6_K32_P8_N8/model-2250' #K=32

# Dont forget to load the eigen values, eigen vectors and mean


# PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/netvlad_k48/db2/siamese_dimred/model-400' #PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'



INPUT_IMAGE_TOPIC = '/semi_keyframes' #this is t be used for launch
PARAM_CALLBACK_SKIP = 2
PARAM_K = 16
PARAM_FPS = 25 #used in ros.spin() loop

BASE__DUMP = '/home/mpkuse/Desktop/a/drag_nap'
DEBUG_IMAGE_PUBLISH = True
DEBUG_PLOTTER = False

def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )

def publish_image( PUB, cv_image, t=None ):

    data_type = 'bgr8'
    if len(cv_image.shape) == 2:
        data_type = 'mono8'

    msg_frame = CvBridge().cv2_to_imgmsg( cv_image, data_type )
    if t is not None:
        msg_frame.header.stamp = t
    PUB.publish( msg_frame )


#--- Nap Msg Creation ---#
def make_nap_msg( i_curr, i_prev, edge_color=None):
    """ Uses global variables S_timestamp, sim_scores_logistic
    """
    nap_msg = NapMsg() #edge msg
    nap_msg.c_timestamp = S_timestamp[i_curr]
    nap_msg.prev_timestamp = S_timestamp[i_prev]
    nap_msg.goodness = sim_scores_logistic[i_prev]

    if edge_color is None:
        edge_color = (0,1.0,0)

    if len(edge_color) != 3:
        edge_color = (0,1.0,0)

    nap_msg.color_r = edge_color[0] #default color is green
    nap_msg.color_g = edge_color[1]
    nap_msg.color_b = edge_color[2]
    return nap_msg

def make_nap_visual_msg( i_curr, i_prev, str_curr, str_prev ):
    """ Uses global variables S_timestamp, sim_scores_logistic, S_thumbnail
    """
    nap_visual_edge_msg = NapVisualEdgeMsg()
    nap_visual_edge_msg.c_timestamp = S_timestamp[i_curr]
    nap_visual_edge_msg.prev_timestamp = S_timestamp[i_prev]
    nap_visual_edge_msg.goodness = sim_scores_logistic[i_prev]
    nap_visual_edge_msg.curr_image = CvBridge().cv2_to_imgmsg( S_thumbnail[i_curr].astype('uint8'), "bgr8" )
    nap_visual_edge_msg.prev_image = CvBridge().cv2_to_imgmsg( S_thumbnail[i_prev].astype('uint8'), "bgr8" )
    nap_visual_edge_msg.curr_label = str_curr #str(i_curr) #+ '::%d,%d' %(nInliers,nMatches)
    nap_visual_edge_msg.prev_label = str_prev #str(i_prev)

    return nap_visual_edge_msg

#--- END Nap Msg ---#

def robust_daisy():
    """ Wrapper for the robust scoring set of calls. Based on
        `test_daisy_improve_dec2017.py'

        Uses, VV, curr_im, prev_im, curr_m_im, __lut_curr_im, __lut_prev_im

        Returns:
            a) _2way: selected_curr_i, selected_prev_i
            b) _3way:
    """
    _2way = None
    _3way = None
    _info = ''
    ROBUST_DAISY_IMSHOW = False

    VV.set_image( curr_im, 1 ) #set current image
    VV.set_image( prev_im, 2 )# set previous image (at this stage dont need lut_raw to be set as it is not used by release_candidate_match2_guided_2way() )

    selected_curr_i, selected_prev_i, sieve_stat = VV.release_candidate_match2_guided_2way( feat2d_curr, feat2d_prev )
    #
    #
    # # # min/ max
    # # if (float(min(feat2d_curr.shape[1],feat2d_prev.shape[1])) / max(feat2d_curr.shape[1],feat2d_prev.shape[1])) < 0.70:
    # #     match2_total_score -= 3
    # #     print 'nTracked features are very different.'
    # #
    #
    match2_total_score = VV.sieve_stat_to_score( sieve_stat ) #remember to do min/max scoring. ie. reduce score if nTracked features are very different in both frames
    print '=X=Total_score : ', match2_total_score, '=X='
    _info += '=X=Total_score : '+ str(match2_total_score)+ '=X=\n'
    _info += 'After 2way_matching, n=%d\n' %( len(selected_curr_i) )

    if ROBUST_DAISY_IMSHOW:
        xcanvas_2way = VV.plot_2way_match( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i]),  enable_lines=True )
        cv2.imshow( 'xcanvas_2way', xcanvas_2way )


    # Rules
    if match2_total_score > 3:
        # Accept this match and move on
        print 'Accept this match and move on'
        print tcol.OKGREEN, 'Accept (Strong)', tcol.ENDC
        _info += tcol.OKGREEN+ 'a: Accept (Strong)'+ tcol.ENDC + '\n'
        _2way = (selected_curr_i,selected_prev_i)

    if match2_total_score > 2 and match2_total_score <= 3 and len(selected_curr_i) > 20:
        # Boundry case, if you see sufficient number of 2way matches, also accpt 2way match
        print 'Boundary case, if you see sufficient number of 2way matches, also accept 2way match'
        print tcol.OKGREEN, 'Accept', tcol.ENDC
        _info += tcol.OKGREEN+ 'b: Accept'+ tcol.ENDC+'\n'

        _2way = (selected_curr_i,selected_prev_i)


    if match2_total_score >= 0.5 and match2_total_score <= 3:
        # Try 3way. But plot 2way and 3way.
        # Beware, 3way match function returns None when it has early-rejected the match
        print 'Attempt robust_3way_matching()'

        # set-data
        VV.set_image( curr_m_im, 3 )  #set curr-1 image
        VV.set_lut_raw( __lut_curr_im, 1 ) #set lut of curr and prev
        VV.set_lut_raw( __lut_prev_im, 2 )
        # VV.set_lut( curr_lut, 1 ) #only needed for in debug mode of 3way match
        # VV.set_lut( prev_lut, 2 ) #only needed for in debug mode of 3way match

        # Attempt 3way match
        # q1,q2,q3: pts_curr, pts_prev, _pts_curr_m,
        # q4      : per_match_vote,
        # q5      : (dense_match_quality, after_vote_match_quality)
        # See GeometricVerification class to know more on this function.
        q1,q2,q3,q4,q5 = VV.robust_match3way()
        print 'dense_match_quality     : ', q5[0]
        print 'after_vote_match_quality: ', q5[1]
        _info += 'After 3way_matching:\n'
        _info += 'dense_match_quality:%4.2f\n' %(q5[0])
        _info += 'after_vote_match_quality:%4.2f\n' %(q5[1])


        if q1 is None:
            print 'Early Reject from robust_match3way()'
            print tcol.FAIL, 'Reject', tcol.ENDC
            _info += 'Early Reject from robust_match3way()\n'
            _info += tcol.FAIL+ 'c: Reject'+ tcol.ENDC+'\n'
            _3way = None

        else:
            print 'nPts_3way_match     : ', q1.shape
            print 'Accept 3way match'
            print tcol.OKGREEN, 'Accept', tcol.ENDC
            _info += 'n3way_matches: %s' %( str(q1.shape) ) + '\n'
            _info += tcol.OKGREEN+ 'c: Accept'+ tcol.ENDC + '\n'
            if ROBUST_DAISY_IMSHOW:
                gridd = VV.plot_3way_match( VV.im1, np.array(q1), VV.im2, np.array(q2), VV.im3, np.array(q3) )
                cv2.imshow( '3way Matchi', gridd )
            #fill up _3way
            _3way = (q1,q2,q3)



    if match2_total_score < 0.5:
        # Reject (don't bother computing 3way)
        print 'Reject 2way matching, and do not compute 3way matching'
        print tcol.FAIL, 'Reject (Strong)', tcol.ENDC
        _info += tcol.FAIL+ 'd: Reject (Strong)'+ tcol.ENDC+'\n'
        _2way = None
        _3way = None


    if ROBUST_DAISY_IMSHOW:
        cv2.waitKey(10)
    return _2way, _3way, _info



# #--- Geometry and Matching ---#
# def match3way_daisy( curr_im, prev_im, curr_m_im,    __lut_curr_im, __lut_prev_im ):
#     """ Gives out 3 way matching 3 3xN matrix (i think)"""
#     DEBUG = True #on enable, writes images to disk
#     # Step-1: Compute dense matches between curr and prev --> SetA
#     VV.set_im( curr_im, prev_im )
#     VV.set_im_lut_raw( S_lut_raw[i_curr], S_lut_raw[i_prev] )
#
#     pts_curr, pts_prev, mask_c_p = VV.daisy_dense_matches()
#     if DEBUG:
#         xcanvas_c_p = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask_c_p)
#         fname = '/home/mpkuse/Desktop/a/drag_nap/%d.jpg' %(loop_index)
#         print 'Write(match3way_daisy) : ', fname
#         cv2.imwrite( fname, xcanvas_c_p )
#
#
#     # Step-2: Match expansion
#     _pts_curr_m = VV.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p, curr_m_im  )
#     masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
#     masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
#
#     if DEBUG:
#         gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )
#         fname = '/home/mpkuse/Desktop/a/drag_nap/%d_3way.jpg' %(loop_index)
#         print 'Write(match3way_daisy) : ', fname
#         cv2.imwrite(fname, gridd )
#
#     assert( len(masked_pts_curr) == len(masked_pts_prev) )
#     assert( len(masked_pts_curr) == len(_pts_curr_m) )
#
#     masked_pts_curr = np.array(masked_pts_curr) #Nx2
#     masked_pts_prev = np.array( masked_pts_prev )
#     _pts_curr_m =  np.array(_pts_curr_m )
#     return  masked_pts_curr, masked_pts_prev, _pts_curr_m
#
#
# def match3way_gms( curr_im, prev_im, curr_m_im ):
#     start3way = time.time()
#     pts_C, pts_P, pts_Cm = GMS.match3( curr_im, prev_im, curr_m_im )
#     print 'time elapsed GMS.match3 : %4.2f' %( 1000. * (time.time() - start3way) )
#     print 'match3way_gms() : pts_C.shape', pts_C.shape
#     print 'match3way_gms() : pts_P.shape', pts_P.shape
#     print 'match3way_gms() : pts_Cm.shape', pts_Cm.shape
#     return np.transpose(pts_C), np.transpose(pts_P), np.transpose(pts_Cm)
#
#
# def match2_guided_gms( curr_im, feature_factory_index, prev_im ):
#     feat2d_normed = feature_factory.features[ feature_factory_index ] #3xN
#     K = feature_factory.K
#
#     pts_curr = np.dot( K, feat2d_normed )
#     print 'match2: Input pts : ', pts_curr.shape
#     curr_annotate = curr_im.copy()
#     for i in range( pts_curr.shape[1] ):
#         xpt = np.int0( pts_curr[0:2,i] )
#         cv2.circle( curr_annotate, (xpt[0], xpt[1]), 2, (55,255,55), -1 )
#     #
#     # fname = '/home/mpkuse/Desktop/a/drag_nap/%d.png' %(feature_factory_index)
#     # print 'Writing file : ', fname
#     # cv2.imwrite( fname, dst )
#
#     pts2_curr, pts2_prev = GMS.match2_guided( curr_im, pts_curr[0:2,:], prev_im )
#
#     # dst = GMS.plot_point_sets( curr_im, pts2_curr, prev_im, pts2_prev )
#     dst = GMS.plot_point_sets( curr_annotate, pts2_curr, prev_im, pts2_prev )
#     fname = '/home/mpkuse/Desktop/a/drag_nap/%d.png' %(feature_factory_index)
#     print 'Writing file : ', fname
#     cv2.imwrite( fname, dst )
#
#
#
#
#
#
#     pts2x_curr, pts2x_prev = GMS.match2( curr_im, prev_im )
#
#     # dst = GMS.plot_point_sets( curr_im, pts2_curr, prev_im, pts2_prev )
#     dst = GMS.plot_point_sets( curr_annotate, pts2x_curr, prev_im, pts2x_prev )
#     fname = '/home/mpkuse/Desktop/a/drag_nap/%d_match2.png' %(feature_factory_index)
#     print 'Writing file : ', fname
#     cv2.imwrite( fname, dst )
#
#
#     print 'match2: Output pts : ', pts2_curr.shape
#     return np.transpose( pts2_curr), np.transpose( pts2_prev )
#
#
#
#
#
# #---           END        ---#


########### Init PlaceRecognitionNetvlad ##########
place_mod = PlaceRecognitionNetvlad(\
                                    PARAM_MODEL,\
                                    PARAM_CALLBACK_SKIP=PARAM_CALLBACK_SKIP,\
                                    PARAM_K = PARAM_K
                                    )


vins_config_yaml = rospy.get_param( '/nap/config_file')
print 'VINS-CONFIG-YAML', vins_config_yaml
feature_factory = FeatureFactory(vins_config_yaml)

############# GEOMETRIC VERIFICATION #################
VV = GeometricVerification()
# GMS = GmsRobe(n=10000) #only used in geometry functions defined in this file.


################### Init Node and Topics ############
rospy.init_node( 'nap_geom_node', log_level=rospy.INFO )

# Input Images
rospy.Subscriber( INPUT_IMAGE_TOPIC, Image, place_mod.callback_image )
rospy.loginfo( 'Subscribed to '+INPUT_IMAGE_TOPIC )

# Tracked Features
# TRACKED_FEATURE_TOPIC = '/feature_tracker/feature'
TRACKED_FEATURE_TOPIC = '/vins_estimator/keyframe_point'
rospy.Subscriber( TRACKED_FEATURE_TOPIC, PointCloud, feature_factory.tracked_features_callback )
rospy.loginfo( 'Subscribed to '+TRACKED_FEATURE_TOPIC )


# raw edges
pub_edge_msg = rospy.Publisher( '/raw_graph_edge', NapMsg, queue_size=1000 )
rospy.loginfo( 'Publish to /raw_graph_edge' )

# raw visual edges
# pub_visual_edge_msg = rospy.Publisher( '/raw_graph_visual_edge', NapVisualEdgeMsg, queue_size=1000 )
# pub_visual_edge_cluster_assgn_msg = rospy.Publisher( '/raw_graph_visual_edge_cluster_assgn', NapVisualEdgeMsg, queue_size=1000 )
# rospy.loginfo( 'Publish to /raw_graph_visual_edge' )


# Time - debug
pub_time_queue_size = rospy.Publisher( '/time/queue_size', Float32, queue_size=1000)
pub_time_desc_comp = rospy.Publisher( '/time/netvlad_comp', Float32, queue_size=1000)
pub_time_dot_scoring = rospy.Publisher( '/time/dot_scoring', Float32, queue_size=1000)
pub_time_seq_merging = rospy.Publisher( '/time/seq_merging', Float32, queue_size=1000)
pub_time_geometric_verification = rospy.Publisher( '/time/geometric_verify', Float32, queue_size=1000)
pub_time_publish = rospy.Publisher( '/time/publish', Float32, queue_size=1000)
pub_time_total_loop = rospy.Publisher( '/time/total', Float32, queue_size=1000)

# Cluster Assignment - raw and falsecolormap
colorLUT = ColorLUT()

if DEBUG_IMAGE_PUBLISH:
    pub_cluster_assgn_falsecolormap = rospy.Publisher( '/debug/cluster_assignment', Image, queue_size=10 )
    rospy.loginfo( 'Publish to /debug/cluster_assignment')

    # pub_cluster_assgn_raw = rospy.Publisher( '/nap/cluster_assignment', Image, queue_size=10 )
    # rospy.loginfo( 'Publish to /nap/cluster_assignment')

    pub_feat2d_matching = rospy.Publisher( '/debug/featues2d_matching', Image, queue_size=10 )
    # pub_2way_matching = rospy.Publisher( '/debug/3way_matching', Image, queue_size=10 )
    # pub_3way_matching = rospy.Publisher( '/debug/2way_matching', Image, queue_size=10 )
    rospy.loginfo( 'Publish to /debug/featues2d_matching')


#################### Init Plotter #####################
if DEBUG_PLOTTER:
    plotter = FastPlotter(n=3)
    plotter.setRange( 0, yRange=[0,1] )
    plotter.setRange( 1, yRange=[0,1] )
    plotter.setRange( 2, yRange=[0,1] )


##################### Main Loop ########################
rate = rospy.Rate(PARAM_FPS)



S_word = []

S_timestamp = [] #np.zeros( 25000, dtype=rospy.Time )
S_thumbnail = []
S_thumbnail_full_res = []
S_lut = [] #only for debug, the cluster assgnment image (list of false color)
S_lut_raw = [] # raw 1-channel cluster assignment
loop_index = -1
startTotalTime = time.time()


loop_candidates = []
loop_candidates2 = []

if BASE__DUMP is not None:
    geometric_verify_log_fp = open( BASE__DUMP+'/geometric_verify.log', 'w' )
else:
    geometric_verify_log_fp = None

while not rospy.is_shutdown():
    rate.sleep()

    publish_time( pub_time_total_loop, 1000.*(time.time() - startTotalTime) ) #this has been put like to use startTotalTime from prev iteration
    startTotalTime = time.time()
    #------------------- Queue book-keeping---------------------#
    rospy.logdebug( '---Queue Size : %d, %d' %( place_mod.im_queue.qsize(), place_mod.im_timestamp_queue.qsize()) )
    if place_mod.im_queue_full_res is not None:
        rospy.logdebug( '---Full Res Queue : %d' %(place_mod.im_queue_full_res.qsize())  )
    if place_mod.im_queue.qsize() < 1 and place_mod.im_timestamp_queue.qsize() < 1:
        rospy.logdebug( 'Empty Queue...Waiting' )
        continue
    publish_time( pub_time_queue_size, place_mod.im_queue.qsize() )


    # Get Image & TimeStamp from the queue
    im_raw = place_mod.im_queue.get()
    print 'im.size : ', im_raw.shape
    if place_mod.im_queue_full_res is not None:
        im_raw_full_res = place_mod.im_queue_full_res.get()
        print 'im_full_res.size : ', im_raw_full_res.shape
    im_raw_timestamp = place_mod.im_timestamp_queue.get()

    feature_factory_index = feature_factory.find_index( im_raw_timestamp )
    print 'feature_factory_index', feature_factory_index
    # if feature_factory_index >= 0:
        # match2_guided( im_raw, feature_factory_index, None)


    loop_index += 1
    #---------------------------- END  -----------------------------#


    #---------------------- Descriptor Extractor ------------------#
    startDescComp = time.time()
    rospy.logdebug( 'NetVLAD Computation' )
    # d_CHAR, d_WORD = place_mod.extract_reduced_descriptor(im_raw)
    d_WORD = place_mod.extract_descriptor(im_raw)



    publish_time( pub_time_desc_comp, 1000.*(time.time() - startDescComp) )
    rospy.logdebug( 'Word Shape : %s' %(d_WORD.shape) )


    #---------------------------- END  -----------------------------#

    #-------------------------- Storage  ----------------------------#
    # Note: Storage of S_word, S_timestamp, S_thumbnail,
    #       (optional) S_thumbnail_full_res, S_lut, S_lut_raw

    rospy.logdebug( 'Storage of S_word, S_timestamp, S_thumbnail, ...' )
    S_lut_raw.append( place_mod.Assgn_matrix[0,:,:]  )

    if DEBUG_IMAGE_PUBLISH: #Set this to true to publish false color Assgn_image and publish it.
        lut = colorLUT.lut( place_mod.Assgn_matrix[0,:,:] )
        S_lut.append( lut )
        publish_image( pub_cluster_assgn_falsecolormap, lut, t=im_raw_timestamp )
        # publish_image( pub_cluster_assgn_raw, place_mod.Assgn_matrix[0,:,:].astype('uint8'), t=im_raw_timestamp )


    S_word.append( d_WORD )
    S_timestamp.append( im_raw_timestamp )
    # S_thumbnail.append(  cv2.resize( im_raw.astype('uint8'), (128,96) ) )#, fx=0.2, fy=0.2 ) )
    S_thumbnail.append(  cv2.resize( im_raw.astype('uint8'), (320,240) ) )#, fx=0.2, fy=0.2 ) )

    if place_mod.im_queue_full_res is not None:
        S_thumbnail_full_res.append( im_raw_full_res.astype('uint8') )

    #---------------------------- END  -----------------------------#


    #------------------- Score Computation (Brute Force)----------------#
    rospy.logdebug( 'Score Computation (Brute Force)' )
    startScoreCompTime = time.time()

    # DOT_word = np.dot( S_word[0:loop_index+1,:], S_word[loop_index,:] )
    DOT_word = np.dot( S_word[0:loop_index+1], np.transpose(S_word[loop_index]) )

    sim_scores = np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) ) #minimum is added to ensure dot product doesnt go beyond 1.0 as it sometimes happens because of numerical issues, which inturn causes issues with sqrt

    sim_scores_logistic = place_mod.logistic( sim_scores ) #convert the raw Similarity scores above to likelihoods

    publish_time( pub_time_dot_scoring, 1000.*(time.time() - startScoreCompTime) )
    #---------------------------- END  -----------------------------#


    # --------- PLOT Sim Score of current wrt all prev ---------#
    if DEBUG_PLOTTER:
        # Plot sim_scores
        plotter.set_data( 0, range(len(DOT_word)), DOT_word, title="DOT_word"  )
        plotter.set_data( 1, range(len(sim_scores)), sim_scores, title="sim_scores = sqrt( 1-dot )"  )
        plotter.set_data( 2, range(len(sim_scores_logistic)), sim_scores_logistic, title="sim_scores_logistic"  )

        plotter.spin()


    #----------------- Grid Filter / Temporal Fusion -------------------#
    if loop_index < 2: #let data accumulate
        continue

    # Ideally should have grid-filter here using `sim_scores_logistic`
    #------------------------------ END  -------------------------------#


    #-------------------------------------------------------------------#
    #------------- Publish Colocation (on loop closure ) ---------------#
    #-------------------------------------------------------------------#


    #----------------------- Candidates (BF NN) ------------------------#
    L = loop_index #alias
    # Determination of edge using `sim_scores_logistic`
    argT = np.where( sim_scores_logistic[1:L] > 0.54 )
    if len(argT ) < 1:
        continue

    print '---'
    print 'Found %d candidates above the thresh' %(len(argT[0]))

    _now_edges = []
    for aT in argT[0]: #Iterate through each match and collect candidates
        # Avoid matches from near current
        if float(S_timestamp[L].to_sec() - S_timestamp[aT].to_sec())<10.  or aT < 5:
            continue

        # nMatches, nInliers = do_geometric_verification( L-1, aT)
        # Do simple verification using,  S_thumbnail[i_curr] and S_thumbnail[i_prev] and class GeometricVerification
        # VV.set_im( S_thumbnail[L], S_thumbnail[aT] )
        # nMatches, nInliers = VV.simple_verify(features='orb')
        # Another possibility is to not do any verification here. And randomly choose 1 pair for 3way matching.
        nMatches = 25
        nInliers = np.random.randint(10, 40) #25

        # Record this match in a file
        # print '%d<--->%d' %(L-1,aT)
        # note, do not use loop_candidates, it is for file-logging. Instead use `_now_edges`
        loop_candidates.append( [L-1, aT, sim_scores_logistic[aT], nMatches, nInliers] ) #here was L in original

        if nInliers > 0:
            _now_edges.append( (L-1, aT, sim_scores_logistic[aT], nMatches, nInliers) )



    #_now_edges : candidates for L-1. If they are sufficient in number. May be select 1 randomly
    #             to publish, or based on some criteria using NetVLAD_Assignment_mat
    # Make a decision on where to use
    #       op_mode=10 : Do nothing, just publish 2 timestamps
    #       op_mode=20 : Guided matching (only with gms-matcher)
    #       op_mode=30 : 3-way matching


    startPublish = time.time()
    if len(_now_edges) < 1: #configurable, minimum support
        continue

    # Randomly pick a candidate of several
    pick = np.random.randint( 0, len(_now_edges) )

    i_curr = _now_edges[pick][0]
    i_prev = _now_edges[pick][1]
    i_sim_score = _now_edges[pick][2]
    i_matches = _now_edges[pick][3]
    i_inliers = _now_edges[pick][4]
    loop_candidates2.append( [i_curr, i_prev, i_sim_score, i_matches, i_inliers] ) #Only selected candidates


    nap_msg = make_nap_msg( i_curr, i_prev, (0.6,1.0,0.6) ) # puts in msg.c_timestamp, msg.prev_timestamp, msg.goodness
    nap_msg.n_sparse_matches = i_inliers #Not required


    ################## Fill feature matching part of nap_msg ###########################
    # Task is to fill up nap msg.

    #
    # Collect Image Data for processing
    #
    curr_im = S_thumbnail[i_curr].astype('uint8')
    prev_im = S_thumbnail[i_prev].astype('uint8')
    curr_m_im = S_thumbnail[i_curr-1].astype('uint8')
    t_curr = S_timestamp[i_curr]
    t_prev = S_timestamp[i_prev]
    t_curr_m = S_timestamp[i_curr-1]

    __lut_curr_im = S_lut_raw[i_curr]
    __lut_prev_im = S_lut_raw[i_prev]


    #
    # Collect Features data
    #
    feat2d_curr_idx = feature_factory.find_index( t_curr )
    feat2d_prev_idx = feature_factory.find_index( t_prev )

    feat2d_curr_normed = feature_factory.features[feat2d_curr_idx ]
    feat2d_prev_normed = feature_factory.features[feat2d_prev_idx ]


    feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
    feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] )

    feat3d_curr = feature_factory.point3d[feat2d_curr_idx]

    # feat2d_curr_global_idx = feature_factory.global_index[feat2d_curr_idx]
    # feat2d_prev_global_idx = feature_factory.global_index[feat2d_prev_idx]

    # This call uses VV, and other variables (global)
    _2way, _3way, _info_txt = robust_daisy()
    if geometric_verify_log_fp is not None:
        geometric_verify_log_fp.write( '\n---i_curr=%5d, i_prev=%5d---\n' %(i_curr, i_prev) )
        geometric_verify_log_fp.write( _info_txt )

    if _2way is not None:
        # Step-1a: Get indices of matches
        selected_curr_i, selected_prev_i = _2way

        # Step-1b: Save as image
        if BASE__DUMP is not None or DEBUG_IMAGE_PUBLISH:

            if selected_curr_i.shape[0] > 0:
                xcanvas_2way = VV.plot_2way_match( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i]),  enable_lines=True )
            else:
                # xcanvas_2way = np.zeros( (100,100), dtype=np.uint8)
                xcanvas_2way = VV.plot_2way_match( curr_im, None, prev_im, None,  enable_lines=True )

            if DEBUG_IMAGE_PUBLISH:
                publish_image( pub_feat2d_matching, xcanvas_2way,  t=im_raw_timestamp )

            if BASE__DUMP is not None:
                _xfname = '%s/%d_%d_2way.png' %(BASE__DUMP, i_curr, i_prev )
                print 'Writing ', _xfname
                cv2.imwrite( _xfname,  xcanvas_2way )

        # Step-2: Fill up nap-msg
        nap_msg = make_nap_msg( i_curr, i_prev, (0.6,1.0,0.6) )
        nap_msg.op_mode = 20
        nap_msg.t_curr = t_curr
        nap_msg.t_prev = t_prev
        # feat2d_curr_global_idx.shape : 96,
        # feat2d_curr_normed.shape : 3x96
        # selected_curr_i.shape: 21,
        #    out of 96 tracked features 21 were selected
        for h in range( len(selected_curr_i) ):
            _u = feat2d_curr_normed[ 0:2, selected_curr_i[h] ]
            _U = feat3d_curr[0:3, selected_curr_i[h] ]
            _g_idx = -100#feat2d_curr_global_idx[ selected_curr_i[h] ]
            # nap_msg.curr will be 2X length, where nap_msg.prev will be X length.
            nap_msg.curr.append( Point32(_u[0], _u[1], _g_idx) )
            nap_msg.curr.append( Point32(_U[0], _U[1], _U[2])  )

            _u = feat2d_prev_normed[ 0:2, selected_prev_i[h] ]
            _g_idx = -100#feat2d_prev_global_idx[ selected_prev_i[h] ]
            nap_msg.prev.append( Point32(_u[0], _u[1], _g_idx) )

        # Step-3: Publish nap-msg
        pub_edge_msg.publish( nap_msg )


    if _3way is not None:
        # Step-1a: Get co-ordinates of matches
        xpts_curr, xpts_prev, xpts_currm = _3way

        # Step-1b: Save image for debugging
        if BASE__DUMP is not None or DEBUG_IMAGE_PUBLISH:
            gridd = VV.plot_3way_match( curr_im, xpts_curr, prev_im, xpts_prev, curr_m_im, xpts_currm, enable_lines=False, enable_text=True )
            if DEBUG_IMAGE_PUBLISH:
                publish_image( pub_feat2d_matching, gridd,  t=im_raw_timestamp )


            if BASE__DUMP is not None:
                _xfname = '%s/%d_%d_3way.png' %(BASE__DUMP, i_curr, i_prev )
                print 'Writing ', _xfname
                cv2.imwrite( _xfname,  gridd )

        # Step-2: Fill up nap msg
        nap_msg = make_nap_msg( i_curr, i_prev, (0.6,1.0,0.6) )
        nap_msg.op_mode = 29
        nap_msg.t_curr = t_curr
        nap_msg.t_prev = t_prev
        nap_msg.t_curr_m = t_curr_m
        for ji in range( len(xpts_curr) ): #len(xpts_curr) is same as xpts_curr.shape[0]
            pt_curr = xpts_curr[ji]
            pt_prev = xpts_prev[ji]
            pt_curr_m = xpts_currm[ji]

            nap_msg.curr.append(   Point32(pt_curr[0], pt_curr[1],-1) )
            nap_msg.prev.append(   Point32(pt_prev[0], pt_prev[1],-1) )
            nap_msg.curr_m.append( Point32(pt_curr_m[0], pt_curr_m[1],-1) )

        pub_edge_msg.publish( nap_msg )

    publish_time( pub_time_publish, 1000.*(time.time() - startPublish) )
    #TODO: publish_image( pub_2way_matching, xcanvas_2way,  t=im_raw_timestamp )
    # based on a flag for visualization
    continue

    # #
    # # Attempt Guided 2way matching (old)
    # #
    # startSet = time.time()
    # VV.set_image( curr_im, 1 ) #set current image
    # VV.set_image( prev_im, 2 )# set previous image (at this stage dont need lut_raw to be set as it is not used by release_candidate_match2_guided_2way() )
    # print 'set_image, ch=1 and ch=2 : %4.2f (ms)' %( 1000. * (time.time() - startSet) )
    #
    # startT = time.time()
    # selected_curr_i, selected_prev_i, sieve_stat = VV.release_candidate_match2_guided_2way( feat2d_curr, feat2d_prev )
    # print 'matcher.release_candidate_match2_guided_2way() : %4.2f (ms)' %(1000. * (time.time() - startT) )
    # print 'guided 2way matches : ', selected_curr_i.shape[0], selected_prev_i.shape[0]
    # n_guided_2way = selected_curr_i.shape[0]
    #
    # if n_guided_2way > 0:
    #     xcanvas_2way = VV.plot_2way_match( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i]),  enable_lines=True )
    # else:
    #     # xcanvas_2way = np.zeros( (100,100), dtype=np.uint8)
    #     xcanvas_2way = VV.plot_2way_match( curr_im, None, prev_im, None,  enable_lines=True )
    # publish_image( pub_feat2d_matching, xcanvas_2way, t=im_raw_timestamp )
    # if BASE__DUMP is not None:
    #     _xfname = '%s/%d_%d_2way.png' %(BASE__DUMP, i_curr, i_prev )
    #     print 'Writing ', _xfname
    #     cv2.imwrite( _xfname,  xcanvas_2way )
    #
    #
    # if( n_guided_2way > 20 ):
    #     # if we get more than 20 matches set these matches in nap_msg and publish
    #     nap_msg.op_mode = 20
    #     nap_msg.t_curr = t_curr
    #     nap_msg.t_prev = t_prev
    #
    #     # feat2d_curr_global_idx.shape : 96,
    #     # feat2d_curr_normed.shape : 3x96
    #     # selected_curr_i.shape: 21,
    #     #    out of 96 tracked features 21 were selected
    #     for h in range( len(selected_curr_i) ):
    #         _u = feat2d_curr_normed[ 0:2, selected_curr_i[h] ]
    #         _U = feat3d_curr[0:3, selected_curr_i[h] ]
    #         _g_idx = -100#feat2d_curr_global_idx[ selected_curr_i[h] ]
    #         # nap_msg.curr will be 2X length, where nap_msg.prev will be X length.
    #         nap_msg.curr.append( Point32(_u[0], _u[1], _g_idx) )
    #         nap_msg.curr.append( Point32(_U[0], _U[1], _U[2])  )
    #
    #         _u = feat2d_prev_normed[ 0:2, selected_prev_i[h] ]
    #         _g_idx = -100#feat2d_prev_global_idx[ selected_prev_i[h] ]
    #         nap_msg.prev.append( Point32(_u[0], _u[1], _g_idx) )
    #
    #     pass
    #     publish_image( pub_2way_matching, xcanvas_2way,  t=im_raw_timestamp )
    #
    # else:
    #     #
    #     # Attempt 3way Matching (old)
    #     #
    #     # if few matches, than attempt a 3way matching
    #     startSet = time.time()
    #     VV.set_image( curr_m_im, 3 )  #set curr-1 image
    #     print 'set_image ch=3 : %4.2f (ms)' %( 1000. * (time.time() - startSet) )
    #
    #     VV.set_lut_raw( __lut_curr_im, 1 ) #set lut of curr and prev
    #     VV.set_lut_raw( __lut_prev_im, 2 )
    #     # lut for curr-1 is not set as it is not used.
    #
    #
    #     # these will be 3 co-ordinate point sets
    #     start3way = time.time()
    #     xpts_curr, xpts_prev, xpts_currm = VV.release_candidate_match3way() #this function reuses daisy for im1, and im2, just 1 daisy computation inside.
    #     print 'n3way matches : ', xpts_curr.shape
    #     print 'matcher.release_candidate_match3way() : %4.2f (ms)' %(1000. * (time.time() - start3way) )
    #
    #     gridd = VV.plot_3way_match( curr_im, xpts_curr, prev_im, xpts_prev, curr_m_im, xpts_currm, enable_lines=False, enable_text=True )
    #     publish_image( pub_feat2d_matching, gridd, t=im_raw_timestamp )
    #     if BASE__DUMP is not None:
    #         _xfname = '%s/%d_%d_3way.png' %(BASE__DUMP, i_curr, i_prev )
    #         print 'Writing ', _xfname
    #         cv2.imwrite( _xfname,  gridd )
    #
    #
    #     # Set in nam_msg
    #     # code.interact( local=locals() )
    #     # xpts_curr.shape : 57x2
    #     # xpts_prev.shape : 57x2
    #     # xpts_currm.shape : 57x2
    #
    #     nap_msg.op_mode = 29
    #     nap_msg.t_curr = t_curr
    #     nap_msg.t_prev = t_prev
    #     nap_msg.t_curr_m = t_curr_m
    #
    #     for ji in range( len(xpts_curr) ): #len(xpts_curr) is same as xpts_curr.shape[0]
    #         pt_curr = xpts_curr[ji]
    #         pt_prev = xpts_prev[ji]
    #         pt_curr_m = xpts_currm[ji]
    #
    #         nap_msg.curr.append(   Point32(pt_curr[0], pt_curr[1],-1) )
    #         nap_msg.prev.append(   Point32(pt_prev[0], pt_prev[1],-1) )
    #         nap_msg.curr_m.append( Point32(pt_curr_m[0], pt_curr_m[1],-1) )
    #
    #     publish_image( pub_3way_matching, gridd,  t=im_raw_timestamp )
    # # continue

    #############################################

    # pub_edge_msg.publish( nap_msg )


    # Comment following 2 lines to not debug-publish loop-candidate
    # nap_visual_edge_msg = make_nap_visual_msg( i_curr, i_prev, "%d,%d" %(i_curr,i_inliers), str(i_prev) )
    # pub_visual_edge_msg.publish( nap_visual_edge_msg )


    # publish_time( pub_time_publish, 1000.*(time.time() - startPublish) )
    #-------------------------------- END  -----------------------------#


print 'Quit...!'
if BASE__DUMP is None:
    print 'BASE__DUMP param is None, indicating, NO_DEBUG'
    print 'Not writing out debug information.'
    quit()

print 'Writing ', BASE__DUMP+'/S_word.npy'
print 'Writing ', BASE__DUMP+'/S_timestamp.npy'
print 'Writing ', BASE__DUMP+'/S_thumbnail.npy'
print 'Writing ', BASE__DUMP+'/S_thumbnail_lut.npy'
print 'Writing ', BASE__DUMP+'/S_thumbnail_lut_raw.npy'

# TODO: write these data only if variable exisit. use 1-line if here.
np.save( BASE__DUMP+'/S_word.npy', S_word[0:loop_index+1] )
np.save( BASE__DUMP+'/S_timestamp.npy', S_timestamp[0:loop_index+1] )
np.save( BASE__DUMP+'/S_thumbnail.npy', np.array(S_thumbnail) )
np.save( BASE__DUMP+'/S_thumbnail_lut.npy', np.array(S_lut) )
np.save( BASE__DUMP+'/S_thumbnail_lut_raw.npy', np.array(S_lut_raw) )

if place_mod.im_queue_full_res is not None:
    print 'Writing ', BASE__DUMP+'/S_thumbnail_full_res.npy'
    np.save( BASE__DUMP+'/S_thumbnail_full_res.npy', np.array(S_thumbnail_full_res) )
else:
    print 'Not writing full res images'


print 'Writing Loop Candidates : ', BASE__DUMP+'/loop_candidates.csv'
np.savetxt( BASE__DUMP+'/loop_candidates.csv', loop_candidates, delimiter=',', comments='NAP loop_candidates' )
print 'Writing Loop Candidates : ', BASE__DUMP+'/loop_candidates2.csv'
np.savetxt( BASE__DUMP+'/loop_candidates2.csv', loop_candidates2, delimiter=',', comments='NAP loop_candidates picked' )


print 'Writing as pickle, FeatureFactory'
feature_factory.dump_to_file( BASE__DUMP+'/FeatureFactory')

if geometric_verify_log_fp is not None:
    print 'Writing file: ', BASE__DUMP+ '/geometric_verify.log'
    geometric_verify_log_fp.close()