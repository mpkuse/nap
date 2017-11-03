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
    for feature matching a) op_mode=10. It is easily matchable by pose-graph-opt node from DB features.
    b) op_mode=20. Forcefully matched features given out (gms_guided_match2).
    c) op_mode=30. 3way match given out.

    The descriptor comparison can be in either i) Brute force or ii) Using FAISS
    (product quantization). The matchings can be done with either daisy or gms-matcher.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 3rd Apr, 2017
        Edition : 2 (of nap_time_node.py)
        Edition : 4 (3rd Nov, 2017)
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

from PlaceRecognitionNetvlad import PlaceRecognitionNetvlad
from FastPlotter import FastPlotter

from GeometricVerification import GeometricVerification

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
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_b20_resnet/model-3750' # trained similar to above but with a resnet neural net
# Dont forget to load the eigen values, eigen vectors and mean


# PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/netvlad_k48/db2/siamese_dimred/model-400' #PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'

PARAM_NETVLAD_WORD_DIM = 16384#12288 # If these are not compatible with tensorfloaw model files program will fail
# PARAM_NETVLAD_CHAR_DIM = 256


INPUT_IMAGE_TOPIC = '/semi_keyframes' #this is t be used for launch
PARAM_CALLBACK_SKIP = 2

PARAM_FPS = 25


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

#--- Geometry and Matching ---#
def match3way_daisy( curr_im, prev_im, curr_m_im,    __lut_curr_im, __lut_prev_im ):
    """ Gives out 3 way matching 3 3xN matrix (i think)"""
    DEBUG = True
    # Step-1: Compute dense matches between curr and prev --> SetA
    VV.set_im( curr_im, prev_im )
    VV.set_im_lut_raw( S_lut_raw[i_curr], S_lut_raw[i_prev] )

    pts_curr, pts_prev, mask_c_p = VV.daisy_dense_matches()
    xcanvas_c_p = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask_c_p)
    if DEBUG:
        fname = '/home/mpkuse/Desktop/a/drag_nap/%d.jpg' %(loop_index)
        print 'Write(match3way_daisy) : ', fname
        cv2.imwrite( fname, xcanvas_c_p )


    # Step-2: Match expansion
    _pts_curr_m = VV.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p, curr_m_im  )
    masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
    masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )

    gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )
    if DEBUG:
        fname = '/home/mpkuse/Desktop/a/drag_nap/%d_3way.jpg' %(loop_index)
        print 'Write(match3way_daisy) : ', fname
        cv2.imwrite(fname, gridd )

    assert( len(masked_pts_curr) == len(masked_pts_prev) )
    assert( len(masked_pts_curr) == len(_pts_curr_m) )

    return np.array(masked_pts_curr) ,np.array( masked_pts_prev ), np.array(_pts_curr_m )




#---           END        ---#

########### Init PlaceRecognitionNetvlad ##########
place_mod = PlaceRecognitionNetvlad(\
                                    PARAM_MODEL,\
                                    PARAM_CALLBACK_SKIP=PARAM_CALLBACK_SKIP,\
                                    PARAM_K = 64
                                    )


############# GEOMETRIC VERIFICATION #################
VV = GeometricVerification()



################### Init Node and Topics ############
rospy.init_node( 'nap_geom_node', log_level=rospy.INFO )
rospy.Subscriber( INPUT_IMAGE_TOPIC, Image, place_mod.callback_image )
rospy.loginfo( 'Subscribed to '+INPUT_IMAGE_TOPIC )

# raw edges
pub_edge_msg = rospy.Publisher( '/raw_graph_edge', NapMsg, queue_size=1000 )
rospy.loginfo( 'Publish to /raw_graph_edge' )

# raw visual edges
pub_visual_edge_msg = rospy.Publisher( '/raw_graph_visual_edge', NapVisualEdgeMsg, queue_size=1000 )
pub_visual_edge_cluster_assgn_msg = rospy.Publisher( '/raw_graph_visual_edge_cluster_assgn', NapVisualEdgeMsg, queue_size=1000 )
rospy.loginfo( 'Publish to /raw_graph_visual_edge' )


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

pub_cluster_assgn_falsecolormap = rospy.Publisher( '/debug/cluster_assignment', Image, queue_size=10 )
rospy.loginfo( 'Publish to /debug/cluster_assignment')

pub_cluster_assgn_raw = rospy.Publisher( '/nap/cluster_assignment', Image, queue_size=10 )


#################### Init Plotter #####################
plotter = FastPlotter(n=3)
plotter.setRange( 0, yRange=[0,1] )
plotter.setRange( 1, yRange=[0,1] )
plotter.setRange( 2, yRange=[0,1] )


##################### Main Loop ########################
rate = rospy.Rate(PARAM_FPS)

# S_word = np.zeros( (25000,8192) ) #word
# S_word = np.zeros( (25000,PARAM_NETVLAD_WORD_DIM) ) #word-48
S_word = []

S_timestamp = [] #np.zeros( 25000, dtype=rospy.Time )
S_thumbnail = []
S_thumbnail_full_res = []
S_lut = [] #only for debug, the cluster assgnment image (list of false color)
S_lut_raw = [] # raw 1-channel cluster assignment
loop_index = -1
startTotalTime = time.time()


loop_candidates = []

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

    if True: #Set this to true to publish false color Assgn_image and publish it.
        lut = colorLUT.lut( place_mod.Assgn_matrix[0,:,:] )
        S_lut.append( lut )
        S_lut_raw.append( place_mod.Assgn_matrix[0,:,:]  )
        publish_image( pub_cluster_assgn_falsecolormap, lut, t=im_raw_timestamp )
        publish_image( pub_cluster_assgn_raw, place_mod.Assgn_matrix[0,:,:].astype('uint8'), t=im_raw_timestamp )


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
        nMatches = 25
        nInliers = np.random.randint(10, 40) #25
        print 'Setting random inlier count'
        # Another possibility is to not do any verification here. And randomly choose 1 pair for 3way matching.


        # Record this match in a file
        # print '%d<--->%d' %(L-1,aT)
        # note, do not use loop_candidates, it is for file-logging. Instead use `_now_edges`
        loop_candidates.append( [L-1, aT, sim_scores_logistic[aT], nMatches, nInliers] ) #here was L in original

        if nInliers > 0:
            _now_edges.append( (L-1, aT, nInliers) )



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
    i_inliers = _now_edges[pick][2]

    nap_msg = make_nap_msg( i_curr, i_prev, (0.6,1.0,0.6) )
    nap_msg.n_sparse_matches = i_inliers #Not required

    ###### Decide op_mode ######
    decided_op_mode = 29;
    ############################

    if decided_op_mode == 29:
        # 3-way matching
        nap_msg.op_mode = 29


        # Step-0 : Collect Images
        curr_im = S_thumbnail[i_curr].astype('uint8')
        prev_im = S_thumbnail[i_prev].astype('uint8')
        curr_m_im = S_thumbnail[i_curr-1].astype('uint8')
        t_curr = S_timestamp[i_curr]
        t_prev = S_timestamp[i_prev]
        t_curr_m = S_timestamp[i_curr-1]
        #Imp Note : curr-1 is actually curr-PARAM_CALLBACK_SKIP. However posegraph opt will have all the keyframes. Best practice I think is to also put 3 timestamps of the images used.

        __lut_curr_im = S_lut_raw[i_curr]
        __lut_prev_im = S_lut_raw[i_prev]


        #
        # Step-1 : Daisy
        pts3_curr, pts3_prev, pts3_currm = match3way_daisy(curr_im, prev_im, curr_m_im,    __lut_curr_im, __lut_prev_im  )
        # print 'pts3_curr.shape', pts3_curr.shape    # Nx2
        # print 'pts3_prev.shape', pts3_prev.shape    # Nx2
        # print 'pts3_currm.shape', pts3_currm.shape  # Nx2
        # TODO consider returning a 2xN numpy matrix instead of list

        #
        # Step-2 : Set into nap_msg (complete the nap msg with 3 timestamps and co-ordinates)
        nap_msg.t_curr = t_curr
        nap_msg.t_prev = t_prev
        nap_msg.t_curr_m = t_curr_m

        for ji in range( len(pts3_curr) ):
            pt_curr = pts3_curr[ji]
            pt_prev = pts3_prev[ji]
            pt_curr_m = pts3_currm[ji]

            nap_msg.curr.append(   Point32(pt_curr[0], pt_curr[1],-1) )
            nap_msg.prev.append(   Point32(pt_prev[0], pt_prev[1],-1) )
            nap_msg.curr_m.append( Point32(pt_curr_m[0], pt_curr_m[1],-1) )



    if decided_op_mode == 10:
        # Nothing, No co-ordinates to pass
        nap_msg.op_mode = 10
        pass

    if decided_op_mode == 20:
        # Do guided matching using tracked features and prev image.
        # Finally the msg will contain 2-way guided match between curr and prev in normalized-image co-ordinates
        nap_msg.op_mode = 20
        pass



    pub_edge_msg.publish( nap_msg )


    # Comment following 2 lines to not debug-publish loop-candidate
    nap_visual_edge_msg = make_nap_visual_msg( i_curr, i_prev, "%d,%d" %(i_curr,i_inliers), str(i_prev) )
    pub_visual_edge_msg.publish( nap_visual_edge_msg )





    #
    # # Old - This seem unneccesarily complicated code.
    # if len(_now_edges) > 0:
    #
    #     sorted_by_inlier_count = sorted( _now_edges, key=lambda tup: tup[2] )
    #     for each_edge in sorted_by_inlier_count[-1:]: #publish only top-1 candidate
    #     # for each_edge in sorted_by_inlier_count: #publish all candidates
    #         i_curr = each_edge[0]
    #         i_prev = each_edge[1]
    #         i_inliers = each_edge[2]
    #
    #
    #         nap_msg = make_nap_msg( i_curr, i_prev, (0.6,1.0,0.6) )
    #         nap_msg.n_sparse_matches = i_inliers
    #
    #         #TODO: If nInliners more than 20 simply publish edge.
    #         #       If nInliers less than 20 attempt a 3-way match. Fill in the 3-way match in nap_msg
    #         if i_inliers < 200: #later make it 20
    #             #
    #             # Do 3-Way Matching
    #             #
    #             print tcol.OKBLUE, S_thumbnail[i_curr].astype('uint8').shape
    #             print S_thumbnail[i_prev].astype('uint8').shape, tcol.ENDC
    #             curr_im = S_thumbnail[i_curr].astype('uint8')
    #             prev_im = S_thumbnail[i_prev].astype('uint8')
    #             curr_m_im = S_thumbnail[i_curr-1].astype('uint8')
    #             t_curr = S_timestamp[i_curr]
    #             t_prev = S_timestamp[i_prev]
    #             t_curr_m = S_timestamp[i_curr-1]
    #             #Imp Note : curr-1 is actually curr-PARAM_CALLBACK_SKIP. However posegraph opt will have all the keyframes. Best practice I think is to also put 3 timestamps of the images used.
    #
    #             # Step-1: Compute dense matches between curr and prev --> SetA
    #             VV.set_im( curr_im, prev_im )
    #             VV.set_im_lut_raw( S_lut_raw[i_curr], S_lut_raw[i_prev] )
    #
    #
    #             pts_curr, pts_prev, mask_c_p = VV.daisy_dense_matches()
    #             # xcanvas_c_p = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask_c_p)
    #             # print 'Write : ', '/home/mpkuse/Desktop/a/%d.jpg' %(loop_index)
    #             # cv2.imwrite( '/home/mpkuse/Desktop/a/%d.jpg' %(loop_index), xcanvas_c_p )
    #
    #
    #             # Step-2: Match expansion
    #             # TODO: Before expanding matches, try cv2.correctMatches() which minimizes the reprojection errors. Try it out, might help reduce false matches even more based on reprojection.
    #             _pts_curr_m = VV.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p, curr_m_im  )
    #
    #             masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
    #             masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
    #
    #             # gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )
    #             # print 'Write : ',  '/home/mpkuse/Desktop/a/%d_3way.jpg' %(loop_index)
    #             # cv2.imwrite( '/home/mpkuse/Desktop/a/%d_3way.jpg' %(loop_index), gridd )
    #
    #
    #
    #             # Fill the nap message with 3-way matches.
    #             # Relative pose was not computed here on purpose. This was because to Triangulate,
    #             # we need SLAM pose between curr and curr-1. So instead of subscribing it here, we do it in pose-graph-opt node
    #             for ji in range( len(_pts_curr_m) ):
    #                 pt_curr = masked_pts_curr[ji]
    #                 pt_prev = masked_pts_prev[ji]
    #                 pt_curr_m = _pts_curr_m[ji]
    #
    #                 nap_msg.curr.append(   Point32(pt_curr[0], pt_curr[1],-1) )
    #                 nap_msg.prev.append(   Point32(pt_prev[0], pt_prev[1],-1) )
    #                 nap_msg.curr_m.append( Point32(pt_curr_m[0], pt_curr_m[1],-1) )
    #
    #             nap_msg.t_curr = t_curr
    #             nap_msg.t_prev = t_prev
    #             nap_msg.t_curr_m = t_curr_m
    #
    #             nap_msg.op_mode = 29 #Signal that the msg contains 3-way match
    #
    #
    #         else:
    #             nap_msg.op_mode = 10 #Signal that the msg does not contain 3-way, neither does it contain any matching data.
    #         pub_edge_msg.publish( nap_msg )
    #
    #
    #         # Comment following 2 lines to not debug-publish loop-candidate
    #         nap_visual_edge_msg = make_nap_visual_msg( i_curr, i_prev, "%d,%d" %(i_curr,i_inliers), str(i_prev) )
    #         pub_visual_edge_msg.publish( nap_visual_edge_msg )
    #


    # TODO Determination of edge using char instead of word TODO`sim_scores_logistic`



    publish_time( pub_time_publish, 1000.*(time.time() - startPublish) )
    #-------------------------------- END  -----------------------------#


print 'Quit...!'
BASE__DUMP = '/home/mpkuse/Desktop/a/drag_nap'
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