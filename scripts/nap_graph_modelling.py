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

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 3rd Apr, 2017
        Edition : 2 (of nap_time_node.py)
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


# Consider removing, not in use
from GraphMerging import Node
from GraphMerging import get_gid
from GraphMerging import NonSeqMergeThread

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

# INPUT_IMAGE_TOPIC = '/camera/image_raw' #point grey
# INPUT_IMAGE_TOPIC = '/dji_sdk/image_raw_resized'
# INPUT_IMAGE_TOPIC = '/youtube_camera/image'
# INPUT_IMAGE_TOPIC = '/android/image'
# INPUT_IMAGE_TOPIC = '/mv_29900616/image_raw'
INPUT_IMAGE_TOPIC = '/semi_keyframes' #this is t be used for launch
PARAM_CALLBACK_SKIP = 2

PARAM_FPS = 25


#### Loading the PCA matrix. In particular top 600 precomputed eigen vectors and eig values
print 'Loading the precomputed top eigen values, eigen vectors and mean'
ipca_uu = np.load( PKG_PATH+'/tf.logs/netvlad_k64_tokyoTM/db_xl/eig_val_top600.npy') #currently not in use. TODO, remove this as in the future we plan to use FAISS for product quantization to improve scalability
ipca_vv = np.load( PKG_PATH+'/tf.logs/netvlad_k64_tokyoTM/db_xl/eig_vec_top600.npy')
ipca_learned_mean = np.load( PKG_PATH+'/tf.logs/netvlad_k64_tokyoTM/db_xl/M_mean.npy')
ipca_P = np.dot( np.diag( ipca_uu[-50:]), np.transpose(ipca_vv[:,-50:]) )
#### END PCA

def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )

def publish_image( PUB, cv_image, t=None ):
    msg_frame = CvBridge().cv2_to_imgmsg( cv_image, "bgr8" )
    if t is not None:
        msg_frame.header.stamp = t
    PUB.publish( msg_frame )



#--- Geometric Verification Functions ---#
# Before publishing do a geometric verification with essential matrix
def lowe_ratio_test( kp1, kp2, matches_org ):
    """ Input keypoints and matches. Compute keypoints like : kp1, des1 = orb.detectAndCompute(im1, None)
    Compute matches like : matches_org = flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2) #find 2 nearest neighbours

    Returns 2 Nx2 arrays. These arrays contains the correspondences in images. """
    __pt1 = []
    __pt2 = []
    for m in matches_org:
        if m[0].distance < 0.8 * m[1].distance: #good match
            # print 'G', m[0].trainIdx, 'th keypoint from im1 <---->', m[0].queryIdx, 'th keypoint from im2'

            _pt1 = np.array( kp1[ m[0].queryIdx ].pt ) #co-ordinate from 1st img
            _pt2 = np.array( kp2[ m[0].trainIdx ].pt )#co-ordinate from 2nd img corresponding
            #now _pt1 and _pt2 are corresponding co-oridnates

            __pt1.append( np.array(_pt1) )
            __pt2.append( np.array(_pt2) )

            # cv2.circle( im1, (int(_pt1[0]), int(_pt1[1])), 3, (0,255,0), -1 )
            # cv2.circle( im2, (int(_pt2[0]), int(_pt2[1])), 3, (0,255,0), -1 )

            # cv2.circle( im1, _pt1, 3, (0,255,0), -1 )
            # cv2.circle( im2, _pt2, 3, (0,255,0), -1 )
            # cv2.imshow( 'im1', im1 )
            # cv2.imshow( 'im2', im2 )
            # cv2.waitKey(10)
    __pt1 = np.array( __pt1)
    __pt2 = np.array( __pt2)
    return __pt1, __pt2

def do_geometric_verification( i_L, i_aT ):
    matches_org = flann.knnMatch(S_kp_desc[i_L].astype('float32'),S_kp_desc[i_aT].astype('float32'),k=2) #find 2 nearest neighbours
    __pt1, __pt2 = lowe_ratio_test( S_kp[i_L], S_kp[i_aT], matches_org )

    nMatches = __pt1.shape[0]
    nInliers = 0

    # Do essential Matrix verification only if u have more than 20 matches
    if nMatches > 20:
        E, mask = cv2.findEssentialMat( __pt1, __pt2 )
        if mask is not None:
            nInliers = mask.sum()
    return nMatches, nInliers
#--- END Geometric verification functions ---#


#--- Nap Msg Creation ---#
def make_nap_msg( i_curr, i_prev, edge_color=None):

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

########### Init PlaceRecognitionNetvlad ##########
place_mod = PlaceRecognitionNetvlad(\
                                    PARAM_MODEL,\
                                    PARAM_CALLBACK_SKIP=PARAM_CALLBACK_SKIP,\
                                    PARAM_K = 64
                                    )

# place_mod.load_siamese_dim_red_module( PARAM_MODEL_DIM_RED, PARAM_NETVLAD_WORD_DIM, 1024, PARAM_NETVLAD_CHAR_DIM  )

############# GEOMETRIC VERIFICATION #################
VV = GeometricVerification()


# Consider removal. All this is taken care off in class GeometricVerification
# Key point detector (geometric verification)
orb = cv2.ORB_create()

#flann
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

S_kp = [] #ORB keypoints of every image
S_kp_desc = [] #descriptors of every image

################### Init Node and Topics ############
rospy.init_node( 'nap_geom_node', log_level=rospy.INFO )
rospy.Subscriber( INPUT_IMAGE_TOPIC, Image, place_mod.callback_image )
rospy.loginfo( 'Subscribed to '+INPUT_IMAGE_TOPIC )

# raw nodes
pub_node_msg   = rospy.Publisher( '/raw_graph_node', NapNodeMsg, queue_size=1000 )
rospy.loginfo( 'Publish to /raw_graph_node' )

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

# Images - debug
pub_cluster_assgn_falsecolormap = rospy.Publisher( '/debug/cluster_assignment', Image, queue_size=10 )
rospy.loginfo( 'Publish to /debug/cluster_assignment')


#################### Init Plotter #####################
plotter = FastPlotter(n=5)
plotter.setRange( 0, yRange=[0,1] )
plotter.setRange( 1, yRange=[0,1] )
plotter.setRange( 2, yRange=[0,1] )
plotter.setRange( 3, yRange=[0,1] )
plotter.setRange( 4, yRange=[0,1] )
from ColorLUT import ColorLUT
colorLUT = ColorLUT()


##################### Main Loop ########################
rate = rospy.Rate(PARAM_FPS)
# S_char = np.zeros( (25000,PARAM_NETVLAD_CHAR_DIM) ) #char
S_char = []

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

#for graph based merge
all_nodes = [] #list of all nodes
internal_e = {} #dict of internal energy at each node key-frame (1st)
n_components = {} #dict of num of components in each segment
uniq_color_list = np.genfromtxt( PKG_PATH+'/scripts/DistinctColorsUChar.csv', delimiter=',')

loop_candidates = []

while not rospy.is_shutdown():
    rate.sleep()

    publish_time( pub_time_total_loop, 1000.*(time.time() - startTotalTime) ) #this has been put like to use startTotalTime from prev iteration
    startTotalTime = time.time()
    #------------------- Queue book-keeping---------------------#
    rospy.loginfo( '---Queue Size : %d, %d' %( place_mod.im_queue.qsize(), place_mod.im_timestamp_queue.qsize()) )
    if place_mod.im_queue_full_res is not None:
        rospy.loginfo( '---Full Res Queue : %d' %(place_mod.im_queue_full_res.qsize())  )
    if place_mod.im_queue.qsize() < 1 and place_mod.im_timestamp_queue.qsize() < 1:
        rospy.logdebug( 'Empty Queue...Waiting' )
        continue
    publish_time( pub_time_queue_size, place_mod.im_queue.qsize() )


    # Get Image & TimeStamp
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



    # PCA - As suggested in
    # Jegou, Herve, and Ondrej Chum. "Negative evidences and co-occurences in image retrieval: The benefit of PCA and whitening." Computer Vision-ECCV 2012 (2012): 774-787.
    d_CHAR = np.dot( ipca_P, d_WORD-ipca_learned_mean )
    d_CHAR = np.dot( ipca_P, d_WORD )
    d_CHAR = d_CHAR / np.linalg.norm( d_CHAR )


    publish_time( pub_time_desc_comp, 1000.*(time.time() - startDescComp) )
    rospy.logdebug( 'Word Shape : %s' %(d_WORD.shape) )
    # rospy.logdebug( 'Char Shape : %s' %(d_CHAR.shape) )
    print 'Main Thread : ', len(all_nodes)

    if True: #Set this to true to false color Assgn_image and publish it.
        lut = colorLUT.lut( place_mod.Assgn_matrix[0,:,:] )
        S_lut.append( lut )
        S_lut_raw.append( place_mod.Assgn_matrix[0,:,:]  )
        publish_image( pub_cluster_assgn_falsecolormap, lut, t=im_raw_timestamp )

    #---------------------------- END  -----------------------------#

    #------------------- Storage & Score Computation ----------------#
    rospy.logdebug( 'Storage & Score Computation' )
    startScoreCompTime = time.time()
    # S_char[loop_index,:] = d_CHAR # OLD Way of pri-allocating a large array, worth considering
    # S_word[loop_index,:] = d_WORD
    # S_timestamp[loop_index] = im_raw_timestamp

    #TODO: With the newest nn scheme, dot product of all is no more needed however S_word need
    # to be stored along with ORB features as well for geometric verification.
    kp1, des1 = orb.detectAndCompute(im_raw, None)
    S_kp.append( kp1 )
    S_kp_desc.append( des1 )

    S_char.append( d_CHAR )
    S_word.append( d_WORD )
    S_timestamp.append( im_raw_timestamp )
    # S_thumbnail.append(  cv2.resize( im_raw.astype('uint8'), (128,96) ) )#, fx=0.2, fy=0.2 ) )
    S_thumbnail.append(  cv2.resize( im_raw.astype('uint8'), (320,240) ) )#, fx=0.2, fy=0.2 ) )

    if place_mod.im_queue_full_res is not None:
        S_thumbnail_full_res.append( im_raw_full_res.astype('uint8') )

    # DOT = np.dot( S_char[0:loop_index+1,:], S_char[loop_index,:] )
    # DOT_word = np.dot( S_word[0:loop_index+1,:], S_word[loop_index,:] )
    # DOT = np.dot( S_char[0:loop_index+1], np.transpose(S_char[loop_index]) )
    DOT_word = np.dot( S_word[0:loop_index+1], np.transpose(S_word[loop_index]) )
    DOT_char = np.dot( S_char[0:loop_index+1], np.transpose(S_char[loop_index]) )
    DOT_char = np.convolve( DOT_char, np.array([1,1,1,1,1,1,1])/7., 'same' )

    sim_scores = np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) ) #minimum is added to ensure dot product doesnt go beyond 1.0 as it sometimes happens because of numerical issues, which inturn causes issues with sqrt
    sim_scores_char =  np.sqrt( 1.0 - np.minimum(1.0, DOT_char ) )

    sim_scores_logistic = place_mod.logistic( sim_scores ) #convert the raw Similarity scores above to likelihoods
    sim_scores_logistic_char = place_mod.logistic( sim_scores_char )

    publish_time( pub_time_dot_scoring, 1000.*(time.time() - startScoreCompTime) )
    #---------------------------- END  -----------------------------#


    # --------- PLOT Sim Score of current wrt all prev ---------#
    # Plot sim_scores
    plotter.set_data( 0, range(len(DOT_word)), DOT_word, title="DOT_word"  )
    plotter.set_data( 1, range(len(sim_scores)), sim_scores, title="sim_scores = sqrt( 1-dot )"  )
    plotter.set_data( 2, range(len(sim_scores_logistic)), sim_scores_logistic, title="sim_scores_logistic"  )
    plotter.set_data( 3, range(len(DOT_char)), DOT_char, title="DOT_char"  )
    plotter.set_data( 4, range(len(sim_scores_logistic_char)), sim_scores_logistic_char, title="sim_scores_logistic_char"  )
    # if len(sim_scores_logistic) > 50:
    #     c_sum = np.cumsum(sim_scores_logistic[:-49])
    #     c_sum /= c_sum[-1]
    #     plotter.set_data( 3, range(len(c_sum)), c_sum, title="cumsum(logisic)"  )

    plotter.spin()

    # #----------------------- Sequencial Merge ----------------------#
    # timeSeqMerge = time.time()
    # if loop_index == 0:
    #     all_nodes.append( Node(uid=0) )
    #     current_component_start = [0]
    #     current_component_end = []
    #     inserted_lsh_k1 = []
    #     inserted_lsh_k2 = []
    #
    #     from sklearn.neighbors import LSHForest
    #     lshf = LSHForest( random_state=42, n_estimators=50 )
    #     print 'Init LSHForest'
    #
    # #compute dot product cost, note this step needs say 50 prev S_Words only
    # else:
    #     window_size = 20
    #     li = loop_index
    #     g_DOT_word = np.dot( S_word[max(0,li-window_size):li], np.transpose(S_word[li]) )
    #     g_DOT_index = range(max(0,li-window_size),li)
    #     g_sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, g_DOT_word ) )
    #     g_wt = 1.0 - place_mod.logistic( g_sim_scores ) #measure of dis-similarity. 0 means very similar.
    #
    #
    #
    #     # Next is code for sequencial merge
    #     all_nodes.append( Node(uid=loop_index) )
    #     for j_ind,w in enumerate(g_wt):
    #         if w>0.3:
    #             continue
    #
    #         gid_i = get_gid( all_nodes[li] )
    #         e_i = 0
    #
    #         j = g_DOT_index[j_ind]
    #         gid_j = get_gid( all_nodes[j] )
    #         e_j = internal_e[gid_j] if internal_e.has_key(gid_j) else 0.0
    #
    #         n_i = n_components[gid_i] if n_components.has_key(gid_i) else 1
    #         n_j = n_components[gid_j] if n_components.has_key(gid_j) else 1
    #
    #         kappa = 0.38
    #         # print 'gid_i=%3d gid_j=%3d' %(gid_i, gid_j)
    #         # print 'w=%4.4f, ei=%4.4f, ej=%4.4f' %(w, e_i+kappa/n_i, e_j+kappa/n_j )
    #         if w < min(e_i+kappa/n_i, e_j+kappa/n_j):
    #             internal_e[gid_j] = w
    #             n_components[gid_j] = n_j + 1
    #             all_nodes[li].parent = all_nodes[j]
    #
    #     # If new segment has started
    #     if all_nodes[loop_index].parent is None:
    #         current_component_start.append(loop_index)
    #         current_component_end.append( loop_index )
    #         # print 'current_component_start', current_component_start
    #         # print 'current_component_end', current_component_end
    #         print 'New Components starts at loop_index=%d' %(loop_index)
    #
    #
    #         # a. Compute centroid of just-passed components
    #         #    ie. k1:=current_component_start[-2]
    #         #        k2:=current_component_start[-1]
    #
    #         # b. Insert centroid into LSH index
    #
    #         # LSH index initialized at 0th frame
    #         _k1 = current_component_start[-2]
    #         _k2 = current_component_end[-1]
    #
    #
    #         if (_k2 - _k1 > 15) or _k1 < 15:
    #             # Only insert this component if it is atleast of lenth 15
    #             inserted_lsh_k1.append(_k1)
    #             inserted_lsh_k2.append(_k2)
    #             print tcol.OKGREEN, 'Insert centroid( %d-->%d ) into LSHForest' %(_k1, _k2), tcol.ENDC
    #             centeroid_of_island = np.array(S_word[_k1:_k2]).mean(axis=0)
    #             norm_of_centeroid = np.linalg.norm( centeroid_of_island )
    #             # centeroid_of_island = centeroid_of_island / norm_of_centeroid
    #             lshf.partial_fit( centeroid_of_island.reshape(1,-1) )
    #
    #
    #
    #     elif len(current_component_start)>2: #bussiness as usual (ie. do nearest neighbour query ~30ms )
    #         # Do query on the LSH. Usually about 30ms but can be imporved 7x (4ms) using the newest NIPS2015 paper.
    #         # But currently using an old implementation from sklearn as FALCONN does not have a dynamic insert into it
    #         nn_dis, nn_indx = lshf.radius_neighbors( S_word[loop_index].reshape(1,-1), 0.05 )
    #         # print 'LSH_NN : ', nn_dis, nn_indx
    #         print tcol.OKBLUE,
    #         for _nn_c, _nn_indx in enumerate(nn_indx[0]):
    #             if loop_index - inserted_lsh_k1[_nn_indx] > 100 :
    #                 __a = loop_index
    #                 __b = inserted_lsh_k1[_nn_indx]
    #                 __c = inserted_lsh_k2[_nn_indx]
    #                 print '%2d] (%4d) <--> (%4d,%4d) : %0.4f' %(_nn_c,__a, __b, __c, nn_dis[0][_nn_c])
    #                 loop_candidates.append( [__a, __b, __c, nn_dis[0][_nn_c] ] )
    #
    #
    #                 # Publish Edge
    #                 nap_msg = NapMsg()
    #                 nap_msg.c_timestamp = rospy.Time.from_sec( float(S_timestamp[__a].to_sec()) )
    #                 nap_msg.prev_timestamp = rospy.Time.from_sec( float(S_timestamp[__b].to_sec()) )
    #                 # nap_msg.goodness = sim_scores_logistic[aT]
    #                 nap_msg.color_r = 1.0
    #                 nap_msg.color_g = 1.0
    #                 nap_msg.color_b = 1.0
    #                 pub_edge_msg.publish( nap_msg )
    #         print tcol.ENDC
    #
    #
    #
    # publish_time( pub_time_seq_merging, 1000.*(time.time() - timeSeqMerge) )
    # #---------------------------- END  -----------------------------#

    # # Plot sim_scores
    # plotter.set_data( 0, range(len(DOT)), DOT, title="DOT"  )
    # plotter.set_data( 1, range(len(DOT_word)), DOT_word, title="DOT_word"  )
    # plotter.set_data( 2, range(len(sim_scores)), sim_scores, title="sim_scores"  )
    # plotter.set_data( 3, range(len(sim_scores_logistic)), sim_scores_logistic, title="sim_scores_logistic"  )
    # plotter.spin()

    #-------------- Graph Build/Publish Node Msg  ------------------#
    # print 'Added Node : ', loop_index

    node_msg = NapNodeMsg()
    # node_msg.node_timestamp = rospy.Time.from_sec( float(im_raw_timestamp.to_sec()) )
    node_msg.node_timestamp = im_raw_timestamp
    node_msg.node_label = str(loop_index)
    # n_gid = get_gid(all_nodes[loop_index])
    node_msg.node_label_str = str(loop_index)
    #
    # n_colors = uniq_color_list.shape[0]
    node_msg.color_r = 1.#uniq_color_list[n_gid%n_colors][0]
    node_msg.color_g = 1.#uniq_color_list[n_gid%n_colors][1]
    node_msg.color_b = 1.#uniq_color_list[n_gid%n_colors][2]
    pub_node_msg.publish( node_msg )



    #------------------------------ END  -------------------------------#

    #----------------- Grid Filter / Temporal Fusion -------------------#
    if loop_index < 2: #let data accumulate
        continue
    #------------------------------ END  -------------------------------#



    #------------- Publish Colocation (on loop closure ) ---------------#
    #------------------------------ Add Edge ---------------------------#
    L = loop_index #alias
    startGeometricVerification = time.time()
    # Determination of edge using `sim_scores_logistic`
    argT = np.where( sim_scores_logistic[1:L] > 0.54 )
    if len(argT ) < 1:
        continue

    _now_edges = []
    for aT in argT[0]: #Iterate through each match and collect candidates
        # Avoid matches from near current
        if float(S_timestamp[L].to_sec() - S_timestamp[aT].to_sec())<10.  or aT < 5:
            continue

        # nMatches, nInliers = do_geometric_verification( L-1, aT)
        # Do simple verification using,  S_thumbnail[i_curr] and S_thumbnail[i_prev] and class GeometricVerification
        VV.set_im( S_thumbnail[L], S_thumbnail[aT] )
        nMatches, nInliers = VV.simple_verify(features='orb')
        # nMatches = 25
        # nInliers = 25
        # Another possibility is to not do any verification here. And randomly choose 1 pair for 3way matching.


        # Record this match in a file
        # print '%d<--->%d' %(L-1,aT)
        loop_candidates.append( [L, aT, sim_scores_logistic[aT], nMatches, nInliers] )

        if nInliers > 0:
            _now_edges.append( (L-1, aT, nInliers) )


    publish_time( pub_time_geometric_verification, 1000.*(time.time() - startGeometricVerification) )

    startPublish = time.time()
    if len(_now_edges) > 0:

        sorted_by_inlier_count = sorted( _now_edges, key=lambda tup: tup[2] )
        for each_edge in sorted_by_inlier_count[-1:]: #publish only top-1 candidate
        # for each_edge in sorted_by_inlier_count: #publish all candidates
            i_curr = each_edge[0]
            i_prev = each_edge[1]
            i_inliers = each_edge[2]


            nap_msg = make_nap_msg( i_curr, i_prev, (0.6,1.0,0.6) )
            nap_msg.n_sparse_matches = i_inliers

            #TODO: If nInliners more than 20 simply publish edge.
            #       If nInliers less than 20 attempt a 3-way match. Fill in the 3-way match in nap_msg
            if i_inliers < 200: #later make it 20
                #
                # Do 3-Way Matching
                #
                print tcol.OKBLUE, S_thumbnail[i_curr].astype('uint8').shape
                print S_thumbnail[i_prev].astype('uint8').shape, tcol.ENDC
                curr_im = S_thumbnail[i_curr].astype('uint8')
                prev_im = S_thumbnail[i_prev].astype('uint8')
                curr_m_im = S_thumbnail[i_curr-1].astype('uint8')
                t_curr = S_timestamp[i_curr]
                t_prev = S_timestamp[i_prev]
                t_curr_m = S_timestamp[i_curr-1]
                #Imp Note : curr-1 is actually curr-PARAM_CALLBACK_SKIP. However posegraph opt will have all the keyframes. Best practice I think is to also put 3 timestamps of the images used.

                # Step-1: Compute dense matches between curr and prev --> SetA
                VV.set_im( curr_im, prev_im )
                VV.set_im_lut_raw( S_lut_raw[i_curr], S_lut_raw[i_prev] )


                pts_curr, pts_prev, mask_c_p = VV.daisy_dense_matches()
                xcanvas_c_p = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask_c_p)

                # print 'Write : ', '/home/mpkuse/Desktop/a/%d.jpg' %(loop_index)
                # cv2.imwrite( '/home/mpkuse/Desktop/a/%d.jpg' %(loop_index), xcanvas_c_p )


                # Step-2: Match expansion
                # TODO: Before expanding matches, try cv2.correctMatches() which minimizes the reprojection errors. Try it out, might help reduce false matches even more based on reprojection.
                _pts_curr_m = VV.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p, curr_m_im  )

                masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
                masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
                gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )

                # print 'Write : ',  '/home/mpkuse/Desktop/a/%d_3way.jpg' %(loop_index)
                # cv2.imwrite( '/home/mpkuse/Desktop/a/%d_3way.jpg' %(loop_index), gridd )



                # Fill the nap message with 3-way matches.
                # Relative pose was not computed here on purpose. This was because to Triangulate,
                # we need SLAM pose between curr and curr-1. So instead of subscribing it here, we do it in pose-graph-opt node
                for ji in range( len(_pts_curr_m) ):
                    pt_curr = masked_pts_curr[ji]
                    pt_prev = masked_pts_prev[ji]
                    pt_curr_m = _pts_curr_m[ji]

                    nap_msg.curr.append(   Point32(pt_curr[0], pt_curr[1],-1) )
                    nap_msg.prev.append(   Point32(pt_prev[0], pt_prev[1],-1) )
                    nap_msg.curr_m.append( Point32(pt_curr_m[0], pt_curr_m[1],-1) )

                nap_msg.t_curr = t_curr
                nap_msg.t_prev = t_prev
                nap_msg.t_curr_m = t_curr_m



            pub_edge_msg.publish( nap_msg )


            # Comment following 2 lines to not debug-publish loop-candidate
            nap_visual_edge_msg = make_nap_visual_msg( i_curr, i_prev, "%d,%d" %(i_curr,i_inliers), str(i_prev) )
            pub_visual_edge_msg.publish( nap_visual_edge_msg )



    # TODO Determination of edge using char instead of word TODO`sim_scores_logistic`



    publish_time( pub_time_publish, 1000.*(time.time() - startPublish) )
    #-------------------------------- END  -----------------------------#


# Save S_word, S_char, S_thumbnail
print 'Quit...!'
print 'Writing ', PKG_PATH+'/DUMP/S_word.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_char.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_timestamp.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_thumbnail.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_thumbnail_lut.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_thumbnail_lut_raw.npy'

# TODO: write these data only if variable exisit. use 1-line if here.
np.save( PKG_PATH+'/DUMP/S_word.npy', S_word[0:loop_index+1] )
np.save( PKG_PATH+'/DUMP/S_char.npy', S_char[0:loop_index+1] )
np.save( PKG_PATH+'/DUMP/S_timestamp.npy', S_timestamp[0:loop_index+1] )
np.save( PKG_PATH+'/DUMP/S_thumbnail.npy', np.array(S_thumbnail) )
np.save( PKG_PATH+'/DUMP/S_thumbnail_lut.npy', np.array(S_lut) )
np.save( PKG_PATH+'/DUMP/S_thumbnail_lut_raw.npy', np.array(S_lut_raw) )

if place_mod.im_queue_full_res is not None:
    print 'Writing ', PKG_PATH+'/DUMP/S_thumbnail_full_res.npy'
    np.save( PKG_PATH+'/DUMP/S_thumbnail_full_res.npy', np.array(S_thumbnail_full_res) )
else:
    print 'Not writing full res images'


print 'Writing Loop Candidates : ', PKG_PATH+'/DUMP/loop_candidates.csv'
np.savetxt( PKG_PATH+'/DUMP/loop_candidates.csv', loop_candidates, delimiter=',', comments='NAP loop_candidates' )
