#!/usr/bin/env python

"""
This is a full multiprocess implementation of `script/nap_robustdaisy_bf.py`.
At the core is the python multiprocessing package. For interprocess communication
we make use of multiprocessing.Manager.list(), .dict() and .Queue().
Everything is like a producer-consumer model.

Note that FeatureFactory and ImageReceiver have been changed to make them threadsafe.
Also changes in Neural Network factory. Image normalization is done in GPU now.
This is actually slightly different computation than we used while training.
Works fine, but ideally retaining is need!


A process for GPU
N processes for pose computation
A process for score computation
main process to publish and subscribes to input topics

- main process subscribes to key-frame images and builds up the queue Qt and Qi
- GPU process consumes the queue Qt, Qi and make the list() S_thumbnails, S_netvlad, S_timestamp
- score computation process does a score computation whenever new items are available in S_netvlad.
  This produces the queue Qd, Queue of putative candidates
- Multiple queues of cpu_process (pose computation) consumes Qd and
  produces nap_msg queues which inturn is published by main.


 TODO:
 - Get 3way matching working.
 - Still has issues of orphan process when the main is immediately closed.


----------------------------------------------------------------------------------------
Copy-pasting messages from nap_robustdaisy_bf.py which is the
single threaded version
----------------------------------------------------------------------------------------
        Subscribes to images topic for every key-frame (or semi key frame) images.
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
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------


        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 3rd Apr, 2017
        Edition : 2 (of nap_time_node.py)
        Edition : 4 (nap_daisy_bf.py 3rd Nov, 2017)
        Edition : 5 (nap_robustdaisy_bf.py 25th Dec, 2017)
        Edition : 6 (nap_multiproc_node.py 10th Apr, 2018)

"""


import rospy
import rospkg

import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
import time
import code
import os
import sys


from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud

from nap.msg import NapMsg


from multiprocessing import Process, Queue, Manager



import tensorflow as tf
import tensorflow.contrib.slim as slim

TF_MAJOR_VERSION = int(tf.__version__.split('.')[0])
TF_MINOR_VERSION = int(tf.__version__.split('.')[1])


from CartWheelFlow import VGGDescriptor
from FeatureFactory import FeatureFactory
from ImageReceiver import ImageReceiver
from GeometricVerification import GeometricVerification

try:
    from DaisyMeld.daisymeld import DaisyMeld
except:
    print 'If you get this error, your DaisyMeld wrapper is not properly setup. You need to set DaisyMeld in LD_LIBRARY_PATH. and PYTHONPATH contains parent of DaisyMeld'
    print 'See also : https://github.com/mpkuse/daisy_py_wrapper'
    print 'Do: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/catkin_ws/src/nap/scripts/DaisyMeld'
    print 'do: export PYTHONPATH=$PYTHONPATH:$HOME/catkin_ws/src/nap/scripts'
    quit()

from ColorLUT import ColorLUT

import TerminalColors
tcolor = TerminalColors.bcolors()
tcol = tcolor

################# UTIL Functions #################
def xprint( msg, threadId ):
    threadId = str(threadId)
    # if threadId.find( 'worker_score_computation') > 0 :
        # return
    if threadId.find( 'worker_cpu') > 0 :
        return
    if threadId.find( 'worker_gpu') > 0 :
        return


    print '[%s]' %( str(threadId) ), msg

def publish_time( PUB, time_ms ):
    if PUB is  None:
        return

    PUB.publish( Float32(time_ms) )


def publish_image( PUB, cv_image, t=None ):
    if PUB is None:
        return

    data_type = 'bgr8'
    if len(cv_image.shape) == 2:
        data_type = 'mono8'

    msg_frame = CvBridge().cv2_to_imgmsg( cv_image, data_type )
    if t is not None:
        msg_frame.header.stamp = t
    PUB.publish( msg_frame )



## 'x' can also be a vector
def logistic(  x ):
    #y = np.array(x)
    #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
    # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)
    filt = [0.1,0.2,0.4,0.2,0.1]
    if len(x) < len(filt):
        return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

    y = np.convolve( np.array(x), filt, 'same' )
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)




############## END UTIL Functions ##################



############# Utils of GEOMETRIC VERIFY (CPU) #############
def find_index( timestamp_ary, stamp ):
    # print 'find_index'
    del_duration = rospy.Duration.from_sec( 0.001 ) #1ms

    for i in range( len(timestamp_ary) ):
        t = timestamp_ary[i]
        # print (t - stamp)
        if (t - stamp) < del_duration and (t - stamp) > -del_duration:
            return i
    return -1


def plot_feat2d( im, feat2d, color=(255,0,255) ):
    """ feat2d: 2xN or 3xN """
    xcanvas = im.copy()
    for xi in range( feat2d.shape[1] ):
        cv2.circle( xcanvas, tuple( np.int0(feat2d[0:2,xi]) ), 2, color )
    return xcanvas


def robust_daisy_wrapper( VV, D, DAISY_flags  ):
    """ Wrapper for the robust scoring set of calls. Based on
        `test_daisy_improve_dec2017.py'

        Uses, VV, curr_im, prev_im, curr_m_im, __lut_curr_im, __lut_prev_im

        DAISY_flags is a dict() which holds the flags

        Returns:
            a) _2way: selected_curr_i, selected_prev_i
            b) _3way:
    """
    THREAD_NAME = '%5d-robust_daisy_wrapper' %( os.getpid() )
    _2way = None
    _3way = None
    _info = ''
    PRINT_DEBUG_MSG = False


    #TODO If in prev iternation im_curr was the same don't reset it. This will save computations
    VV.set_image( D['im_curr'], 1 ) #set current image
    VV.set_image( D['im_prev'], 2 )# set previous image (at this stage dont need lut_raw to be set as it is not used by release_candidate_match2_guided_2way() )

    selected_curr_i, selected_prev_i, sieve_stat = VV.release_candidate_match2_guided_2way(\
                      D['feat2d_curr'],\
                      D['feat2d_prev']\
                    )
    match2_total_score = VV.sieve_stat_to_score( sieve_stat )
    if PRINT_DEBUG_MSG:
        xprint( '=X=Total_score : '+ str(match2_total_score)+ '=X=', THREAD_NAME )

    _info += '=X=Total_score : '+ str(match2_total_score)+ '=X=\n'
    _info += 'After 2way_matching, n=%d\n' %( len(selected_curr_i) )



    # Rules
    # Rules
    if match2_total_score > 3:
        # Accept this match and move on
        if PRINT_DEBUG_MSG:
            xprint( 'Accept this match and move on' , THREAD_NAME )
            xprint( tcol.OKGREEN+ 'Accept (Strong)'+ tcol.ENDC, THREAD_NAME )
        _info += tcol.OKGREEN+ 'a: Accept (Strong)'+ tcol.ENDC + '\n'
        _2way = (selected_curr_i,selected_prev_i)

    if match2_total_score > 2 and match2_total_score <= 3 and len(selected_curr_i) > 20:
        # Boundry case, if you see sufficient number of 2way matches, also accpt 2way match
        if PRINT_DEBUG_MSG:
            xprint( 'Boundary case, if you see sufficient number of 2way matches, also accept 2way match', THREAD_NAME )
            xprint( tcol.OKGREEN+ 'Accept'+ tcol.ENDC )
        _info += tcol.OKGREEN+ 'b: Accept'+ tcol.ENDC+'\n'

        _2way = (selected_curr_i,selected_prev_i)


    if match2_total_score >= 0.5 and match2_total_score <= 3 and DAISY_flags['ENABLE_3WAY']: #currently 3 way is disabled. TODO
        # Try 3way. But plot 2way and 3way.
        # Beware, 3way match function returns None when it has early-rejected the match
        xprint( '@@@@@@@@@@@@@ Attempt robust_3way_matching()', THREAD_NAME )

        # set-data
        VV.set_image( D['im_curr_m'], 3 )  #set curr-1 image
        VV.set_lut_raw( D['__lut_curr_im'], 1 ) #set lut of curr and prev
        VV.set_lut_raw( D['__lut_prev_im'], 2 )
        # VV.set_lut( curr_lut, 1 ) #only needed for in debug mode of 3way match
        # VV.set_lut( prev_lut, 2 ) #only needed for in debug mode of 3way match

        # Attempt 3way match
        # q1,q2,q3: pts_curr, pts_prev, _pts_curr_m,
        # q4      : per_match_vote,
        # q5      : (dense_match_quality, after_vote_match_quality)
        # See GeometricVerification class to know more on this function.
        q1,q2,q3,q4,q5 = VV.robust_match3way()
        xprint( 'dense_match_quality     : '+ str(q5[0]), THREAD_NAME )
        xprint( 'after_vote_match_quality: '+ str(q5[1]), THREAD_NAME )
        _info += 'After 3way_matching:\n'
        _info += 'dense_match_quality:%4.2f\n' %(q5[0])
        _info += 'after_vote_match_quality:%4.2f\n' %(q5[1])


        if q1 is None:
            xprint( 'Early Reject from robust_match3way()', THREAD_NAME )
            xprint( tcol.FAIL+ 'Reject'+ tcol.ENDC, THREAD_NAME )
            _info += 'Early Reject from robust_match3way()\n'
            _info += tcol.FAIL+ 'c: Reject'+ tcol.ENDC+'\n'
            _3way = None

        else:
            xprint( 'nPts_3way_match     : '+ str(q1.shape) , THREAD_NAME )
            xprint( 'Accept 3way match', THREAD_NAME )
            xprint( tcol.OKGREEN+ 'Accept'+ tcol.ENDC, THREAD_NAME )
            _info += 'n3way_matches: %s' %( str(q1.shape) ) + '\n'
            _info += tcol.OKGREEN+ 'c: Accept'+ tcol.ENDC + '\n'
            #fill up _3way
            _3way = (q1,q2,q3)



    if match2_total_score < 0.5:
        # Reject (don't bother computing 3way)
        if PRINT_DEBUG_MSG:
            xprint( 'Reject 2way matching, and do not compute 3way matching', THREAD_NAME )
            xprint( tcol.FAIL+ 'Reject (Strong)'+ tcol.ENDC )
        _info += tcol.FAIL+ 'd: Reject (Strong)'+ tcol.ENDC+'\n'
        _2way = None
        _3way = None


    return _2way, _3way, _info


#--- Nap Msg Creation ---#
def make_nap_msg( t_curr, t_prev, edge_color=None):
    """ Uses global variables S_timestamp, sim_scores_logistic
    """
    nap_msg = NapMsg() #edge msg
    nap_msg.c_timestamp = t_curr#S_timestamp[i_curr]
    nap_msg.prev_timestamp = t_prev#S_timestamp[i_prev]
    # nap_msg.goodness = sim_scores_logistic[i_prev]

    # edge_color is not in use. but fill it for legacy reasons. May be some code somewhere uses it. Not sure!
    if edge_color is None:
        edge_color = (0,1.0,0)

    if len(edge_color) != 3:
        edge_color = (0,1.0,0)

    nap_msg.color_r = edge_color[0] #default color is green
    nap_msg.color_g = edge_color[1]
    nap_msg.color_b = edge_color[2]

    return nap_msg
############################ END ################################




def worker_gpu( process_flags, Qi, Qt, \
                S_thumbnails, S_timestamp, S_netvlad, S_lut_raw,\
                Q_time_netvlad, Q_cluster_assgn_falsecolormap ):
    """ Consumes the Queue Qi and Qt to produce Qd.
    Qi contains the received images, Qt contains the corresponding timestamps.
    S_thumbnails and S_netvlad are also filled up. These are multiprocessing lists()
    Qd contains the loop closure pair, ie. i_curr, i_prev, ie. a pair of image indices
    """
    THREAD_NAME = '%5d-worker_gpu' %( os.getpid() )

    ###
    ### Init Tensorflow
    ###
    NET_TYPE = "resnet6"
    PARAM_K = 16
    PARAM_model_restore = base_path+'/tf.models/D/model-8000'


    # Create tf queue #TODO

    colorLUT = ColorLUT()


    tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
    is_training = tf.placeholder( tf.bool, [], name='is_training')
    vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
    tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )

    try:
        status = process_flags['MAIN_ENDED']
        if status:
            xprint( 'close gpu process', THREAD_NAME )
            Qi.close()
            Qt.close()

            Q_time_netvlad.close()
            if Q_cluster_assgn_falsecolormap is not None:
                Q_cluster_assgn_falsecolormap.close()
            xprint( 'terminate gpu process', THREAD_NAME )
            return
    except:
        xprint( 'exception encountered.1', THREAD_NAME )
        Qi.close()
        Qt.close()

        Q_time_netvlad.close()
        if Q_cluster_assgn_falsecolormap is not None:
            Q_cluster_assgn_falsecolormap.close()
        xprint( 'terminate gpu process', THREAD_NAME )
        return

    sess = tf.Session()
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver = tf.train.Saver()
    tensorflow_saver.restore( sess, PARAM_model_restore )

    # while Qi.qsize() > 0 or flag:
    # while True:
    while process_flags['MAIN_ENDED'] is False:
        xprint( 'qsize:%d' %(Qi.qsize() ), THREAD_NAME )


        #--------------------------- GET-----------------------#
        try:
            im = Qi.get_nowait()
            timestamp = Qt.get_nowait()
        except:
            time.sleep( 0.2 )
            continue
        #--------------------------- END-----------------------#


        #------------------------- Do NetVLAD -----------------#
        # GPU Computations on `im`
        startNetVLAD = time.time()

        im_batch = np.expand_dims( im.astype('float32'), 0 ) #unnormalized batch. Nx240x320x3
        feed_dict = {tf_x : im_batch,\
                     is_training:True,\
                     vgg_obj.initial_t: 0
                    }
        tff_vlad_word, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm], feed_dict=feed_dict)
        Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1

        Q_time_netvlad.put( 1000.*(time.time() - startNetVLAD ) )

        #--------------------------- END-----------------------#

        if Q_cluster_assgn_falsecolormap is not None:
            lut = colorLUT.lut( Assgn_matrix[0,:,:] )
            xprint( '(false colormap) lut.shape'+str( lut.shape ), THREAD_NAME )

            # Write qsize in lut image
            # Q_cluster_assgn_falsecolormap.put( lut )
            debug_txt_image = np.zeros((30, lut.shape[1],lut.shape[2]), dtype=np.uint8)
            toTxt = "gpu-q: %d" %(Qi.qsize())
            cv2.putText(debug_txt_image,toTxt, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255) )
            lut_debug = np.concatenate( (lut, debug_txt_image), axis=0 )
            Q_cluster_assgn_falsecolormap.put( lut_debug )



        #-------------------------- Storage  ---------------------------#
        # Dump into S_thumbnails, S_timestamp, S_netvlad
        S_thumbnails.append( im )
        S_timestamp.append( timestamp )
        S_netvlad.append( tff_vlad_word )
        S_lut_raw.append( Assgn_matrix[0,:,:] )
        # TODO Perhaps also need S_lut_raw ie. Assgn_matrix
        #--------------------------- END-----------------------#



    xprint( 'close gpu process', THREAD_NAME )
    Qi.close()
    Qt.close()

    Q_time_netvlad.close()
    if Q_cluster_assgn_falsecolormap is not None:
        Q_cluster_assgn_falsecolormap.close()
    xprint( 'terminate gpu process', THREAD_NAME )



def worker_score_computation( process_flags, Qd, S_timestamp, S_netvlad, Q_time_netvlad_etc, Q_scores_plot ):
    """
        periodically checks if new image-frame was converted to netvlad.
        If yes, do the score computation < N[last] , N[i] > \forall i=0,1...
        This produces the candidate list which is enqueued in the Queue Qd.

    """
    THREAD_NAME = '%5d-worker_score_computation' %( os.getpid() )

    rate = rospy.Rate( 15 )
    curr_item_idx = 0 #of S_netvlad
    while process_flags['MAIN_ENDED'] is False:
        try:
            curr_len = len(S_netvlad)
        except:
            # Manager has closed connection
            break
        # xprint( "curr_item_idx=%d" %(curr_item_idx), THREAD_NAME )
        if (curr_item_idx+1) >= curr_len:
            # Nothing new added by GPU so do nothing.
            # xprint( "nothing new. len(S_netvlad)=%d" %(len(S_netvlad)) ,  THREAD_NAME  )
            rate.sleep()
            continue

        # Score computation
        #------------------- Score Computation (Brute Force)----------------#
        # Find loop candidates and put idx_curr and idx_prev in  Qd
        if curr_item_idx < 30:
            curr_item_idx = curr_item_idx + 1
            # xprint( "too few items", THREAD_NAME )
            continue


        # All set. Dot computation < X[curr_item_idx] , X[0:curr_item_idx-1] >
        startETC = time.time()
        xprint( "Do scoring < X[%d] , X[0:%d] >" %( curr_item_idx, curr_item_idx-1) , THREAD_NAME )


        _A = np.asarray(S_netvlad[0:curr_item_idx-1]) #_A for some reason is Nx1x4096
        _A = _A[:,0,:] #now Nx4096
        _B = np.asarray(S_netvlad[curr_item_idx]) # 1x4096
        # xprint( '_A.shape: %s' %( str( _A.shape) ), THREAD_NAME )
        # xprint( '_B.shape: %s' %( str( _B.shape) ), THREAD_NAME )

        DOT_word = np.dot( _A, np.transpose(_B) )
        DOT_word = DOT_word[:,0]
        sim_scores = np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) ) #minimum is added to ensure dot product doesnt go beyond 1.0 as it sometimes happens because of numerical issues, which inturn causes issues with sqrt
        sim_scores_logistic = logistic( sim_scores ) #convert the raw Similarity scores above to likelihoods


        # xprint( 'DOT_word.shape=%s' %(  str(DOT_word.shape) ), THREAD_NAME )
        # xprint( 'sim_scores_logistic.shape %s' %(str(sim_scores_logistic.shape) ), os.getpid() )

        # Uses opencv 3.3 for cv2.plot
        if Q_scores_plot is not None:
            # pass
            # DIY plotting
            from Plot2Mat import Plot2Mat
            xprint( 'DIY plotting', THREAD_NAME )
            r = Plot2Mat( )
            r.plot( sim_scores_logistic, line_color=(0,255,0) )




            # Opencv Plot need opencv 3.3
            # r = cv2.plot.Plot2d_create( np.array( range(len(sim_scores_logistic)) ).astype('float64'), sim_scores_logistic.astype('float64') )
            # r.setMinY( 0. )
            # r.setMaxY( 1. )
            # r.setInvertOrientation( True )
            # Qp = r.render()



        # Top-5 from (0 to N-25), followed by thresholding Thresh.
        keep_top_n = 20
        skip_last_n = 25
        score_thresh = 0.5
        max_enqueue = 4
        if Qd.qsize() > 20:
            max_enqueue = 1
        if Qd.qsize() > 15:
            max_enqueue = 2

        _QQW = [(i,sim_scores_logistic[i]) for i in np.argsort(sim_scores_logistic[0:-skip_last_n])[-keep_top_n:]]
        __QQW = [sim_scores_logistic[i] for i in np.argsort(sim_scores_logistic[0:-skip_last_n])[-keep_top_n:]]
        n_found = (np.array(__QQW) > score_thresh).sum()
        n_items_enqueued = 0
        _items_enqueued = []
        for idx, scr in _QQW:
            if scr < score_thresh:
                continue

            latestTimeStamp = S_timestamp[curr_item_idx].to_sec()
            # Avoid matches from near current
            if (latestTimeStamp -  S_timestamp[idx].to_sec()) <10.  or idx < 5:
                continue

            if n_items_enqueued > max_enqueue: #if already enough put into the queue, den quit
                break

            i_curr = curr_item_idx
            i_prev = idx

            hj = np.array( _items_enqueued )
            if len( hj ) > 0:
                __l = abs(np.array(_items_enqueued ) - i_prev)
                # if min(__l) < 5  :
                if (__l < 10).sum() > 2: #dont let more than 2 nearby
                    continue # dont enqueue if something very near to i_prev is already enqueued

            xprint( 'Enqueue(%d,%d)' %(i_curr, i_prev), THREAD_NAME )



            Qd.put( (i_curr, i_prev) )
            n_items_enqueued = n_items_enqueued + 1
            _items_enqueued.append( i_prev )


        # xprint( 'Found %d candidates above the thresh' %(len(argT) ), THREAD_NAME  )
        xprint( 'Found=%d, Enqueued=%d' %(n_found, n_items_enqueued), THREAD_NAME )
        Qp = r.mark(  np.array( _items_enqueued )  )

        if Q_scores_plot is not None:
            Q_scores_plot.put( Qp.astype('uint8') )
        # cv2.imwrite( '/home/mpkuse/Pictures/zzz12_%d.png' %(curr_item_idx), Qp )



        # Increment, log and continue
        curr_item_idx = curr_item_idx + 1
        Q_time_netvlad_etc.put( 1000.*(time.time() - startETC ) )
        continue
        #-----------------------------------------------------#


    xprint( 'close queues', THREAD_NAME )
    Qd.close()
    Q_time_netvlad_etc.close()
    Q_scores_plot.close()
    xprint( 'Terminate', THREAD_NAME )



def worker_cpu( process_flags, Qd, S_thumbnails, S_timestamp, S_lut_raw,\
                            FF, Q_2way_napmsg, Q_3way_napmsg, Q_time_cpu, Q_match_im_canvas, Q_match_im3way_canvas ):
    """ Consumes the Queue Qd. Qd contains the candidate loop closure pair indices.
        cpu does the geometric verification using the pair indices and looking up those
        images from S_timestamp.

        TODO: This part also needs access to point features. Need to workout how to do that
    """
    THREAD_NAME = '%5d-worker_cpu' %( os.getpid() )

    # dai1 = DaisyMeld(240, 320, 0)
    # dai2 = DaisyMeld(240, 320, 0)
    VV = GeometricVerification()
    VV.disable_printing()



    prev_qsize = -1
    while process_flags['MAIN_ENDED'] is False:
        ###
        ### Dequeue
        ###

        curr_qsize = Qd.qsize()
        # Heuristic. If qsize gets more than 10 dequeue multiple. Essentually throwing away matches without checking
        try:
            if curr_qsize <= 10:
                g = Qd.get_nowait()
            else:
                for h in range( int(curr_qsize/10.)  ):
                    g = Qd.get_nowait()
        except:
            # xprint( 'Sleep for 0.2', THREAD_NAME )
            time.sleep( 0.2 )
            continue
        # try:
        #     g = Qd.get(timeout=1)
        # except:
        #     continue # No more. Adjust this as need be. beware of sleep at start of this function
        xprint( '----------------\ni_curr: %d, i_prev: %d' %( g[0], g[1] ), THREAD_NAME )

        startKMeans = time.time()


        ###
        ### Assemble Data
        ###
        # Collect all the required things to compute the match

        # Images
        i_curr = g[0]
        i_prev = g[1]
        im_curr = S_thumbnails[ i_curr ]
        im_prev = S_thumbnails[ i_prev ]
        im_curr_m = S_thumbnails[ i_curr-1 ]


        # Timestamps
        t_curr   = S_timestamp[ i_curr ]
        t_prev   = S_timestamp[ i_prev ]
        t_curr_m = S_timestamp[ i_curr-1 ]


        # Features (From FeatureFactor)
        # ? The way to get corresponding features is to search timestamp `t_curr` with
        #   FeatureFactor.find_index().
        # xprint( 'len(feature_factory.timestamp)=%d' %( len(FF['timestamp']) ), THREAD_NAME )
        feat2d_curr_idx = find_index( FF['timestamp'], t_curr  )
        feat2d_prev_idx = find_index( FF['timestamp'], t_prev  )
        assert feat2d_curr_idx >= 0 , "This is a fatal error for geometry. tracked features corresponding to this image not found."
        assert feat2d_prev_idx >= 0 , "This is a fatal error for geometry. tracked features corresponding to this image not found."

        feat2d_curr_normed = FF['features'][feat2d_curr_idx ]
        feat2d_prev_normed = FF['features'][feat2d_prev_idx ]

        feat2d_curr = np.dot( FF['K'], feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
        feat2d_prev = np.dot( FF['K'], feature_factory.features[feat2d_prev_idx ] )

        feat3d_curr = FF['point3d'][feat2d_curr_idx]

        daisy_data = {}
        daisy_data['i_curr'] = i_curr
        daisy_data['i_prev'] = i_prev
        daisy_data['im_curr'] = im_curr
        daisy_data['im_prev'] = im_prev

        daisy_data['t_curr'] = t_curr
        daisy_data['t_prev'] = t_prev

        daisy_data['feat2d_curr'] = feat2d_curr
        daisy_data['feat2d_prev'] = feat2d_prev
        daisy_data['feat3d_curr'] = feat3d_curr

        # Data for 3way match
        daisy_data['t_curr_m'] = t_curr_m
        daisy_data['im_curr_m'] = im_curr_m
        daisy_data['__lut_curr_im'] = S_lut_raw[ i_curr ]
        daisy_data['__lut_prev_im'] = S_lut_raw[ i_prev ]
        xprint( 'len of S_lut_raw %d' %( len(S_lut_raw) ), THREAD_NAME  )


        if False: # Timing info.
            hsgh0 = 'i_curr(%3d) <--> feat2d_curr_idx(%3d)' %( i_curr, feat2d_curr_idx)
            hsgh0x = 'i_curr(%s) <--> feat2d_curr_idx(%s)' %( t_curr, FF['timestamp'][feat2d_curr_idx] )
            xprint( hsgh0, THREAD_NAME )
            xprint( hsgh0x, THREAD_NAME )

            hsgh1 = 'i_prev(%3d) <--> feat2d_prev_idx(%3d)' %( i_prev, feat2d_prev_idx )
            hsgh1x = 'i_prev(%s) <--> feat2d_prev_idx(%s)' %( t_prev, FF['timestamp'][feat2d_prev_idx] )
            xprint( hsgh1, THREAD_NAME )
            xprint( hsgh1x, THREAD_NAME )

        ###
        ### Do Robust Daisy
        ###
        _2way = None
        _3way = None
        _info = ''
        DAISY_flags = {}
        DAISY_flags['ENABLE_3WAY'] = False
        _2way, _3way, _info = robust_daisy_wrapper( VV, daisy_data, DAISY_flags=DAISY_flags )
        xprint( _info, THREAD_NAME )


        ###
        ### Put Nap Msg on publish queue
        ###
        if _2way is not None:
            # Step-1a: Get indices of matches
            selected_curr_i, selected_prev_i = _2way
            nap_msg = make_nap_msg( t_curr, t_prev, (0.6,1.0,0.6) )
            nap_msg.op_mode = 20
            nap_msg.t_curr = t_curr
            nap_msg.t_prev = t_prev

            # Step-2: Fill up nap-msg
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

            Q_2way_napmsg.put( nap_msg )


        if _3way is not None:
            # Step-1a: Get co-ordinates of matches
            xpts_curr, xpts_prev, xpts_currm = _3way

            # Step-2: Fill up nap msg
            nap_msg = make_nap_msg( t_curr, t_prev, (0.6,1.0,0.6) )
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

            Q_3way_napmsg.put( nap_msg )




        ###
        ### Image Matching Info into DEBUG queue
        ###
        if Q_match_im_canvas is not None:
            A = plot_feat2d( im_curr, feat2d_curr, (0,0,255) )
            B = plot_feat2d( im_prev, feat2d_prev, (0,0,255) )

            cv2.putText(A, str(i_curr), (30,30), \
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1);
            cv2.putText(B, str(i_prev), (30,30), \
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1);

            #TODO: Write image index on the images
            if _2way is not None:
                selected_curr_i, selected_prev_i = _2way
                xcanvas_2way = VV.plot_2way_match( A, np.int0(feat2d_curr[0:2,selected_curr_i]), B, np.int0(feat2d_prev[0:2,selected_prev_i]),  enable_lines=True )
                # fname =  '/home/nvidia/Pictures/%d_%d.jpg' %(g[0], g[1])
                # xprint( fname, THREAD_NAME )
                # cv2.imwrite( fname, xcanvas_2way )

                # Q_match_im_canvas.put( xcanvas_2way )

                debug_txt_image = np.zeros((int(xcanvas_2way.shape[0]/4), xcanvas_2way.shape[1],xcanvas_2way.shape[2]), dtype=np.uint8)
                toTxt = "cpu-q: %d" %(Qd.qsize())
                cv2.putText(debug_txt_image,toTxt, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) , 3 )
                xcanvas_2way_debug = np.concatenate( (xcanvas_2way, debug_txt_image), axis=0 )
                Q_match_im_canvas.put( xcanvas_2way_debug )


            if _3way is not None:
                q1, q2, q3 = _3way
                gridd = VV.plot_3way_match( VV.im1, np.array(q1), VV.im2, np.array(q2), VV.im3, np.array(q3) )
                # fname =  '/home/nvidia/Pictures/3way_%d_%d.jpg' %(g[0], g[1])
                # xprint( fname, THREAD_NAME )
                # cv2.imwrite( fname, gridd )

                Q_match_im3way_canvas.put( gridd )



        Q_time_cpu.put(   1000.*(time.time() - startKMeans)  )
        prev_qsize = curr_qsize

    xprint( 'Loop Ended. Now close queues of cpu', THREAD_NAME )
    Qd.close()
    Q_2way_napmsg.close()
    Q_3way_napmsg.close()
    Q_time_cpu.close()
    if Q_match_im_canvas is not None:
        Q_match_im_canvas.close()
    if Q_match_im3way_canvas is not None:
        Q_match_im3way_canvas.close()
    xprint( 'Terminate cpu_process', THREAD_NAME )




##########
#  Main #
#########
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    base_path = rospack.get_path( 'nap' )


    PARAMS = {}

    PARAMS['TRACKED_FEATURE_TOPIC'] = '/vins_estimator/keyframe_point'
    PARAMS['OUTPUT_EDGES_TOPIC'] = '/raw_graph_edge'
    PARAMS['PARAM_CALLBACK_SKIP'] = 3
    PARAMS['INPUT_IMAGE_TOPIC'] = '/semi_keyframes'
    PARAMS['VINS_CONFIG_YAML_FNAME'] = rospy.get_param( '/nap/config_file')
    PARAMS['N_CPU'] = 4

    # Local launch
    # PARAMS['INPUT_IMAGE_TOPIC'] = '/vins_estimator/keyframe_image'
    # PARAMS['VINS_CONFIG_YAML_FNAME'] = rospack.get_path( 'nap' )+'/slam_data/blackbox4.yaml' #TODO: read this paramter from the parameter server.See nap_daisy_bf.py
    # PARAMS['N_CPU'] = 4

    #TWEAK NetVLAD param in worker_gpu

    ###
    ### INIT Ros node
    ###
    rospy.init_node( 'nap_multiproc_node' )


    ###
    ### Shared global data holders
    ###
    manager = Manager()
    S_thumbnails = manager.list()
    S_timestamp = manager.list()
    S_netvlad = manager.list()
    S_lut_raw = manager.list() # The cluster-assignment map from NetVLAD's layer.

    # These flags will determine ending the loops in threads
    process_flags = manager.dict()
    process_flags['MAIN_ENDED'] = False

    Qd = Queue() # This queue will contains loop-closure candidates. Produced by `worker_score_computation`. consumed by p_cpu

    Q_2way_napmsg = Queue()
    Q_3way_napmsg = Queue()

    ###
    ### Core Objects
    ###
    feature_factory = FeatureFactory( PARAMS['VINS_CONFIG_YAML_FNAME'], manager )
    image_receiver =  ImageReceiver(  PARAMS['PARAM_CALLBACK_SKIP']    )

    FEAT_FACT_SEMAPHORES = manager.dict()
    FEAT_FACT_SEMAPHORES['timestamp'] = feature_factory.timestamp
    FEAT_FACT_SEMAPHORES['features'] = feature_factory.features
    FEAT_FACT_SEMAPHORES['global_index'] = feature_factory.global_index
    FEAT_FACT_SEMAPHORES['point3d'] = feature_factory.point3d
    FEAT_FACT_SEMAPHORES['K'] = feature_factory.K
    FEAT_FACT_SEMAPHORES['K_org'] = feature_factory.K_org


    ###
    ### Subscribers
    ###
    # Input Images
    rospy.Subscriber( PARAMS['INPUT_IMAGE_TOPIC'], Image, image_receiver.color_image_callback )
    rospy.loginfo( 'Subscribed to '+PARAMS['INPUT_IMAGE_TOPIC'] )

    # Tracked Features
    rospy.Subscriber( PARAMS['TRACKED_FEATURE_TOPIC'], PointCloud, feature_factory.tracked_features_callback )
    rospy.loginfo( 'Subscribed to '+PARAMS['TRACKED_FEATURE_TOPIC'] )



    ###
    ### Publishers
    ###
    # raw edges
    pub_edge_msg = rospy.Publisher( PARAMS['OUTPUT_EDGES_TOPIC'], NapMsg, queue_size=100 )
    rospy.loginfo( 'Publish to %s' %(PARAMS['OUTPUT_EDGES_TOPIC']) )


    ###
    ### Time/Queue Publishers
    ###         Note: Make any of these publishers to `None` to not publish this topic. Don't make time queues as None
    ###         Note: Make the Queues to None to disable the associated processing to get() and put().
    pub_Qi_size = rospy.Publisher( '/time/Qi_qsize', Float32, queue_size=1000 )
    pub_Qt_size = rospy.Publisher( '/time/Qt_qsize', Float32, queue_size=1000 )
    pub_Qd_size = rospy.Publisher( '/time/Qd_qsize', Float32, queue_size=1000 )

    # Time queues and publishers
    Q_time_netvlad = Queue()
    Q_time_netvlad_etc = Queue()
    pub_time_netvlad = rospy.Publisher( '/time/netvlad', Float32, queue_size=1000)
    pub_time_netvlad_etc = rospy.Publisher( '/time/netvlad_etc', Float32, queue_size=1000)

    Q_time_cpu = Queue()
    pub_time_cpu = rospy.Publisher( '/time/cpu', Float32, queue_size=1000)


    # Score Plot Queue and Publisher
    Q_scores_plot  = Queue()
    pub_scores_plot = rospy.Publisher( '/debug/scores_plot', Image, queue_size=10 )

    # Matched images Queue and Publisher.
    Q_match_im_canvas = Queue()
    pub_match_im_canvas = rospy.Publisher( '/debug/featues2d_matching', Image, queue_size=10 )
    Q_match_im3way_canvas = Queue() # 3way matching
    pub_match_im3way_canvas = rospy.Publisher( '/debug/featues_matching_3way', Image, queue_size=10 )


    # False colormap from Neural Network
    Q_cluster_assgn_falsecolormap = Queue()
    pub_cluster_assgn_falsecolormap = rospy.Publisher( '/debug/cluster_assignment', Image, queue_size=10 )


    ###
    ### Launch Processes
    ###
    p_gpu = Process( target=worker_gpu, name="gpu_process", args=\
                (process_flags, \
                 image_receiver.im_queue,\
                 image_receiver.im_timestamp_queue,\
                 S_thumbnails, S_timestamp, S_netvlad, S_lut_raw,\
                 Q_time_netvlad, Q_cluster_assgn_falsecolormap\
                )\
                 )

    assert PARAMS['N_CPU'] >= 1 , "N_CPU param need to be a positive number"
    n_cpu_procs = int(PARAMS['N_CPU'])
    cpu_jobs = []
    for i_cpu in range(n_cpu_procs):
        p_cpu = Process( target=worker_cpu, name="cpu%d_process" %(i_cpu), args=\
                        (   process_flags,\
                            Qd,\
                            S_thumbnails,\
                            S_timestamp,\
                            S_lut_raw,\
                            FEAT_FACT_SEMAPHORES,\
                            Q_2way_napmsg, Q_3way_napmsg,\
                            Q_time_cpu, Q_match_im_canvas, Q_match_im3way_canvas\
                        )\
                    )
        cpu_jobs.append( p_cpu )

    p_scoring = Process( target=worker_score_computation, name="scoring_process" , args=\
                    (   process_flags,\
                        Qd, S_timestamp, S_netvlad, Q_time_netvlad_etc, Q_scores_plot\
                    )\
                )

    # p_gpu.daemon = True
    # for i_cpu in range(n_cpu_procs):
    #     cpu_jobs[i_cpu].daemon = True
    # p_scoring.daemon = True

    p_gpu.start()
    p_scoring.start()

    # p_cpu.start()
    for i_cpu in range(n_cpu_procs):
        cpu_jobs[i_cpu].start()



    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            publish_time( pub_time_netvlad, Q_time_netvlad.get_nowait() )
            # xprint( 'MAIN1', os.getpid() )
        except:
            pass

        try:
            publish_time( pub_time_netvlad_etc, Q_time_netvlad_etc.get_nowait() )
            # xprint( 'MAIN1', os.getpid() )
        except:
            pass

        try:
            publish_time( pub_time_cpu, Q_time_cpu.get_nowait() )
            # xprint( 'MAIN1', os.getpid() )
        except:
            pass


        publish_time( pub_Qi_size, image_receiver.im_queue.qsize() )
        publish_time( pub_Qt_size, image_receiver.im_timestamp_queue.qsize() )
        publish_time( pub_Qd_size, Qd.qsize() )


        # Debug Image Publishers
        try:
            # print "MAIN: ", Q_scores_plot.qsize()
            publish_image( pub_scores_plot, Q_scores_plot.get_nowait() )
            # print 'MAIN: publsihed scrores. '
        except:
            pass
        # except Exception as e:
            # print(e)

        try:
            publish_image( pub_match_im_canvas, Q_match_im_canvas.get_nowait() )
        except:
            pass

        try:
            publish_image( pub_match_im3way_canvas, Q_match_im3way_canvas.get_nowait() )
        except:
            pass

        try:
            publish_image( pub_cluster_assgn_falsecolormap, Q_cluster_assgn_falsecolormap.get_nowait() )
        except:
            pass


        # Publish Nap Msg
        # xprint( 'napmsg.qsize: %d %d' %(Q_2way_napmsg.qsize(), Q_3way_napmsg.qsize()) , 'MAIN' )
        try:
            pub_edge_msg.publish( Q_2way_napmsg.get_nowait() )
            pub_edge_msg.publish( Q_3way_napmsg.get_nowait() )
        except:
            pass

        rate.sleep()

    process_flags['MAIN_ENDED'] = True

    # Write Info to files (for debugging)
    print 'Main Ended'
    try:
        # BASE__DUMP = '/home/mpkuse/Desktop/bundle_adj'
        BASE__DUMP = rospy.get_param( '/nap/debug_output_dir')
        if BASE__DUMP is not None:
            print 'Writing ', BASE__DUMP+'/S_netvlad.npy'
            print 'Writing ', BASE__DUMP+'/S_timestamp.npy'
            print 'Writing ', BASE__DUMP+'/S_thumbnails.npy'
            print 'Writing ', BASE__DUMP+'/S_thumbnail_lut_raw.npy'
            np.save( BASE__DUMP+'/S_netvlad.npy', np.array(S_netvlad) )
            np.save( BASE__DUMP+'/S_timestamp.npy', np.array(S_timestamp) )
            np.save( BASE__DUMP+'/S_thumbnails.npy', np.array(S_thumbnails) )
            np.save( BASE__DUMP+'/S_lut_raw.npy', np.array(S_lut_raw) )
    except:
        print 'ROSPARAM `/nap/debug_output_dir` not found so not writing debug info'







    # Close Queues
    image_receiver.qclose()
    Qd.close()
    Q_time_netvlad.close()
    Q_time_netvlad_etc.close()
    Q_time_cpu.close()
    Q_2way_napmsg.close()
    Q_3way_napmsg.close()
    Q_scores_plot.close()
    if Q_match_im_canvas is not None:
        Q_match_im_canvas.close()
    if Q_match_im3way_canvas is not None:
        Q_match_im3way_canvas.close()
    if Q_cluster_assgn_falsecolormap is not None:
        Q_cluster_assgn_falsecolormap.close()




    if Q_scores_plot is not None:
        Q_scores_plot.close()
    if Q_match_im_canvas is not None:
        Q_match_im_canvas.close()
    if Q_cluster_assgn_falsecolormap is not None:
        Q_cluster_assgn_falsecolormap.close()



    print 'waiting for other processes to finish'
    p_gpu.join(timeout=.5)
    # p_gpu.terminate()
    print 'p_gpu joined'
    # p_cpu.join()
    for i_cpu in range(n_cpu_procs):
        cpu_jobs[i_cpu].join(timeout=.5)
        # cpu_jobs[i_cpu].terminate()
    print 'p_cpu joined'
    p_scoring.join(timeout=.5)
    # p_scoring.terminate()
    print 'p_scoring joined'

    manager.shutdown()
    print 'Ending Main'
    # sys.exit(0)
    # os._exit(0)
