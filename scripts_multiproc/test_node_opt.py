#!/usr/bin/env python
import rospy
import rospkg
from std_msgs.msg import Float32

import threading
# import multiprocessing

import cv2
import numpy as np
import time
import code
import Queue

import tensorflow as tf
import tensorflow.contrib.slim as slim

TF_MAJOR_VERSION = int(tf.__version__.split('.')[0])
TF_MINOR_VERSION = int(tf.__version__.split('.')[1])


from CartWheelFlow import VGGDescriptor

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


def xprint( msg, threadId ):
    print '[%02d]' %(threadId), msg


def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )


import signal
import sys
def signal_handler( signal, frame ):
    global flag_continue
    print 'You pressed CTRL+C!'
    flag_continue = False


    # sys.exit(0)
#########
## This will have 3 threads.
# th1 will read files from images at a fixed rate say 40 fps and put it in a queue, Qw
# th2 will consume Qw and do GPU computation on elements of Qw. Additionally this will populate Q'
# th3 will consume Q'
########

def worker_image_reading():
    # This thread reads images from file @40fps and dumps into Queue `Qw `
    global pub_time_reader
    global Qw
    global base_path
    global flag_reading_complete

    rate = rospy.Rate( 10 )
    i = 0
    while flag_continue:
        startTime = time.time()
        fname = base_path+'/some_images/%d.png' %(i)

        if i % 100 == 0 :
            xprint( 'READ Image %s'  %(fname), 0 )

        IM = cv2.imread( fname )
        if IM == None or i>600:
            xprint( 'All images over. Exit this thread', 0 )
            flag_reading_complete = True
            break

        Qw.put( IM )
        rate.sleep()
        i = i+1
        publish_time( pub_time_reader, 1000.*(time.time() - startTime) )




def worker_gpu():
    # This thread consumes `Qw` and produces `Q' `
    global base_path
    global Qw
    global Qd
    global S_thumbnails, S_netvlad

    ###
    ### Init Tensorflow
    ###
    NET_TYPE = "resnet6"
    PARAM_K = 16
    PARAM_model_restore = base_path+'/tf.models/D/model-8000'


    # Create tf queue


    tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
    is_training = tf.placeholder( tf.bool, [], name='is_training')
    vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
    tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )

    sess = tf.Session()
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver = tf.train.Saver()
    tensorflow_saver.restore( sess, PARAM_model_restore )


    while flag_continue:

        try:
            img = Qw.get( timeout=1 )
        except:
            xprint( 'Qw is empty', 1 )
            continue

        startNetVLAD = time.time()
        im_batch = np.expand_dims( img.astype('float32'), 0 ) #unnormalized batch. Nx240x320x3

        feed_dict = {tf_x : im_batch,\
                     is_training:True,\
                     vgg_obj.initial_t: 0
                    }
        tff_vlad_word, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm], feed_dict=feed_dict)
        Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1

        # feed_dict = {tf_x: im_batch }
        # tff_vlad_word = sess.run( [tf_vlad_word ], feed_dict=feed_dict)



        # Dump in S_
        S_thumbnails.append( img )
        S_netvlad.append( tff_vlad_word )

        # Make an element in Qd (simulate loop detection)
        if len(S_thumbnails) < 11:
            continue

        i_curr = np.random.randint( 8, len(S_thumbnails) )
        i_prev = np.random.randint( 2, i_curr )
        T = {}
        T['i_curr'] = i_curr
        T['i_prev'] = i_prev
        Qd.put( (i_curr, i_prev) )

        publish_time( pub_time_netvlad, 1000.*(time.time() - startNetVLAD) )



    xprint( 'Terminate thread `worker_gpu`', 1 )



def worker_cpu():
    # This thread consumes `Q'` and does CPU processing on it

    global S_thumbnails
    global Qd

    dai1 = DaisyMeld(240, 320, 0)
    dai2 = DaisyMeld(240, 320, 0)

    while flag_continue:
        try:
            g = Qd.get( timeout=1 )
        except:
            xprint( 'Qd is empty', 2 )
            continue

        xprint( 'qsize: %d, i_curr: %d, i_prev: %d' %( Qd.qsize(), g[0], g[1] ), 1 )

        startKMeans = time.time()

        im_curr = S_thumbnails[ g[0] ]
        im_prev = S_thumbnails[ g[1] ]

        # Daisy starts
        # im_curr32 = im_curr[:,:,0].copy().astype( 'float32' )
        im_curr32 = im_curr[:,:,0].astype( 'float32' )
        dai1.do_daisy_computation( im_curr32 )
        vi1 = dai1.get_daisy_view()

        # im_prev32 = im_prev[:,:,0].copy().astype( 'float32' )
        im_prev32 = im_prev[:,:,0].astype( 'float32' )
        dai2.do_daisy_computation( im_prev32 )
        vi2 = dai2.get_daisy_view()
        # Daisy Ends

        # X = np.random.random( (1000,1000) )
        # e = np.linalg.eig( X )
        time.sleep(.3)

        publish_time( pub_time_kmeans, 1000.*(time.time() - startKMeans) )



def monitor_qsize():
    global pub_Qw_size
    global pub_Qd_size

    rate = rospy.Rate( 5 )
    while flag_continue:
        Qw_qsize = Qw.qsize()
        Qd_qsize = Qd.qsize()

        publish_time( pub_Qw_size, Qw_qsize )
        publish_time( pub_Qd_size, Qd_qsize )

        rate.sleep()


##########
#  Main #
#########
# if __name__ == "__main__":

###
### INIT Ros node
###
rospy.init_node( 'test_node' )
rospack = rospkg.RosPack()
base_path = rospack.get_path( 'tx2_standalone_test' )

###
### Global Holders
###
S_thumbnails = []
S_netvlad = []


###
### Time Publisher
###
pub_Qw_size = rospy.Publisher( '/time/Qw_qsize', Float32, queue_size=1000)
pub_Qd_size = rospy.Publisher( '/time/Qd_qsize', Float32, queue_size=1000)


pub_time_reader = rospy.Publisher( '/time/reader', Float32, queue_size=1000)
pub_time_netvlad = rospy.Publisher( '/time/netvlad', Float32, queue_size=1000)
pub_time_kmeans = rospy.Publisher( '/time/kmeans', Float32, queue_size=1000)


###
### Queues
###
Qw = Queue.Queue() # Read images queue
Qd = Queue.Queue() # Queues of 2 image indices


signal.signal( signal.SIGINT, signal_handler )
###
### Threads
###
flag_reading_complete = False
flag_continue = True

# th0 = multiprocessing.Process(name='worker_image_reading', target=worker_image_reading)
# th1 = multiprocessing.Process(name='worker_gpu', target=worker_gpu)
# th2 = multiprocessing.Process(name='worker_cpu', target=worker_cpu)
# th_monitor = multiprocessing.Process(name='monitor_qsize', target=monitor_qsize )

th0 = threading.Thread( target=worker_image_reading )
th1 = threading.Thread( target=worker_gpu )
th2 = threading.Thread( target=worker_cpu )
th_monitor = threading.Thread( target=monitor_qsize )

th0.start()
th1.start()
th2.start()
th_monitor.start()
print '[MAIN] All Threads started'

th0.join()
th1.join()
th2.join()
th_monitor.join()
print '[MAIN] All Threads ended'
