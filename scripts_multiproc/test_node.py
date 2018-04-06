#!/usr/bin/env python
import rospy
import rospkg
import cv2
from std_msgs.msg import Float32


import cv2
import numpy as np
import os
import time
import code
import argparse
import sys
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

# from association_map import zNormalize
# from association_map import normalize_batch
## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)

def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        for ch in range(im_batch.shape[3]):
                im_batch_normalized[b,:,:,ch] = zNormalize( im_batch[b,:,:,ch])

    return im_batch_normalized


def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )

def consume_queue():
    global S_thumbnails
    global task_queue
    global pub_qsize, pub_time_kmeans
    global XFT
    global dai1, dai2
    global base_path

    while XFT:
        qsize = task_queue.qsize()
        publish_time( pub_qsize, qsize )
        try:
            g = task_queue.get(timeout=1)
            xprint( 'qsize: %d, i_curr: %d, i_prev: %d' %( qsize, g[0], g[1] ), 1 )

            startKMeans = time.time()
            im_curr = S_thumbnails[ g[0] ]
            im_prev = S_thumbnails[ g[1] ]

            # Daisy starts
            im_curr32 = im_curr[:,:,0].copy().astype( 'float32' )
            im_curr32 = im_curr[:,:,0].astype( 'float32' )
            dai1.do_daisy_computation( im_curr32 )
            vi1 = dai1.get_daisy_view()

            # im_prev32 = im_prev[:,:,0].copy().astype( 'float32' )
            im_prev32 = im_prev[:,:,0].astype( 'float32' )
            dai2.do_daisy_computation( im_prev32 )
            vi2 = dai2.get_daisy_view()

            # r = np.random.randint( 10000000, 90000000 )
            # cv2.imwrite( base_path+'/debug_images/%d_1daisy.png' %(r), (255*vi1[:,:,0]).astype('uint8') )
            # cv2.imwrite( base_path+'/debug_images/%d_1org.png' %(r), im_curr.astype('uint8') )
            #
            # cv2.imwrite( base_path+'/debug_images/%d_1daisy.png' %(r), (255*vi2[:,:,0]).astype('uint8') )
            # cv2.imwrite( base_path+'/debug_images/%d_1org.png' %(r), im_prev.astype('uint8') )

            # Daisy ends


            # Kmeans starts
            # Z1 = np.float32( im_curr.reshape( (-1,3) ) )
            # Z2 = np.float32( im_prev.reshape( (-1,3) ) )
            #
            # # Kmeans params
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # K = 8
            #
            # # do Kmeans for image 1
            # ret1,label1,center1=cv2.kmeans(Z1,K,None,criteria,10,cv2.KMEANS_PP_CENTERS 	)
            #
            #
            #
            # ret2,label2,center2=cv2.kmeans(Z2,K,label1,criteria,1,cv2.KMEANS_USE_INITIAL_LABELS)
            # Kmeans ends
            publish_time( pub_time_kmeans, 1000.*(time.time() - startKMeans) )



        except:
            print 'thread:', 'empty'

###
### INIT Ros node
###
rospy.init_node( 'test_node' )
rate = rospy.Rate( 30 )
rospack = rospkg.RosPack()
base_path = rospack.get_path( 'tx2_standalone_test' )


###
### Time Publisher
###
pub_qsize = rospy.Publisher( '/time/qsize', Float32, queue_size=1000)

pub_time_netvlad = rospy.Publisher( '/time/netvlad', Float32, queue_size=1000)
pub_time_kmeans = rospy.Publisher( '/time/kmeans', Float32, queue_size=1000)



###
### Init tensorflow
###
NET_TYPE = "resnet6"
PARAM_K = 16
PARAM_model_restore = base_path+'/tf.models/D/model-8000'

tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )

sess = tf.Session()

# print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
# tensorflow_saver = tf.train.Saver()
# tensorflow_saver.restore( sess, PARAM_model_restore )



###
### Main while-loop
###
S_thumbnails = []
S_netvlad = []
task_queue = Queue.Queue()
dai1 = DaisyMeld(240, 320, 0)
dai2 = DaisyMeld(240, 320, 0)

XFT = True



###
### Launch another thread
###
import threading
th = threading.Thread( target=consume_queue )
th.start()

i = 0
while not rospy.is_shutdown():
    print i, base_path+'/some_images/%d.png' %(i)

    # Read Image
    IM = cv2.imread( base_path+'/some_images/%d.png' %(i) )
    if IM == None or i>200:
        xprint( 'All images over, wait for the queue to be empty', 0 )
        while task_queue.qsize() > 1:
            xprint( 'Sleep for 3 sec', 0 )
            time.sleep( 3 )
        xprint( 'break', 0 )
        break


    i = i + 1

    # # NetVLAD
    im_batch = np.expand_dims( IM.astype('float32'), 0 )
    # im_batch_normalized = normalize_batch( im_batch ) #normalization is done inside computational-graph using tf.image.per_image_standardization()
    im_batch_normalized = im_batch

    # feed_dict = {tf_x : im_batch_normalized,\
    #              is_training:True,\
    #              vgg_obj.initial_t: 0
    #             }
    #


    startNetvlad = time.time()
    tff_vlad_word = sess.run( [tf_vlad_word],  feed_dict={tf_x: im_batch_normalized} )


    # tff_vlad_word, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm], feed_dict=feed_dict)
    # Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1

    publish_time( pub_time_netvlad, 1000.*(time.time() - startNetvlad) )


    S_netvlad.append( tff_vlad_word )
    # colorLUT = ColorLUT()
    # lut = colorLUT.lut( Assgn_matrix[0,:,:] )
    # cv2.imshow( 'IM', IM )
    # cv2.imshow( 'association map', cv2.resize( lut, (320,240) ) )
    # cv2.imwrite( base_path+'/debug_images/%d_im.jpg' %(i), IM )
    # cv2.imwrite( base_path+'/debug_images/%d_amap.jpg' %(i), cv2.resize( lut, (320,240) ) )


    # Dump in S_thumbnails
    S_thumbnails.append( IM )


    # Randomly choose any 2 idx from S_thumbnails as put it onto a queue
    if len(S_thumbnails) < 50:
        continue

    i_curr = np.random.randint( 10, len(S_thumbnails) )
    i_prev = np.random.randint( 2, i_curr )
    T = {}
    T['i_curr'] = i_curr
    T['i_prev'] = i_prev
    task_queue.put( (i_curr, i_prev) )
    xprint(  'QSize: %d' %( task_queue.qsize() ),  0  )

    # cv2.waitKey(1)
    rate.sleep()

print 'Waiting to join'
XFT = False
th.join()
