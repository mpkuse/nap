#!/usr/bin/env python
import rospy
import rospkg
from std_msgs.msg import Float32

from multiprocessing import Process, Queue, Manager


import cv2
import numpy as np
import time
import code
import os

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

################ Util functions ####################
def xprint( msg, threadId ):
    print '[%02d]' %(threadId), msg

def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )
####################################################


def image_reader( Qw, base_path, Q_time_imread ):
    """ produces the queue `Qw` """

    rate = rospy.Rate(30)
    i = 0
    while True:
        startTime = time.time()
        fname = base_path+'/some_images/%d.png' %(i)

        if i % 10 == 0 :
            xprint( 'READ Image %s'  %(fname), os.getpid() )

        IM = cv2.imread( fname )
        if IM == None or i>200:
            xprint( 'All images over. Exit this thread', os.getpid() )
            Qw.close()
            Q_time_imread.close()
            return

        Qw.put( IM )
        rate.sleep()
        i = i+1
        # publish_time( pub_time_reader, 1000.*(time.time() - startTime) )
        Q_time_imread.put(   1000.*(time.time() - startTime)  )
    return


def worker_gpu( Qw, Qd, base_path, S_thumbnails, S_netvlad, Q_time_gpu  ):
    """ Consumes Qw, produces Qd. Also fills up S_thumbnails, S_netvlad"""

    ###
    ### Init Tensorflow
    ###
    NET_TYPE = "resnet6"
    PARAM_K = 16
    PARAM_model_restore = base_path+'/tf.models/D/model-8000'


    # Create tf queue #TODO


    tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
    is_training = tf.placeholder( tf.bool, [], name='is_training')
    vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
    tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )

    sess = tf.Session()
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver = tf.train.Saver()
    tensorflow_saver.restore( sess, PARAM_model_restore )



    while Qw.qsize() > 0 :
        startGPU = time.time()
        xprint( 'qsize:%d' %(Qw.qsize() ), os.getpid() )
        im = Qw.get()

        # GPU Computations on `im `
        im_batch = np.expand_dims( im.astype('float32'), 0 ) #unnormalized batch. Nx240x320x3
        feed_dict = {tf_x : im_batch,\
                     is_training:True,\
                     vgg_obj.initial_t: 0
                    }
        tff_vlad_word, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm], feed_dict=feed_dict)
        Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1


        # Put the results in lists S_thumbnails, S_netvlad
        S_thumbnails.append( im )
        S_netvlad.append( tff_vlad_word )



        # Make an element in Qd (simulate loop detection)
        if len(S_thumbnails) < 11:
            continue

        i_curr = np.random.randint( 9, len(S_thumbnails) )
        i_prev = np.random.randint( 2, i_curr )
        T = {}
        T['i_curr'] = i_curr
        T['i_prev'] = i_prev
        Qd.put( (i_curr, i_prev) )



        Q_time_gpu.put(   1000.*(time.time() - startGPU)  )

    print 'close()'
    Qw.close()
    Qd.close()
    Q_time_gpu.close()

def worker_cpu( Qd, S_thumbnails, Q_time_cpu ):
    """ Consumes Qd. read-only access to S_thumbnails """

    g = Qd.get() #block until the queue has something

    dai1 = DaisyMeld(240, 320, 0)
    dai2 = DaisyMeld(240, 320, 0)

    while True:
        try:
            g = Qd.get(timeout=1)
        except:
            break # No more. Adjust this as need be. beware of sleep at start of this function
        xprint( 'i_curr: %d, i_prev: %d' %( g[0], g[1] ), os.getpid() )

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
        # time.sleep( 0.4 )

        Q_time_cpu.put(   1000.*(time.time() - startKMeans)  )





    Qd.close()
    Q_time_cpu.close()



##########
#  Main #
#########
if __name__ == "__main__":
    ###
    ### INIT Ros node
    ###
    rospy.init_node( 'test_node' )
    rospack = rospkg.RosPack()
    base_path = rospack.get_path( 'tx2_standalone_test' )


    n_cpu_procs = 2

    ###
    ### Queues
    ###
    Qw = Queue()
    Qd = Queue()
    Q_time_imread = Queue()
    Q_time_gpu = Queue()

    Q_time_cpu = []  #----(ncpu)
    for pr in range(n_cpu_procs):
        Q_time_cpu.append( Queue() )




    ###
    ### Shared global data holders
    ###
    manager = Manager()
    S_thumbnails = manager.list()
    S_netvlad = manager.list()

    ###
    ### Time publishers
    ###
    pub_Qw_size = rospy.Publisher( '/time/Qw_qsize', Float32, queue_size=1000)
    pub_Qd_size = rospy.Publisher( '/time/Qd_qsize', Float32, queue_size=1000)


    pub_time_reader = rospy.Publisher( '/time/reader', Float32, queue_size=1000)
    pub_time_netvlad = rospy.Publisher( '/time/netvlad', Float32, queue_size=1000)
    pub_time_kmeans = [] #----(ncpu)
    for pr in range(n_cpu_procs):
        tmp = rospy.Publisher( '/time/kmeans%d' %(pr), Float32, queue_size=1000)
        pub_time_kmeans.append( tmp )



    ###
    ### Launch Processes
    ###
    p_imagereader = Process( target=image_reader, args=( Qw,base_path, Q_time_imread )   )
    p_gpu = Process( target=worker_gpu, args=(Qw, Qd, base_path, S_thumbnails, S_netvlad, Q_time_gpu) )
    p_cpu = [] #----(ncpu)
    for pr in range(n_cpu_procs):
        tmp = Process( target=worker_cpu, args=(Qd, S_thumbnails, Q_time_cpu[pr]) )
        p_cpu.append( tmp )


    print 'Launch process. Main PID=%d' %(os.getpid() )
    p_imagereader.start()
    p_gpu.start()
    for i_cpu in p_cpu: #----(ncpu)
        i_cpu.start()
    print 'All proc started. Waiting to join() '


    rate = rospy.Rate(20)
    # while True:
    while not rospy.is_shutdown():
        xprint( 'MAIN Q_time_imread=%02d Q_time_gpu=%02d ' %(Q_time_imread.qsize(), Q_time_gpu.qsize() ), os.getpid() )
        try:
            publish_time( pub_time_reader, Q_time_imread.get_nowait() )
            # xprint( 'MAIN1', os.getpid() )
        except:
            pass

        try:
            publish_time( pub_time_netvlad, Q_time_gpu.get_nowait() )
            # xprint( 'MAIN2', os.getpid() )
        except:
            pass

        for pr in range( n_cpu_procs):
            try:
                publish_time( pub_time_kmeans[pr], Q_time_cpu[pr].get_nowait() )
                # xprint( 'MAIN2', os.getpid() )
            except:
                pass

        publish_time( pub_Qw_size, Qw.qsize() )
        publish_time( pub_Qd_size, Qd.qsize() )
        rate.sleep()

    print 'Main Done0'

    p_imagereader.join()
    print 'p_imagereader.join()'
    p_gpu.join()
    print 'p_gpu.join()'
    for i_cpu in p_cpu: #----(ncpu)
        i_cpu.join()
        print 'cpu.join()'

    print 'Main Done'
    code.interact( local=locals() )
