from multiprocessing import Queue
import cv2
cv2.ocl.setUseOpenCL(False)
from cv_bridge import CvBridge, CvBridgeError

class ImageReceiver:
    """ This class holds the queue of received images """
    def __init__(self, PARAM_CALLBACK_SKIP, keep_full_resolution=False):
        """
        PARAM_CALLBACK_SKIP : Keep one of `PARAM_CALLBACK_SKIP` number of frames
        """
        self.PARAM_CALLBACK_SKIP = PARAM_CALLBACK_SKIP
        self.call_q = 0

        #
        # Setup internal Image queue
        #note that these are multiprocessing.Queues and not Queue.Queue
        self.im_queue = Queue()
        if keep_full_resolution:
            self.im_queue_full_res = Queue()
        else:
            self.im_queue_full_res = None #Queue.Queue() #Uncomment this if you need access to full resolution images
        self.im_timestamp_queue = Queue()

    def _print( self, msg ):
        pass
        # print "[ImageReceiver]", msg


    def color_image_callback( self, data ):
        """ ROS Image Topic callback """
        n_SKIP = self.PARAM_CALLBACK_SKIP

        self._print( 'Received Image (call_q=%4d): %d,%d' %( self.call_q, data.height, data.width ) )
        cv_image = CvBridge().imgmsg_to_cv2( data, 'bgr8' )

        if self.call_q%n_SKIP == 0: #only use 1 out of 10 images
            # self.im_queue.put( cv_image )
            self._print( 'Enqueue Image and Timestamp. size : '+ str(cv_image.shape) )
            if self.im_queue_full_res is not None:
                    self.im_queue_full_res.put(cv_image )
            self.im_queue.put(cv2.resize(cv_image, (320,240) ) )
            self.im_timestamp_queue.put(data.header.stamp)
        self.call_q = self.call_q + 1

    def qclose(self):
        self.im_queue.close()
        self.im_timestamp_queue.close() #note that these are multiprocessing.Queues and not Queue.Queue
        if self.im_queue_full_res is not None:
            self.im_queue_full_res.close()
