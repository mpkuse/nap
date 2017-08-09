""" Class for Geometric Verification of an image pair/pairs.
    It has a simple verifier based on Lowe's Ratio test and Essential Matrix Test.

    This is a testbed to develop methods for wide angle semantic matching
    algorithms

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 9th August, 2017
"""


import numpy as np
import cv2
import code
cv2.ocl.setUseOpenCL(False)

class GeometricVerification:
    def __init__(self):
        _x = 0
        self.reset()
        self.orb = cv2.ORB_create()
        # self.orb = cv2.xfeatures2d.SURF_create( 400, 5, 5 )

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

    def reset(self):
        self.im1 = None
        self.im2 = None

        self.im1_lut = None
        self.im2_lut = None

    def set_im(self,im1, im2):
        self.im1 = im1
        self.im2 = im2

    def set_im_lut(self,im1_lut, im2_lut):
        self.im1_lut = im1_lut
        self.im2_lut = im2_lut

    def set_im_lut_raw(self,im1_lut_raw, im2_lut_raw):
        self.im1_lut_raw = im1_lut_raw
        self.im2_lut_raw = im2_lut_raw

    def _lowe_ratio_test( self, kp1, kp2, matches_org ):
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

    def simple_verify(self):
        # Simple verification using ORB keypoints and essential matrix
        # 1. Extract Keypoints and descriptor
        kp1, des1 = self.orb.detectAndCompute( self.im1, None )
        kp2, des2 = self.orb.detectAndCompute( self.im2, None )
        # code.interact( local=locals() )
        # kp1, des1 = self.orb.detectAndCompute( self.im1, (self.im1_lut_raw==23).astype('uint8') )
        # kp2, des2 = self.orb.detectAndCompute( self.im2, (self.im2_lut_raw==23).astype('uint8') )

        # 2. Lowe's Ratio Test
        cv2.imshow( 'kp1', cv2.drawKeypoints( self.im1, kp1, None ) )
        cv2.imshow( 'kp2', cv2.drawKeypoints( self.im2, kp2, None ) )
        matches_org = self.flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2) #find 2 nearest neighbours
        __pt1, __pt2 = self._lowe_ratio_test( kp1, kp2, matches_org )
        nMatches = __pt1.shape[0]
        nInliers = 0


        # 3. Essential Matrix test
        E, mask = cv2.findEssentialMat( __pt1, __pt2 )
        if mask is not None:
            nInliers = mask.sum()

        print 'nMatches : ', nMatches
        print 'nInliers : ', nInliers

    def obliq_geometry_verify(self ):

        # Playing with masked detections
        _i = []
        for i in range(64):
            print i, (self.im1_lut_raw==i).sum()
            _i.append((self.im1_lut_raw==i).sum())

        # 1. Extract Keypoints and descriptor
        # kp1, des1 = self.orb.detectAndCompute( self.im1, None )
        # kp2, des2 = self.orb.detectAndCompute( self.im2, None )
        # code.interact( local=locals() )
        print 'np.argmax(_i)) : ', np.argmax(_i)
        kp1, des1 = self.orb.detectAndCompute( self.im1, (self.im1_lut_raw==np.argmax(_i)).astype('uint8') )
        kp2, des2 = self.orb.detectAndCompute( self.im2, (self.im2_lut_raw==np.argmax(_i)).astype('uint8') )

        # 2. Lowe's Ratio Test
        cv2.imshow( 'kp1', cv2.drawKeypoints( self.im1, kp1, None ) )
        cv2.imshow( 'kp2', cv2.drawKeypoints( self.im2, kp2, None ) )
        matches_org = self.flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2) #find 2 nearest neighbours
        __pt1, __pt2 = self._lowe_ratio_test( kp1, kp2, matches_org )
        nMatches = __pt1.shape[0]
        nInliers = 0


        # 3. Essential Matrix test
        if len(__pt1) > 0 and len(__pt2) > 0:
            E, mask = cv2.findEssentialMat( __pt1, __pt2 )
            if mask is not None:
                nInliers = mask.sum()

        print 'nMatches : ', nMatches
        print 'nInliers : ', nInliers
