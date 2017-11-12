""" This script is used to develop guided 2-way match using Daisy

        Will load the loop closure data which includes,
        - images
        - neural net cluster assignments
        - list of loop-closure candidates (giving index of prev and curr)
        - Feature factory (per keyframe tracked-features data from VINS)

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 7th Nov, 2017
"""


import numpy as np
import cv2
import code
import time
import sys

from operator import itemgetter
cv2.ocl.setUseOpenCL(False)

from GeometricVerification import GeometricVerification
from ColorLUT import ColorLUT

from DaisyMeld.daisymeld import DaisyMeld

import TerminalColors
tcol = TerminalColors.bcolors()
from FeatureFactory import FeatureFactory


class DaisyExMatcher:
    def __init__(self):
        self.dai = []
        self.dai.append( DaisyMeld( 240, 320, 0 ) )
        self.dai.append( DaisyMeld( 240, 320, 0 ) )

        index_params = dict( algorithm=0, trees=5 )
        search_params = dict( checks=20 )
        self.flann = cv2.FlannBasedMatcher( index_params, search_params )

    def index2rc( self, idx ):
        w = 320
        return idx/320, idx% 320
    def index2xy( self, idx ):
        w = 320
        return idx% 320, idx/320

    def get_daisy_view(self, im, ch ):
        im32 = im[:,:,0].copy().astype( 'float32' )
        self.dai[ch].do_daisy_computation( im32 )
        return self.dai[ch].get_daisy_view()

    def match2_guided( self, im_curr, pts_curr, im_prev ):
        startD = time.time()
        daisy_curr = self.get_daisy_view( im_curr, ch=0 )
        daisy_prev = self.get_daisy_view( im_prev, ch=1 )
        print '2 daisy performed in %4.2f (ms): ' % (1000.*(time.time() - startD))

        print 'nPoints : ', pts_curr.shape[1]
        nPoints = pts_curr.shape[1]

        pt_idx = 3
        startT = time.time()
        for pt_idx in range(nPoints):
            # print '---', pt_idx
            startA = time.time()
            pt_i = []
            # Add neighouring pts
            for u in [-2,0,2]:
                for v in [-2,0,-2]:
                    _pt = [u+int(pts_curr[0,pt_idx]) , v+int(pts_curr[1,pt_idx])  ]
                    if _pt[0] >= 0 and _pt[0] < im_curr.shape[1] and _pt[1] >=0 and _pt[1] < im_curr.shape[0]:
                        pt_i.append( _pt )
            pt_i = np.array( pt_i )

            try:
                A = daisy_curr[ pt_i[:,1], pt_i[:,0], : ] #9x20
            except:
                #
                continue
            print 'A.elapsed (ms) : %4.2f' %(1000.*(time.time() - startA) )

            # pt_i = np.array( [ int(pts_curr[0,pt_idx]), int(pts_curr[1,pt_idx]) ] )
            # A = daisy_curr[ np.int0(pts_curr[1,0:10]), np.int0(pts_curr[0,0:10]), : ] #5x20

            # A = np.expand_dims( daisy_curr[ int(pts_curr[1,pt_idx]), int(pts_curr[0,pt_idx]), : ], 0) #1x20
            xNx = daisy_prev.shape[0]*daisy_prev.shape[1]
            B = daisy_prev.reshape( xNx, -1 ) #w*h x 20
            # B_sampled = B[range(0,xNx,2),:]
            # code.interact( local=locals() )

            startB = time.time()
            matches = self.flann.knnMatch( A, B, k=1 )
            print 'B.elapsed (ms) : %4.2f' %(1000.*(time.time() - startB) )

            startC = time.time()
            Q = []
            for j in range(pt_i.shape[0]):
                for i in range(1):
                    # print matches[j][i].queryIdx, matches[j][i].trainIdx, matches[j][i].distance, self.index2rc(matches[j][i].trainIdx)
                    # Q.append( self.index2xy(matches[j][i].trainIdx) )
                    Q.append( self.index2xy(matches[j][i].trainIdx) )
            Q = np.transpose( np.array(Q) )
            # print Q
            print 'C.elapsed (ms) : %4.2f' %(1000.*(time.time() - startC) )

            # continue

            cv2.imshow( 'daisy_curr', daisy_curr[:,:,0] )
            cv2.imshow( 'daisy_prev', daisy_prev[:,:,0] )

            # dst = self.points_overlay( im_curr, pts_curr, enable_text=True, show_index=[pt_idx] )
            dst = self.points_overlay( im_curr, np.transpose(pt_i), enable_text=True )
            dst2 = self.points_overlay( im_prev, Q, enable_text=True, show_index=None )
            cv2.imshow( 'im_curr pts overlay', dst )
            cv2.imshow( 'im_prev nn pts overlay', dst2 )
            cv2.waitKey(0)

        print 'Matching Time Elapsed : %4.2f' %( 1000.* (time.time() - startT) )
        code.interact( local=locals() )



    def match2_guided_2pointset( self, im_curr, pts_curr, im_prev, pts_prev):
        """ Given images with tracked feature2d pts. Returns matched points
            using the voting mechanism. Loosely inspired from GMS-matcher

            im_curr : Current Image
            pts_curr : Detected points in current image (3xN) or (2xN)
            im_prev : Previous Image
            pts_prev : Detected points in prev image (3xM) or (2xN)

            Note: points in pts_curr and pts_prev might be different in count.
        """
        daisy_curr = self.get_daisy_view( im_curr, ch=0 )
        daisy_prev = self.get_daisy_view( im_prev, ch=1 )
        DEBUG = True


        W = 1
        # _R = range( -W, W+1 )
        _R = [-4,0,4]
        if DEBUG:
            print 'im_curr.shape', im_curr.shape
            print 'im_prev.shape', im_prev.shape
            print 'pts_curr.shape', pts_curr.shape
            print 'pts_prev.shape', pts_prev.shape
            print 'neighbourhood: ', _R

        # wxw around each of curr
        A = []    # Nd x 20. Nd is usually ~ 2000
        A_i = []  # Nd x 1 (index)
        A_pt = [] # Nd x 2 (every 2d pt)
        for pt_i in range( pts_curr.shape[1] ):
            for u in _R:#range(-W, W+1):
                for v in _R:#range(-W,W+1):
                    pt = np.int0( pts_curr[0:2, pt_i ] ) + np.array([u,v])
                    if pt[1] < 0 or pt[1] >= im_curr.shape[0] or pt[0] < 0 or pt[0] >= im_curr.shape[1]:
                        continue
                    # print pt_i, pt, im_curr.shape
                    A.append( daisy_curr[ pt[1], pt[0], : ] )
                    A_pt.append( pt )
                    A_i.append( pt_i )

        # wxw around each of prev
        B = []
        B_i = []
        B_pt = []
        for pt_i in range( pts_prev.shape[1] ):
            for u in -R:#range(-W,W+1):
                for v in -R:#range(-W,W+1):
                    pt = np.int0( pts_prev[0:2, pt_i ] )+ np.array([u,v])
                    if pt[1] < 0 or pt[1] >= im_prev.shape[0] or pt[0] < 0 or pt[0] >= im_prev.shape[1]:
                        continue
                    # print pt_i, pt
                    B.append( daisy_prev[ pt[1], pt[0], : ] )
                    B_pt.append( pt )
                    B_i.append( pt_i )


        if DEBUG:
            print 'len(A)', len(A)
            print 'len(B)', len(B)

        #
        # FLANN Matching
        startflann = time.time()
        matches = self.flann.knnMatch( np.array(A), np.array(B), k=1 )
        if DEBUG:
            print 'time elaspsed for flann : %4.2f (ms)' %(1000.0 * (time.time() - startflann) )


        # Loop over each match and do voting
        startvote = time.time()
        vote = np.zeros( ( pts_curr.shape[1], pts_prev.shape[1] ) )
        for mi, m in enumerate(matches):
            # m[0].queryIdx <--> m[0].trainIdx
            # m[1].queryIdx <--> m[1].trainIdx
            # .
            # m[k].queryIdx <--> m[k].trainIdx

            # print mi, m[0].queryIdx, m[0].trainIdx, m[0].distance
            # print A_i[m[0].queryIdx], B_i[ m[0].trainIdx ]

            for k in range( len(m) ):
                ptA = A_pt[ m[k].queryIdx ]
                ptB = B_pt[ m[k].trainIdx ]

                iA = A_i[ m[k].queryIdx ]
                iB = B_i[ m[k].trainIdx ]
                # print iA, iB
                vote[ iA, iB ] += 1.0/(1.0+k*k)

        if DEBUG:
            print 'time elaspsed for voting : %4.2f (ms)' %( 1000. * (time.time() - startvote) )


            # cv2.imshow( 'curr_overlay', self.points_overlay( im_curr, np.expand_dims(ptA,1)) )
            # cv2.imshow( 'prev_overlay', self.points_overlay( im_prev, np.expand_dims(ptB,1)) )
            # cv2.waitKey(0)
        selected_A = []
        selected_A_i = []
        selected_B = []
        selected_B_i = []
        startSelect = time.time()
        for i in range( vote.shape[0] ):
            iS = vote[i,:]
            nz = iS.nonzero()
            if sum(iS) <= 0:
                continue
            iS /=  sum(iS)




            top = iS[nz].max()
            toparg = iS[nz].argmax()
            top2 = self.second_largest( iS[nz] )

            if DEBUG:
                print i, nz, iS[nz]
                print 'toparg=',toparg, 'top=',round(top,2), 'top2=',round(top2,2)

            if top/top2 >= 2.0 or top2 < 0:

                # i <---> nz[0]
                # i <---> nz[1]
                # ...
                ptxA = np.int0(   np.expand_dims( pts_curr[0:2,i], 1 )   ) # current point in curr
                ptxB = np.int0(   pts_prev[0:2,nz][:,0,:]    )             # All candidates
                ptyB = np.int0(   np.expand_dims( ptxB[0:2,toparg], 1 )   )# top candidate only

                selected_A.append( ptxA )
                selected_B.append( ptyB )

                selected_A_i.append( i )
                selected_B_i.append( nz[0][toparg] )

                if DEBUG and False:
                    print 'ACCEPTED'
                    cv2.imshow( 'curr_overlay', self.points_overlay( im_curr, ptxA, enable_text=True ))
                    cv2.imshow( 'prev_overlay', self.points_overlay( im_prev, ptxB, enable_text=True ))
                    cv2.imshow( 'prev_overlay top', self.points_overlay( im_prev, ptyB, enable_text=True ))
                    cv2.waitKey(0)

        selected_A = np.transpose(  np.array( selected_A )[:,:,0]  )
        selected_B = np.transpose(  np.array( selected_B )[:,:,0]  )

        if DEBUG:
            print 'time elaspsed for selection from voting : %4.2f (ms)' %( 1000. * (time.time() - startSelect) )

            # cv2.imshow( 'xxx' , self.plot_point_sets( im_curr, np.int0(pts_curr[0:2,selected_A_i]), im_prev, np.int0(pts_prev[0:2,selected_B_i] ) ) )

        #
        # Fundamental Matrix Text
        startFundamentalMatrixTest = time.time()
        E, mask = cv2.findFundamentalMat( np.transpose( selected_A ), np.transpose( selected_B ),param1=5 )
        if mask is not None:
            nInliers = mask.sum()

        masked_pts_curr = np.transpose( np.array( list( selected_A[:,i] for i in np.where( mask[:,0] == 1 )[0] ) ) )
        masked_pts_prev = np.transpose( np.array( list( selected_B[:,i] for i in np.where( mask[:,0] == 1 )[0] ) ) )

        if DEBUG:
            print 'nInliers : ', nInliers
            print 'time elaspsed for fundamental matrix test : %4.2f (ms)' %( 1000. * (time.time() - startFundamentalMatrixTest) )


        masked_selected_A_i = list( selected_A_i[q] for q in np.where( mask[:,0] == 1 )[0] )
        masked_selected_B_i = list( selected_B_i[q] for q in np.where( mask[:,0] == 1 )[0] )



        if DEBUG:
            cv2.imshow( 'curr_overlay', self.points_overlay( im_curr, pts_curr) )
            cv2.imshow( 'prev_overlay', self.points_overlay( im_prev, pts_prev) )
            cv2.imshow( 'selected', self.plot_point_sets( im_curr, selected_A, im_prev, selected_B) )
            cv2.imshow( 'selected+fundamentalmatrixtest', self.plot_point_sets( im_curr, masked_pts_curr, im_prev, masked_pts_prev) )

            cv2.imshow( 'yyy selected+fundamentalmatrixtest', self.plot_point_sets( im_curr, np.int0(pts_curr[0:2,masked_selected_A_i]), im_prev, np.int0(pts_prev[0:2,masked_selected_B_i])  ) )


        # Return the masked index in the original
        return np.array(masked_selected_A_i), np.array(masked_selected_B_i)

    def second_largest(self,numbers):
        count = 0
        m1 = m2 = float('-inf')
        for x in numbers:
            count += 1
            if x > m2:
                if x >= m1:
                    m1, m2 = x, m1
                else:
                    m2 = x
        return m2 if count >= 2 else -1 #None

    # Image, pts 2xN
    def points_overlay(self, im_org, pts, color=(255,0,0), enable_text=False, show_index=None ):
        im = im_org.copy()
        if len(im.shape) == 2:
            im = cv2.cvtColor( im, cv2.COLOR_GRAY2BGR )

        if pts.shape[0] == 3: #if input is 3-row mat than  it is in homogeneous so perspective divide
            pts = pts / pts[2,:]

        color_com = ( 255 - color[0] , 255 - color[1], 255 - color[2] )

        if show_index is None:
            rr = range( pts.shape[1] )
        else:
            rr = show_index

        for i in rr:
            cv2.circle(  im, tuple(np.int0(pts[0:2,i])), 3, color, -1 )
            if enable_text:
                cv2.putText( im, str(i), tuple(np.int0(pts[0:2,i])), cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )

        return im



    def plot_point_sets( self, im1, pt1, im2, pt2, mask=None, color=(255,0,0), enable_text=True, enable_lines=False ):
        """ pt1, pt2 : 2xN """
        xcanvas = np.concatenate( (im1, im2), axis=1 )
        for xi in range( pt1.shape[1] ):
            if (mask is not None) and (mask[xi,0] == 0):
                continue

            pta = tuple(pt1[0:2,xi])
            ptb = tuple(np.array(pt2[0:2,xi]) + [im1.shape[1],0])

            cv2.circle( xcanvas, pta, 4, color )
            cv2.circle( xcanvas, ptb, 4, color )


            color_com = ( 255 - color[0] , 255 - color[1], 255 - color[2] )

            if enable_text:
                cv2.putText( xcanvas, str(xi), pta, cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )
                cv2.putText( xcanvas, str(xi), ptb, cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )

            if enable_lines:
                cv2.line( xcanvas, pta, ptb, (255,0,0) )
        return xcanvas


# Image, pts 2xN
def points_overlay( im_org, pts, color=(255,0,0), enable_text=False, show_index=None ):
    im = im_org.copy()
    if len(im.shape) == 2:
        im = cv2.cvtColor( im, cv2.COLOR_GRAY2BGR )

    if pts.shape[0] == 3: #if input is 3-row mat than  it is in homogeneous so perspective divide
        pts = pts / pts[2,:]

    color_com = ( 255 - color[0] , 255 - color[1], 255 - color[2] )

    if show_index is None:
        rr = range( pts.shape[1] )
    else:
        rr = show_index

    for i in rr:
        cv2.circle(  im, tuple(np.int0(pts[0:2,i])), 3, color, -1 )
        if enable_text:
            cv2.putText( im, str(i), tuple(np.int0(pts[0:2,i])), cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )

    return im


def plot_point_sets( im1, pt1, im2, pt2, mask=None, color=(255,0,0), enable_text=True, enable_lines=False ):
    """ pt1, pt2 : 2xN """
    xcanvas = np.concatenate( (im1, im2), axis=1 )
    for xi in range( pt1.shape[1] ):
        if (mask is not None) and (mask[xi,0] == 0):
            continue

        pta = tuple(pt1[0:2,xi])
        ptb = tuple(np.array(pt2[0:2,xi]) + [im1.shape[1],0])

        cv2.circle( xcanvas, pta, 4, color )
        cv2.circle( xcanvas, ptb, 4, color )


        color_com = ( 255 - color[0] , 255 - color[1], 255 - color[2] )

        if enable_text:
            cv2.putText( xcanvas, str(xi), pta, cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )
            cv2.putText( xcanvas, str(xi), ptb, cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )

        if enable_lines:
            cv2.line( xcanvas, pta, ptb, (255,0,0) )
    return xcanvas

#
# Specify Files
#
BASE__DUMP = '/home/mpkuse/Desktop/a/drag_nap'
KF_TIMSTAMP_FILE_NPY = BASE__DUMP+'/S_timestamp.npy'
IMAGE_FILE_NPY = BASE__DUMP+'/S_thumbnail.npy'
IMAGE_FILE_NPY_lut = BASE__DUMP+'/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = BASE__DUMP+'/S_thumbnail_lut_raw.npy'

LOOP_CANDIDATES_NPY = BASE__DUMP+'/loop_candidates.csv'

FEATURE_FACTORY = BASE__DUMP+'/FeatureFactory'

#
# Load Files
#
print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
S_timestamp = np.load(KF_TIMSTAMP_FILE_NPY)
print 'Reading : ', IMAGE_FILE_NPY_lut
S_thumbnails_lut = np.load(IMAGE_FILE_NPY_lut)
print 'Reading : ', IMAGE_FILE_NPY_lut_raw
S_thumbnails_lut_raw = np.load(IMAGE_FILE_NPY_lut_raw)
print 'S_thumbnails.shape : ', S_thumbnails.shape


print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )


feature_factory = FeatureFactory()
feature_factory.load_from_pickle( FEATURE_FACTORY )

#
# #
# # Loop Over every frame
# for i in range( S_thumbnails.shape[0] ):
#     curr_im = S_thumbnails[i, :,:,:]
#     t_curr = S_timestamp[i]
#
#     feat2d_curr_idx = feature_factory.find_index( t_curr )
#     feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
#     print feat2d_curr.shape
#
#     cv2.imshow( 'im', points_overlay( curr_im, feat2d_curr) )
#     cv2.waitKey(30)
# quit()

# #
# # Loop over every loop-closure
# for li,l in enumerate(loop_candidates):
#     curr = int(l[0])
#     prev = int(l[1])
#
#     t_curr = S_timestamp[curr]
#     t_prev = S_timestamp[prev]
#
#
#     print li, curr, prev
#     feat2d_curr_idx = feature_factory.find_index( t_curr )
#     feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
#
#     feat2d_prev_idx = feature_factory.find_index( t_prev )
#     feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] ) #3xN in homogeneous cords
#     print feat2d_curr.shape, feat2d_prev.shape
#
#
# quit()

# which loop_candidates to load
if len(sys.argv) == 2:
    i=int(sys.argv[1])
else:
    i = 0

for i in range( len(loop_candidates) ):
    print '==='
    l = loop_candidates[i]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])


    print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)


    curr_im = S_thumbnails[curr, :,:,:]
    prev_im = S_thumbnails[prev, :,:,:]
    curr_lut = S_thumbnails_lut[curr,:,:]
    prev_lut = S_thumbnails_lut[prev,:,:]
    t_curr = S_timestamp[curr]
    t_prev = S_timestamp[prev]

    feat2d_curr_idx = feature_factory.find_index( t_curr )
    feat2d_prev_idx = feature_factory.find_index( t_prev )

    feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
    feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] )
    print feat2d_curr.shape
    print feat2d_prev.shape



    # matcher = DaisyExMatcher()
    matcher_geom = GeometricVerification()
    # matcher.match2_guided( curr_im, feat2d_curr, prev_im )
    # quit()
    startT = time.time()
    selected_curr_i, selected_prev_i = matcher_geom.match2_guided_2pointset( curr_im, feat2d_curr, prev_im, feat2d_prev )
    print 'matcher.match2_guided_2pointset() : %4.2f (ms)' %(1000. * (time.time() - startT) )
    print selected_curr_i.shape
    print selected_prev_i.shape
    cv2.imshow( 'main selected+fundamentalmatrixtest', plot_point_sets( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i])  ) )
    cv2.imshow( 'curr_im', points_overlay( curr_im, feat2d_curr) )
    cv2.imshow( 'prev_im', points_overlay( prev_im, feat2d_prev) )



    #
    # cv2.imshow( 'curr', curr_im )
    # cv2.imshow( 'prev', prev_im )
    #
    # cv2.imshow( 'curr_lut', curr_lut )
    # cv2.imshow( 'prev_lut', prev_lut )
    cv2.waitKey(0)
