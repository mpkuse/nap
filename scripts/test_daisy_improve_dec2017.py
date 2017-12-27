""" An attempt to improve the 3way matching false positive.
    Especially looking at a simple mechanism to identify false matches
    early. Will start from current implementation of geometryverify
    and improve upon it.


        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 13th Dec, 2017
"""



import numpy as np
import cv2
import code
import time
import sys
import matplotlib.pyplot as plt
from operator import itemgetter
cv2.ocl.setUseOpenCL(False)

from GeometricVerification import GeometricVerification
from ColorLUT import ColorLUT

from DaisyMeld.daisymeld import DaisyMeld

import TerminalColors
tcol = TerminalColors.bcolors()
from FeatureFactory import FeatureFactory

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



#
# Specify Files
#
BASE__DUMP = '/home/mpkuse/Desktop/a/drag_nap'
KF_TIMSTAMP_FILE_NPY = BASE__DUMP+'/S_timestamp.npy'
IMAGE_FILE_NPY = BASE__DUMP+'/S_thumbnail.npy'
IMAGE_FILE_NPY_lut = BASE__DUMP+'/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = BASE__DUMP+'/S_thumbnail_lut_raw.npy'

LOOP_CANDIDATES_NPY = BASE__DUMP+'/loop_candidates2.csv'

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

# # Loop Over every frame
# for i in range( S_thumbnails.shape[0] ):
#     curr_im = S_thumbnails[i, :,:,:]
#     t_curr = S_timestamp[i]
#     lut_raw = S_thumbnails_lut_raw[i,:,:]
#     lut = S_thumbnails_lut[i,:,:,:]
#
#     feat2d_curr_idx = feature_factory.find_index( t_curr )
#     feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
#     print feat2d_curr.shape
#     print 'lut_raw.uniq ', len( np.unique(lut_raw[:]))
#
#     cv2.imshow( 'im', points_overlay( curr_im, feat2d_curr) )
#     cv2.imshow( 'lut', lut.astype('uint8') )
#
#     # plt.hist( lut_raw[:], 64 )
#     # plt.show( False )
#     cv2.waitKey(0)
#
# quit()
###############################################################################
VV = GeometricVerification()
human_a = 0
human_s = 0
human_d = 0
human_f = 0
# for i in [10]: #range( len(loop_candidates) ):
for i in range( len(loop_candidates)):
    print '==='
    l = loop_candidates[i]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])


    print tcol.OKBLUE, '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches), tcol.ENDC
    t_curr = S_timestamp[curr]
    t_prev = S_timestamp[prev]

    curr_im = S_thumbnails[curr, :,:,:]
    prev_im = S_thumbnails[prev, :,:,:]

    curr_lut = S_thumbnails_lut[curr,:,:,:]
    curr_lut_raw = S_thumbnails_lut_raw[curr,:,:]
    prev_lut = S_thumbnails_lut[prev,:,:,:]
    prev_lut_raw = S_thumbnails_lut_raw[prev,:,:]

    feat2d_curr_idx = feature_factory.find_index( t_curr )
    feat2d_prev_idx = feature_factory.find_index( t_prev )

    feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
    feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] )
    print feat2d_curr.shape
    print feat2d_prev.shape

    # cv2.imshow( 'curr_im', curr_im )
    # cv2.imshow( 'prev_im', prev_im )
    cv2.imshow( 'curr_im', points_overlay( curr_im, feat2d_curr) )
    cv2.imshow( 'prev_im', points_overlay( prev_im, feat2d_prev) )

    cv2.imshow( 'curr_lut', curr_lut )
    cv2.imshow( 'prev_lut', prev_lut )


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


    xcanvas_2way = VV.plot_2way_match( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i]),  enable_lines=True )
    cv2.imshow( 'xcanvas_2way', xcanvas_2way )

    if match2_total_score > 3:
        # Accept this match and move on
        print 'Accept this match and move on'
        print tcol.OKGREEN, 'Accept (Strong)', tcol.ENDC
        pass

    if match2_total_score > 2 and match2_total_score <= 3 and len(selected_curr_i) > 20:
        # Boundry case, if you see sufficient number of 2way matches, also accpt 2way match
        print 'Boundary case, if you see sufficient number of 2way matches, also accept 2way match'
        print tcol.OKGREEN, 'Accept', tcol.ENDC


    if match2_total_score >= 0.5 and match2_total_score <= 3:
        # Try 3way. But plot 2way and 3way.
        # Beware, 3way match function returns None when it has early-rejected the match
        print 'Attempt robust_3way_matching()'

        # set-data
        curr_m_im = S_thumbnails[curr-1,:,:,:]

        VV.set_image( curr_m_im, 3 )  #set curr-1 image
        VV.set_lut_raw( curr_lut_raw, 1 ) #set lut of curr and prev
        VV.set_lut_raw( prev_lut_raw, 2 )
        VV.set_lut( curr_lut, 1 ) #only needed for in debug mode of 3way match
        VV.set_lut( prev_lut, 2 ) #only needed for in debug mode of 3way match

        # Attempt 3way match
        q1,q2,q3,q4,q5 = VV.robust_match3way()
        print 'dense_match_quality     : ', q5[0]
        print 'after_vote_match_quality: ', q5[1]

        if q1 is None:
            print 'Early Reject from robust_match3way()'
            print tcol.FAIL, 'Reject', tcol.ENDC

        else:
            print 'nPts_3way_match     : ', q1.shape
            print 'Accept 3way match'
            print tcol.OKGREEN, 'Accept', tcol.ENDC
            gridd = VV.plot_3way_match( VV.im1, np.array(q1), VV.im2, np.array(q2), VV.im3, np.array(q3) )
            cv2.imshow( '3way Matchi', gridd )



    if match2_total_score < 0.5:
        # Reject (don't bother computing 3way)
        print 'Reject 2way matching, and do not compute 3way matching'
        print tcol.FAIL, 'Reject (Strong)', tcol.ENDC

    key_press = cv2.waitKey(0)
    if key_press == ord('a'):
        print tcol.HEADER, '[HUMAN-a] False match accepted by geometry', tcol.ENDC
        human_a += 1
    if key_press == ord('s'):
        print tcol.HEADER, '[HUMAN-s] True  match rejected by geometry', tcol.ENDC
        human_s += 1
    if key_press == ord('d'):
        print tcol.HEADER, '[HUMAN-d] Yes-Yes case', tcol.ENDC
        human_d += 1
    if key_press == ord('f'):
        print tcol.HEADER, '[HUMAN-f] No-No case', tcol.ENDC
        human_f += 1


print tcol.HEADER, '[HUMAN-a:%d] False match accepted by geometry' %(human_a), tcol.ENDC
print tcol.HEADER, '[HUMAN-s:%d] True  match rejected by geometry' %(human_s), tcol.ENDC
print tcol.HEADER, '[HUMAN-d:%d] Yes-Yes case' %(human_d), tcol.ENDC
print tcol.HEADER, '[HUMAN-f:%d] No-No case' %(human_f), tcol.ENDC
