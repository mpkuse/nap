""" Inspect loop candidates one at a time
        In particular reader for file loop_candidates.npy which is written in
        main nap node as :

        loop_candidates.append( [L, aT, sim_scores_logistic[aT], nMatches, nInliers] )

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 6th July, 2017
"""

import numpy as np
import cv2
import code
cv2.ocl.setUseOpenCL(False)

from GeometricVerification import GeometricVerification


IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail.npy'
IMAGE_FILE_NPY_lut = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut_raw.npy'
LOOP_CANDIDATES_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/loop_candidates.csv'

print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
S_thumbnails_lut = np.load(IMAGE_FILE_NPY_lut)
S_thumbnails_lut_raw = np.load(IMAGE_FILE_NPY_lut_raw)
print 'S_thumbnails.shape : ', S_thumbnails.shape
# for i in range(S_thumbnails.shape[0]):
#     print i, 'of', S_thumbnails.shape[0]
#     cv2.imshow( 'win', S_thumbnails[i,:,:,:] )
#     if cv2.waitKey(0) == 27:
#         break
# quit()

print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )

__c = -1
__p = 0
flag = False
VV = GeometricVerification()
for i,l in enumerate(loop_candidates):
    # [ curr, prev, score, nMatches, nConsistentMatches]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])
    # if __c != curr:
    if (curr - __c) > 3 :
        print '---'
        __c = curr
        __p = -1

    if (prev - __p) > 5:
        print '.'
        flag = True
    __p = prev

    print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)
    # if nMatches > 10:
    # if score > 0.0:
    if flag is True and score>0.4:
        flag = False
        cv2.imshow( 'curr', S_thumbnails[curr, :,:,:] )
        cv2.imshow( 'prev', S_thumbnails[prev, :,:,:] )

        # cv2.imshow( 'curr_lut', cv2.resize( S_thumbnails_lut[curr, :,:,:], (320,240)) )
        # cv2.imshow( 'prev_lut', cv2.resize( S_thumbnails_lut[prev, :,:,:], (320,240)) )
        #
        # alpha_curr = 0.45*S_thumbnails[curr, :,:,:] + 0.35*cv2.resize( S_thumbnails_lut[curr, :,:,:], (320,240))
        # alpha_prev = 0.45*S_thumbnails[prev, :,:,:] + 0.35*cv2.resize( S_thumbnails_lut[prev, :,:,:], (320,240))
        # cv2.imshow( 'alpha_curr', alpha_curr.astype('uint8') )
        # cv2.imshow( 'alpha_prev', alpha_prev.astype('uint8') )
        # # cv2.imwrite( '%d.png' %(curr), S_thumbnails[curr, :,:,:])
        # # cv2.imwrite( '%d.png'%(prev), S_thumbnails[prev, :,:,:])
        #
        #
        #
        # VV.set_im( S_thumbnails[curr, :,:,:] , S_thumbnails[prev, :,:,:] )
        # VV.set_im_lut( cv2.resize(S_thumbnails_lut[curr, :,:,:], (320,240), interpolation=cv2.INTER_NEAREST), cv2.resize(S_thumbnails_lut[prev, :,:,:], (320,240), interpolation=cv2.INTER_NEAREST))
        # VV.set_im_lut_raw( cv2.resize(S_thumbnails_lut_raw[curr, :,:], (320,240), interpolation=cv2.INTER_NEAREST), cv2.resize(S_thumbnails_lut_raw[prev, :,:], (320,240), interpolation=cv2.INTER_NEAREST))
        # VV.obliq_geometry_verify(  )
        # # VV.simple_verify()


        if cv2.waitKey(0) == 27:
            break
