""" Inspect loop candidates one at a time
        In particular reader for file loop_candidates.npy which is written in
        main nap node as :

        loop_candidates.append( [L, aT, sim_scores_logistic[aT], nMatches, nInliers] )

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 6th July, 2017
"""

import numpy as np
import cv2

# IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail.npy'
IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut.npy'
LOOP_CANDIDATES_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/loop_candidates.csv'

print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
print 'S_thumbnails.shape : ', S_thumbnails.shape
# for i in range(S_thumbnails.shape[0]):
#     print i, 'of', S_thumbnails.shape[0]
#     cv2.imshow( 'win', S_thumbnails[i,:,:,:] )
#     if cv2.waitKey(0) == 27:
#         break
# quit()

print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )


for i,l in enumerate(loop_candidates):
    # [ curr, prev, score, nMatches, nConsistentMatches]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])
    print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)
    # if nMatches > 10:
    if score > 0.6:
        cv2.imshow( 'curr', S_thumbnails[curr, :,:,:] )
        cv2.imshow( 'prev', S_thumbnails[prev, :,:,:] )
        # cv2.imwrite( '%d.png' %(curr), S_thumbnails[curr, :,:,:])
        # cv2.imwrite( '%d.png'%(prev), S_thumbnails[prev, :,:,:])

        if cv2.waitKey(0) == 27:
            break
