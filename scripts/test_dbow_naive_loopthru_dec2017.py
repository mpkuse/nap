""" Inspection script for drag_mpkuse_dbow.
    Loops over each dbow-naive candidates for manual marking for confusion-matrix

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 30th Jan, 2018
"""
import numpy as np
import cv2
import TerminalColors
tcol = TerminalColors.bcolors()

## Data Files
BASE__DUMP = '/home/mpkuse/Desktop/a/drag_mpkuse_dbow/'
LOOP_LIST = BASE__DUMP + 'loop_candidates_dbow.csv'
IMAGE_FILE_NPY = BASE__DUMP+'/S_thumbnail_dbow.npy'

print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)

print 'Reading : ', LOOP_LIST
loop_candidates = np.loadtxt( LOOP_LIST, delimiter=',' )


human_a = 0
human_s = 0
human_d = 0
human_f = 0
for i in range( len(loop_candidates)):
    print '=== Loop Candidate #', i+1, 'of %d===' %(len(loop_candidates))
    l = loop_candidates[i]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])

    cv2.imshow( 'A', cv2.resize( S_thumbnails[curr,:,:,:].astype('uint8'), (0,0), fx=0.5, fy=0.5 ) )
    cv2.imshow( 'B', cv2.resize( S_thumbnails[prev,:,:,:].astype('uint8'), (0,0), fx=0.5, fy=0.5 ) )


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
