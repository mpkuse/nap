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
import time
from operator import itemgetter

cv2.ocl.setUseOpenCL(False)

try:
    from DaisyMeld.daisymeld import DaisyMeld
except:
    print 'If you get this error, your DaisyMeld wrapper is not properly setup. You need to set DaisyMeld in LD_LIBRARY_PATH. and PYTHONPATH contains parent of DaisyMeld'
    print 'See also : https://github.com/mpkuse/daisy_py_wrapper'
    print 'Do: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/catkin_ws/src/nap/scripts_multiproc/DaisyMeld'
    print 'do: export PYTHONPATH=$PYTHONPATH:$HOME/catkin_ws/src/nap/scripts_multiproc'
    quit()
from ColorLUT import ColorLUT




import TerminalColors
tcol = TerminalColors.bcolors()

class GeometricVerification:
    def __init__(self):
        _x = 0
        self.reset()
        self.orb = cv2.ORB_create() #used in simple_verify()
        self.surf = cv2.xfeatures2d.SURF_create( 400, 5, 5 )

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

        # self.dai = DaisyMeld()
        self.dai1 = DaisyMeld( 240, 320, 0 ) #v2 of py_daisy_wrapper. You really need 3 as the memory is allocated and owned by C++
        self.dai2 = DaisyMeld( 240, 320, 0 )
        self.dai3 = DaisyMeld( 240, 320, 0 )

        self.PRINTING = True

    def enable_printing( self ):
        self.PRINTING = True

    def disable_printing( self ):
        self.PRINTING = False

    def reset(self):
        self.im1 = None
        self.im2 = None
        self.im3 = None

        self.im1_lut = None
        self.im2_lut = None
        self.im3_lut = None

        self.im1_lut_raw = None
        self.im2_lut_raw = None
        self.im3_lut_raw = None

    #### Newest setting functions in an effort to dis-entangle.
    def crunch_daisy( self, ch ):
        if ch == 1:
            assert( self.im1 is not None )
            xim_gray = cv2.cvtColor( self.im1, cv2.COLOR_BGR2GRAY ).astype('float32')
            self.dai1.do_daisy_computation( xim_gray )
            return None

        if ch == 2:
            assert( self.im2 is not None )
            xim_gray = cv2.cvtColor( self.im2, cv2.COLOR_BGR2GRAY ).astype('float32')
            self.dai2.do_daisy_computation( xim_gray )
            return None

        if ch == 3:
            assert( self.im3 is not None )
            xim_gray = cv2.cvtColor( self.im3, cv2.COLOR_BGR2GRAY ).astype('float32')
            self.dai3.do_daisy_computation( xim_gray )
            return None

        print '[FATAL-ERROR] GeometricVerification::do_daisy() got wrong ch'
        quit()

    def view_daisy( self, ch ):
        if ch== 1:
            return self.dai1.get_daisy_view()
        if ch== 2:
            return self.dai2.get_daisy_view()
        if ch== 3:
            return self.dai3.get_daisy_view()

        print '[FATAL-ERROR] GeometricVerification::view_daisy() got wrong ch'



    def set_image( self, image, ch, enable_dense_daisy=True ):
        if ch==1:
            self.im1 = image
            if enable_dense_daisy:
                self.crunch_daisy(ch=1)
            return

        if ch==2:
            self.im2 = image
            if enable_dense_daisy:
                self.crunch_daisy(ch=2)
            return

        if ch==3:
            self.im3 = image
            if enable_dense_daisy:
                self.crunch_daisy(ch=3)
            return

        print '[FATAL-ERROR] GeometricVerification::set_image'
        quit()


    def set_lut( self, lut, ch ):
        if ch==1:
            self.im1_lut = lut
            return

        if ch==2:
            self.im2_lut = lut
            return

        if ch==3:
            self.im3_lut = lut
            return

        print '[FATAL-ERROR] GeometricVerification::set_image'
        quit()


    def set_lut_raw( self, lut_raw, ch ):
        if ch==1:
            self.im1_lut_raw = lut_raw
            return

        if ch==2:
            self.im2_lut_raw = lut_raw
            return

        if ch==3:
            self.im3_lut_raw = lut_raw
            return

        print '[FATAL-ERROR] GeometricVerification::set_image'
        quit()

    ############################


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


        __pt1 = np.array( __pt1)
        __pt2 = np.array( __pt2)
        return __pt1, __pt2

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


    def _print_time(self, msg, startT, endT):
        return
        print tcol.OKBLUE, '%8.2f :%s (ms)'  %( 1000. * (endT - startT), msg ), tcol.ENDC



    def prominent_clusters( self, im_no=-1 ):
        # print tcol.OKGREEN, 'prominent_clusters : Uses im?,im?_lut,im?_lut_raw', tcol.ENDC

        if im_no == 1:
            im = self.im1
            im_lut_raw = self.im1_lut_raw
        elif im_no==2:
            im = self.im2
            im_lut_raw = self.im2_lut_raw
        else:
            print tcol.FAIL, 'fail. In valid im_no in prominent_clusters()', tcol.ENDC
            return None


        # TODO Consider passing these as function arguments
        K = 64 #number of clusters in netvlad
        retain_n_largest = 7 #top-n of the largest k(s)
        retain_min_pix = 30 #retains a blobs if area is atleast this number
        printing = False

        # Enumerating all clusters
        cluster_list = []
        startTime = time.time()
        total_pix = im_lut_raw.shape[0] * im_lut_raw.shape[1]
        for k in range(K):
            # print k, (S_thumbnails_lut_raw[curr, :,: ] == k).sum()
            cluster_list.append( (k, (im_lut_raw == k).sum()/float(total_pix)) )

        # Sort clusters by size
        cluster_list.sort( key=itemgetter(1), reverse=True )
        if printing:
            print 'Time taken to find top 5 clusters (ms) :', 1000.*(time.time() - startTime)

        # Iterate through top-5 clusters
        Z = np.zeros( im_lut_raw.shape, dtype='uint8' ) # This is a map of celected cluster Z[i,j] is the label of the cluster for pixel i,j. 0 is to ignore
        for (k, k_size ) in cluster_list[0:retain_n_largest]:
            if printing:
                print 'cluster_k=%d of size %f' %(k, k_size)

            bin_im = (im_lut_raw == k).astype('uint8') * 255
            # cv2.imshow( 'bin_im', bin_im )
            n_compo, compo_map, compo_stats, compo_centrum = cv2.connectedComponentsWithStats( bin_im, 4, cv2.CV_32S )
            if printing:
                print 'perform cca. has %d components' %(n_compo)
            n_large = 0
            for ci in range( 1,n_compo):
                # print ci, compo_stats[ci,:]
                if compo_stats[ci,4] > retain_min_pix: #retain this blob if it has certain area
                    n_large += 1
                    # cv2.imshow( 'compo', (compo_map == ci).astype('uint8')*255 )
                    # cv2.waitKey(0)
                    # code.interact( local=locals() )
                    Z = Z + k* (compo_map==ci).astype('uint8')
                    # if cv2.waitKey(0) == ord('c'):
                        # break
            if printing:
                print '%d were large enough' %(n_large)

        if printing:
            print 'Total time (ms) :', 1000.*(time.time() - startTime)
        return Z

    ## Z is the CCA map. Make sure you give in the largest clusters.
    def get_rotated_rect( self, Z ):
        printing = False
        compute_occupied = True


        all_k_in_Z = np.unique( Z )
        Q = {}
        for kv in all_k_in_Z[1:]: # Iterate over the clusters, all_k_in_Z[0], ie label=0 means unassigned
            if printing:
                print 'kv = ', kv
            kv_im_bool = (Z==kv)
            kv_im = 255*kv_im_bool.astype('uint8')
            if( kv_im.sum() == 0 ):
                continue

            Q[str(kv)] = []


            # Find contours in this binary image
                # issue: findContours may not return exactly the same number of contours as n_components of a cca. This
                # at the point when there are holes inside a connected-component. So, although area of a bounding box
                # is accurate, the number of pixels marked in bounding box may not be correct. Alternative way to do this
                # is to crop out the bounding box and count the marked pixels

            _, contours, hierarchy = cv2.findContours( kv_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
            if printing:
                print 'len(contours) = ', len(contours)
            contour_i = 0
            for contour in contours: #iterate over contours
                contour = contour.reshape( -1, 2 )

                assert contour.shape[1] == 2, "contour's shape should be Nx2"
                if contour.shape[0] < 10:
                    continue

                contour_i += 1
                # if printing:
                    # print 'contour.shape', contour.shape
                rect = cv2.minAreaRect( contour )
                if printing :
                    print contour_i, ') area of bounded rect %4.2f' %(rect[1][0] * rect[1][1])
                    print 'occupied pix : ', (kv_im/255.).sum() #this is wrong. it will be area of all the pixels of this label and not currect connected component.

                box = cv2.boxPoints( rect )
                box = np.int0(box)


                if compute_occupied == True:
                    XC = np.zeros( kv_im.shape, dtype='uint8' )
                    cv2.fillConvexPoly( XC, box, (255) )
                    # if u need more speed, just ignore computation of occupied_area per contour.
                    occupied_area = (XC.astype('bool') & kv_im_bool).sum() #(kv_im/255.).sum() #number of pixels (/255 is since the binary image was multiplied by 255 for CCA)
                else:
                    occupied_area = 0.


                bounded_area = rect[1][0] * rect[1][1] #bounding area (in pixels) of the bounding rectangle


                Q[str(kv)].append( (box, occupied_area, bounded_area) )



            # cv2.imshow( 'kv', kv_im)
            # cv2.waitKey(0)

        return Q


    # Called by daisy_dense_matches() to loop over matches.
    def analyze_dense_matches( self, H1, H2, matches, return_lowe_ratio=False ):
        M = []
        # Lowe's ratio test
        for i, (m,n) in enumerate( matches ):
            #print i, m.trainIdx, m.queryIdx, m.distance, n.trainIdx, n.queryIdx, n.distance
            if m.distance < 0.8 * n.distance: #originally 0.8. Making it lower will tighten
                r = float(m.distance)/n.distance
                M.append( (m,r)  )

        # print '%d of %d pass lowe\'s ratio test' %( len(M), len(matches) )

        # plot
        # canvas = np.concatenate( (self.im1, self.im2), axis=1 )
        pts_A = []
        pts_B = []
        pts_lowe_ratios = []

        # -a- lowe's ratios are scored as. Note: lower ratio is better here:
        #       note, this scoring is done in call analyze_dense_matches() on per match basis
        #              x <= 0.67  ---> +2
        #       0.67 < x <= 0.72  ---> +1
        #       0.72 < x <= 0.75  ---> +0.5
        #       0.75 < x <= 0.8   ---> +0

        for (m,r) in M: #good matches
            # print m.trainIdx, m.queryIdx, m.distance

            a = H1[1][m.queryIdx]*4
            b = H1[0][m.queryIdx]*4

            c = H2[1][m.trainIdx]*4
            d = H2[0][m.trainIdx]*4

            pts_A.append( (a,b) )
            pts_B.append( (c,d) )


            if r <= 0.67:
                r_score = 2.0
            else:
                if r > 0.67 and r <= 0.72:
                    r_score = 1.0
                else:
                    if r > 0.72 and r <= 0.75:
                        r_score = 0.5
                    if r > 0.75 :
                        r_score = 0

            pts_lowe_ratios.append( r_score )


        if return_lowe_ratio:
            return pts_A, pts_B, pts_lowe_ratios

        return pts_A, pts_B




    # This function will compute the daisy matches, given the cluster assignments
    # from netvlad and dense daisy. Need to set_im() and set_im_lut() before calling this
    def daisy_dense_matches(self, DEBUG=False):
        # DEBUG = True # in debug mode 4 things are returned, : m1, m2, mask and [xcanvas]
        #               in non-debug mode 3 things r returned
        assert self.im1 is not None, "GeometricVerification.daisy_dense_matches(): im1 was not set. "
        assert self.im2 is not None, "GeometricVerification.daisy_dense_matches(): im2 was not set. "
        assert self.im1_lut_raw is not None, "GeometricVerification.daisy_dense_matches(): im1_lut_raw was not set. "
        assert self.im2_lut_raw is not None, "GeometricVerification.daisy_dense_matches(): im2_lut was not set. "

        if DEBUG:
            assert self.im1_lut is not None, "GeometricVerification.daisy_dense_matches(): im1_lut was not set. "
            assert self.im2_lut is not None, "GeometricVerification.daisy_dense_matches(): im2_lut was not set. "


        # Get prominent_clusters
        startProminentClusters = time.time()
        Z_curr = self.prominent_clusters(im_no=1)
        Z_prev = self.prominent_clusters(im_no=2)
        self._print_time( 'Prominent clusters', startProminentClusters, time.time() )


        # Step-1 : Get Daisy at every point
        startDaisy = time.time()
        # Old code
        # D_curr = self.get_whole_image_daisy( im_no=1 )
        # D_prev = self.get_whole_image_daisy( im_no=2 )
        # self.daisy_im1 = D_curr
        # self.daisy_im2 = D_prev

        # new code (12th Nov, assumes daisy was already computed, just get views). Possibly dirty
        D_curr = self.view_daisy( ch=1 ) #self.my_daisy( im_curr, ch=0 )
        D_prev = self.view_daisy( ch=2 ) #my_daisy( im_prev, ch=1 )
        self._print_time( 'Daisy (2 images)', startDaisy, time.time() )

        # Step-2 : Given a k which is in both images, compare clusters with daisy. To do that do NN followd by Lowe's ratio test etc
        startDenseFLANN = time.time()
        Z_curr_uniq = np.unique( Z_curr )[1:] #from 1 to avoid 0 which is for no assigned cluster
        Z_prev_uniq = np.unique( Z_prev )[1:]
        # print Z_curr_uniq #list of uniq k
        # print Z_prev_uniq


        # Prepare FLANN Matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        # Loop over intersection, and merge all the pts from each of the k
        pts_A = []
        pts_B = []
        k_intersection = set(Z_curr_uniq).intersection( set(Z_prev_uniq) )
        xcanvas_array = []
        for k in k_intersection:
            H_curr = np.where( Z_curr==k ) #co-ordinates. these co-ordinates need testing
            desc_c = np.array( D_curr[ H_curr[0]*4, H_curr[1]*4 ] ) # This is since D_curr is (240,320) and H_curr is (60,80)

            H_prev = np.where( Z_prev==k ) #co-ordinates #remember , Z_prev is (80,60)
            desc_p = np.array( D_prev[ H_prev[0]*4, H_prev[1]*4 ] )

            matches = flann.knnMatch(desc_c.astype('float32'),desc_p.astype('float32'),k=2)

            _pts_A, _pts_B = self.analyze_dense_matches(  H_curr, H_prev, matches )

            pts_A += _pts_A
            pts_B += _pts_B

            # DEBUG
            if DEBUG:
                print 'k=%d returned %d matches' %(k, len(_pts_A) )
                xim1 = self.s_overlay( self.im1, np.int0(Z_curr==k), 0.7 )
                xim2 = self.s_overlay( self.im2, np.int0(Z_prev==k), 0.7 )
                # xcanvas = self.plot_point_sets( self.im1, _pts_A, self.im2, _pts_B)
                xcanvas = self.plot_point_sets( xim1, _pts_A, xim2, _pts_B)
                cv2.imshow( 'xcanvas', xcanvas)
                cv2.waitKey(0)
                xcanvas_array.append( xcanvas )
            # END Debug


        self._print_time( 'Dense FLANN over common k=%s' %(str(k_intersection)), startDenseFLANN, time.time() )


        # DEBUG, checking pts_A, pts_B
        if DEBUG:
            print 'Total Matches : %d' %(len(pts_A))
            xcanvas = self.plot_point_sets( self.im1, pts_A, self.im2, pts_B)
            cv2.imshow( 'full_xcanvas', xcanvas)
            cv2.waitKey(0)

        # End DEBUG


        # Step-3 : Essential Matrix Text
        E, mask = cv2.findFundamentalMat( np.array( pts_A ), np.array( pts_B ), param1=5 )
        if DEBUG:
            print 'Total Dense Matches : ', len(pts_A)
            print 'Total Verified Dense Matches : ', mask.sum()
        # code.interact( local=locals() )


        if DEBUG:
            return pts_A, pts_B, mask, xcanvas_array
        else:
            return pts_A, pts_B, mask

    def plot_point_sets( self, im1, pt1, im2, pt2, mask=None, enable_text=False ):
        xcanvas = np.concatenate( (im1, im2), axis=1 )
        for xi in range( len(pt1) ):
            if (mask is not None) and (mask[xi,0] == 0):
                continue

            cv2.circle( xcanvas, pt1[xi], 4, (255,0,255) )
            ptb = tuple(np.array(pt2[xi]) + [im1.shape[1],0])
            cv2.circle( xcanvas, ptb, 4, (255,0,255) )
            cv2.line( xcanvas, pt1[xi], ptb, (255,0,0) )


            if enable_text:
                color_com = (0,0,255)
                cv2.putText( xcanvas, str(xi), pt1[xi], cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )
                cv2.putText( xcanvas, str(xi), ptb, cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )
        return xcanvas


    def s_overlay( self, im1, Z, alpha=0.5 ):
        lut = ColorLUT()
        out_imcurr = lut.lut( Z )

        out_im = alpha*im1 + (1 - alpha)*cv2.resize(out_imcurr,  (im1.shape[1], im1.shape[0])  )
        return out_im.astype('uint8')

    def plot_2way_match( self, im1, pt1, im2, pt2, mask=None, color=(255,0,0), enable_text=True, enable_lines=False ):
        """ pt1, pt2 : 2xN """
        xcanvas = np.concatenate( (im1, im2), axis=1 )
        if pt1 is None or pt2 is None:
            return xcanvas

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

    # grid : [ [curr, prev], [curr-1  X ] ]
    def plot_3way_match( self, curr_im, pts_curr, prev_im, pts_prev, curr_m_im, pts_curr_m, enable_lines=True, enable_text=True ):
        """ pts_curr, pts_prev, pts_curr_m: Nx2 """
        DEBUG = False

        zero_image = np.zeros( curr_im.shape, dtype='uint8' )
        cv2.putText( zero_image, str(len(pts_curr)), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255 )


        r1 = np.concatenate( ( curr_im, prev_im ), axis=1 )
        r2 = np.concatenate( ( curr_m_im, zero_image ), axis=1 )
        gridd = np.concatenate( (r1,r2), axis=0 )

        if DEBUG:
            ncorrect = 0
            for xi in range( len(pts_curr) ):
                gridd_debug = np.concatenate( (r1,r2), axis=0 )

                cv2.circle( gridd_debug, pts_curr[xi], 4, (0,255,0) )
                ptb = tuple(np.array(pts_prev[xi]) + [curr_im.shape[1],0])
                cv2.circle( gridd_debug, ptb, 4, (0,255,0) )
                cv2.line( gridd_debug, pts_curr[xi], ptb, (255,0,0) )

                ptb = tuple(np.array(pts_curr_m[xi]) + [0,curr_im.shape[0]])
                cv2.circle( gridd_debug, ptb, 4, (0,255,0) )
                cv2.line( gridd_debug, pts_curr[xi], ptb, (255,30,255) )

                cv2.imshow( 'gridd', gridd_debug )

                key = cv2.waitKey(0)
                print '%d of %d q:quit ; c:mark as correct match ; <space>:continue' %(xi, len(pts_curr))
                if ( key & 0xFF) == ord( 'q' ):
                    break

                if ( key & 0xFF) == ord( 'c' ):
                    ncorrect += 1

            print 'Total Marked as correct : %d of %d' %(ncorrect, len(pts_curr))




        # print 'pts_curr.shape   ', pts_curr.shape
        # print 'pts_prev.shape   ', pts_prev.shape
        # print 'pts_curr_m.shape ', pts_curr_m.shape
        # all 3 should have same number of points

        for xi in range( pts_curr.shape[0] ):
            pta = pts_curr[xi,0:2]
            ptb = pts_prev[xi,0:2] + [curr_im.shape[1],0]
            ptc = pts_curr_m[xi,0:2] +  [0,curr_im.shape[0] ]

            cv2.circle( gridd, tuple(pta), 4, (0,255,0) )
            cv2.circle( gridd, tuple(ptb), 4, (0,255,0) )
            cv2.circle( gridd, tuple(ptc), 4, (0,255,0) )


            if enable_text:
                cv2.putText( gridd, str(xi), tuple(pta), cv2.FONT_HERSHEY_SIMPLEX, .3,  (255,0,255) )
                cv2.putText( gridd, str(xi), tuple(ptb), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,0,255)  )
                cv2.putText( gridd, str(xi), tuple(ptc), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,0,255)  )

            if enable_lines:
                cv2.line( gridd, tuple(pta), tuple(ptb), (255,0,0) )
                cv2.line( gridd, tuple(pta), tuple(ptc), (255,30,255) )

        # old code
        #     cv2.circle( gridd, pts_curr[xi], 4, (0,255,0) )
        #     ptb = tuple(np.array(pts_prev[xi]) + [curr_im.shape[1],0])
        #     cv2.circle( gridd, ptb, 4, (0,255,0) )
        #     cv2.line( gridd, pts_curr[xi], ptb, (255,0,0) )
        #
        # # for xi in range( len(pts_curr) ):
        #     # cv2.circle( gridd, pts_curr[xi], 4, (0,255,0) )
        #     ptb = tuple(np.array(pts_curr_m[xi]) + [0,curr_im.shape[0]])
        #     cv2.circle( gridd, ptb, 4, (0,255,0) )
        #     cv2.line( gridd, pts_curr[xi], ptb, (255,30,255) )



        return gridd


    # Given the matches between curr and prev. expand these matches onto curr-1.
    # Main purpose is to get a 3-way matches for pose computation. The way this
    # is done is for each match between curr and prev, find the NN in curr_m_im
    # around the neighbourhood of 40x40 in daisy space.
    #
    #   pts_curr, pts_prev, mask_c_p : output from daisy_dense_matches()
    #   curr_m_im : curr-1 image (self.im3)
    def expand_matches_to_curr_m( self, pts_curr, pts_prev, mask_c_p,   curr_m_im):

        DEBUG = False
        PARAM_W = 20

        masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
        masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )

        _pts_curr = np.array( masked_pts_curr )

        # Old code
        # D_pts_curr = self.daisy_im1[ _pts_curr[:,1], _pts_curr[:,0], : ] #assuming im1 is curr

        # New code
        _daisy_im1 = self.view_daisy( ch=1 )
        D_pts_curr = _daisy_im1[  _pts_curr[:,1], _pts_curr[:,0], : ]


        if DEBUG:
            print 'using a neighbooirhood of %d' %(2*PARAM_W + 1)
            xcanvas_c_p = self.plot_point_sets( self.im1, pts_curr, self.im2, pts_prev, mask_c_p)
            cv2.imshow( 'xcanvas_c_p', xcanvas_c_p)

        #
        # Alternate Step-2:: Find the matched points in curr (from step-1) in c-1. Using a
        # bounding box around each point
        #   INPUTS : curr_m_im, _pts_curr, D_pts_curr
        #   OUTPUT : _pts_curr_m
        internalDaisy = time.time()
        # old code
        # daisy_c_m = self.static_get_daisy_descriptor_mat(  curr_m_im  )#Daisy of (curr-1)

        # new code
        # self.crunch_daisy( ch=3 ) #if image is set daisy is automatically computed
        daisy_c_m = self.view_daisy( ch=3 )

        if DEBUG:
            print 'in expand_matches_to_curr_m, daisy took %4.2f ms' %( 1000. * (time.time() - internalDaisy) )


        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        _pts_curr_m = []
        for h in range(_pts_curr.shape[0]): # loop over each match (between curr and prev)
            # get daisy descriptors in WxW neighbourhood. For a co-ordinate in curr
            # I assume itz match can be found in a WxW neighbourhood in curr-1 image.
            # This assumption is valid since curr and curr-1 are adjustcent keyframes.
            # NOTE: All this can be confusing, be careful!

            _W = PARAM_W #20
            row_range = range( max(0,_pts_curr[h,1]-_W),  min(curr_m_im.shape[0], _pts_curr[h,1]+_W) )
            col_range = range( max(0,_pts_curr[h,0]-_W),  min(curr_m_im.shape[1], _pts_curr[h,0]+_W) )
            g = np.meshgrid( row_range, col_range )
            positions = np.vstack( map(np.ravel, g) ) #2x400, these are in cartisian-indexing-convention


            D_positions = daisy_c_m[positions[0,:], positions[1,:],:] #400x20
            if DEBUG:
                print 'D_positions.shape :', D_positions.shape


            if DEBUG:
                y = self.im1.copy() #curr_im
                z = curr_m_im.copy() #curr_m_im
                for p in range( positions.shape[1] ):
                    cv2.circle( y, (positions[1,p], positions[0,p]), 1, (255,0,0) )

                cv2.circle( y, (_pts_curr[h,0],_pts_curr[h,1]), 1, (0,0,255) )
                cv2.imshow( 'curr_im_debug', y )





            matches = flann.knnMatch(np.expand_dims(D_pts_curr[h,:], axis=0).astype('float32'),D_positions.astype('float32'),k=1)
            matches = matches[0][0]

            if DEBUG:
                print 'match : %d <--%4.2f--> %d' %(matches.queryIdx, matches.distance, matches.trainIdx )
                cv2.circle( z, tuple(positions[[1,0],matches.trainIdx]), 2, (0,0,255) )
                cv2.imshow( 'curr-1 debug', z )
                print 'Press <Space> for next match'
                cv2.waitKey(0)


            _pts_curr_m.append( tuple(positions[[1,0],matches.trainIdx]) )


        return _pts_curr_m



        ##################### Guided Matching (added 12th Nov, 2017) ############

    def release_candidate_match2_guided_2way( self, pts_curr, pts_prev, return_per_match_vote_score=False):
        """ Given images with tracked feature2d pts. Returns matched points
            using the voting mechanism. Loosely inspired from GMS-matcher

            im_curr : Current Image (get from im1)
            pts_curr : Detected points in current image (3xN) or (2xN)
            im_prev : Previous Image (get from im2)
            pts_prev : Detected points in prev image (3xM) or (2xN)



            returns :
            2 1d arrays. the masked index in the original.
            Basically, the index_i in pts_curr <--> index_j in pts_prev as 1d arrays

            Note: points in pts_curr and pts_prev could be different in count.
            These are the detected features in 2 views in image co-ordinates (ie. NOT normalized co-ordinates)
        """
        #
        # Algorithm Params

        # a) neighbourhood
        W = 1
        # _R = range( -W, W+1 )
        _R = [-4,0,4]


        # b) number of NN
        #
        K_nn = 1


        # Enabling this might cause issue. Some debug overlay functions are missing.
        # Caution!
        DEBUG = False

        assert( self.im1 is not None )
        assert( self.im2 is not None )
        im_curr = self.im1
        im_prev = self.im2

        # self.crunch_daisy( ch=1 ) #if image is set daisy is automatially computed
        # self.crunch_daisy( ch=2 )

        daisy_curr = self.view_daisy( ch=1 ) #self.my_daisy( im_curr, ch=0 )
        daisy_prev = self.view_daisy( ch=2 ) #my_daisy( im_prev, ch=1 )



        if DEBUG:
            print 'im_curr.shape', im_curr.shape
            print 'im_prev.shape', im_prev.shape
            print 'pts_curr.shape', pts_curr.shape
            print 'pts_prev.shape', pts_prev.shape
            print 'neighbourhood: ', _R

        #
        # Step-1a: wxw around each of curr
        #
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

        #
        # Step-1b: wxw around each of prev
        #
        B = []
        B_i = []
        B_pt = []
        for pt_i in range( pts_prev.shape[1] ):
            for u in _R:#range(-W,W+1):
                for v in _R:#range(-W,W+1):
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
        # Step-2: FLANN Matching
        #
        startflann = time.time()
        matches = self.flann.knnMatch( np.array(A), np.array(B), k=K_nn )
        if DEBUG:
            print 'time elaspsed for flann : %4.2f (ms)' %(1000.0 * (time.time() - startflann) )


        #
        # Step-3a: Loop over each match and do voting
        #
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
                vote[ iA, iB ] += 1.0/(1.0+k*k) # vote from 2nd nn is 1/4 the 1st nn and 3rd is 1/9th and so on

        if DEBUG:
            print 'time elaspsed for voting : %4.2f (ms)' %( 1000. * (time.time() - startvote) )

        # # consider removal
        # # Step-4b: Scale the actual votes with the priors
        # #
        # if prior_vote_matrix is not None:
        #     assert prior_vote_matrix.shape[0] == vote.shape[0]
        #     assert prior_vote_matrix.shape[1] == vote.shape[1]
        #     print 'do vote scaling'
        #     code.interact( local=locals(), banner='vote scaling')
        #     vote = np.multiply( vote / vote.sum(axis=1), prior_vote_matrix ) #`vote/sum(vote) ` normalizes the votes as a probability measure


        #
        # Step-4: Evaluating Votes (or Scaled votes)
        #
        selected_A = []
        selected_A_i = []
        selected_B = []
        selected_B_i = []
        selected_score = []
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

            if top/top2 >= 1.3 or top2 < 0: #smaller ratio will mean looser constraint

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

                selected_score.append( top )
                # if top2<0:
                #     selected_score.append( 10 ) #this means all votes are going to one match
                # else:
                #     selected_score.append( top/top2 )

                if DEBUG and False:
                    print 'ACCEPTED'
                    cv2.imshow( 'curr_overlay', self.points_overlay( im_curr, ptxA, enable_text=True ))
                    cv2.imshow( 'prev_overlay', self.points_overlay( im_prev, ptxB, enable_text=True ))
                    cv2.imshow( 'prev_overlay top', self.points_overlay( im_prev, ptyB, enable_text=True ))
                    cv2.waitKey(0)

        if len(selected_A) == 0:
            return np.array([]), np.array([])

        selected_A = np.transpose(  np.array( selected_A )[:,:,0]  )
        selected_B = np.transpose(  np.array( selected_B )[:,:,0]  )

        if DEBUG:
            print 'time elaspsed for selection from voting : %4.2f (ms)' %( 1000. * (time.time() - startSelect) )

            # cv2.imshow( 'xxx' , self.plot_point_sets( im_curr, np.int0(pts_curr[0:2,selected_A_i]), im_prev, np.int0(pts_prev[0:2,selected_B_i] ) ) )

        #
        # Step-5: Fundamental Matrix Text
        #
        startFundamentalMatrixTest = time.time()
        E, mask = cv2.findFundamentalMat( np.transpose( selected_A ), np.transpose( selected_B ),param1=5 )
        if mask is not None:
            nInliers = mask.sum()
        else: #mask is None (no inliers)
            return np.array([]), np.array([])

        if self.PRINTING:
            print 'A: %s ; B: %s ; nInliers:%d' %(str(selected_A.shape), str(selected_B.shape), nInliers)
        masked_pts_curr = np.transpose( np.array( list( selected_A[:,i] for i in np.where( mask[:,0] == 1 )[0] ) ) )
        masked_pts_prev = np.transpose( np.array( list( selected_B[:,i] for i in np.where( mask[:,0] == 1 )[0] ) ) )

        if DEBUG:
            print 'nInliers : ', nInliers
            print 'time elaspsed for fundamental matrix test : %4.2f (ms)' %( 1000. * (time.time() - startFundamentalMatrixTest) )


        masked_selected_A_i = list( selected_A_i[q] for q in np.where( mask[:,0] == 1 )[0] )
        masked_selected_B_i = list( selected_B_i[q] for q in np.where( mask[:,0] == 1 )[0] )

        masked_selected_score = list( selected_score[q]  for q in np.where( mask[:,0] == 1 )[0] )

        if DEBUG:
            cv2.imshow( 'curr_overlay', self.points_overlay( im_curr, pts_curr) )
            cv2.imshow( 'prev_overlay', self.points_overlay( im_prev, pts_prev) )
            cv2.imshow( 'selected', self.plot_point_sets( im_curr, selected_A, im_prev, selected_B) )
            cv2.imshow( 'selected+fundamentalmatrixtest', self.plot_point_sets( im_curr, masked_pts_curr, im_prev, masked_pts_prev) )

            cv2.imshow( 'yyy selected+fundamentalmatrixtest', self.plot_point_sets( im_curr, np.int0(pts_curr[0:2,masked_selected_A_i]), im_prev, np.int0(pts_prev[0:2,masked_selected_B_i])  ) )


        # Return the masked index in the original
        # return np.array(masked_selected_A_i), np.array(masked_selected_B_i)
        if return_per_match_vote_score is False:
            return np.array(masked_selected_A_i), np.array(masked_selected_B_i), (min(pts_curr.shape[1], pts_prev.shape[1]), selected_A.shape[1], nInliers)
        else:
            # return np.array(masked_selected_A_i), np.array(masked_selected_B_i), (min(pts_curr.shape[1], pts_prev.shape[1]), selected_A.shape[1], nInliers), selected_score
            try:
                # code.interact( local=locals() , banner="before returning votes")
                _stat_ = (min(pts_curr.shape[1], pts_prev.shape[1]), selected_A.shape[1], nInliers)
                return np.array(masked_selected_A_i), np.array(masked_selected_B_i), _stat_, np.array(masked_selected_score)#max(masked_selected_score)
            except:
                code.interact( local=locals(), banner="Exception occured in voting" )



    def release_candidate_match3way( self ):
        assert( self.im1 is not None and self.im2 is not None and self.im3 is not None )
        assert( self.im1_lut_raw is not None and self.im2_lut_raw is not None )

        DEBUG = False
        pts_curr, pts_prev, mask_c_p = self.daisy_dense_matches()
        if DEBUG:
            xcanvas_c_p = self.plot_point_sets( self.im1, pts_curr, self.im2, pts_prev, mask_c_p)
            # fname = '/home/mpkuse/Desktop/a/drag_nap/%d.jpg' %(loop_index)
            # print 'Write(match3way_daisy) : ', fname
            # cv2.imwrite( fname, xcanvas_c_p )
            cv2.imshow( 'xcanvas_c_p', xcanvas_c_p )

        # Step-2: Match expansion
        _pts_curr_m = self.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p, self.im3  )
        masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
        masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )

        if DEBUG:
            gridd = self.plot_3way_match( self.im1, masked_pts_curr, self.im2, masked_pts_prev, self.im3, _pts_curr_m )
            # fname = '/home/mpkuse/Desktop/a/drag_nap/%d_3way.jpg' %(loop_index)
            # print 'Write(match3way_daisy) : ', fname
            # cv2.imwrite(fname, gridd )
            cv2.imshow( 'gridd0', gridd )

        assert( len(masked_pts_curr) == len(masked_pts_prev) )
        assert( len(masked_pts_curr) == len(_pts_curr_m) )

        masked_pts_curr = np.array(masked_pts_curr) #Nx2
        masked_pts_prev = np.array( masked_pts_prev )
        _pts_curr_m =  np.array(_pts_curr_m )
        return  masked_pts_curr, masked_pts_prev, _pts_curr_m

    ###########################################################################

    def sieve_stat_to_score( self, sieve_stat ):
        if self.PRINTING:
            print 'Tracked Features              :', sieve_stat[0]
            print 'Features retained post voting :', sieve_stat[1]
            print 'Features retained post f-test :', sieve_stat[2]

        # Finding-1: if atleast 20% of the tracked points of min(feat2d_curr, feat2d_prev)
        # are retained than it indicates that this one is probably a true match.

        # Finding-2: If tracked features in both images are dramatically different,
        # then, most likely it is a false match and hence need to be rejected.



        match2_voting_score = float(sieve_stat[1]) / sieve_stat[0] #how many remain after voting. More retained means better confident I am. However less retained doesn't mean it is wrong match. Particularly, there could be less overlaping area and the tracked features are uniformly distributed on image space.
        match2_tretained_score = float(sieve_stat[2]) / sieve_stat[0] #how many remain at the end
        match2_geometric_score  = (sieve_stat[1] - sieve_stat[2]) / sieve_stat[1]#how many were eliminated by f-test. lesser is better here. If few were eliminated means that the matching after voting is more geometrically consistent
        if self.PRINTING:
            print 'match2_voting_score: %4.2f; ' %(match2_voting_score),
            print 'match2_tretained_score: %4.2f; ' %(match2_tretained_score),
            print 'match2_geometric_score: %4.2f' %(match2_geometric_score)



        match2_total_score = 0.
        if match2_voting_score > 0.5:
            match2_total_score += 1.0

        if match2_tretained_score > 0.25:
            match2_total_score += 2.5
        else:
            if match2_tretained_score > 0.2 and match2_tretained_score <= 0.25:
                match2_total_score += 1.0
            if match2_tretained_score > 0.15 and match2_tretained_score <= 0.2:
                match2_total_score += 0.5


        if match2_geometric_score > 0.55:
            match2_total_score -= 1.
        else:
            if match2_geometric_score < 0.4 and match2_geometric_score >= 0.3:
                match2_total_score += 1.0
            if match2_geometric_score < 0.3 and match2_geometric_score >= 0.2:
                match2_total_score += 1.5
            if match2_geometric_score < 0.2 :
                match2_total_score += 1.5

        # min/ max
        # if (float(min(feat2d_curr.shape[1],feat2d_prev.shape[1])) / max(feat2d_curr.shape[1],feat2d_prev.shape[1])) < 0.70:
            # match2_total_score -= 3
            # print 'nTracked features are very different.'

        if self.PRINTING:
            print '==Total_score : ', match2_total_score, '=='
        return match2_total_score



    def robust_match3way( self, DEBUG=False ):
        """

        Returns:
            pts_curr, pts_prev, _pts_curr_m,  --> 3 point sets in curr, prev and curr-1
            per_match_vote,                   --> Match quality of each, a number between 0,5
            (dense_match_quality, after_vote_match_quality) --> 2-scalars to give quality after each of the 2 steps involved

            if pts_curr is None indicates that dense_match_quality was too low to proceed further and hence this match
            was REJECTed after step-1.

            Often times match can also be REJECTed after voting, ie. after step-2. In that case as well
            pts_curr will be zero.
        """
        assert( self.im1 is not None and self.im2 is not None and self.im3 is not None )
        assert( self.im1_lut_raw is not None and self.im2_lut_raw is not None )

        # DEBUG = False
        # Step-1: Daisy Dense Match.
        # Enabling DEBUG for this step will show matches in each prominent clusters. Will be useful to generate results
        pts_curr, pts_prev, mask_c_p, pt_match_quality_scores = self.daisy_dense_matches_with_scores(DEBUG=False)

        #######################################################################
        ## If there are not enough high-quality matches, consider giving up! ##
        #######################################################################
        npts_score_gt_2 = (np.array(pt_match_quality_scores) >= 2).sum() #0.5 (top-half)
        npts_score_gt_3 = (np.array(pt_match_quality_scores) >= 3).sum() #0.75 (top-quarter)
        dense_match_quality = 0
        # (-inf,35): +0; [35,50): +1 ; [50,inf]: +2
        if npts_score_gt_2 > 35 and npts_score_gt_2 <= 50:
            dense_match_quality += 1
        else:
            if npts_score_gt_2 > 50 :
                dense_match_quality += 2

        # (-inf,20): +0; [20,35): +1 ; [35,inf]: +2
        if npts_score_gt_3 > 20 and npts_score_gt_3 <= 35:
            dense_match_quality += 1
        else:
            if npts_score_gt_3 > 35 :
                dense_match_quality += 2


        if DEBUG:
            print '===Step-1: Daisy Dense Match==='

            # xcanvas_c_p = self.plot_point_sets( self.im1, pts_curr, self.im2, pts_prev, mask_c_p)
            xcanvas_c_p = self.plot_point_sets_with_quality_scores( self.im1, pts_curr, self.im2, pts_prev, pt_match_quality_scores, mask_c_p, enable_text=True)
            print 'nPts                : ', len(pts_curr)
            print 'nPts.score > 2      : ', (np.array(pt_match_quality_scores) >= 2).sum()
            print 'nPts.score > 3      : ', (np.array(pt_match_quality_scores) >= 3).sum()
            print 'Dense Match Quality : ', dense_match_quality

            print 'Point Set Match Quality : %4.2f' %( np.mean(pt_match_quality_scores) )
            # fname = '/home/mpkuse/Desktop/a/drag_nap/%d.jpg' %(loop_index)
            # print 'Write(match3way_daisy) : ', fname
            # cv2.imwrite( fname, xcanvas_c_p )
            cv2.imshow( 'xcanvas_c_p__daisy_dense_matches', xcanvas_c_p )



        # if dense_match_quality is less than 2, there is no point continuing. Possibly reject
        if dense_match_quality < 2 :
            print 'REJECT after daisy_dense_matches(), as the quality of dense-matches is too low'
            return None, None, None, None, (dense_match_quality,0)
        #############################################
        ## end evaluating quality of dense matches ##
        #############################################



        # Step-1.5: Filter Dense Matches with voting
        #   Ideally, these dense matches with daisy+netvlad_mask need to be
        #   filtered using GMS-style-voting scheme. Note that fundamentalmatrix-test
        #   is already performed in self.daisy_dense_matches(). Possibly using
        #   the percentage of matches eliminated from original matching can be of
        #   value.
        #
        if DEBUG:
            print 'f-test: (%d/%d) : %4.2f' %( mask_c_p.sum(), len(mask_c_p), float(mask_c_p.sum()) / len(mask_c_p))
        masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
        masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
        masked_pt_match_quality_scores = list( pt_match_quality_scores[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )

        masked_pts_curr_2N = np.transpose( np.array( masked_pts_curr ) )
        masked_pts_prev_2N = np.transpose( np.array( masked_pts_prev ) )
        # code.interact( local=locals() )
        # Without prior
        # ind_curr, ind_prev, _stat_ = self.release_candidate_match2_guided_2way( masked_pts_curr_2N, masked_pts_prev_2N)

        # With prior
        prior_vote_matrix = None#.01 + .25*np.diag(masked_pt_match_quality_scores)
        # code.interact( local=locals() )
        ind_curr, ind_prev, _stat_, per_match_vote = self.release_candidate_match2_guided_2way( masked_pts_curr_2N, masked_pts_prev_2N, return_per_match_vote_score=True)

        #
        # Scale pt_match_quality_scores here
        _tmp = np.array(masked_pt_match_quality_scores)[ind_curr]
        per_match_vote = np.multiply( 1.5*_tmp, per_match_vote )

        #########  XX Eval Quality after Voting XX ##########

        npts_score_gt_2 = (np.array(per_match_vote) >= 2.).sum() #0.5 (top-half)
        npts_score_gt_3 = (np.array(per_match_vote) >= 3.).sum() #0.75 (top-quarter)
        after_vote_match_quality = 0
        # (-inf,35): +0; [35,50): +1 ; [50,inf]: +2
        if npts_score_gt_2 > 25 and npts_score_gt_2 <= 40:
            after_vote_match_quality += 1
        else:
            if npts_score_gt_2 > 40 :
                after_vote_match_quality += 2

        # (-inf,20): +0; [20,35): +1 ; [35,inf]: +2
        if npts_score_gt_3 > 10 and npts_score_gt_3 <= 25:
            after_vote_match_quality += 1
        else:
            if npts_score_gt_3 > 25 :
                after_vote_match_quality += 2



        #####################################################


        if len(ind_curr) != 0:
            pts_curr = list( map(tuple,np.transpose(masked_pts_curr_2N[:,ind_curr])) )
            pts_prev = list( map(tuple,np.transpose(masked_pts_prev_2N[:,ind_prev])) )
            mask_c_p = np.ones( (len(pts_curr),1), dtype=mask_c_p.dtype )
        else:
            pts_curr = []
            pts_prev = []
            mask_c_p = []




        if DEBUG:
            # xcanvas_c_p = self.plot_point_sets( self.im1, pts_curr, self.im2, pts_prev, mask_c_p)
            xcanvas_c_p = self.plot_point_sets_with_quality_scores( self.im1, pts_curr, self.im2, pts_prev, per_match_vote, mask_c_p, enable_text=True)
            print '=== Step-1.5: Voting ==='
            # print '_stat_', _stat_
            # self.sieve_stat_to_score( _stat_ )
            print 'nPts returned by voting : ', len(pts_curr)
            print 'average per_match_vote  : %4.2f' %(np.mean(per_match_vote))

            print 'nPts.score > 2           : ', npts_score_gt_2
            print 'nPts.score > 3           : ', npts_score_gt_3
            print 'After Vote Match Quality : ', after_vote_match_quality
            print 'per_match_vote', per_match_vote

            # fname = '/home/mpkuse/Desktop/a/drag_nap/%d.jpg' %(loop_index)
            # print 'Write(match3way_daisy) : ', fname
            # cv2.imwrite( fname, xcanvas_c_p )
            cv2.imshow( 'xcanvas_c_p__after_voting', xcanvas_c_p )
            # code.interact( local=locals() )



        if after_vote_match_quality < 2 :
            print 'REJECT after voting ie. release_candidate_match2_guided_2way(), as the quality after voting was too low'
            return None, None, None, None, (dense_match_quality,after_vote_match_quality)


        # Step-2: Match expansion
        _pts_curr_m = self.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p, self.im3  )

        # code.interact( local=locals() )
        if DEBUG:
            gridd = self.plot_3way_match( self.im1, np.array(pts_curr), self.im2, np.array(pts_prev), self.im3, np.array(_pts_curr_m) )
            # fname = '/home/mpkuse/Desktop/a/drag_nap/%d_3way.jpg' %(loop_index)
            # print 'Write(match3way_daisy) : ', fname
            # cv2.imwrite(fname, gridd )
            cv2.imshow( 'gridd0', gridd )
            cv2.waitKey(10)

        assert( len(pts_curr) == len(pts_prev) )
        assert( len(pts_curr) == len(_pts_curr_m) )

        pts_curr = np.array(pts_curr) #Nx2
        pts_prev = np.array( pts_prev )
        _pts_curr_m =  np.array(_pts_curr_m )

        # code.interact( local=locals() )
        return  pts_curr, pts_prev, _pts_curr_m, per_match_vote, (dense_match_quality, after_vote_match_quality)



    # This function will compute the daisy matches, given the cluster assignments
    # from netvlad and dense daisy. Need to set_im() and set_im_lut() before calling this
    def daisy_dense_matches_with_scores(self, DEBUG=False):
        # DEBUG = True # in debug mode 4 things are returned, : m1, m2, mask and [xcanvas]
        #               in non-debug mode 3 things r returned
        assert self.im1 is not None, "GeometricVerification.daisy_dense_matches(): im1 was not set. "
        assert self.im2 is not None, "GeometricVerification.daisy_dense_matches(): im2 was not set. "
        assert self.im1_lut_raw is not None, "GeometricVerification.daisy_dense_matches(): im1_lut_raw was not set. "
        assert self.im2_lut_raw is not None, "GeometricVerification.daisy_dense_matches(): im2_lut was not set. "


        if DEBUG:
            assert self.im1_lut is not None, "GeometricVerification.daisy_dense_matches(): im1_lut was not set. "
            assert self.im2_lut is not None, "GeometricVerification.daisy_dense_matches(): im2_lut was not set. "


        # Get prominent_clusters
        startProminentClusters = time.time()
        Z_curr = self.prominent_clusters(im_no=1)
        Z_prev = self.prominent_clusters(im_no=2)
        self._print_time( 'Prominent clusters', startProminentClusters, time.time() )


        # Step-1 : Get Daisy at every point
        startDaisy = time.time()
        # Old code
        # D_curr = self.get_whole_image_daisy( im_no=1 )
        # D_prev = self.get_whole_image_daisy( im_no=2 )
        # self.daisy_im1 = D_curr
        # self.daisy_im2 = D_prev

        # new code (12th Nov, assumes daisy was already computed, just get views). Possibly dirty
        D_curr = self.view_daisy( ch=1 ) #self.my_daisy( im_curr, ch=0 )
        D_prev = self.view_daisy( ch=2 ) #my_daisy( im_prev, ch=1 )
        self._print_time( 'Daisy (2 images)', startDaisy, time.time() )

        # Step-2 : Given a k which is in both images, compare clusters with daisy. To do that do NN followd by Lowe's ratio test etc
        startDenseFLANN = time.time()
        Z_curr_uniq = np.unique( Z_curr )[1:] #from 1 to avoid 0 which is for no assigned cluster
        Z_prev_uniq = np.unique( Z_prev )[1:]
        # print Z_curr_uniq #list of uniq k
        # print Z_prev_uniq


        # Prepare FLANN Matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        # Loop over intersection, and merge all the pts from each of the k
        pts_A = []
        pts_B = []
        pt_match_quality_scores = [] #equal to size of pts_A or pts_B.
        k_intersection = set(Z_curr_uniq).intersection( set(Z_prev_uniq) )
        xcanvas_array = []
        if DEBUG:
            print 'k_intersection: ', k_intersection
        for k in k_intersection:
            H_curr = np.where( Z_curr==k ) #co-ordinates. these co-ordinates need testing
            desc_c = np.array( D_curr[ H_curr[0]*4, H_curr[1]*4 ] ) # This is since D_curr is (240,320) and H_curr is (60,80)

            H_prev = np.where( Z_prev==k ) #co-ordinates #remember , Z_prev is (80,60)
            desc_p = np.array( D_prev[ H_prev[0]*4, H_prev[1]*4 ] )

            matches = flann.knnMatch(desc_c.astype('float32'),desc_p.astype('float32'),k=2)
            _pts_A, _pts_B, _pts_lowe_ratio = self.analyze_dense_matches(  H_curr, H_prev, matches, return_lowe_ratio=True )

            # Scores to assess the quality of matches for this segment
            # -a- lowe's ratios are scored as. Note: lower ratio is better here:
            #       note, this scoring is done in call analyze_dense_matches() on per match basis
            #              x <= 0.67  ---> +2
            #       0.67 < x <= 0.72  ---> +1
            #       0.72 < x <= 0.75  ---> +0.5
            #       0.75 < x <= 0.8   ---> +0
            #
            # -b- percentage of dense points retained. Usually more than 10% is a good sign
            #          0 < x <= 10  --->  +0
            #         10 < x <= 15  --->  +1
            #         15 < x <= 30  --->  +1.5
            #         30 < x        --->  +2
            mean_of_lowe_ratios = np.mean( _pts_lowe_ratio )
            retained_percent = 0
            if len(_pts_A) != 0:
                retained_percent = float(len(_pts_A)) * 100. /desc_c.shape[0]

            if retained_percent > 0 and retained_percent <= 10:
                retained_percent_score = 0.
            else:
                if retained_percent > 10 and retained_percent <= 15:
                    retained_percent_score = 1.
                else:
                    if retained_percent > 15 and retained_percent <= 30:
                        retained_percent_score = 1.5
                    else:
                        retained_percent_score = 2.
            # End of Scoring


            pts_A += _pts_A
            pts_B += _pts_B
            # pts_lowe_ratio += _pts_lowe_ratio #Old way
            pt_match_quality_scores += list( np.array(_pts_lowe_ratio) + np.array(retained_percent_score) )
            # DEBUG
            if DEBUG:
                print '-_-_-_'
                print 'k=', k
                print 'Input Pts: ', desc_c.shape[0]
                print 'Retained : %d matches; %4.2f percent' %( len(_pts_A),  retained_percent)
                print 'Mean %4.2f' %(mean_of_lowe_ratios)
                xim1 = self.s_overlay( self.im1, np.int0(Z_curr==k), 0.7 )
                xim2 = self.s_overlay( self.im2, np.int0(Z_prev==k), 0.7 )
                # xcanvas = self.plot_point_sets( self.im1, _pts_A, self.im2, _pts_B)
                xcanvas = self.plot_point_sets( xim1, _pts_A, xim2, _pts_B, enable_text=True)
                cv2.imshow( 'xcanvas', xcanvas)
                cv2.waitKey(10)
                xcanvas_array.append( xcanvas )
                code.interact( local=locals(), banner='In daisy_dense_matches_with_scores()' )
            # END Debug


        self._print_time( 'Dense FLANN over common k=%s' %(str(k_intersection)), startDenseFLANN, time.time() )


        # DEBUG, checking pts_A, pts_B
        if DEBUG:
            print 'Total Matches : %d' %(len(pts_A))
            xcanvas = self.plot_point_sets( self.im1, pts_A, self.im2, pts_B)
            cv2.imshow( 'full_xcanvas', xcanvas)
            cv2.waitKey(0)

        # End DEBUG


        # Step-3 : Essential Matrix Text
        E, mask = cv2.findFundamentalMat( np.array( pts_A ), np.array( pts_B ), param1=5 )
        if DEBUG:
            print 'Total Dense Matches : ', len(pts_A)
            print 'Total Verified Dense Matches : ', mask.sum()
        # code.interact( local=locals() )


        return pts_A, pts_B, mask, pt_match_quality_scores



    def plot_point_sets_with_quality_scores( self, im1, pt1, im2, pt2, pt_match_quality_scores, mask=None, enable_text=False ):
        """
            Color coding according to pt_match_quality_scores.
            0 : E7D9D9 (rgb:231,217,217)
            1 : E7B7B7 (231,183,183)
            2 : E78A8A (231,138,138)
            3 : E75050 (231,80,80)
            4 : E70000 (231,0,0)
            5 : FF0000 (255,0,0)
        """
        xcolor = []
        xcolor.append( (217,217,231) )
        xcolor.append( (183,183,231) )
        xcolor.append( (138,138,231) )
        xcolor.append( ( 80, 80,231) )
        xcolor.append( (  0,  0,231) )
        xcolor.append( (  0,  0,255) )

        assert( len(pt1) == len(pt2) )
        assert( len(pt1) == len(pt_match_quality_scores) )
        xcanvas = np.concatenate( (im1, im2), axis=1 )
        for xi in range( len(pt1) ):
            if (mask is not None) and (mask[xi,0] == 0):
                continue

            cv2.circle( xcanvas, pt1[xi], 4, (255,0,255) )
            ptb = tuple(np.array(pt2[xi]) + [im1.shape[1],0])
            cv2.circle( xcanvas, ptb, 4, (255,0,255) )

            c_indx = min(5,int( np.floor(pt_match_quality_scores[xi]) ) )
            cv2.line( xcanvas, pt1[xi], ptb, xcolor[ c_indx ] )


            if enable_text:
                color_com = (0,0,255)
                cv2.putText( xcanvas, str(xi), pt1[xi], cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )
                cv2.putText( xcanvas, str(xi), ptb, cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )
        return xcanvas
