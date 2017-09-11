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
    print 'If you get this error, your DaisyMeld wrapper is not properly setup. You need to set DaisyMeld in LD_LIBRARY_PATH. '
    print 'See also : https://github.com/mpkuse/daisy_py_wrapper'
from ColorLUT import ColorLUT




import TerminalColors
tcol = TerminalColors.bcolors()

class GeometricVerification:
    def __init__(self):
        _x = 0
        self.reset()
        self.orb = cv2.ORB_create()
        self.surf = cv2.xfeatures2d.SURF_create( 400, 5, 5 )

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

        self.dai = DaisyMeld()

    def reset(self):
        self.im1 = None
        self.im2 = None

        self.im1_lut = None
        self.im2_lut = None

    def set_im(self,im1, im2):
        self.im1 = im1
        self.im2 = im2
        self.daisy_im1 = None
        self.daisy_im2 = None

    def set_im_lut(self,im1_lut, im2_lut):
        self.im1_lut = im1_lut
        self.im2_lut = im2_lut

    def set_im_lut_raw(self,im1_lut_raw, im2_lut_raw):
        self.im1_lut_raw = im1_lut_raw
        self.im2_lut_raw = im2_lut_raw

    # TODO. Define another function to set full resolution images

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

    def _print_time(self, msg, startT, endT):
        return
        print tcol.OKBLUE, '%8.2f :%s (ms)'  %( 1000. * (endT - startT), msg ), tcol.ENDC

    # Features : ['orb', 'surf']
    def simple_verify(self, features='orb', debug_images=False):
        print tcol.OKGREEN, 'This simple_verify() uses im1, im2 only', tcol.ENDC

        # Simple verification using ORB keypoints and essential matrix
        # 1. Extract Keypoints and descriptor
        startFeatures = time.time()
        if features == 'orb':
            kp1, des1 = self.orb.detectAndCompute( self.im1, None )
            kp2, des2 = self.orb.detectAndCompute( self.im2, None )
        elif features == 'surf':
            kp1, des1 = self.surf.detectAndCompute( self.im1, None )
            kp2, des2 = self.surf.detectAndCompute( self.im2, None )
        else:
            print 'INVALID FEATURE TYPE....quit()'
            quit()
        self._print_time( 'feature ext '+features, startFeatures, time.time() )

        # code.interact( local=locals() )
        # kp1, des1 = self.orb.detectAndCompute( self.im1, (self.im1_lut_raw==23).astype('uint8') )
        # kp2, des2 = self.orb.detectAndCompute( self.im2, (self.im2_lut_raw==23).astype('uint8') )

        # 2. Lowe's Ratio Test
        # cv2.imshow( 'kp1', cv2.drawKeypoints( self.im1, kp1, None ) )
        # cv2.imshow( 'kp2', cv2.drawKeypoints( self.im2, kp2, None ) )
        startFLANN = time.time()
        matches_org = self.flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2) #find 2 nearest neighbours
        self._print_time( 'flann matcher', startFLANN, time.time() )

        startLoweRatioTest = time.time()
        __pt1, __pt2 = self._lowe_ratio_test( kp1, kp2, matches_org )
        nMatches = __pt1.shape[0]
        self._print_time( 'lowe ratio', startLoweRatioTest, time.time() )


        # 3. Essential Matrix test
        nInliers = 0
        startEssentialMatTest = time.time()
        if __pt1.shape[0] < 10: #only attempt to compute fundamental matrix if atleast 8 correspondences
            if debug_images == True:
                return nMatches, nInliers,  np.concatenate( (self.im1, self.im2), axis=1 )
            else:
                return nMatches, nInliers


        # E, mask = cv2.findEssentialMat( __pt1, __pt2 )
        E, mask = cv2.findFundamentalMat( __pt1, __pt2,param1=5 )
        if mask is not None:
            nInliers = mask.sum()

        self._print_time( 'essential mat test', startEssentialMatTest, time.time() )

        print 'nMatches   : ', nMatches
        print 'nInliers   : ', nInliers

        self._print_time( 'total elapsed in simple_verify (ms): ', startFeatures, time.time() )

        # Make debug images
        if debug_images == True:
            xpt1 = __pt1
            xpt2 = __pt2

            # Epilines
            if E is not None:
                # lines1 = cv2.computeCorrespondEpilines( xpt1, 1, E ) #Nx1x3
                # lines2 = cv2.computeCorrespondEpilines( xpt2, 2, E )
                # epilines_p1_l2 = self.debug_draw_epilines( self.im1, self.im2, __pt1, __pt2, lines1, lines2, mask )
                # epilines_l1_p1 = self.debug_draw_epilines( self.im2, self.im1, __pt2, __pt1, lines2, lines1, mask, invert_concat=True )
                #
                # cv2.imshow( 'epilines_p1_l2', epilines_p1_l2 )
                # cv2.imshow( 'epilines_l1_p1', epilines_l1_p1 )
                pass


            # Draw Point matches
            canvas = self.debug_draw_matches( self.im1, __pt1, self.im2, __pt2, mask ) #im1 is the original input image, __pt1 is an Nx2 matrix of co-ordinates. mask is output from the essential mat computation
            return nMatches, nInliers, canvas

        return nMatches, nInliers


    def debug_draw_epilines(self,ximg1,ximg2,pts1,pts2, lines1, lines2, mask, invert_concat=False):
        img1 = ximg1.copy()
        img2 = ximg2.copy()

        _i = 0
        for _i in range( pts1.shape[0] ):
            if mask[_i,0] == 0:
                continue
            p1 = tuple(np.int0(pts1[_i,:]))
            p2 = tuple(np.int0(pts2[_i,:]))
            # l1: 2 points. corresponding to p1 but on img2
            l1 = [ ( 0, np.int0(-lines1[_i,0,2]/lines1[_i,0,1]) ), ( np.int0(-lines1[_i,0,2]/lines1[_i,0,0]), 0) ]

            color = tuple( np.random.randint(0,255,3).tolist() )
            cv2.circle( img1, p1, 2, color ) #plot 1st pt of im1
            cv2.circle( img2, p2, 2, color ) #plot 1st pt of im2
            cv2.line( img2, l1[0], l1[1], color ) #draw epiline corresponding to p1

        if invert_concat == False:
            epilines = np.concatenate( (img1, img2), axis=1 )
        else:
            epilines = np.concatenate( (img2, img1), axis=1 )
        return epilines
        # cv2.imshow( 'epilines', np.concatenate( (img1, img2), axis=1 ) )
        # cv2.waitKey(0)
        # code.interact( local=locals() )



    def debug_draw_matches( self, im1, pt1, im2, pt2, mask ):
        canvas = np.concatenate( (im1, im2), axis=1 )
        cv2.putText( canvas, 'nMatches: %03d' %(pt1.shape[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) )
        cv2.putText( canvas, 'nInliers: %03d' %(mask.sum()), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) )
        for p in range( pt1.shape[0] ): #loop over pt1 (and pt2)
            if mask[p,0] == 0:
                continue
            p1 = tuple(np.int0(pt1[p,:]))
            p2 = tuple(np.int0(pt2[p,:]) + [ 320,0 ] )

            cv2.circle( canvas, p1, 2, (255,0,0) )
            cv2.circle( canvas, p2, 2, (0,0,255) )
            cv2.line( canvas, p1, p2, (0,0,255) )
            # cv2.arrowedLine( canvas, p1, p2, (0,0,255) )

        return canvas
        code.interact( local=locals() )
        cv2.imshow( 'canvas', canvas )
        cv2.waitKey(0)





    # Mark for removal
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


    def prominent_clusters( self, im_no=-1 ):
        # print tcol.OKGREEN, 'prominent_clusters : Uses im?,im?_lut,im?_lut_raw', tcol.ENDC

        if im_no == 1:
            im = self.im1
            im_lut = self.im1_lut
            im_lut_raw = self.im1_lut_raw
        elif im_no==2:
            im = self.im2
            im_lut = self.im2_lut
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

    ## Given an image (im) and a 4x2 matrix defining 4 co-ordinate points (rotated rect) on the image.
    ## Computes the affine transform between these 4 points to a right-angled rectangle while maintaining
    ## the aspect ratio. Returns a 100 x a*100 or a*100 x 100 image.
    def affine_warp( self, im, contours, n_cols=100 ):
        #find which dim is larger
        _dima = np.sqrt( np.linalg.norm( contours[0] - contours[1] ) )
        _dimb = np.sqrt( np.linalg.norm( contours[1] - contours[2] ) )
        aspect = max( _dima/_dimb, _dimb/_dima )
        if _dima > _dimb:
            # print 'dima is bigger'
            from_ = contours[0:3]
            to_ = np.array( [ [0,0], [np.int(aspect*n_cols),0], [np.int(aspect*n_cols), n_cols] ] )
            # print 'from_', from_
            # print 'to_', to_
            M = cv2.getAffineTransform( from_.astype('float32'), to_.astype('float32') )
            OU = np.zeros( (n_cols,np.int(n_cols*aspect) ) )
            # OU = np.zeros( (500,500) )
            OU = cv2.warpAffine( im, M, tuple(OU.shape) )
            print 'OU.shape', OU.shape
            return OU

        else:
            # print 'dimb is bigger'
            from_ = contours[0:3]
            A = aspect
            to_ = np.array( [ [0,0], [0, n_cols], [np.int(A*n_cols), n_cols]  ]  )
            M = cv2.getAffineTransform( from_.astype('float32'), to_.astype('float32') )
            OU = np.zeros( (np.int(n_cols*aspect), n_cols ) )
            # OU = np.zeros( (500,500) )
            OU = np.swapaxes( cv2.warpAffine( im, M, tuple(OU.shape) ), 0, 1 )
            print 'OU.shape', OU.shape
            return OU


    # Return the daisy image for input image
    def static_get_daisy_descriptor_mat( self, xim ):
        xim_gray = cv2.cvtColor( xim, cv2.COLOR_BGR2GRAY ).astype('float64')
        output = self.dai.hook( xim_gray.flatten(), xim_gray.shape )
        output = np.array( output ).reshape( xim_gray.shape[0], xim_gray.shape[1], -1 )
        return output

    def get_whole_image_daisy(self, im_no):
        # cv2.imshow( 'do_daisy_im1', self.im1 )
        # cv2.imshow( 'do_daisy_im2', self.im2 )


        startD = time.time()
        printing = False
        if im_no == 1:
            im1_gray = cv2.cvtColor( self.im1, cv2.COLOR_BGR2GRAY ).astype('float64')
        elif im_no == 2:
            im1_gray = cv2.cvtColor( self.im2, cv2.COLOR_BGR2GRAY ).astype('float64')
        else:
            print 'ERROR in get_whole_image_daisy. im_no has to be 1 or 2'
            quit()

        #NOTE: Have DaisyMeld/ is LD_LIBRARY_PATH
        output = self.dai.hook( im1_gray.flatten(), im1_gray.shape )
        output = np.array( output ).reshape( im1_gray.shape[0], im1_gray.shape[1], -1 )

        if printing:
            print im1_gray.dtype, im1_gray.shape, output.shape, 'daisy in (ms) %4.2f' %(1000. * (time.time() - startD) )
        # cv2.imshow( 'do_daisy_im1', output[:,:,0] )
        return output


    def analyze_dense_matches( self, H1, H2, matches ):
        M = []
        # Lowe's ratio test
        for i, (m,n) in enumerate( matches ):
            #print i, m.trainIdx, m.queryIdx, m.distance, n.trainIdx, n.queryIdx, n.distance
            if m.distance < 0.8 * n.distance:
                M.append( m )

        # print '%d of %d pass lowe\'s ratio test' %( len(M), len(matches) )

        # plot
        canvas = np.concatenate( (self.im1, self.im2), axis=1 )
        pts_A = []
        pts_B = []

        for m in M: #good matches
            # print m.trainIdx, m.queryIdx, m.distance

            a = H1[1][m.queryIdx]*4
            b = H1[0][m.queryIdx]*4

            c = H2[1][m.trainIdx]*4
            d = H2[0][m.trainIdx]*4

            pts_A.append( (a,b) )
            pts_B.append( (c,d) )

            # cv2.circle( canvas, (a,b), 4, (0,0,255) )
            # cv2.circle( canvas, (c+self.im1.shape[1],d), 4, (0,0,255) )
            # cv2.line( canvas, (a,b), (c+self.im1.shape[1],d), (0,255,0) )
        # cv2.imshow( 'canvas_dense', canvas )
        # return canvas, pts_A, pts_B


        return pts_A, pts_B




    # This function will compute the daisy matches, given the cluster assignments
    # from netvlad and dense daisy. Need to set_im() and set_im_lut() before calling this
    def daisy_dense_matches(self):
        DEBUG = False


        # Get prominent_clusters
        startProminentClusters = time.time()
        Z_curr = self.prominent_clusters(im_no=1)
        Z_prev = self.prominent_clusters(im_no=2)
        self._print_time( 'Prominent clusters', startProminentClusters, time.time() )


        # Step-1 : Get Daisy at every point
        startDaisy = time.time()
        D_curr = self.get_whole_image_daisy( im_no=1 )
        D_prev = self.get_whole_image_daisy( im_no=2 )
        self.daisy_im1 = D_curr
        self.daisy_im2 = D_prev
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
        print 'Total Dense Matches : ', len(pts_A)
        print 'Total Verified Dense Matches : ', mask.sum()
        # code.interact( local=locals() )


        return pts_A, pts_B, mask


    def plot_point_sets( self, im1, pt1, im2, pt2, mask=None ):
        xcanvas = np.concatenate( (im1, im2), axis=1 )
        for xi in range( len(pt1) ):
            if (mask is not None) and (mask[xi,0] == 0):
                continue

            cv2.circle( xcanvas, pt1[xi], 4, (255,0,255) )
            ptb = tuple(np.array(pt2[xi]) + [im1.shape[1],0])
            cv2.circle( xcanvas, ptb, 4, (255,0,255) )
            cv2.line( xcanvas, pt1[xi], ptb, (255,0,0) )
        return xcanvas


    def s_overlay( self, im1, Z, alpha=0.5 ):
        lut = ColorLUT()
        out_imcurr = lut.lut( Z )

        out_im = alpha*im1 + (1 - alpha)*cv2.resize(out_imcurr,  (im1.shape[1], im1.shape[0])  )
        return out_im.astype('uint8')




    # grid : [ [curr, prev], [curr-1  X ] ]
    def plot_3way_match( self, curr_im, pts_curr, prev_im, pts_prev, curr_m_im, pts_curr_m ):

        r1 = np.concatenate( ( curr_im, prev_im ), axis=1 )
        r2 = np.concatenate( ( curr_m_im, np.zeros( curr_im.shape, dtype='uint8' ) ), axis=1 )
        gridd = np.concatenate( (r1,r2), axis=0 )

        print 'pts_curr.shape   ', len(pts_curr)
        print 'pts_prev.shape   ', len(pts_prev)
        print 'pts_curr_m.shape ', len(pts_curr_m)
        # all 3 should have same number of points

        for xi in range( len(pts_curr) ):
            cv2.circle( gridd, pts_curr[xi], 4, (0,255,0) )
            ptb = tuple(np.array(pts_prev[xi]) + [curr_im.shape[1],0])
            cv2.circle( gridd, ptb, 4, (0,255,0) )
            cv2.line( gridd, pts_curr[xi], ptb, (255,0,0) )

        for xi in range( len(pts_curr) ):
            # cv2.circle( gridd, pts_curr[xi], 4, (0,255,0) )
            ptb = tuple(np.array(pts_curr_m[xi]) + [0,curr_im.shape[0]])
            cv2.circle( gridd, ptb, 4, (0,255,0) )
            cv2.line( gridd, pts_curr[xi], ptb, (255,30,255) )


        return gridd


    # Given the matches between curr and prev. expand these matches onto curr-1.
    # Main purpose is to get a 3-way matches for pose computation. The way this
    # is done is for each match between curr and prev, find the NN in curr_m_im
    # around the neighbourhood of 40x40 in daisy space.
    #
    #   pts_curr, pts_prev, mask_c_p : output from daisy_dense_matches()
    #   curr_m_im : curr-1 image
    def expand_matches_to_curr_m( self, pts_curr, pts_prev, mask_c_p,   curr_m_im):

        DEBUG = False
        PARAM_W = 20

        masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
        masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )

        _pts_curr = np.array( masked_pts_curr )

        D_pts_curr = self.daisy_im1[ _pts_curr[:,1], _pts_curr[:,0], : ] #assuming im1 is curr

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
        daisy_c_m = self.static_get_daisy_descriptor_mat(  curr_m_im  )#Daisy of (curr-1)
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
