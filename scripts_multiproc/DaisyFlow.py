""" This is meant for the local bundle feature tracking using Daisy

        The basic idea is that once you have a dense match say A<--->B,
        we track the matched featues in A-1, A-2, ... and B-1, B-2, ...

        Once we have feature visibilities in nearby frames we solve a
        local bundle adjustment problem (with ceres) to get good
        estimates of the poses. In the earlier edition, we used just 3 frames
        viz A, A-1, and B which resulted in poor triangulations and bad
        estimates.

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


class DaisyFlow:
    def __init__(self):
        # setup multi daisys
        self.dai = []
        for i in range(4):
            self.xprint( '__init__', 'Setup daisy [%d]' %(i) )
            self.dai.append( DaisyMeld( 240, 320, 0 ) )

        self.uim = {}
        self.uim_lut = {}


    def xprint( self, header, msg ):
        return
        if header == 'daisy_dense_matches_with_scores':
            return
        print '[%s] %s' %(header, msg)

    def _print_time(self, header, msg, startT, endT):
        # return
        s = tcol.OKBLUE, '%8.2f :%s (ms)'  %( 1000. * (endT - startT), msg ), tcol.ENDC
        self.xprint( header, s )


    ################## Utils ########################
    def _second_largest(self,numbers):
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

    ####################### DAISY ######################################

    def _crunch_daisy( self, ch, d_ch ):
        """ do daisy computation using daisy[d_ch] """
        assert( self.uim[ch] is not None )
        ch = int(ch)
        d_ch = int(d_ch)
        assert ch in self.uim.keys(), "%d is not in self.uim.keys()" %(ch)
        assert d_ch < len(self.dai), "DaisyMeld is not allocated for index %d" %(d_ch)


        self.xprint( '_crunch_daisy', 'compute for self.uim[%d] using self.dai[%d]' %(ch, d_ch) )
        if len(self.uim[ch].shape) == 3:
            xim_gray = cv2.cvtColor( self.uim[ch], cv2.COLOR_BGR2GRAY ).astype('float32')
        else:
            xim_gray = self.uim[ch].astype('float32')

        start = time.time()
        self.dai[d_ch].do_daisy_computation( xim_gray )
        self.xprint( '_crunch_daisy', 'do_daisy_computation in %f ms' %( (1000. * (time.time() - start)) ) )

    def _view_daisy( self, d_ch ):
        d_ch = int(d_ch)
        assert d_ch < len(self.dai), "DaisyMeld is not allocated for index %d. So cannot view" %(d_ch)

        return self.dai[d_ch].get_daisy_view()


    def set_image( self, image, ch, d_ch ):
        """ Sets the image in channel ch. Do daisy computation using d_ch.
        It is OK to set a color image here, bcoz in _crunch_daisy() we have a rgb2gray.
        """
        ch = int(ch)
        d_ch = int(d_ch)
        self.xprint( 'set_image', 'set image_(%s) into channel %d' %(str(image.shape), ch ) )
        self.uim[ch] = image.astype('uint8')

        if d_ch >= 0 :
            self._crunch_daisy( ch, d_ch )

    def set_lut( self, lut, ch ):
        """ sets lut in channel ch"""
        ch = int(ch)
        self.uim_lut[ch] = lut

    def imshow_daisy( self, d_ch, cv_win_name="cv_window", daisy_depth=7 ):
        """ show 0th channel of daisy[d_ch] """
        X = self._view_daisy( d_ch )
        self.xprint( 'imshow_daisy', 'X.shape: %s. Showing %dth channel of daisy' %( str(X.shape), daisy_depth ) )

        cv2.imshow( str(cv_win_name), X[:,:,daisy_depth] )

    #####################################################################################


    ########################### Plotting ################################################
    def plot_point_sets( self, im1, pt1, im2, pt2, mask=None, enable_text=False, mark=None, markmin=None, markmax=None ):
        assert len(pt1) == len(pt2)
        xcanvas = np.concatenate( (im1, im2), axis=1 )
        if mark is not None:
            assert len(mark) == len(pt1)
            lut = ColorLUT()


        for xi in range( len(pt1) ):
            if (mask is not None) and (mask[xi,0] == 0):
                continue

            cv2.circle( xcanvas, tuple(pt1[xi]), 4, (255,0,255) )
            ptb = tuple(np.array(pt2[xi]) + [im1.shape[1],0])
            cv2.circle( xcanvas, ptb, 4, (255,0,255) )

            if mark is None:
                line_color = (255,0,0)
            else:
                # c = lut.float_2_rgb( mark[xi], mark.min(), mark.max() )
                if markmin is None or markmax is None:
                    c = lut.float_2_rgb( mark[xi], 0, 20 )
                else :
                    c = lut.float_2_rgb( mark[xi], markmin, markmax )
                line_color = 255*np.array(c).astype('uint8')
                line_color = ( int(line_color[2]), int(line_color[1]), int(line_color[0]) )

            cv2.line( xcanvas, tuple(pt1[xi]), ptb, line_color)  #true color is blue


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


    ############################# HELPERS FOR real computation ########################
    def _prominent_clusters( self, ch ):
        # print tcol.OKGREEN, 'prominent_clusters : Uses im?,im?_lut,im?_lut_raw', tcol.ENDC

        assert ch in self.uim.keys()
        assert ch in self.uim_lut.keys()

        im = self.uim[ch]
        im_lut_raw = self.uim_lut[ch]


        # TODO Consider passing these as function arguments

        K = im_lut_raw.max()+1  #number of clusters in netvlad
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
            print 'Time taken to find top clusters (ms) :', 1000.*(time.time() - startTime)

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


    # Called by daisy_dense_matches() to loop over matches.
    def _ratiotest_on_matches( self, H1, H2, matches, return_lowe_ratio=False ):
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


    def _sieve_stat_to_score( self, sieve_stat, PRINTING=False ):
        if PRINTING:
            print 'DaisyFlow.py/_sieve_stat_to_score'
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
        if PRINTING:
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
        if match2_tretained_score > 0.5:
            match2_total_score += 0.5 #extra points if large quantity of points are retained.

        if match2_geometric_score > 0.55:
            match2_total_score -= 1.
        else:
            if match2_geometric_score < 0.4 and match2_geometric_score >= 0.3:
                match2_total_score += 1.0
            if match2_geometric_score < 0.3 and match2_geometric_score >= 0.2:
                match2_total_score += 1.5
            if match2_geometric_score < 0.2 :
                match2_total_score += 1.5


        # if absolute number of tracked points  TODO

        if PRINTING:
            print '==Total_score : ', match2_total_score, '=='
        return match2_total_score




    ############################### PUBLIC ##############################################

    def daisy_dense_matches( self, ch0, d_ch0, ch1, d_ch1 ):
        """ Do daisy_dense_matches() for images self.uim[ch0] <--> self.uim[ch1].
            Assume their daisy-descriptors can be retrived as self._view_daisy[d_ch0] and
            self._view_daisy[d_ch1] respectively.

            netvlad lut is also required

            ch0   : name of the 0th image (number) will use self.uim[ch0]
            d_ch0 : where the daisy-descriptors of 0th image is stored, will use self.dai[d_ch0]
            ch1   : name of the 1st image (number) will use self.uim[ch1]
            d_ch1 : where the daisy-descriptors of 1st image is stored. will use self.dai[d_ch1]

            returns
                pts_A : [ (123,32), (111,112), ... ] co-ordinates in 1st image
                pts_B : [ (123,32), (111,112), ... ] matches of pts_A in 2nd image
                pt_match_quality_scores : Score (from 0.5-2) of each match as per lowe's ratio test

                All the returned are already filtered as per f-test. Hence we can delete
                the mask after this call.

        """

        assert self.uim[ch0] is not None
        assert self.uim[ch1] is not None


        # Get prominent clusters
        startProminentClusters = time.time()
        Z_curr = self._prominent_clusters(ch0)
        Z_prev = self._prominent_clusters(ch1)
        # self.xprint( 'daisy_dense_matches_with_scores', 'Prominent clusters computation took (ms): %f' %( 1000. * ( time.time() - startProminentClusters  ) ) )
        self._print_time( 'daisy_dense_matches_with_scores', 'Prominent clusters', startProminentClusters, time.time() )



        # Step-1: Get Daisy at every point
        startDaisy = time.time()
        D_curr = self._view_daisy( d_ch=d_ch0 )
        D_prev = self._view_daisy( d_ch=d_ch1 )
        self._print_time( 'daisy_dense_matches_with_scores', 'Daisy (2 images)', startDaisy, time.time() )



        # Step-2 : Given a k which is in both images, compare clusters with daisy. To do that do NN followd by Lowe's ratio test etc
        startDenseFLANN = time.time()
        Z_curr_uniq = np.unique( Z_curr )[1:] #from 1 to avoid 0 which is for no assigned cluster
        Z_prev_uniq = np.unique( Z_prev )[1:]


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
        self.xprint( 'daisy_dense_matches_with_scores', 'step2, Z_curr_uniq: %s; Z_prev_uniq: %s' %( str(Z_curr_uniq), str(Z_prev_uniq) ))
        self.xprint( 'daisy_dense_matches_with_scores', 'step2, k_intersection: %s.' %( str(k_intersection)))

        for k in k_intersection:
            H_curr = np.where( Z_curr==k ) #co-ordinates. these co-ordinates need testing
            desc_c = np.array( D_curr[ H_curr[0]*4, H_curr[1]*4 ] ) # This is since D_curr is (240,320) and H_curr is (60,80)

            H_prev = np.where( Z_prev==k ) #co-ordinates #remember , Z_prev is (80,60)
            desc_p = np.array( D_prev[ H_prev[0]*4, H_prev[1]*4 ] )

            matches = flann.knnMatch(desc_c.astype('float32'),desc_p.astype('float32'),k=2)
            _pts_A, _pts_B, _pts_lowe_ratio = self._ratiotest_on_matches(  H_curr, H_prev, matches, return_lowe_ratio=True )


            # Analysis on how many remained after lowe's ratio test
            mean_of_lowe_ratios = np.mean( _pts_lowe_ratio )
            retained_percent = 0
            if len(_pts_A) != 0:
                retained_percent = float(len(_pts_A)) * 100. /desc_c.shape[0]

            self.xprint( 'daisy_dense_matches_with_scores', 'for cluster k=%d, lowe retained_percent: %f' %( k, retained_percent ) )

            pts_A += _pts_A
            pts_B += _pts_B
            pt_match_quality_scores += _pts_lowe_ratio #list( np.array(_pts_lowe_ratio) )
            # DEBUG
            if False:
                print '-_-_-_'
                print 'k=', k
                print 'Input Pts: ', desc_c.shape[0]
                print 'Retained : %d matches; %4.2f percent' %( len(_pts_A),  retained_percent)
                print 'Mean %4.2f' %(mean_of_lowe_ratios)
                xim1 = self.s_overlay( self.uim[ch0], np.int0(Z_curr==k), 0.7 )
                xim2 = self.s_overlay( self.uim[ch1], np.int0(Z_prev==k), 0.7 )
                xcanvas = self.plot_point_sets( xim1, _pts_A, xim2, _pts_B, enable_text=True)
                cv2.imshow( 'xcanvas', xcanvas)
                cv2.waitKey(0)
                xcanvas_array.append( xcanvas )
                # code.interact( local=locals(), banner='In daisy_dense_matches_with_scores()' )
            # END Debug

            # code.interact( local=locals(), banner="" )

        # DEBUG, checking pts_A, pts_B
        if False:
            print 'Total Matches : %d' %(len(pts_A))
            xcanvas = self.plot_point_sets( self.uim[ch0], pts_A, self.uim[ch1], pts_B)
            cv2.imshow( 'full_xcanvas', xcanvas)
            cv2.waitKey(0)

        self._print_time( 'daisy_dense_matches_with_scores', 'Dense FLANN over common k=%s' %(str(k_intersection)), startDenseFLANN, time.time() )


        # Step-3 : Essential Matrix Text
        E, mask = cv2.findFundamentalMat( np.array( pts_A ), np.array( pts_B ), param1=5 )

        self.xprint( 'daisy_dense_matches_with_scores', 'Total Dense Matches : %d' %len(pts_A) )
        self.xprint( 'daisy_dense_matches_with_scores', 'Total Verified Dense Matches : %d' %mask.sum() )
        if False:
            xcanvas = self.plot_point_sets( self.uim[ch0], pts_A, self.uim[ch1], pts_B, mask )
            cv2.imshow( 'full_xcanvas_f_test', xcanvas)
            cv2.waitKey(0)



        masked_pts_A = list( pts_A[i] for i in np.where( mask[:,0] == 1 )[0] )
        masked_pts_B = list( pts_B[i] for i in np.where( mask[:,0] == 1 )[0] )
        masked_pt_match_quality_scores = list( pt_match_quality_scores[i] for i in np.where( mask[:,0] == 1 )[0] )


        return masked_pts_A, masked_pts_B, masked_pt_match_quality_scores

    def make_dilated_mask_from_pts( self,pts, dims, win_size=32 ):
        pts = np.array(pts)
        m = np.zeros( dims, dtype='uint8' )
        m[ pts[:,1], pts[:,0] ] = 255

        # dilate
        m_out = cv2.dilate( m, np.ones((win_size,win_size), dtype='uint8') )

        return m_out

    def expand_matches( self, ch0, d_ch0,  pts, chx, d_chx ):
        """ using co-ordinates pts in image pointed by ch0 (corresponding daisy-descriptors in d_ch0),
        this function will expand the match to another image (in temporal proximity) pointed
        by ch1 (corresponding daisy in d_ch1).

        pts : [ (32,22), (120,43), .... ]
        ch0, d_ch0, chx, d_chx are integers
        """
        PARAM_W = 30
        DEBUG = False


        # retrive images
        __im0 = self.uim[ch0]
        __im1 = self.uim[chx]


        _pts = np.array(pts)# np.array( masked_pts ). note these are (x,y) or (col,row)

        dilated_mask = self.make_dilated_mask_from_pts( _pts, __im0.shape[0:2], win_size=32 )
        _pts_chx_rows, _pts_chx_cols = np.where( dilated_mask > 0 )

        # code.interact( local=locals() )

        # retrive daisy descriptors
        D_curr = self._view_daisy( d_ch=d_ch0 )
        D_pts_curr = D_curr[  _pts[:,1], _pts[:,0], : ] # Nx20 retrive daisy of curr only at specified pts.
        D_currm = self._view_daisy( d_ch=d_chx ) # 240x320x20 dense daisy of currm (ie. chx, d_chx)
        D_pts_currm = D_currm[ _pts_chx_rows, _pts_chx_cols, : ] # daisy descriptors of image currm based on the dilated mask computed above

        # make FLANN index for D_currm. search for D_pts_curr in this index
        startFLANN = time.time()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)


        # Using all
        # matches = flann.knnMatch( D_pts_curr, D_currm.reshape( -1, 20 ), k=2 ) #N matches.
        matches = flann.knnMatch( D_pts_curr, D_pts_currm, k=2 ) #N matches.

        #for speed just down-sample the image.A Another way could be selective sample. ie.
        # only put points in the tree which are in the mask. A mask can be obtained
        # marking the points D_pts_curr on a black image. Dilate this image in a say 40x40 neighbourhood.
        # Only put points in the mask for KD-tree instead of currently everything.
        # matches = flann.knnMatch( D_pts_curr, D_currm.reshape( -1, 20 ), k=2 ) #N matches.
        # print 'expand_matches() flann took: %4.2f' %(  1000. * (time.time() - startFLANN) )

        __q_pt = []
        __t_pt = []
        __lowe_ratio = []
        __topNN_dist = []
        for m in matches: #looping over all the matches
            qIdx = m[0].queryIdx
            tIdx = m[0].trainIdx
            lowe_ratio = m[0].distance / m[1].distance

            qPt = _pts[qIdx] # this is x,y or (col,row). To plot of circles we need x,y
            # tPt = np.unravel_index( tIdx, D_currm.shape[0:2] ) #this is row,col
            tPt = (_pts_chx_rows[tIdx], _pts_chx_cols[tIdx] ) #this is row,col
            tPt = ( tPt[1], tPt[0] ) #convert to (x,y)

            __q_pt.append( qPt )
            __t_pt.append( tPt )
            __lowe_ratio.append( lowe_ratio )
            __topNN_dist.append( m[0].distance )

            if DEBUG : #verified ...correct
                print qIdx, tIdx, 'lowe_ratio=%.2f, topNN_dist=%2.2f' %(lowe_ratio, m[0].distance )
                print qPt, tPt

                _q = __im0.copy() # query Image
                _t = __im1.copy() # train image
                cv2.circle( _q, tuple(qPt), 2, (255,0,0), -1 )
                cv2.circle( _t, tuple(tPt), 2, (255,0,0), -1 )
                # cv2.circle( _t, tuple((tPt[1], tPt[0])), 1, (255,0,0) )
                cv2.imshow( '_q', _q )
                cv2.imshow( '_t', _t )
                cv2.waitKey(0)


        # Essential Matrix Text. Ideally should use the Fundamental matrix
        # from the odometry measurements. As we know that these image are going to be
        # consicutive
        E, mask = cv2.findFundamentalMat( np.array( __q_pt ), np.array( __t_pt ), param1=5 )


        return __t_pt, __topNN_dist, __lowe_ratio, mask




    def guided_matches( self, ch0, d_ch0, pts0, ch1, d_ch1, pts1 ):
        """ Given two images and tracked points on them. Produce matching on these
            pointsets.

            pts0 : 3xN
            pts1 : 3xM (2d uv points) homogeneous

            returns:
            Returns 3 values
            - indices of the pts0 and pts1 which are retained.
            - score (by heuristics). Read code to know details.

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
        PRINTING = False
        return_per_match_vote_score = False

        assert self.uim[ch0] is not None
        assert self.uim[ch1] is not None


        im_curr = self.uim[ch0]
        im_prev = self.uim[ch1]
        pts_curr = pts0
        pts_prev = pts1

        # self.crunch_daisy( ch=1 ) #if image is set daisy is automatially computed
        # self.crunch_daisy( ch=2 )

        daisy_curr = self._view_daisy( d_ch=d_ch0 )
        daisy_prev = self._view_daisy( d_ch=d_ch1 )



        if PRINTING:
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
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch( np.array(A), np.array(B), k=K_nn )
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
            top2 = self._second_largest( iS[nz] )

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
                # if True:
                    code.interact( local=locals(), banner="guided_matches:")
                    print 'ACCEPTED'
                    cv2.imshow( 'curr_overlay', self.points_overlay( im_curr, ptxA, enable_text=True ))
                    cv2.imshow( 'prev_overlay', self.points_overlay( im_prev, ptxB, enable_text=True ))
                    cv2.imshow( 'prev_overlay top', self.points_overlay( im_prev, ptyB, enable_text=True ))
                    cv2.waitKey(0)

        if len(selected_A) == 0:
            return np.array([]), np.array([]), None

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
            return np.array([]), np.array([]), None

        if PRINTING:
            print 'A: %s ; B: %s ; nInliers:%d' %(str(selected_A.shape), str(selected_B.shape), nInliers)
        masked_pts_curr = np.transpose( np.array( list( selected_A[:,i] for i in np.where( mask[:,0] == 1 )[0] ) ) )
        masked_pts_prev = np.transpose( np.array( list( selected_B[:,i] for i in np.where( mask[:,0] == 1 )[0] ) ) )

        if PRINTING:
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



        per_match_vote_score = np.array(masked_selected_score)


        # _stat_ to score

        # return np.array(masked_selected_A_i), np.array(masked_selected_B_i), _stat_, np.array(masked_selected_score)#max(masked_selected_score)
        _stat_ = (min(pts_curr.shape[1], pts_prev.shape[1]), selected_A.shape[1], nInliers)

        match2_total_score = self._sieve_stat_to_score( _stat_, PRINTING )
        if PRINTING:
            print '=X=Total_score : '+ str(match2_total_score)+ '=X='

        # Rules time now !

        return np.array(masked_selected_A_i), np.array(masked_selected_B_i), match2_total_score




if __name__=="__main__":
    print 'hello'
