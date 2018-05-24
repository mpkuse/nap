 #
 # Implementation of a storage class for dense tracks. This will store the following:
 #    f{i} ==> feature id
 #    428 ==> a_idx
 #    51  ==> b_idx
 #
 #    A] An indicator matrix. On which features are visible where
 #        (428,51)   (51,50)   (50,49) ...    (428,427)   (427,426)  ...
 #    f1
 #    f2
 #    .
 #    .
 #    .
 #    f100
 #
 #
 #    B] image co-ordinates of all the features in all the images in question
 #            428   427   426 ...     51   50   49 ...
 #    f1   (72,212)
 #    f2   (91,32)
 #    .
 #    .
 #    .
 #    f100
 #


import numpy as np
import code

class DenseFeatureTracks:
    def __init__(self):
        self.raw_data = {}
        self.verbosity = 1
        self.i_curr = -1 # Meta info
        self.i_prev = -1

    def set_verbosity( self, verbosity ):
        self.verbosity = int(verbosity)

    def reset(self):
        self.raw_data = {}
        self.visibility_table = {}
        self.feature_list = {}
        self.pair_type = {}

    def _xprint( self, header, msg ):
        # return
        if self.verbosity == 0:
            return

        print '[%s] %s' %(header, msg)


    def set( self, idx_1, pts_1, idx_2, pts_2, mask, TYPE ):
        """ idx_1, idx_2 : Scalars indicating global_idx of the image
            pts_1, pts_2 : Nx2 indicating co-ordinates.
            mask : mask from f-test in expand_matches()

            TYPE: int32.
                    -1 : Dense Match
                     1 : i_prev + j  (j is +ve)
                     2 : i_prev - j  (j is +ve)
                     3 : i_curr - j  (j is +ve)
        """

        idx = (idx_1,idx_2)#'%d,%d' %(idx_1,idx_2)
        assert idx not in self.raw_data.keys(), "Seem to be already present. This is an error"
        self.raw_data[idx] = {}
        self.raw_data[idx][ 'mask'] = np.array(mask)

        self.raw_data[idx][ 'pt0'] = np.array(pts_1)
        self.raw_data[idx][ 'pt1'] = np.array(pts_2)
        self.raw_data[idx][ 'TYPE'] = TYPE

    def verify_ind( self, idx, kappa ):
        """
            example idx: 201
            example kappa : [(200,201), (201,202), (201,409)]
        """

        if kappa[0][0] == idx:
            L = len(  self.raw_data[kappa[0]]['pt0']  )
            L_pt = self.raw_data[kappa[0]]['pt0']
        else:
            L = len(  self.raw_data[kappa[0]]['pt1']  )
            L_pt = self.raw_data[kappa[0]]['pt1']

        self._xprint( 'verify_ind', "Len = %d" %(L) )

        # try:
        for _k in kappa:
            if abs( _k[0] - _k[1] ) != 1:
                continue
            if _k[0] == idx :
                self._xprint( 'verify_ind', 'test %s.pt0' %(str(_k) ) )
                assert len(self.raw_data[_k]['pt0']) == L
                assert( np.linalg.norm(self.raw_data[_k]['pt0'] - L_pt) == 0 )
            else:
                self._xprint( 'verify_ind', 'test %s.pt1' %(str(_k) ) )
                assert len(self.raw_data[_k]['pt1']) == L
                assert( np.linalg.norm(self.raw_data[_k]['pt1'] - L_pt) == 0 )
        # except:
            # code.interact( local=locals(), banner="assertion exception")


        self._xprint( 'verify_ind', 'all OK for id=%d in %s' %( idx, str(kappa ) ) )


    def verify_sanctity(self):
        """ example
                raw_data['294,293']['pt1'] - raw_data['293,292']['pt0'] == 0_{Nx2}

                if this is satisfied for all keys of raw_data ==> raw_data is OK

                also #of tracked features has to be same for all the keys (by design)
                so: len(raw_data[:][pt0] == len(raw_data[:][pt1])  and basically all the ptx has to have same length
        """
        self._xprint( 'verify_sanctity', 'start')
        all_keys = self.raw_data.keys()
        self._xprint( 'verify_sanctity', 'all keys: '+str(all_keys) )

        for k in range( len(all_keys) ):
            this_key = all_keys[k]

            # look for this_key[0] in all_keys in 1st position. assert( this_key.pt0.len == found_key.pt0.len )
            kx = [ item for item in all_keys if item[0] == this_key[0] or item[1] == this_key[0]  ]
            #   note: kx now has all the keys which all contains this_key[0]

            self._xprint( 'verify_sanctity', 'found %d in keys: %s' %(this_key[0], str(kx) ) )


            # deal with kx, this_key[0]
            self.verify_ind( this_key[0], kx )

            # look for this_key[1] in all_keys in 2nd position. assert( this_key.pt1.len == found_key.pt1.len )
            ky = [ item for item in all_keys if item[0] == this_key[1] or item[1] == this_key[1]  ]
            #   note: ky now has all the keys which all contains this_key[1]

            self._xprint( 'verify_sanctity', 'found %d in keys: %s' %(this_key[1], str(ky) ) )


            # deal with ky, this_key[1]
            self.verify_ind( this_key[1], ky )



    def optimize_layout( self ):
        """ This verifies that the data in this makes sense. Then converts
        self.raw_data into self.feature_list and self.visibility_table
        """

        self.verify_sanctity()

        ###
        ### Part - A: Create visibility table
        ### mask in each
        #      49   50,   51,   ...   428,   427, ...
        # 49  [11001..]
        # 50
        # 51
        # .
        # .
        # .
        # 428
        # 428,
        # .
        # .
        visibility_table = {}
        pair_type = {}
        set_type_m1 = {}
        set_type_1 = {}
        set_type_2 = {}
        set_type_3 = {}
        for _k in self.raw_data.keys():
            visibility_table[_k] = self.raw_data[_k]['mask']
            pair_type[_k] = self.raw_data[_k]['TYPE']

            if pair_type[_k] == -1 :
                set_type_m1[_k] = True
            if pair_type[_k] ==  1 :
                set_type_1[_k] = True
            if pair_type[_k] ==  2 :
                set_type_2[_k] = True
            if pair_type[_k] ==  3 :
                set_type_3[_k] = True


        ###
        ### Part - B: Create list of co-ordinates of tracked points in each frame
        ###
        features_list = {}
        for _k in self.raw_data.keys():
            if _k[0] not in features_list.keys() :
                features_list[_k[0]] = self.raw_data[_k]['pt0']

            if _k[1] not in features_list.keys() :
                features_list[_k[1]] = self.raw_data[_k]['pt1']


        self.visibility_table = visibility_table
        self.features_list = features_list
        self.pair_type = pair_type

        self.set_type_m1 = set_type_m1
        self.set_type_1 = set_type_1
        self.set_type_2 = set_type_2
        self.set_type_3 = set_type_3
