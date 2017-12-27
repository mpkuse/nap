""" Computes netvlad vectors for every image of pittsburg street view.
    I am doing this so that I can learn an voronoi on this distribution
    for fast nn-search using faiss

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 7th Dec, 2017
"""
import rospkg

PKG_PATH = rospkg.RosPack().get_path('nap')

PITS_STREETVIEW = '/media/mpkuse/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_b20_resnet/model-3750' # trained similar to above but with a resnet neural net

PARAM_MODEL = PKG_PATH+'/tf2.logs/attempt_resnet6_K16_P8_N8/model-2500'

import numpy as np
import cv2
from PlaceRecognitionNetvlad import PlaceRecognitionNetvlad
import glob
import code
import time
import pickle

import TerminalColors
tcol = TerminalColors.bcolors()



place_mod = PlaceRecognitionNetvlad(\
                                    PARAM_MODEL,\
                                    PARAM_CALLBACK_SKIP=2,\
                                    PARAM_K = 16
                                    )

list_of_images = glob.glob( PITS_STREETVIEW+'*/*.jpg')
list_of_netvlads = []

for i,file_name in enumerate(list_of_images[0:10]):
    s = time.time()
    im = cv2.imread( file_name )
    im = cv2.cvtColor( im, cv2.COLOR_BGR2RGB )

    X = place_mod.extract_descriptor( im )
    list_of_netvlads.append( X )

    print '%d in %4.2fms : %s' %( i, 1000.0*(time.time()-s), file_name )



# store `list_of_images` and `list_of_netvlads`
list_of_netvlads = np.array( list_of_netvlads )
quit()

print 'Writing file: ', PITS_STREETVIEW+'/list_of_images.pickle'
with open( PITS_STREETVIEW+'/list_of_images.pickle', 'wb' ) as fp:
    pickle.dump( list_of_images, fp )

print 'Writing file: ', PITS_STREETVIEW+'/list_of_netvlads.pickle'
with open( PITS_STREETVIEW+'/list_of_netvlads.pickle', 'wb' ) as fp:
    pickle.dump( list_of_netvlads, fp )


# Read it back
# list_of_images = pickle.load( open( PITS_STREETVIEW+'/list_of_images.pickle', 'rb' ) )
# list_of_netvlads = pickle.load( open( PITS_STREETVIEW+'/list_of_netvlads.pickle', 'rb' ) )
