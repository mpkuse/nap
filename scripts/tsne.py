""" Tries with TSNE on higher dimensional data """

import numpy as np
from sklearn.manifold import TSNE
import cv2

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()


    x, y = np.atleast_1d(x, y)
    artists = []
    i = 0
    for x0, y0 in zip(x, y):
        im = OffsetImage( cv2.cvtColor( image[i,:,:,:], cv2.COLOR_RGB2BGR ), zoom=zoom)
        i = i+1
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

#### LOAD Data ####
S_char = np.load( PKG_PATH+'/DUMP/'+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH+'/DUMP/'+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH+'/DUMP/'+'S_thumbnail.npy' )#N x 96 x 128 x 3


#### TSNE ####
model = TSNE( n_components=2, random_state=0, perplexity=20, metric='cosine', verbose=1 )
out = model.fit_transform( S_char )


#### Visualize #####
fig, ax = plt.subplots()
imscatter(out[:,0], out[:,1], S_thumbs, zoom=0.5, ax=ax)
ax.plot(out[:,0], out[:,1], 'r.')
plt.show()
