""" Idea from Pedro's famous graph segmentation to be applied on my problem
    Think of all the scenes as island. The task being similar to image segmentation

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 16th May, 2017
"""
import numpy as np
import cv2
import networkx as nx
import code
import time
import json
import pickle
#
import TerminalColors
tcol = TerminalColors.bcolors()

from FastPlotter import FastPlotter

import matplotlib.pyplot as plt

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

## 'x' can also be a vector
def logistic( x ):
    #y = np.array(x)
    #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
    # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)
    if len(x) < 3:
        return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

    y = np.convolve( np.array(x), [0.25,0.5,0.25], 'same' )
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)

class Node:
    def __init__(self, uid, parent=None):
        self.uid = uid
        self.parent = parent

    def __repr__(self):
        return '(u=%-4d, g=%-4d)' %(self.uid,get_gid(self))



def get_gid( node, verbose=False ):
    while node.parent is not None:
        node = node.parent
    return node.uid


def get_gid_path( node, verbose=False ):
    path = []
    if verbose:
        print 'Path from (%d) : ' %(node.uid),

    while node.parent is not None:
        if verbose:
            print '(%-3d)--' %(node.uid),

        path.append(node.uid)
        node = node.parent
    if verbose:
        print ''
    path.append(node.uid)
    return node.uid, path



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

def gps_layout2d( G, gps_t, gps_x, gps_y ):
    pos = {}
    for node_id in G.node:
        # print node_id, ': ', G.node[node_id]['time_stamp']
        time = long(G.node[node_id]['time_stamp'])
        m = abs(gps_t - time).argmin()
        # print gps_x[m], gps_y[m]
        pos[node_id] = [gps_x[m], gps_y[m]]
        #code.interact( local=locals() )
    return pos

def save_json( out_file_name, data ):
    print 'Writing ', out_file_name
    with open( out_file_name, 'w') as f:
        json.dump( data, f )

def load_json( file_name ):
    print 'Loading ', file_name
    with open( file_name ) as f:
        my_dict =  json.load(f)
        return  {int(k):float(v) for k,v in my_dict.items() }

#----- Load Data -----#
folder = '/DUMP/amsterdam_walk/'
S_char = np.load( PKG_PATH+  folder+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH+  folder+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH+folder+'S_thumbnail.npy' )#N x 96 x 128 x 3
S_timestamp = np.load( PKG_PATH + folder+'S_timestamp.npy' )
#-------- END --------#


if False:
    plotter = FastPlotter(1,200,200)
    plotter.setRange( 0, yRange=[0,1] )
    all_nodes = []
    internal_e = {} #associate array
    n_components = {}
    for i in range(S_word.shape[0]):
        #assume at ith stage, all previous S are available only.
        if i==0: #no edges from 0th node to previous, ie. no previous nodes
            all_nodes.append( Node(uid=0))
            continue;

        startTime = time.time()

        #now there is atleast 1 prev nodes

        #compute dot product cost
        window_size = 50
        DOT_word = np.dot( S_word[max(0,i-window_size):i,:], S_word[i,:] )
        DOT_index = range(max(0,i-window_size),i)
        sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) )
        wt = 1.0 - logistic( sim_scores ) #measure of dis-similarity. 0 means very similar.

        all_nodes.append( Node(uid=i) )
        for j_ind,w in enumerate(wt):
            if w>0.3:
                continue

            gid_i = get_gid( all_nodes[i] )
            e_i = 0

            j = DOT_index[j_ind]
            gid_j = get_gid( all_nodes[j] )
            e_j = internal_e[gid_j] if internal_e.has_key(gid_j) else 0.0

            n_i = n_components[gid_i] if n_components.has_key(gid_i) else 1
            n_j = n_components[gid_j] if n_components.has_key(gid_j) else 1

            kappa = 0.22
            # print 'gid_i=%3d gid_j=%3d' %(gid_i, gid_j)
            # print 'w=%4.4f, ei=%4.4f, ej=%4.4f' %(w, e_i+kappa/n_i, e_j+kappa/n_j )
            if w < min(e_i+kappa/n_i, e_j+kappa/n_j):
                internal_e[gid_j] = w
                n_components[gid_j] = n_j + 1
                all_nodes[i].parent = all_nodes[j]




        #
        _past_key_frames = sorted(internal_e)
        print _past_key_frames
        _past_mid_frames = []
        for n_id,n in enumerate( _past_key_frames[0:-1] ):
            # print n_id, n
            _past_mid_frames.append( int((n+_past_key_frames[n_id+1])/2.) )
        print _past_mid_frames
        _past_DOT_word = np.dot( S_word[_past_mid_frames,:] , S_word[i,:] )
        _past_sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, _past_DOT_word ) )
        _past_wt = 1.0 - logistic( _past_sim_scores ) #measure of dis-similarity. 0 means very similar.
        print _past_wt
        if len(_past_wt) > 0 :
            plotter.set_data( 0, _past_mid_frames, _past_wt )
            plotter.spin()




        print i, 'of', S_word.shape[0], tcol.OKBLUE, 'Done in (ms) : ', np.round( (time.time() - startTime )*1000.,2 ), tcol.ENDC
        # code.interact( banner='---End of %d---' %(i), local=locals() )

        # print_stats( all_nodes, internal_e, n_components )
        thumb = S_thumbs[i,:,:,:]
        cv2.putText( thumb, str(get_gid(all_nodes[i])), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )
        cv2.imshow( 'win', thumb )
        cv2.waitKey(10)

    save_json(PKG_PATH+folder+'/internal_e.json', internal_e )
    save_json(PKG_PATH+folder+'/n_components.json', n_components )
    with open( PKG_PATH+folder+'all_nodes.pickle', 'w' ) as f:
        pickle.dump( all_nodes, f )

else:
    internal_e = load_json( PKG_PATH+folder+'/internal_e.json' )
    n_components = load_json( PKG_PATH+folder+'/n_components.json' )
    with open( PKG_PATH+folder+'all_nodes.pickle', 'r' ) as f:
        all_nodes = pickle.load( f )

#--------------- Analysis of Key Frames ---------------------#
# Vars :
#   internal_e : associate array of internal energies of each components
#   n_components : number of elements in each components



key_frames =  sorted(internal_e)

code.interact( local=locals() )
# Draw intra graph (path to gids)
for k in range(len(key_frames)-1):
    intragraphStartTime = time.time()
    H = nx.Graph()
    for i in range( key_frames[k], key_frames[k+1] ):
        gid, path = get_gid_path( all_nodes[i] )
        H.add_path( path )
    pagerank = nx.pagerank(H)
    print 'Intragraph constructed in %4.2f ms' %(1000.*(time.time()-intragraphStartTime))

    print 'Number of nodes : ', H.number_of_nodes()
    if H.number_of_nodes() > 30:
        pagerank = nx.pagerank(H)
        print sorted( pagerank, key=pagerank.get )[-10:]

        nx.draw( H, pos=nx.spring_layout(H), with_labels=True)
        plt.show()
        code.interact( local=locals() )






quit()


#f, axarr = plt.subplots(3)
for k in range(len(key_frames)-1):
    c_with = int(np.floor( 0.5*(key_frames[k]+key_frames[k+1]) ))
    p0 = np.dot( S_word[ key_frames[k]:key_frames[k+1] ], np.transpose( S_word[ key_frames[k] ] ) )
    pn = np.dot( S_word[ key_frames[k]:key_frames[k+1] ], np.transpose( S_word[ c_with ] ) )
    pN = np.dot( S_word[ key_frames[k]:key_frames[k+1] ], np.transpose( S_word[ key_frames[k+1]-1 ] ) )

    plt.plot( range(key_frames[k],key_frames[k+1]), p0 )
    plt.plot( range(key_frames[k],key_frames[k+1]), pn )
    plt.plot( range(key_frames[k],key_frames[k+1]), pN )
    plt.show()



quit()
G = nx.Graph()
for k in key_frames:
    G.add_node( k, n_compo=n_components[k], int_e=internal_e[k], time_stamp=str(S_timestamp[k]) )


C  = np.dot( S_word[key_frames,:], np.transpose(S_word[key_frames,:]) )


gps_t, gps_x, gps_y, gps_z = np.loadtxt( PKG_PATH+'/DUMP/GPS_track.csv', dtype={'names':('t','x','y','z'), 'formats':('i8', 'f4', 'f4', 'f4') }, delimiter=',', unpack=True)
pos1 = gps_layout2d( G, gps_t, gps_x, gps_y )
nx.draw_networkx( G, pos1, font_size=10, width=0.5 )
plt.show()
