# this pulls in all the phis and performs PCA on it
import constants
import cPickle
import os
import numpy
import re
import utilities

from sklearn.decomposition import PCA

def create_FV_matrix(images_to_use=None, get_indices=False):
    input_dir = constants.FV_DIR
    if images_to_use is None:
        fv_files = [name for name in os.listdir(input_dir)]
        fvs = numpy.zeros((67584, len(fv_files))) 
    else:
        fv_files = images_to_use
        fvs = numpy.zeros((67584, len(images_to_use)))

    if get_indices: images_to_index = dict()
    for i, name in enumerate(fv_files):
        input_file = os.path.join(input_dir,name)
        fv = utilities.hydrate_fv_from_file(input_file)
        fvs[:, i] = fv
        if get_indices: images_to_index[name] = i

    if get_indices: return fvs, images_to_index

    return fvs

def init_W_with_indices(images_to_use=None, verbose=False):
    if verbose: print 'Initializing W. Returning image mapping in FV matrix...'
    fvs, images_to_index = create_FV_matrix(images_to_use, get_indices=True)
    W = __perform_PCA(fvs, verbose)
    return W, fvs, images_to_index

def init_W(images_to_use=None, verbose=False):
    if verbose: print 'Initializing W.'
    fvs = create_FV_matrix(images_to_use, get_indices=False)
    if verbose: print 'Finished FV Matrix Construction'
    W = __perform_PCA(fvs, verbose)
    print 'Pickling W...'
    cPickle.dump(W, open(constants.W_MATRIX_FILE, 'wb'))
    return W

def __perform_PCA(fvs, verbose=False):
    if verbose: print 'Start on PCA...'
    pca = PCA(n_components=128, whiten=True)
    W = pca.fit_transform(fvs).T
    if verbose: print 'PCA complete.'
    return W

if __name__ == "__main__":
    init_W()
