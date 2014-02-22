# this pulls in all the phis and performs PCA on it
import constants
import cPickle
import os
import numpy
import re
import utilities

from sklearn.decomposition import PCA

def create_FV_matrix(dataset, get_indices=False):

    images = dataset.get_train_images()

    fvs = numpy.zeros((67584, len(images)))
    fvs = fvs.astype(numpy.float32, copy=False)

    if get_indices: images_to_index = dict()
    for i, name in enumerate(images):
        input_file = dataset.get_fv_file_for_image(name)
        fv = utilities.hydrate_fv_from_file(input_file)
        fvs[:, i] = fv
        if get_indices: images_to_index[name] = i

    if get_indices: return fvs, images_to_index

    return fvs

def init_W_with_indices(dataset, verbose=False):
    if verbose: print 'Initializing W. Returning image mapping in FV matrix...'
    fvs, images_to_index = create_FV_matrix(dataset, get_indices=True)
    W = __perform_PCA(fvs, verbose)
    return W, fvs, images_to_index

def init_W(dataset, verbose=False):
    if verbose: print 'Initializing W.'
    fvs = create_FV_matrix(dataset, get_indices=False)
    if verbose: print 'Finished FV Matrix Construction'
    W = __perform_PCA(fvs, verbose)
    return W

def __perform_PCA(fvs, verbose=False):
    if verbose: print 'Start on PCA...'
    pca = PCA(n_components=128, whiten=True)
    W = pca.fit_transform(fvs).T
    if verbose: print 'PCA complete.'
    return W

if __name__ == "__main__":
    init_W()
