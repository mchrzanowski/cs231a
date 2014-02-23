# this pulls in all the phis and performs PCA on it
import constants
import cPickle
import dA
import os
import numpy
import random
import utilities

from sklearn.decomposition import PCA

def create_FV_matrix(dataset, debug=False):

    images = dataset.get_train_images()
    if debug: images = random.sample(images, 200)

    fvs = numpy.zeros((constants.FV_DIM, len(images)))
    fvs = fvs.astype(numpy.float32, copy=False)

    images_to_index = dict()
    for i, name in enumerate(images):
        input_file = dataset.get_fv_file_for_image(name)
        fv = utilities.hydrate_fv_from_file(input_file)
        fvs[:, i] = fv
        images_to_index[name] = i

    return fvs, images_to_index

def init(dataset, debug=False, verbose=False):
    if verbose: print 'Initializing W. Heuristic: PCA'
    fvs, images_to_index = create_FV_matrix(dataset, debug)
    W = __perform_PCA(fvs, verbose)
    return W, fvs, images_to_index

def init_deepnet(dataset, debug=False, verbose=False):
    if verbose: print 'Initializing W. Heuristic: dA -> PCA'
    fvs, images_to_index = create_FV_matrix(dataset, debug)
    da, fvs = __deep_learning(fvs, verbose)
    W = __perform_PCA(fvs, verbose)
    return W, da, fvs, images_to_index

def __deep_learning(fvs, verbose):
    if verbose: print 'Start training denoising autoencoder...'
    da = dA.train(fvs, hidden_units=fvs.shape[0] // 20,
        learning_rate=0.01, training_epochs=15,
        batch_size=20, corruption_level=0.3, verbose=verbose)
    dfvs = numpy.zeros((da.n_hidden, fvs.shape[1]))
    for i in xrange(fvs.shape[1]):
        val = da.get_hidden_values(fvs[:, i])
        dfvs[:, i] = val.eval()
    if verbose:
        print 'Pre-DA Dim: ', fvs.shape
        print 'Post-DA Dim: ', dfvs.shape
    return da, dfvs

def __perform_PCA(fvs, verbose=False):
    if verbose: print 'Start on PCA...'
    pca = PCA(n_components=128, whiten=True)
    W = pca.fit_transform(fvs).T
    if verbose: print 'PCA complete.'
    return W
