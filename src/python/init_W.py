# this pulls in all the phis and performs PCA on it
import constants
import cPickle
import os
import numpy
import re
import utilities

from sklearn.decomposition import RandomizedPCA

def init_W(images_to_use=None, verbose=False):

    input_dir = constants.FV_DIR
    if images_to_use is None:
        fv_files = [name for name in os.listdir(input_dir)]
        fvs = numpy.zeros((67584, len(fv_files))) 
    else:
        fv_files = images_to_use
        fvs = numpy.zeros((67584, len(images_to_use)))

    image_to_index = dict()
    for i, name in enumerate(fv_files):
        input_file = os.path.join(input_dir,name)
        fv = utilities.hydrate_fv_from_file(input_file)
        fvs[:, i] = fv
        image_to_index[name] = i

    if verbose: print('Finished FV Matrix Construction')

    cPickle.dump(fvs, open(constants.FV_FILE, 'wb'))
    cPickle.dump(image_to_index, open(constants.IMAGE_TO_INDEX_FILE, 'wb'))
    del image_to_index

    if verbose: print('Finished pickling FV matrix and image mapping.')
    if verbose: print('Start on PCA...')
    
    pca = RandomizedPCA(n_components=128, whiten=True)
    W = pca.fit_transform(fvs).T

    if verbose: print('PCA complete.')
    cPickle.dump(W, open(constants.W_MATRIX_FILE, 'wb'))

if __name__ == "__main__":
    init_W()
