import constants
import cPickle
import os
import random
import utilities

from numpy.linalg import norm
from numpy import dot, outer

def subgradient_optimization(W, dataset, fvs=None, image_to_index=None, w_eta=0.5, b_eta=10, iterations=1000000, cache=True, verbose=False):
    b = 0
    if verbose: print 'Begin Subgradient Gradient Descent Learning...'
    for i in xrange(iterations):
        if verbose and i % 1000 == 0: print 'Iteration: %s' % i
        if random.random() > 0.5:
            sample = dataset.get_same_person_train_sample()
            y = +1
        else:
            sample = dataset.get_diff_person_train_sample()
            y = -1

        img1, img2 = sample

        if fvs is None or image_to_index is None:
            fv1 = utilities.hydrate_fv_from_file(os.path.join(constants.FV_DIR, img1))
            fv2 = utilities.hydrate_fv_from_file(os.path.join(constants.FV_DIR, img2))
        else:
            fv1 = fvs[:, image_to_index[img1]]
            fv2 = fvs[:, image_to_index[img2]]

        fv_diff = fv1 - fv2
        first_op = dot(W, fv_diff)
        dist = norm(first_op, 2) ** 2

        # update W & b
        if y * (b - dist) < 1:
            W -= w_eta * y * outer(first_op, fv_diff)
            b += b_eta * y

    if verbose: print 'Optimization Complete!'
    if cache: cPickle.dump(W, open(constants.W_MATRIX_FILE, 'wb'))
    if verbose: print 'Learned b: %s' % b
    if cache: cPickle.dump(b, open(constants.B_FILE, 'wb'))

    return W, b
