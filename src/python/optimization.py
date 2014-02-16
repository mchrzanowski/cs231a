import constants
import os
import random
import utilities

from numpy.linalg import norm
from numpy import dot, outer

def subgradient_optimization(W, training_same, training_diff, fvs=None, image_to_index=None, eta=0.01, iterations=1000000, verbose=False):
    b = 0
    if verbose: print 'Begin Subgradient Gradient Descent Learning...'
    for i in xrange(iterations):
        if verbose and i % 1000 == 0: print 'Iteration: %s' % i
        if random.random() > 0.5:
            sample = random.choice(training_same)
            y = +1
        else:
            sample = random.choice(training_diff)
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
            W -= eta * y * outer(first_op, fv_diff)
            b += eta * y

    if verbose: print 'Optimization Complete!'
    return W
