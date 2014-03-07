import constants
import cPickle
import os
import random
import utilities

from numpy import dot, outer

def ssgd(W, dataset, fvs, image_to_index,
    w_eta=0.5, b_eta=10, iterations=1000000, cache=False, verbose=False, b=None):
    
    if b is None:
        update_b = True
        b = 0
    else:
        update_b = False

    if verbose: print 'Begin Stochastic Subgradient Descent Learning...'
    for i in xrange(1, iterations + 1):
        if verbose and i % 500000 == 0: print 'Iteration: %s' % i
        if random.random() > 0.5:
            sample = dataset.get_same_person_train_sample()
            y = +1
        else:
            sample = dataset.get_diff_person_train_sample()
            y = -1

        img1, img2 = sample        
        fv1 = fvs[:, image_to_index[img1]]
        fv2 = fvs[:, image_to_index[img2]]

        fv_diff = fv1 - fv2
        compressed_fv = dot(W, fv_diff)
        dist = dot(compressed_fv.T, compressed_fv)

        # update W & b
        if y * (b - dist) <= 1:
            W -= w_eta * y * outer(compressed_fv, fv_diff)
            if update_b:
                b += b_eta * y

    if verbose: print 'Optimization Complete!'
    if verbose: print 'Learned b: %s' % b
    if cache: cPickle.dump((b, W), open(dataset.param_file, 'wb'))

    return W, b
