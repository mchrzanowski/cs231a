import random
from numpy.linalg import norm
from numpy import dot, outer

def subgradient_optimization(W, fvs, person_to_indices, training_same, training_diff, eta=0.01, iterations=1000000, verbose=False):
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

        i = person_to_indices[sample[0]]
        j = person_to_indices[sample[1]]

        fv_diff = fvs[:, i] - fvs[:, j]
        first_op = dot(W, fv_diff)
        dist = norm(first_op, 2) ** 2

        # update W & b
        if y * (b - dist) < 1:
            W -= eta * y * outer(first_op, fv_diff)
            b += eta * y

    if verbose: print 'Optimization Complete!'
    return W
