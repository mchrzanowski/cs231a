import constants
import numpy
import pickle
import random

def main():
    fvs, person_to_indices = pickle.load(open(constants.FV_AND_MAPPING_FILE, 'rb'))
    W = pickle.load(open(constants.W_MATRIX_FILE, 'rb'))

    # tp = really +1, pred +1
    # fp = really -1, pred +1
    # fn = really +1, pred -1
    # tn = really -1, pred -1
    tp = fp = fn = tn = 0
    keys = [person for person in person_to_indices]
    trials = 600
    for i in xrange(trials):
        if i < trials / 2:
            while True:
                person = random.choice(person_to_indices.keys())
                if len(person_to_indices[person]) == 1:
                    continue
                i, j = random.sample(person_to_indices[person], 2)
                y = +1
                break
        else:
            first_person, second_person = random.sample(person_to_indices, 2)
            i = random.choice(person_to_indices[first_person])
            j = random.choice(person_to_indices[second_person])
            y = -1

        dist_i = numpy.dot(W, fvs[:, i])
        dist_j = numpy.dot(W, fvs[:, j])
        dist = numpy.dot(dist_i, dist_j)
        if dist > 1 and y == 1:
            tp += 1
        elif dist > 1 and y == -1:
            fp += 1
        elif dist < 1 and y == 1:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    print 'Total: %s' % total
    print 'Precision: %s' % (tp / (1e-4 + tp + fp))
    print 'Recall: %s' % (tp / (1e-4 + tp + fn))
    print 'TP: %s' % tp
    print 'FP: %s' % fp
    print 'FN: %s' % fn
    print 'TN: %s' % tn

if __name__ == "__main__":
    main()