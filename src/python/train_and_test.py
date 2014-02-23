import constants
import numpy
import optimization
import os
import pickle
import utilities
import init_W

import dataset_generation

def run(deep_learning=False, debug=False, verbose=False):
    if debug:
        dataset = dataset_generation.DevDataset()
    else:
        dataset = dataset_generation.UnrestrictedDataset(split=1)

    W, b, fvs, images_to_indices, da = train(dataset, deep_learning, debug, verbose)
    test(dataset, W, b, 'Train', da=da, fvs=fvs, images_to_indices=images_to_indices, verbose=verbose)
    test(dataset, W, b, 'Test', da=da, verbose=verbose)

def test(dataset, W, b, type, da=None, fvs=None, images_to_indices=None, verbose=False):

    if verbose: print 'Getting %s Error...' % type
    # tp = really +1, pred +1
    # fp = really -1, pred +1
    # fn = really +1, pred -1
    # tn = really -1, pred -1
    tp = fp = fn = tn = 0.

    if type == 'Train':
        same_data = dataset.gen_same_person_train_samples
        diff_data = dataset.gen_diff_person_train_samples
    elif type == 'Test':
        same_data = dataset.gen_same_person_test_samples
        diff_data = dataset.gen_diff_person_test_samples

    def get_distance(W, b, y, fv1, fv2):
        result = numpy.dot(W, fv1 - fv2)
        dist = numpy.linalg.norm(result, 2) ** 2
        return y * (b - dist)

    labels_and_data = ((+1, same_data), (-1, diff_data))

    for (label, data) in labels_and_data:
        for (img1, img2) in data():

            if images_to_indices is not None and img1 in images_to_indices:
                fv1 = fvs[:, images_to_indices[img1]]
            else:
                img1 = dataset.get_fv_file_for_image(img1)
                fv1 = utilities.hydrate_fv_from_file(img1)
                if da is not None:
                    fv1 = da.get_hidden_values(fv1).eval()

            if images_to_indices is not None and img2 in images_to_indices:
                fv2 = fvs[:, images_to_indices[img2]]
            else:
                img2 = dataset.get_fv_file_for_image(img2)
                fv2 = utilities.hydrate_fv_from_file(img2)
                if da is not None:
                    fv2 = da.get_hidden_values(fv2).eval()

            dist = get_distance(W, b, label, fv1, fv2)
            if dist > 1 and label == +1:
                tp += 1
            elif dist < 1 and label == +1:
                fp += 1
            elif dist > 1 and label == -1:
                tn += 1
            else:
                fn += 1

    total = tp + fp + fn + tn
    print 'Total: %s' % total
    print 'Precision: %s' % (tp / (tp + fp))
    print 'Recall: %s' % (tp / (tp + fn))
    print 'TP: %s' % tp
    print 'FP: %s' % fp
    print 'FN: %s' % fn
    print 'TN: %s' % tn

def train(dataset, deep_learning=False, debug=False, verbose=True):

    if verbose:
        print 'Fisher Vector Directory: %s' % constants.FV_DIR
        print 'Deep-Learning Mode: %s' % deep_learning
        print 'Debug Mode: %s' % debug
        dataset.print_dataset_stats()

    if deep_learning:
        W, da, fvs, images_to_indices = init_W.init_deepnet(dataset, debug, verbose)
    else:
        W, fvs, images_to_indices = init_W.init(dataset, debug, verbose)
        da = None
    if debug:
        W, b = optimization.ssgd(W, dataset, fvs=fvs, iterations=0, cache=False,
            image_to_index=images_to_indices, verbose=verbose)
    else:
        W, b = optimization.ssgd(W, dataset, fvs=fvs,
            image_to_index=images_to_indices, verbose=verbose)
    
    return W, b, fvs, images_to_indices, da

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help="Debug mode.")
    parser.add_argument('-dl', action='store_true', help="Deep Learning mode.")
    parser.add_argument('-v', action='store_true', help="Verbose mode.")
    args = vars(parser.parse_args())
    run(args['dl'], args['d'], args['v'])
