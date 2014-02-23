import constants
import numpy
import optimization
import os
import pickle
import utilities
import init_W

from dataset_generation import RealDataset

def run(in_mem, verbose):
    dataset = RealDataset(split=1)
    W, b = train(dataset, in_mem)
    test(dataset, W, b, 'Train', verbose)
    test(dataset, W, b, 'Test', verbose)

def test(dataset, W, b, type, verbose=False):

    if verbose: print 'Getting %s Error' % type
    # tp = really +1, pred +1
    # fp = really -1, pred +1
    # fn = really +1, pred -1
    # tn = really -1, pred -1
    tp = fp = fn = tn = 0

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

            img1, img2 = map(dataset.get_fv_file_for_image, (img1, img2))
            fv1, fv2 = map(utilities.hydrate_fv_from_file, (img1, img2))
            
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

def train(dataset, in_mem=True, verbose=True):

    if verbose: print 'FV DIR: %s' % constants.FV_DIR

    dataset.print_dataset_stats()
    if in_mem:
        W, fvs, images_to_indices = init_W.init_W_with_indices(dataset, verbose=True)
        W, b = optimization.subgradient_optimization(W, dataset, fvs=fvs,
            image_to_index=images_to_indices, verbose=True)
    else:
        W = init_W.init_W(dataset, verbose=True)
        W, b = optimization.subgradient_optimization(W, dataset, verbose=True)
    
    return W, b

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-nm', action='store_false', help="Be memory-efficient: don't do everything in memory")
    parser.add_argument('-v', action='store_true', help="Verbose mode.")
    args = vars(parser.parse_args())
    run(args['nm'], args['v'])
