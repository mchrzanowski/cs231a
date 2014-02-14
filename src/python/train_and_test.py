import constants
import numpy
import optimization
import pickle
import utilities

from init_W import init_W

def run():
    W = train()
    test(W)

def test(W):

    # tp = really +1, pred +1
    # fp = really -1, pred +1
    # fn = really +1, pred -1
    # tn = really -1, pred -1
    tp = fp = fn = tn = 0

    def hydrate_and_get_distance(W, img, img2):
        fv1 = fv = utilities.hydrate_fv_from_file(img1)
        fv2 = utilities.hydrate_fv_from_file(img2)
        dist1 = numpy.dot(W, fv1)
        dist2 = numpy.dot(W, fv2)
        return numpy.dot(dist1, dist2)

    _, same_pairs, diff_pairs = process_dev_file(constants.DEV_TEST_PAIR_FILE)
    print 'Testing: Same Pairs: %s' % len(same_pairs)
    print 'Testing: Diff Pairs: %s' % len(diff_pairs)

    for (img1, img2) in same_pairs:
        dist = hydrate_and_get_distance(W, img1, img2)
        if dist > 1:
            tp += 1
        else:
            fn += 1

    for (img1, img2) in diff_pairs:
        dist = hydrate_and_get_distance(W, img1, img2)
        if dist > 1:
            fp += 1
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

def train():

    images_required, same_pairs, diff_pairs = process_dev_file(constants.DEV_TRAIN_PAIR_FILE)
    print 'Training: Images Required: %s' % len(images_required)
    print 'Training: Same Pairs: %s' % len(same_pairs)
    print 'Testing: Diff Pairs: %s' % len(diff_pairs)

    init_W(images_to_use=images_required, verbose=True)
    fvs, image_to_index = pickle.load(open(constants.FV_AND_MAPPING_FILE, 'rb'))
    W = pickle.load(open(constants.W_MATRIX_FILE, 'rb'))
    W = optimization.subgradient_optimization(W, fvs, same_pairs, diff_pairs, image_to_index)

    return W

def process_dev_file(filename):
    # read training file in
    images_required = set()
    same_person_pairs = set()
    diff_person_pairs = set()
    with open(filename, 'rb') as f:
        f.readline()    # skip header
        for line in f:
            data = line.strip().split('\t')
            if len(data) == 3:      # same person
                person, img1, img2 = data
                img1 = convert_to_filename(person, img1)
                img2 = convert_to_filename(person, img2)
                same_person_pairs.add((img1, img2))
            else:                   # different people
                person1, img1, person2, img2 = data
                img1 = convert_to_filename(person1, img1)
                img2 = convert_to_filename(person2, img2)
                diff_person_pairs.add((img1, img2))
            images_required.add(img1)
            images_required.add(img2)

    return images_required, same_person_pairs, diff_person_pairs

def convert_to_filename(person, image_num):
    image_num = str(image_num)
    image_num = '0' * (4 - len(image_num)) + image_num  # pre-append zeros
    return person + '_' + image_num + '.jpg'

if __name__ == "__main__":
    run()