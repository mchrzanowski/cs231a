import constants
import numpy
import os
import shutil
import distutils.dir_util

def get_distance(W, b, fv1, fv2):
    result = numpy.dot(W, fv1 - fv2)
    dist = numpy.dot(result.T, result)
    return b - dist

def get_dataset(use_deep_funneled):
    if use_deep_funneled:
        return constants.FV_DF_DIR
    return constants.FV_DIR

def convert_to_param_file(deep_learning, deep_funneled):
    if deep_funneled and deep_learning:
        return constants.W_B_DF_DL_FILE
    if deep_funneled and not deep_learning:
        return constants.W_B_DF_FILE
    if not deep_funneled and deep_learning:
        return constants.W_B_DL_FILE
    return constants.W_B_FILE

def hydrate_fv_from_file(input_file):
    return numpy.genfromtxt(input_file, dtype=numpy.float32, delimiter=',')

def parse_lfw_data_into_train_and_test(src_dir, train_dir, test_dir, split, people_file=constants.PEOPLE_FILE):
    current_split = 0
    with open(people_file, 'rb') as f:
        for i, line in enumerate(f):
            data = line.strip().split('\t')
            if len(data) == 1:                  # new split 
                current_split += 1
            else:  
                person, num = data
                copy_from = os.path.join(src_dir, person)
                #print copy_from
                if current_split == split:      # test set.
                    dst_dir = test_dir
                else:
                    dst_dir = train_dir
                #print dst_dir
                os.system('cp -rf %s %s' % (copy_from, dst_dir))
