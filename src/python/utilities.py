import constants
import numpy
import os
import shutil
import distutils.dir_util

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
