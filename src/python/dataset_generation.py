import constants
import os
import random

class Dataset(object):
    def convert_to_filename(self, person, image_num):
        image_num = str(image_num)
        image_num = '0' * (4 - len(image_num)) + image_num  # pre-append zeros
        return person + '_' + image_num + '.csv'

    def get_fv_file_for_image(self, image):
        return os.path.join(constants.FV_DIR, image)

class DevDataset(Dataset):
    def __init__(self, train_filename=constants.DEV_TRAIN_PAIR_FILE,
    test_filename=constants.DEV_TEST_PAIR_FILE):
        self.train_images, \
        self.train_same_person_pairs, \
        self.train_diff_person_pairs = self.init(train_filename)

        self.test_images, \
        self.test_same_person_pairs, \
        self.test_diff_person_pairs = self.init(test_filename)

    def get_train_images(self):
        return self.train_images

    def get_same_person_train_sample(self):
        return random.choice(self.train_same_person_pairs)

    def gen_same_person_test_samples(self):
        for pair in self.test_same_person_pairs:
            yield pair

    def gen_same_person_train_samples(self):
        for pair in self.train_same_person_pairs:
            yield pair

    def get_diff_person_train_sample(self):
        return random.choice(self.train_diff_person_pairs)

    def gen_diff_person_test_samples(self):
        for pair in self.test_diff_person_pairs:
            yield pair

    def gen_diff_person_train_samples(self):
        for pair in self.train_diff_person_pairs:
            yield pair

    def print_dataset_stats(self):
        print 'Training: Images Required: %s' % len(self.train_images)
        print 'Training: Same Pairs: %s' % len(self.train_same_person_pairs)
        print 'Training: Diff Pairs: %s' % len(self.train_diff_person_pairs)

        print 'Testing: Images Required: %s' % len(self.test_images)
        print 'Testing: Same Pairs: %s' % len(self.test_same_person_pairs)
        print 'Testing: Diff Pairs: %s' % len(self.test_diff_person_pairs)

    def init(self, filename):
        # read training file in
        images_required = set()
        same_person_pairs = list()
        diff_person_pairs = list()
        with open(filename, 'rb') as f:
            f.readline()    # skip header
            for i, line in enumerate(f):
                data = line.strip().split('\t')
                if len(data) == 3:      # same person
                    person, img1, img2 = data
                    img1 = self.convert_to_filename(person, img1)
                    img2 = self.convert_to_filename(person, img2)
                    same_person_pairs.append((img1, img2))
                else:                   # different people
                    person1, img1, person2, img2 = data
                    img1 = self.convert_to_filename(person1, img1)
                    img2 = self.convert_to_filename(person2, img2)
                    diff_person_pairs.append((img1, img2))
                images_required.add(img1)
                images_required.add(img2)

        return images_required, same_person_pairs, diff_person_pairs

class RealDataset(DevDataset):
    def __init__(self, filename=constants.PAIR_FILE, split=random.randint(1, 10)):
        self.split = split
        self.train_images, \
        self.train_same_person_pairs, \
        self.train_diff_person_pairs, \
        self.test_images, \
        self.test_same_person_pairs, \
        self.test_diff_person_pairs = self.init(filename, split)

    def print_dataset_stats(self):
        print 'Split: %s' % self.split
        DevDataset.print_dataset_stats(self)

    def init(self, filename, split):
        '''
        assume splits are 600 a piece, with 300 same pairs first
        and 300 same pairs afterwards
        '''
        assert split >= 1 and split <= 10
        train_images = set()
        test_images = set()
        train_same_pairs = list()
        train_diff_pairs = list()
        test_same_pairs = list()
        test_diff_pairs = list()
        with open(filename, 'rb') as f:
            pairs_per_split = 600
            start_test_index = (split - 1) * pairs_per_split
            end_test_index = split * pairs_per_split
            for i, line in enumerate(f):
                
                if i >= start_test_index and i < end_test_index:
                    images = test_images
                    same_pairs = test_same_pairs
                    diff_pairs = test_diff_pairs
                else:
                    images = train_images
                    same_pairs = train_same_pairs
                    diff_pairs = train_diff_pairs

                data = line.strip().split('\t')
                if len(data) == 3:      # same person
                    person, img1, img2 = data
                    img1 = self.convert_to_filename(person, img1)
                    img2 = self.convert_to_filename(person, img2)
                    same_pairs.append((img1, img2))
                else:                   # different people
                    person1, img1, person2, img2 = data
                    img1 = self.convert_to_filename(person1, img1)
                    img2 = self.convert_to_filename(person2, img2)
                    diff_pairs.append((img1, img2))
                images.add(img1)
                images.add(img2)
        
        return train_images, train_same_pairs, train_diff_pairs, \
            test_images, test_same_pairs, test_diff_pairs

'''
class RandomDataset(Dataset):
    def __init__(self, train_to_test_ratio=0.8):
        self.train_person_to_images, \
        self.test_person_to_images = self.init(train_to_test_ratio)

    def init(self, train_to_test_ratio):
        train_person_to_images = dict()
        test_person_to_images = dict()
        for person_dir in os.listdir(constants.LFW_DIR):
            if random.random() < train_to_test_ratio:
                mapping = train_person_to_images
            else:
                mapping = test_person_to_images

            mapping[person_dir] = list()
            for img_file in os.listdir(os.path.join(constants.LFW_DIR, person_dir)):
                mapping[person_dir].append(img_file)

        return train_person_to_images, test_person_to_images

    def __diff_person_sample(self, mapping):
        first_person, second_person = random.sample(mapping, 2)
        first_img = random.choice(mapping[first_person])
        second_img = random.choice(mapping[second_person])
        return first_img, second_img

    def __same_person_sample(self, mapping):
        while True:
            person = random.choice(mapping.keys())
            if len(mapping[person]) == 1:
                continue
            return random.sample(mapping[person], 2)

    def get_same_person_train_sample(self):
        return self.__same_person_sample(self.train_person_to_images)

    def get_same_person_test_sample(self):
        return self.__same_person_sample(self.test_person_to_images)

    def get_diff_person_train_sample(self):
        return self.__diff_person_sample(self.train_person_to_images)

    def get_diff_person_test_sample(self):
        return self.__diff_person_sample(self.test_person_to_images)

    def get_train_images(self):
        images = list()
        for person in self.train_person_to_images:
            for img in self.train_person_to_images[person]:
                images.append(img)
        return images

    def print_dataset_stats(self):
        print 'Training: Unique Persons: %s' % len(self.train_person_to_images)
        print 'Testing: Unique Persons: %s' % len(self.test_person_to_images)
'''
