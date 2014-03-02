import constants
import os
import random

class Dataset(object):
    def convert_to_filename(self, person, image_num):
        image_num = str(image_num)
        image_num = '0' * (4 - len(image_num)) + image_num  # pre-append zeros
        return person + '_' + image_num + '.csv'

    def get_fv_file_for_image(self, image):
        return os.path.join(self.base_dir, image)

    def print_dataset_stats(self):
        print 'Type: %s' % self.__class__.__name__
        print 'LFW Dataset Directory: %s' % self.base_dir

class DevDataset(Dataset):
    def __init__(self, base_dir, train_filename=constants.DEV_TRAIN_PAIR_FILE,
    test_filename=constants.DEV_TEST_PAIR_FILE):
        self.base_dir = base_dir
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
        Dataset.print_dataset_stats(self)
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

class RestrictedDataset(DevDataset):
    def __init__(self, base_dir, filename=constants.PAIR_FILE, split=random.randint(1, 10)):
        self.base_dir = base_dir
        self.split = split
        self.train_images, \
        self.train_same_person_pairs, \
        self.train_diff_person_pairs, \
        self.test_images, \
        self.test_same_person_pairs, \
        self.test_diff_person_pairs = self.init(filename, split)

    def print_dataset_stats(self):
        Dataset.print_dataset_stats(self)
        print 'Split: %s' % self.split

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

class UnrestrictedDataset(DevDataset):
    def __init__(self, base_dir, filename=constants.PEOPLE_FILE, split=random.randint(1, 10)):
        self.base_dir = base_dir
        self.split = split
        self.train_images, \
        self.train_persons_to_imgs, \
        self.test_images, \
        self.test_persons_to_imgs = self.init(filename, split)

    def get_train_images(self):
        return self.train_images

    def __get_same_person_sample(self, mapping):
        while True:
            person = random.choice(mapping.keys())
            if len(mapping[person]) > 1:
                break
        # sample w/o replacement. 
        imgs = list(mapping[person]) # copy: careful to not change state...
        img1 = random.choice(imgs)
        imgs.remove(img1)
        img2 = random.choice(imgs)
        return img1, img2

    def __get_diff_person_sample(self, mapping):
        population = mapping.keys()     # copy for sampling w/o replacement.
        person1 = random.choice(population)
        img1 = random.choice(mapping[person1])
        population.remove(person1)
        person2 = random.choice(population)
        img2 = random.choice(mapping[person2])
        return img1, img2

    def get_same_person_train_sample(self):
        return self.__get_same_person_sample(self.train_persons_to_imgs)

    def get_diff_person_train_sample(self):
        return self.__get_diff_person_sample(self.train_persons_to_imgs)

    def gen_same_person_test_samples(self, number=300):
        for _ in xrange(number):
            yield self.__get_same_person_sample(self.test_persons_to_imgs)

    def gen_same_person_train_samples(self, number=300):
        for _ in xrange(number):
            yield self.__get_same_person_sample(self.train_persons_to_imgs)

    def gen_diff_person_test_samples(self, number=300):
        for _ in xrange(number):
            yield self.__get_diff_person_sample(self.test_persons_to_imgs)

    def gen_diff_person_train_samples(self, number=300):
        for _ in xrange(number):
            yield self.__get_diff_person_sample(self.train_persons_to_imgs)

    def print_dataset_stats(self):
        Dataset.print_dataset_stats(self)
        print 'Split: %s' % self.split
        train_persons = float(len(self.train_persons_to_imgs))
        train_imgs = float(len(self.train_images))
        print 'Train: Total Persons: %s' % train_persons
        print 'Train: Total Imgs: %s' % train_imgs
        print 'Train: Mean Imgs per Person: %s' % (train_imgs / train_persons)
        test_persons = float(len(self.test_persons_to_imgs))
        test_imgs = float(len(self.test_images))
        print 'Test: Total Persons: %s' % test_persons
        print 'Test: Total Imgs: %s' % test_imgs
        print 'Test: Avg Imgs per Person: %s' % (test_imgs / test_persons)

    def init(self, filename, split):
        '''
        assume splits are 600 a piece, with 300 same pairs first
        and 300 same pairs afterwards
        '''
        assert split >= 1 and split <= 10
        return self.__create_person_to_img_maps(filename, split)

    def __create_person_to_img_maps(self, filename, split):
        train_person_to_imgs = dict()
        test_person_to_imgs = dict()
        train_images = set()
        test_images = set()
        current_split = 0
        with open(filename, 'rb') as f:
            for i, line in enumerate(f):
                
                data = line.strip().split('\t')
                if len(data) == 1:                  # new split 
                    current_split += 1
                else:                               # people data.
                    if current_split == split:      # test set.
                        person_to_img = test_person_to_imgs
                        imgs = test_images
                    else:                           # train.
                        person_to_img = train_person_to_imgs
                        imgs = train_images
                    
                    person, num = data
                    if person not in person_to_img:
                        person_to_img[person] = list()

                    for i in xrange(1, int(num) + 1):
                        img = self.convert_to_filename(person, i)
                        imgs.add(img)
                        person_to_img[person].append(img)

        return train_images, train_person_to_imgs, test_images, test_person_to_imgs
