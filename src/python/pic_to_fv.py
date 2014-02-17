import cv2
from preprocessing import preprocessImage
from dense_sift_generation import createDenseSIFTFeatures
from compress_sift_features import compressSiftFeatures
from feature_augmentation import feature_augmentation
from fv import form_feature_vector_by_mixture_of_gaussians


class FisherVectorGenerator(object):

    def __init__(self, img_file):
        self.img_file = img_file

    def generate(self):
        img = preprocessImage(self.img_file)
        keypts, descriptors = createDenseSIFTFeatures(img)
        print descriptors.shape
        descriptors = compressSiftFeatures(descriptors)
        descriptors = feature_augmentation(descriptors, keypts)
        print descriptors.shape
        fv = form_feature_vector_by_mixture_of_gaussians(descriptors)
        return fv


def main():
    x = '/opt/cs231a/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
    fvg = FisherVectorGenerator(x)
    y = fvg.generate()
    print len(y)

if __name__ == "__main__":
    main()
