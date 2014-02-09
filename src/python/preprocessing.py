import cv2
import constants

def preprocessImage(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, dsize=(constants.RESIZED_IMAGE_HEIGHT, constants.RESIZED_IMAGE_WIDTH))
    return img
