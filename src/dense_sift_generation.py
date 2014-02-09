import cv2

def createDenseSIFTFeatures(image):
    feature_detector = cv2.FeatureDetector_create('Dense')
    feature_detector.setInt('initImgBound', 18)
    feature_detector.setInt('initXyStep', 1)
    feature_detector.setDouble('initFeatureScale', 24)
    feature_detector.setDouble('featureScaleMul', 2 ** 0.5)
    feature_detector.setInt('featureScaleLevels', 5)
    feature_detector.setBool('varyXyStepWithScale', True)
    feature_detector.setBool('varyImgBoundWithScale', True)

    keypts = feature_detector.detect(image)
    
    sift = cv2.SIFT()
    _, descriptors = sift.compute(image, keypts)

    return keypts, descriptors
