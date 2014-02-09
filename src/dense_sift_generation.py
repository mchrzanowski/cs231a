import cv2

def createDenseSIFTFeatures(image):
    '''
    This is weird:
        initXyStep = the stride to use
        initImgBound = the number of rows and cols to exclude
        initFeatureScale = diameter of patch to use for keypt
        featureScaleMul = scale for the feature patch per level
        featureScaleLevels = # of levels to use
        varyXyStepWithScale = scale the xy stride with scaling
        varyImgBoundWithScale = scale the image bounds with the scaling
    '''
    feature_detector = cv2.FeatureDetector_create('Dense')
    feature_detector.setInt('initImgBound', 18)
    feature_detector.setInt('initXyStep', 1)
    feature_detector.setDouble('initFeatureScale', 36)
    feature_detector.setDouble('featureScaleMul', 2 ** 0.5)
    feature_detector.setInt('featureScaleLevels', 5)
    feature_detector.setBool('varyXyStepWithScale', False)
    feature_detector.setBool('varyImgBoundWithScale', True)

    keypts = feature_detector.detect(image)
    
    sift = cv2.SIFT()
    _, descriptors = sift.compute(image, keypts)

    return keypts, descriptors
