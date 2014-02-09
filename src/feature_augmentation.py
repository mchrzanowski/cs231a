# feature augmentation takes in a post PCA's matrix and tacks on some meta information about each column such
# as: coordinates, colors, illumination, etc ... the columns of B are whitened sift descriptors

def feature_augmentation(B, locations, coordinates=True,colors=False,Illumination=False):

    if (type(B).__module__!='numpy'):
        print "Not a valid numpy matrix..."
        return
    else:
        #for x in range(0, B.shape[1]):
        return B	

