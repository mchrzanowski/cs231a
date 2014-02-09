# feature augmentation takes in a post PCA's matrix and tacks on some meta information about each column such
# as: coordinates, colors, illumination, etc ... the columns of B are whitened sift descriptors
import numpy

def feature_augmentation(B, locations, coordinates=True,colors=False,Illumination=False):

    if (type(B).__module__!='numpy'):
        print "Not a valid numpy matrix..."
        return
    else:
        x_vect=[]
        y_vect=[]
        
        for i in xrange(0, B.shape[1]):
            x_vect.append(locations[i].pt[0])
            y_vect.append(locations[i].pt[1])

        B=numpy.concatenate((B, x_vect), axis=0)
        B=numpy.concatenate((B, y_vect), axis=0)
        return B	

