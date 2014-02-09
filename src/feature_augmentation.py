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
        
        for i in xrange(0, B.shape[0]):
            x_vect.append(locations[i].pt[0])
            y_vect.append(locations[i].pt[1])
        print len(x_vect)
        print B.shape
        B=numpy.concatenate((B,numpy.array([numpy.asarray(x_vect)]).T),axis=1)
        B=numpy.concatenate((B,numpy.array([numpy.asarray(y_vect)]).T),axis=1)
        return B	

