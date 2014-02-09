# compress sift takes in a matrix of dx128 and drops it to dx64 and returns it
import numpy
from sklearn.decomposition import RandomizedPCA

def compressSiftFeatures(Q,truncationAmount=64):

	if (type(Q).__module__!='numpy'):
		print "Not a valid numpy matrix..."
		return
	else:
		pca = RandomizedPCA(copy=True, iterated_power=3, n_components=truncationAmount,
			       random_state=None, whiten=True)
		B = pca.fit_transform(Q)
        return B
