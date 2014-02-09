# compress sift takes in a matrix of 128xd and drops it to 64xd and returns it
import numpy
from sklearn.decomposition import RandomizedPCA

def compressSiftFeatures(Q,truncationAmount=64):

	if (type(Q).__module__!='numpy'):
		print "Not a valid numpy matrix..."
		return
	else:
		pca = RandomizedPCA(copy=True, iterated_power=3, n_components=truncationAmount,
			       random_state=None, whiten=True)
		B=pca.fit(Q).components_
		return B


