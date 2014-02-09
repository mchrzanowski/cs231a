# fv.py takes in an augmented matrix of sift descriptors and performs Mixture of Guassians 
import numpy
from sklearn.mixture import GMM

def form_feature_vector_by_mixture_of_gaussians(B,numberOfGaussians=512):

    g = GMM(n_components=numberOfGaussians, n_init=1, n_iter=5, covariance_type='diag', params='mc', init_params='mc')
    print g
    g.fit(B)
    print 'done!'
    means = numpy.round(g.means_, 4)
    covs = numpy.round(g.covars_, 4)
    fisherVectorList = []
    for i in xrange(0,len(means)):
	   fisherVectorList.extend(means[i])
	   fisherVectorList.extend(covs[i])
    return  fisherVectorList
