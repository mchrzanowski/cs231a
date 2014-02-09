# fv.py takes in an augmented matrix of sift descriptors and performs Mixture of Guassians 
import numpy
from sklearn.mixture import GMM
from gmm_specializer.gmm import *



def form_feature_vector_by_mixture_of_gaussians(B,numberOfGaussians=512):

    dim = B.shape
    gmm = GMM(dim(0), dim(1), cvtype='diag')
    gmm.train(B, max_em_iters=10,min_em_iters=2)
   
    means = numpy.round(gmm.components.means,4)
    covs = numpy.round(gmm.components.covars,4)
    
    #g = GMM(n_components=numberOfGaussians,n_iter=10)
    #g.fit(B)
    #means = numpy.round(g.means_, 4)
    #covs = numpy.round(g.covars_, 4)
    
    fisherVectorList = []
    for i in xrange(0,len(means)):
	   fisherVectorList.extend(means[i])
	   fisherVectorList.extend(covs[i])
    return  fisherVectorList
