# this pulls in all the phis and performs PCA on it
import os
import numpy
from sklearn.decomposition import PCA
import pickle
import re
def init_W():

    input_dir='/opt/cs231a/serialized/fvs'
    W = []
    for name in os.listdir(input_dir):
        input_file = os.path.join(input_dir,name)
        with open(input_file) as f:
            #w, h = [float(x) for x in f.readline().split()]
            #array = [[float(x) for x in line.split()] for line in f]
            s= numpy.genfromtxt(input_file, delimiter=',')

            W.append(numpy.true_divide(s,numpy.linalg.norm(s,ord=2)))
           

    W_matrix=numpy.asarray(W)
    print('finished matrix comp')
    
    pca = PCA(n_components=128, whiten=True)
    B = pca.fit_transform(W_matrix.T)
    numpy.savetxt("initd_W_128.csv", B, delimiter=",")
    return B

def create_total_dict():
    input_dir='/opt/cs231a/serialized/fvs'
    fv_dict = dict()
    for name in os.listdir(input_dir):
        input_file = os.path.join(input_dir,name)
        with open(input_file) as f:
            #w, h = [float(x) for x in f.readline().split()]
            #array = [[float(x) for x in line.split()] for line in f]
            s= numpy.genfromtxt(input_file, delimiter=',')
            word1 = "".join(re.findall("[a-zA-Z]+", name))
            word = word1[:-3]
            if word in fv_dict:
                fv_dict[word]= numpy.concatenate((fv_dict[word],numpy.true_divide(s,numpy.linalg.norm(s,ord=2))),axis=0)
            else:
                fv_dict[word]=numpy.true_divide(s,numpy.linalg.norm(s,ord=2))
    return fv_dict






