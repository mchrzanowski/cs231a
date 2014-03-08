#! /opt/local/bin/ python2.7
import os
import numpy
import pickle
import re
#from pyfann import libfann
import pybrain

import random
from numpy.linalg import norm
from numpy import dot

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import cPickle

def this_create_total_dict():
    input_dir='/tmp/fvs_fun/fv_deep_funneled/'
    fv_dict = dict()
    count=0;
    for name in os.listdir(input_dir):
        count=count+1
        input_file = os.path.join(input_dir,name)
        with open(input_file) as f:
            #w, h = [float(x) for x in f.readline().split()]
            #array = [[float(x) for x in line.split()] for line in f]
            s= numpy.genfromtxt(input_file, delimiter=',')
            

            word1 = "".join(re.findall("[a-zA-Z]+", name))
            word = word1[:-3]
            if word in fv_dict:
                fv_dict[word].append([numpy.true_divide(s,numpy.linalg.norm(s,ord=2))])
                #fv_dict[word]= numpy.concatenate(  (fv_dict[word],  numpy.true_divide(s,numpy.linalg.norm(s,ord=2))) ,axis=1)
            else:
                fv_dict[word]=[numpy.true_divide(s,numpy.linalg.norm(s,ord=2))]

            print count
            if count>10000:
                break;
    return fv_dict



if __name__ != "Main":
    connection_rate = 1
    learning_rate = 0.7
    num_input = 2
    num_hidden = 4
    num_output = 1

    desired_error = 0.0001
    max_iterations = 100000
    iterations_between_reports = 1000


    print 'creating total dict'
    fv_dict = this_create_total_dict()
    print 'finished creating dict'


    ######
    # pybrain
    ###########
    epochs = 100
    layerDims = [67584,1024,128,2]

    print "building network"
    net = buildNetwork(*layerDims,hiddenclass=SigmoidLayer,outclass=SoftmaxLayer)
    print "finished building network"
    #net.addInputModule(LinearLayer(67584, 'visible'))
    trainer = BackpropTrainer(net)


    #trainer = pybrain.supervised.trainers.BackpropTrainer(net, ds, learningrate = 0.001, momentum = 0.99)
    ##########
    # fann
    #########
    #ann = libfann.neural_net()
    #ann.create_standard_array((67584,300,100,2))

    #ann.set_learning_rate(learning_rate)
    #ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    interval=100
    count=0
    for i in xrange(max_iterations):
        count+=1
        if random.random() < 0.5 :
            while True:
                person = random.choice(fv_dict.keys())
                if len(fv_dict[person])== 1:
                    continue
                fv_all=fv_dict[person]
                i=random.randint(0,len(fv_all)-1)
                while True:
                    j=random.randint(0,len(fv_all)-1)
                    if i==j:
                        continue
                    else:
                        break
                fv1 = fv_all[i]
                fv2 = fv_all[j]
                if(count%interval == 0):
                    print "count %s: %s at %s and %s at %s" % (count, person,i,person,j)
                y = (1, 0)
                break
        else:
            while True:
                person1 = random.choice(fv_dict.keys())
                person2 = random.choice(fv_dict.keys())
                if person1 is person2:
                    continue
                else:
                    break
           
            fv1_temp = fv_dict[person1]
            fv2_temp = fv_dict[person2]
            
            i=random.randint(0,len(fv1_temp)-1)
            j=random.randint(0,len(fv2_temp)-1)
            if(count%interval == 0):
                print "count %s: %s at %s and %s at %s" % (count, person1,i,person2,j)
            fv1=fv1_temp[i]
            fv2=fv2_temp[j]
            y = (0, 1)

        fv_diff = numpy.atleast_2d(numpy.asarray(fv1) - numpy.asarray(fv2))
        #print fv_diff.shape
        #print len(fv_diff[0].tolist())

        #cPickle.dump([0,1],open('lol','wb'))
        #cPickle.load(open('lol','rb'))
        #####
        # pybrain training
        ######
        dataSet = SupervisedDataSet(67584, 2)
        dataSet.addSample(fv_diff[0].tolist(),y)
        trainer.setData(dataSet)
        #print "training"
        trainer.train()
        ########
        # fann
        ########
        if(count%interval == 0):
            #ann.train(fv_diff[0].tolist(),y)
            print "testing"
            print net.activate(fv_diff[0].tolist())
            print y
            #print ann.run(fv_diff[0].tolist())
            #print y
            print numpy.asarray(fv1)
            print numpy.asarray(fv2)
            if(count%(interval*50) == 0):
                cPickle.dump(net,open('francois_neural_network','wb'))
        

    #ann.save("fv.net")

