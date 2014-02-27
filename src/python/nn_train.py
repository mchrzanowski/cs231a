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
import math
def this_create_total_dict(fromPercent, toPercent):
    input_dir="/tmp/fvs_fun/fv_deep_funneled/"
    total = len(os.listdir(input_dir))
    start_index = int(fromPercent*total)
    end_index =  int(toPercent*total)
    fv_dict = dict()
    count=0;
    
    for name in os.listdir(input_dir)[start_index:end_index]:
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
        #if len(fv_dict.keys())>50:
        #    break;
    return fv_dict




def get_diff_of_fvs(fv_dict,iterations_between_reports,count):
        if random.random() < 0.5 :
            while True:
                person = random.choice(fv_dict.keys())
                if len(fv_dict[person])== 1:
                    #print "bump"
                    continue
                fv_all=fv_dict[person]
                i=random.randint(0,len(fv_all)-1)
                while True:
                    j=random.randint(0,len(fv_all)-1)
                    if i==j:
                        #print "bump"
                        continue
                    else:
                        break
                fv1 = fv_all[i]
                fv2 = fv_all[j]
                if(count%iterations_between_reports == 0):
                    print "count %s: %s at %s and %s at %s" % (count, person,i,person,j)
                y = (1, 0)
                break
        else:
            while True:
                person1 = random.choice(fv_dict.keys())
                person2 = random.choice(fv_dict.keys())
                if person1 is person2:
                    #print "bump"
                    continue
                else:
                    break
           
            fv1_temp = fv_dict[person1]
            fv2_temp = fv_dict[person2]
            
            i=random.randint(0,len(fv1_temp)-1)
            j=random.randint(0,len(fv2_temp)-1)
            if(count%iterations_between_reports == 0):
                print "count %s: %s at %s and %s at %s" % (count, person1,i,person2,j)
            fv1=fv1_temp[i]
            fv2=fv2_temp[j]
            y = (0, 1)

        if (numpy.atleast_2d(fv1).shape!=numpy.atleast_2d(fv2).shape):
            print "errorrr!!!"
            print numpy.atleast_2d(fv1).shape
            print numpy.atleast_2d(fv2).shape
            (fv_diff,y)=get_diff_of_fvs(fv_dict,iterations_between_reports,count)
        else:
            fv_diff = numpy.atleast_2d(numpy.asarray(fv1) - numpy.asarray(fv2))
        return (fv_diff,y)






def test_the_nn(net,max_iterations,train_percent_of_dataset,iterations_between_reports):
    print 'creating total dict'
    fv_dict = this_create_total_dict(train_percent_of_dataset,1.0)
    print 'finished creating dict'
    tp=0
    fp=0
    fn=0
    for i in xrange(max_iterations):
        (fv_diff, y) = get_diff_of_fvs(fv_dict,iterations_between_reports,i)

        res=net.activate(fv_diff[0].tolist())
        if((res[0]>res[1]) & (y[0]>y[1])):
            tp=tp+1
        elif((res[0]<=res[1]) & (y[0]>y[1])):
            fp=fp+1
        elif((res[0]>res[1]) & (y[0]<y[1])):
            fn=fn+1
        ########
        # fann
        ########
        
        #ann.train(fv_diff[0].tolist(),y)
        #print ann.run(fv_diff[0].tolist())
        #print y
        print "i %s: prec %s recall %s" % (i,(tp/(1e-4+fp+tp)),(tp/(1e-4+fn+tp)))
        #print "prec"
        #print (tp/(1e-4+fp+tp))
        #print "recall"
        #print (tp/(1e-4+fn+tp))


def train_the_nn(max_iterations,iterations_between_reports,train_percent_of_dataset,layerDims):

    print 'creating total dict'
    fv_dict = this_create_total_dict(0,train_percent_of_dataset)
    print 'finished creating dict'


    ######
    # pybrain
    ###########

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
    
    
    fn=0
    fp=0
    tp=0
    for i in xrange(max_iterations):
        
        (fv_diff,y) = get_diff_of_fvs(fv_dict,iterations_between_reports,i)

        
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
       
        print "training"
        trainer.train()
        print "finished train"
        res=net.activate(fv_diff[0].tolist())
        print "finished test"
        
        if((res[0]>res[1]) & (y[0]>y[1])):
            tp=tp+1
        elif((res[0]<=res[1]) & (y[0]>y[1])):
            fp=fp+1
        elif((res[0]>res[1]) & (y[0]<y[1])):
            fn=fn+1
        ########
        # fann
        ########
        if(i%iterations_between_reports == 0):
            #ann.train(fv_diff[0].tolist(),y)
            print "testing"
            print net.activate(fv_diff[0].tolist())
            print y
            #print ann.run(fv_diff[0].tolist())
            #print y
            print "prec"
            print (tp/(1e-4+fp+tp))
            print "recall"
            print (tp/(1e-4+fn+tp))
            tp=0
            fp=0
            fn=0


    return net
            
        



if __name__ != "Main":
 
    ######
    # params to fluctuate
    ###########

    max_iterations_test = 10000
    max_iterations_train = 100000
    iterations_between_reports = 1000
    train_percent_of_dataset = 0.75
    layerDims = [67584,1024,128,2]

    print "training the net"
    net = train_the_nn(max_iterations_train,iterations_between_reports,train_percent_of_dataset,layerDims)
    
    print "testing the net"
    test_the_nn(net,max_iterations_test,train_percent_of_dataset,iterations_between_reports)
    
    print "pickling.."
    cPickle.dump(net,open('francois_neural_network','wb'))
    
    print "finished pickling!"





