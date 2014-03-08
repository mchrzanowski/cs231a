#! /opt/local/bin/ python2.7
import os
import numpy
import pickle
import re
import time
#from pyfann import libfann


import random
from numpy.linalg import norm
from numpy import dot
#####
# theano 
#####
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from DBN import DBN
######
# pybrain
###########
#from pybrain.tools.shortcuts import buildNetwork
#from pybrain.structure import SoftmaxLayer
#from pybrain.structure import LinearLayer
#from pybrain.structure import SigmoidLayer
#from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.datasets import SupervisedDataSet

import cPickle
import math



def this_create_total_dict(fromPercent, toPercent):

    input_dir="/tmp/fvs_fun/fv_deep_funneled/"
    #input_dir=""
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

    #####
    # theano 
    #####
    # requires atleast 3 layers
    print "building network"
    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    net = DBN(numpy_rng=numpy_rng, n_ins=layerDims[0],
              hidden_layers_sizes=layerDims[1:-1],
              n_outs=layerDims[-1])
    print "finished building network"
    
    
    ######
    # pybrain
    ###########

    #print "building network"
    #net = buildNetwork(*layerDims,hiddenclass=SigmoidLayer,outclass=SoftmaxLayer)
    #print "finished building network"
    ##net.addInputModule(LinearLayer(67584, 'visible'))
    #trainer = BackpropTrainer(net)

    ##########
    # fann
    #########
    #ann = libfann.neural_net()
    #ann.create_standard_array((67584,300,100,2))

    #ann.set_learning_rate(learning_rate)
    #ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    
    data_set_size = 1000
    gibbs = 2
    batch_size = 10
    pretraining_epochs=100
    finetune_lr=0.1
    pretrain_lr=0.01
    training_epochs=1000,
    
    fn=0
    fp=0
    tp=0
    for ii in xrange(max_iterations):
        train_set_x = []
        train_set_y = []
        validate_set_x = []
        validate_set_y = []
        test_set_x = []
        test_set_y = []
        
        for n in xrange(data_set_size):
            (fv_diff,y) = get_diff_of_fvs(fv_dict,iterations_between_reports,ii)
            train_set_x.append(fv_diff[0].tolist())
            train_set_y.append(y)
        train_set_x=theano.shared(numpy.asarray(train_set_x))
        train_set_y=theano.shared(numpy.asarray(train_set_y))
        
        for n in xrange(data_set_size):
            (fv_diff,y) = get_diff_of_fvs(fv_dict,iterations_between_reports,ii)
            validate_set_x.append(fv_diff[0].tolist())
            validate_set_y.append(y)
        validate_set_x=theano.shared(numpy.asarray(validate_set_x))
        validate_set_y=theano.shared(numpy.asarray(validate_set_y))
        
        for n in xrange(data_set_size):
            (fv_diff,y) = get_diff_of_fvs(fv_dict,iterations_between_reports,ii)
            test_set_x.append(fv_diff[0].tolist())
            test_set_y.append(y)
        test_set_x=theano.shared(numpy.asarray(test_set_x))
        test_set_y=theano.shared(numpy.asarray(test_set_y))
        
        datasets=[[train_set_x,train_set_y],[validate_set_x,validate_set_y],[test_set_x,test_set_y]]
        #print fv_diff.shape
        #print len(fv_diff[0].tolist())

        #cPickle.dump([0,1],open('lol','wb'))
        #cPickle.load(open('lol','rb'))
        #####
        # pybrain training
        ######
        #dataSet = SupervisedDataSet(67584, 2)
        #dataSet.addSample(fv_diff[0].tolist(),y)
        #trainer.setData(dataSet)
        ##print "training"
        #print "training"
        #trainer.train()
        #print "finished train"
        #res=net.activate(fv_diff[0].tolist())
        #print "finished test"
        
        ###########
        # Theano pretraining
        ###########
        print '... getting the pretraining functions'
        
        # We are using CD-1 here
        pretraining_fns = net.pretraining_functions(
                train_set_x=train_set_x,
                batch_size=batch_size,
                k=gibbs)
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        #n_train_batches = len(train_set_x) / batch_size
        
        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        for i in xrange(net.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
    
        end_time = time.clock()
        
        
        
        ########################
        # FINETUNING THE MODEL #
        ########################
    
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = net.build_finetune_functions(
                    datasets=datasets, batch_size=batch_size,
                    learning_rate=finetune_lr)
    
        print '... finetunning the model'
        # early-stopping parameters
        patience = 4 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.    # wait this much longer when a new best is
                                  # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()
    
        done_looping = False
        epoch = 0
    
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
    
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
    
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = test_model()
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
                     (best_validation_loss * 100., test_score * 100.))
        print >> sys.stderr, ('The fine tuning code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time)
                                                  / 60.))
        return net
                                                  
                                                  
                                              
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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

    max_iterations_test = 10
    max_iterations_train = 10
    iterations_between_reports = 1
    train_percent_of_dataset = 0.75
    layerDims = [67584,1024,128,2]

    print "training the net"
    net = train_the_nn(max_iterations_train,iterations_between_reports,train_percent_of_dataset,layerDims)
    
    print "testing the net"
    #test_the_nn(net,max_iterations_test,train_percent_of_dataset,iterations_between_reports)
    
    print "pickling.."
    #cPickle.dump(net,open('francois_neural_network','wb'))
    
    print "finished pickling!"






