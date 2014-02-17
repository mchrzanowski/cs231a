import os
import numpy
import pickle
import re
from pyfann import libfann

import random
from numpy.linalg import norm
from numpy import dot

def this_create_total_dict():
    input_dir='/opt/cs231a/serialized/fvs/'
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
            if count>5000:
                break;
    return fv_dict



if __name__ != "Main":
    connection_rate = 1
    learning_rate = 0.7
    num_input = 2
    num_hidden = 4
    num_output = 1

    desired_error = 0.001
    max_iterations = 100000
    iterations_between_reports = 1000


    print 'creating total dict'
    fv_dict = this_create_total_dict()
    print 'finished creating dict'

    ann = libfann.neural_net()
    ann.create_standard_array((67584,13000,1000,2))

    ann.set_learning_rate(learning_rate)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
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

                print "count %s: %s at %s and %s at %s" % (count, person,i,person,j)
                y = [1, -1]
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
            print "count %s: %s at %s and %s at %s" % (count, person1,i,person2,j)
            fv1=fv1_temp[i]
            fv2=fv2_temp[j]
            y = [-1, 1]

        fv_diff = numpy.atleast_2d(numpy.asarray(fv1) - numpy.asarray(fv2))
        print numpy.asarray(fv1)
        print numpy.asarray(fv2)
        print fv_diff.shape
        print len(fv_diff[0].tolist())

        ann.train(fv_diff[0].tolist(),y)
        print ann.run(fv_diff[0].tolist())
        print y


    ann.save("fv.net")

