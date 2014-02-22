import numpy

def hydrate_fv_from_file(input_file):
    return numpy.genfromtxt(input_file, dtype=numpy.float32, delimiter=',')
