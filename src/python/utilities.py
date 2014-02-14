import numpy

def hydrate_fv_from_file(input_file):
    fv = numpy.genfromtxt(input_file, dtype=numpy.float32, delimiter=',')
    return fv / numpy.linalg.norm(fv, 2)
