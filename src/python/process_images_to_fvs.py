from pic_to_fv import FisherVectorGenerator
import multiprocessing
import os
import pickle
def main():
    output_dir = '/opt/cs231a/serialized/'
    input_dir = '/opt/cs231a/lfw/'
    pool = multiprocessing.Pool()
    for name in os.listdir(input_dir):
        subdir = os.path.join(input_dir, name)
        for image_file in os.listdir(subdir):
            image_file_absolute =  os.path.join(subdir,image_file)
            pool.apply_async( run, args=(FisherVectorGenerator(image_file_absolute),))
    




def run(x_class):
    fisher_vector = x_class.generate()
    output_dir = '/opt/cs231a/serialized/'
    save_path = os.path.join(output_dir,x_class.img_file)
    print 'Saving %s' % save_path
    pickle.dump(fisher_vector,save_path) 





