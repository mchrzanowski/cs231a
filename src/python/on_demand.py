import constants
import cPickle
import utilities
import os

def _generate_fv(file):
    fv_file = '/tmp/fv' 
    if os.path.isfile('/tmp/fv'):
        os.remove(fv_file)
    
    cmd = """cd ../matlab; matlab -nosplash -nodesktop -nojvm -r "remote_create_fv('%s', '/tmp/fv', '%s')" > /dev/null""" % (file,
        constants.PARAM_DIR)
    return_value = os.system(cmd)
    if return_value != 0:
        raise Exception('error from matlab! error code: %d' % return_value)
    
    fv = utilities.hydrate_fv_from_file('/tmp/fv')
    if os.path.isfile('/tmp/fv'):
        os.remove(fv_file)
    if fv.shape[0] != constants.FV_DIM:
        raise Exception('error! fv for file %s of unexpected size. Got: %s. Expected: %s' % (file, fv.shape[0], constants.FV_DIM))
    
    return fv

def decide(file1, file2):
    try:
        fv1, fv2 = map(_generate_fv, [file1, file2])
        b, W = cPickle.load(open(constants.W_B_FILE, 'rb'))
        decision = utilities.get_distance(W, b, fv1, fv2) >= 0
    except Exception as e:
        raise e
    return decision

if __name__ == "__main__":
    x = _generate_fv('~/Downloads/lfw/Aaron_Sorkin/Aaron_Sorkin_0001.jpg')
    print x
