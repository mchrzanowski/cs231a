import constants
from datetime import datetime
import cPickle
import utilities
import os

_W = None
_b = None

def _generate_fvs(matlab_src_dir, param_dir, file1, file2):
    now = int(datetime.now().strftime("%s"))
    fv1_file = '/tmp/fv1_%s' % now
    fv2_file = '/tmp/fv2_%s' % now
    if os.path.isfile(fv1_file): os.remove(fv1_file)
    if os.path.isfile(fv2_file): os.remove(fv2_file)

    cmd = """cd %s; matlab -nosplash -nodesktop -nojvm -r "remote_create_fv('%s', '%s', '%s', '%s', '%s')" > /dev/null""" % (matlab_src_dir,
        file1, fv1_file, file2, fv2_file, param_dir)
    return_value = os.system(cmd)
    if return_value != 0: raise Exception('error from matlab! error code: %d' % return_value)

    fv1 = utilities.hydrate_fv_from_file(fv1_file)
    fv2 = utilities.hydrate_fv_from_file(fv2_file)
    os.remove(fv1_file)
    os.remove(fv2_file)
    if fv1.shape[0] != constants.FV_DIM:
        raise Exception('error! fv for file %s of unexpected size. Got: %s. Expected: %s' % (file1, fv1.shape[0], constants.FV_DIM))
    if fv2.shape[0] != constants.FV_DIM:
        raise Exception('error! fv for file %s of unexpected size. Got: %s. Expected: %s' % (file2, fv2.shape[0], constants.FV_DIM))

    return fv1, fv2

def decide(matlab_dir, param_dir, file1, file2):
    try:
        fv1, fv2 = _generate_fvs(matlab_dir, param_dir, file1, file2)
        global _W, _b
        if _W is None or _b is None:
            _b, _W = cPickle.load(open(os.path.join(param_dir, constants.B_W_FILE), 'rb'))
        decision = utilities.get_distance(_W, _b, fv1, fv2) >= 0
    except Exception as e:
        raise e
    return decision

if __name__ == "__main__":
    x = _generate_fvs(os.path.abspath('../cs231a/src/matlab'), os.path.abspath('../cs231a/params/gmm_params_rooted_df/'),
        '~/Downloads/lfw/Aaron_Sorkin/Aaron_Sorkin_0001.jpg', '~/Downloads/lfw/Aaron_Sorkin/Aaron_Sorkin_0001.jpg')
    print x
