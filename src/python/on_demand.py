import utilities
import os
import urllib2

def _generate_fv(url):
    raw_data = response = urllib2.urlopen(url).read()
    with open('/tmp/img', 'wb') as f:
        f.write(raw_data)
        print 'wrote data'

    cmd = "cd ../matlab; matlab -nosplash -nodesktop -nojvm -r \"remote_create_fv(\'/tmp/img\', \'/tmp/fv\')\"";
    print cmd
    os.system(cmd)
    fv = utilities.hydrate_fv_from_file('/tmp/fv')
    return fv

def generate_fvs(url1, url2):
    fv1, fv2 = map(_generate_fv, [url1, url2])
    return fv1, fv2

if __name__ == "__main__":
    x = _generate_fv('https://math.stanford.edu/inc/img/PalmDrive.png')
    print x.shape
