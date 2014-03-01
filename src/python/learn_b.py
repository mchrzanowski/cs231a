import constants
import dataset_generation
import init_W
import multiprocessing
import optimization
import train_and_test

def cv(dataset, fvs, images_to_indices, b, Winit, debug=False, verbose=False):
    if debug:
        W, _ = optimization.ssgd(Winit, dataset, fvs=fvs,
                image_to_index=images_to_indices, verbose=verbose, b=b, iterations=0, cache=False)
    else:
        W, _ = optimization.ssgd(Winit, dataset, fvs=fvs,
                image_to_index=images_to_indices, verbose=verbose, b=b, cache=False)
    
    tp, fp, fn, tn = train_and_test.test(dataset, W, b, 'Test', verbose=verbose)
    acc = (tp + tn) / float(tp + fp + fn + tn)
    return acc

def run(deep_funneled, debug=False, verbose=False):

    if deep_funneled:
        data_dir = constants.FV_DF_DIR
    else:
        data_dir = constants.FV_DIR

    if debug:
        dataset = dataset_generation.DevDataset(base_dir=data_dir)
    else:
        dataset = dataset_generation.UnrestrictedDataset(base_dir=data_dir, split=1)

    if verbose:
        print 'Debug Mode: %s' % debug
        dataset.print_dataset_stats()

    Winit, fvs, images_to_indices = init_W.init(dataset, debug, verbose)
    pool = multiprocessing.Pool()
    bs = [-100, 0, 50, 100, 200, 400, 800, 1600, 3200, 5400]
    best_accuracy = 0.
    best_b = None
    rets = []

    for b in bs:
        ret = pool.apply_async(cv, (dataset, fvs, images_to_indices, b, Winit, debug, verbose))
        rets.append((b, ret))
    
    for (b, ret) in rets:
        acc = ret.get()
        print 'Candidate b: %s. Accuracy: %s' % (b, acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_b = b
    pool.close()
    pool.join()
    print 'Winner: %s. Accuracy: %s' % (best_b, best_accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help="Debug mode.")
    parser.add_argument('-df', action='store_true', help="Deep-Funneled LFW dataset.")
    parser.add_argument('-v', action='store_true', help="Verbose mode.")
    args = vars(parser.parse_args())
    run(args['df'], args['d'], args['v'])
