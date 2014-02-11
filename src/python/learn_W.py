import constants
import init_W
import optimization
import pickle

def learn():
    W, fvs, person_to_indices = init_W.init_W()
    W = optimization.subgradient_optimization(W, fvs, person_to_indices)
    pickle.dump(W, open(constants.W_MATRIX_FILE, 'wb'))
    pickle.dump((fvs, person_to_indices), open(constants.FV_AND_MAPPING_FILE, 'wb'))

if __name__ == "__main__":
    learn()