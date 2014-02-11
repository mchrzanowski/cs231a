import init_W
import optimization
import pickle

def main():
    W, fvs, person_to_indices = init_W.init_W()
    W = optimization.subgradient_optimization(W, fvs, person_to_indices)
    pickle.dump(W, open('./w_matrix', 'wb'))
    pickle.dump((fvs, person_to_indices), open('./lol', 'wb'))

if __name__ == "__main__":
    main()