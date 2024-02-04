import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import logging
from tsplearn import *

def setup_logging():
    logging.basicConfig(filename='simulation_log.txt', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the TSP simulation with customizable parameters.')
    parser.add_argument('--dictionary_type', type=str, default="separated", help='Can be of type: separated, edge_laplacian or joint')
    parser.add_argument('--n_sim', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--n', type=int, default=40, help='Number of nodes in the graph')
    parser.add_argument('--p', type=float, default=0.162, help='Probability for edge creation in the graph')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')
    parser.add_argument('--sub_size', type=int, default=100, help='Sub-sampling size to decrease complexity')
    parser.add_argument('--m_train', type=int, default=150, help='Number of Train Signals')
    parser.add_argument('--m_test', type=int, default=80, help='Number of Test Signals')
    parser.add_argument('--s', type=int, default=3, help='Number of Kernels (Sub-dictionaries)')
    parser.add_argument('--k', type=int, default=2, help='Polynomial order')
    parser.add_argument('--sparsity', type=float, default=.1, help='Sparsity percentage')
    parser.add_argument('--K0_max', type=int, default=20, help='Max sparsity level')
    parser.add_argument('--n_search', type=int, default=3000, help='Number of searches for best sparse approximation')
    parser.add_argument('--verbose', type=bool, default=False, action='store_true', help='Enable verbose output')
    parser.add_argument('--tol', type=float, default=1e-7, help='Tolerance for patience')
    parser.add_argument('--lambda_', type=float, default=1e-7, help='l2 multiplier for the optimization problem')
    parser.add_argument('--max_iter', type=int, default=100, help='Max number of iterations for the optimization algorithm') 
    parser.add_argument('--patience', type=int, default=1e-7, help='Patience of the optimization algorithm')
    
    return parser.parse_args()

def initialize_simulation_variables(n,s,n_sim,m_train,m_test,n_search=3000):

    # Variables for dictionary generation
    global D_true, D_true_coll, Y_train, Y_test, epsilon_true, c_true, X_train, X_test
    D_true = np.zeros((n, n * s, n_sim))
    D_true_coll = np.zeros((n, n, s, n_sim))
    Y_train = np.zeros((n, m_train, n_sim))
    Y_test = np.zeros((n, m_test, n_sim))
    epsilon_true = np.zeros(n_sim)
    c_true = np.zeros(n_sim)
    X_train = np.zeros((n * s, m_train, n_sim))
    X_test = np.zeros((n * s, m_test, n_sim))
    n_search = n_search 

    # Variables for storing results
    global K0_coll, min_error_fou_train, min_error_fou_test, min_error_sep_train, min_error_sep_test, min_error_edge_train
    global min_error_edge_test, min_error_joint_train, min_error_joint_test, dict_errors, dict_types
    K0_coll = np.arange(5, 26, 4)
    min_error_fou_train = np.zeros((n_sim, len(K0_coll)))
    min_error_fou_test = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_train = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_test = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_train = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_test = np.zeros((n_sim, len(K0_coll)))
    min_error_joint_train = np.zeros((n_sim, len(K0_coll)))
    min_error_joint_test = np.zeros((n_sim, len(K0_coll)))

    dict_errors = {
        "fou": (min_error_fou_train,min_error_fou_test),
        "edge": (min_error_edge_train,min_error_edge_test),
        "joint": (min_error_joint_train,min_error_joint_test),
        "sep": (min_error_sep_train,min_error_sep_test)
        }

    dict_types = {
        "fou": ("Fourier","fourier"),
        "edge": ("Edge Laplacian", "edge_laplacian"),
        "joint": ("Hodge Laplacian","joint"),
        "sep": ("Separated Hodge Laplacian","separated")
        }

def run_simulation(args):

    # Suppress warnings and setup logging
    warnings.filterwarnings('ignore')
    setup_logging()

    # Log the simulation parameters
    logging.info(f"Starting simulation with parameters: {args}")

    # Load the graph with user-defined parameters
    G = EnhancedGraph(n=args.n, p=args.p, seed=args.seed)
    B1 = G.get_b1()
    B2 = G.get_b2()

    # Sub-sampling if needed to decrease complexity
    sub_size = 100
    B1 = B1[:, :sub_size]
    B2 = B2[:sub_size, :]
    B2 = B2[:,np.sum(np.abs(B2), 0) == 3]

    # Laplacians
    Ld = np.matmul(np.transpose(B1), B1, dtype=float)
    Lu = np.matmul(B2, np.transpose(B2), dtype=float)
    L = Lu+Ld

    initialize_simulation_variables(args.n, args.s, args.n_sim, args.m_train, args.m_test, args.n_search)

    for sim in range(args.n_sim):
        best_sparsity = 0
        best_acc = 0

        for _ in tqdm(range(args.n_search)):
            try:
                D_try, _, Y_train_try, Y_test_try, epsilon_try, c_try, X_train_try, X_test_try = create_ground_truth(
                                                                                        args.Lu,
                                                                                        args.Ld,
                                                                                        args.m_train,
                                                                                        args.m_test, 
                                                                                        s=args.s, 
                                                                                        K=args.k, 
                                                                                        K0=args.K0_max, 
                                                                                        dictionary_type=args.dictionary_type, 
                                                                                        sparsity_mode=args.sparsity_mode)
                
                max_possible_sparsity, acc = verify_dic(D_try, Y_train_try, X_train_try, args.K0_max, .7)
                if max_possible_sparsity > best_sparsity:
                    best_sparsity = max_possible_sparsity
                    best_acc = acc
                    D_true[:, :, sim] = D_try
                    Y_train[:, :, sim] = Y_train_try
                    Y_test[:, :, sim] = Y_test_try
                    epsilon_true[sim] = epsilon_try
                    c_true[sim] = c_try
                    X_train[:, :, sim] = X_train_try
                    X_test[:, :, sim] = X_test_try

            except Exception as e:
                print(f"Error during dictionary creation: {e}")
        if args.verbose:
            print(f"...Done! # Best Sparsity: {best_sparsity}")

    logging.info("Dictionary generation completed successfully.")

    for sim in range(args.n_sim):
        c = c_true[sim]  
        epsilon = epsilon_true[sim] 
        for k0_index, k0 in tqdm(enumerate(K0_coll)):
            discard = 1
            while discard == 1:
                
                try:
                    D0, X0, discard = initialize_dic(Lu, Ld, args.s, args.k, Y_train[:, :, sim], k0, args.dictionary_type, c, epsilon, "only_X")
                except:
                    print("Initialization Failed!")

            for d in dict_types.items():
                try:
                    dict_errors[d[0]][0][sim,k0_index], dict_errors[d[0]][1][sim,k0_index], _, _, _ = topological_dictionary_learn(Y_train[:,:,sim], Y_test[:,:,sim],
                                                                                                                                args.k, args.n, args.s, D0, X0,
                                                                                                                                Lu, Ld, d[1][1], c, epsilon, k0,
                                                                                                                                args.lambda_, args.max_iter,
                                                                                                                                args.patience, args.tol)
                    if args.verbose:
                        print(f"Simulation: {sim+1}/{args.n_sim} Sparsity: {k0} Testing {d[1][0]}... Done! Test Error: {dict_errors[d[0]][1][sim,k0_index]}")
                except:
                    print(f'Simulation: {sim+1}/{args.n_sim} Sparsity: {k0} Testing {d[1][0]}... Diverged!')
                    try:
                        dict_errors[d[0]][0][sim,k0_index], dict_errors[d[0]][1][sim,k0_index] = (dict_errors[d[0]][0][sim-1,k0_index]
                                                                                            , dict_errors[d[0]][1][sim-1,k0_index])
                    except:
                        dict_errors[d[0]][0][sim,k0_index], dict_errors[d[0]][1][sim,k0_index] = (dict_errors[d[0]][0][sim+1,k0_index]
                                                                                            , dict_errors[d[0]][1][sim+1,k0_index])

    logging.info("Dictionary learning completed successfully.")

if __name__ == "__main__":
    args = parse_arguments()
    run_simulation(args)
