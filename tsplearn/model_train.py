import scipy.linalg as sla
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from .data_gen import *
from typing import Tuple

def initialize_dic(Lu: np.ndarray,
                   Ld: np.ndarray, 
                   s: int, 
                   K: int, 
                   Y_train: np.ndarray, 
                   K0: int,
                   dictionary_type: str, 
                   c: float, 
                   epsilon: float, 
                   only: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Initialize the dictionary and the signal sparse representation for the alternating
    optimization algorithm.

    Args:
        Lu (np.ndarray): Upper Laplacian matrix
        Ld (np.ndarray): Lower Laplacian matrix
        s (int): Number of kernels (sub-dictionaries).
        K (int): Max order of the polynomial for the single sub-dictionary.
        Y_train (np.ndarray): Training data.
        K0 (int): Sparsity of the signal representation.
        dictionary_type (str): Type of dictionary.
        c (float): Boundary constant from the synthetic data generation process.
        epsilon (float): Boundary constant from the synthetic data generation process.
        only (str): Type of initialization. Can be one of: "only_X", "all", "only_D".

    Returns:
        Tuple[np.ndarray, np.ndarray, bool]: Initialized dictionary, initialized sparse representation, and discard flag value.
    """

    n = Lu.shape[0]
    D = np.zeros((n, n*s))
    X = np.zeros(Y_train.shape)
    X = np.tile(X, (s,1))
    discard = 0

    def _multiplier_search(*arrays, s=s):
        is_okay = 0
        mult = 100
        tries = 0
        while is_okay==0:
            is_okay = 1
            h, c_try, _, tmp_sum_min, tmp_sum_max = generate_coeffs(arrays, s=s, mult=mult)
            if c_try <= c:
                is_okay *= 1
            if tmp_sum_min > c-epsilon:
                is_okay *= 1
                incr_mult = 0
            else:
                is_okay = is_okay*0
                incr_mult = 1
            if tmp_sum_max < c+epsilon:
                is_okay *= 1
                decr_mult = 0
            else:
                is_okay *= 0
                decr_mult = 1
            if is_okay == 0:
                tries += 1
            if tries >3:
                discard = 1
                break
            if incr_mult == 1:
                mult *= 2
            if decr_mult == 1:
                mult /= 2
        return h, discard

    if (only == "only_D") or (only == "all"):
        
        if dictionary_type == "joint":
            Lk, lambda_max_k, lambda_min_k = compute_Lk_and_lambdak(Lu + Ld, K)
            h, discard = _multiplier_search(lambda_max_k, lambda_min_k)
            D = generate_dictionary(h, s, Lk)

        elif dictionary_type == "edge_laplacian":
            Lk, lambda_max_k, lambda_min_k = compute_Lk_and_lambdak(Ld, K)
            h, discard = _multiplier_search(lambda_max_k, lambda_min_k)
            D = generate_dictionary(h, s, Lk)

        elif dictionary_type == "separated":
            Luk, lambda_max_u_k, lambda_min_u_k = compute_Lk_and_lambdak(Lu, K, separated=True)
            Ldk, lambda_max_d_k, lambda_min_d_k = compute_Lk_and_lambdak(Ld, K, separated=True)
            h, discard = _multiplier_search(lambda_max_d_k, lambda_min_d_k, lambda_max_u_k, lambda_min_u_k)
            D = generate_dictionary(h, s, Luk, Ldk)
    
    if (only == "only_X" or only == "all"):
        
        if dictionary_type == "edge_laplacian":
            L = Ld
        else:
            L = Lu+Ld

        _, Dx = sla.eig(L)
        dd = la.norm(Dx, axis=0)
        W = np.diag(1./dd)
        Dx = Dx / la.norm(Dx)  
        Domp = Dx@W
        X = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp.real, x), axis=0, arr=Y_train)
        X = np.tile(X, (s,1))
        
    return D, X, discard


def topological_dictionary_learn(Y_train: np.ndarray,
                                 Y_test: np.ndarray, 
                                 K: int, 
                                 n: int, 
                                 s: int,
                                 D0: np.ndarray, 
                                 X0: np.ndarray, 
                                 Lu: np.ndarray, 
                                 Ld: np.ndarray,
                                 dictionary_type: str, 
                                 c: float, 
                                 epsilon: float, 
                                 K0: int,
                                 lambda_: float = 1e-3, 
                                 max_iter: int = 10, 
                                 patience: int = 10,
                                 tol: float = 1e-7, 
                                 verbose: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dictionary learning algorithm implementation for sparse representations of a signal on complex regular cellular.
    The algorithm consists of an iterative alternating optimization procedure defined in two steps: the positive semi-definite programming step
    for obtaining the coefficients and dictionary based on Hodge theory, and the Orthogonal Matching Pursuit step for constructing 
    the K0-sparse solution from the dictionary found in the previous step, which best approximates the original signal.
    Args:
        Y_train (np.ndarray): Training data.
        Y_test (np.ndarray): Testing data.
        K (int): Max order of the polynomial for the single sub-dictionary.
        n (int): Number of data points (number of nodes in the data graph).
        s (int): Number of kernels (sub-dictionaries).
        D0 (np.ndarray): Initial dictionary.
        X0 (np.ndarray): Initial sparse representation.
        Lu (np.ndarray): Upper Laplacian matrix
        Ld (np.ndarray): Lower Laplacian matrix
        dictionary_type (str): Type of dictionary.
        c (float): Boundary constant from the synthetic data generation process.
        epsilon (float): Boundary constant from the synthetic data generation process.
        K0 (int): Sparsity of the signal representation.
        lambda_ (float, optional): Regularization parameter. Defaults to 1e-3.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        patience (int, optional): Patience for early stopping. Defaults to 10.
        tol (float, optional): Tolerance value. Defaults to 1e-7.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
         minimum training error, minimum testing error, optimal coefficients, optimal testing sparse representation, and optimal training sparse representation.
    """

    # Define hyperparameters
    min_error_train_norm, min_error_test_norm = 1e20, 1e20
    m_test, m_train = Y_test.shape[1], Y_train.shape[1]
    iter_, pat_iter = 1, 0

    if dictionary_type != "fourier":
        if dictionary_type=="joint":
            Lk, _, _ = compute_Lk_and_lambdak(Lu + Ld, K)
        elif dictionary_type=="edge_laplacian":
            Lk, _, _ = compute_Lk_and_lambdak(Ld, K)
        elif dictionary_type=="separated":
            Luk, _, _ = compute_Lk_and_lambdak(Lu, K, separated=True)
            Ldk, _, _ = compute_Lk_and_lambdak(Ld, K, separated=True)

        # Init the dictionary and the sparse representation 
        D_coll = [cp.Constant(D0[:,(n*i):(n*(i+1))]) for i in range(s)]
        Y = cp.Constant(Y_train)
        X_train = X0
        
        while pat_iter < patience and iter_ <= max_iter:
            
            # SDP Step
            # Init constants and parameters
            D_coll = [cp.Constant(np.zeros((n, n))) for i in range(s)] 
            Dsum = cp.Constant(np.zeros((n, n)))
            X = cp.Constant(X_train)
            I = cp.Constant(np.eye(n))
            
            # Define the objective function
            if dictionary_type in ["joint", "edge_laplacian"]:
                # Init the variables
                h = cp.Variable((s, K))
                hI = cp.Variable((s, 1))
                for i in range(0,s):
                    tmp =  cp.Constant(np.zeros((n, n)))
                    for j in range(0,K):
                        tmp += (cp.Constant(Lk[j, :, :]) * h[i,j])
                    tmp += (I*hI[i])
                    D_coll[i] = tmp
                    Dsum += tmp
                D = cp.hstack([D_coll[i]for i in range(s)])
                term1 = cp.square(cp.norm((Y - D @ X), 'fro'))
                term2 = cp.square(cp.norm(h, 'fro')*lambda_)
                term3 = cp.square(cp.norm(hI, 'fro')*lambda_)
                obj = cp.Minimize(term1 + term2 + term3)

            else:
                # Init the variables
                hI = cp.Variable((s, K))
                hS = cp.Variable((s, K))
                hH = cp.Variable((s, 1))
                for i in range(0,s):
                    tmp =  cp.Constant(np.zeros((n, n)))
                    for j in range(0,K):
                        tmp += ((cp.Constant(Luk[j, :, :])*hS[i,j]) + (cp.Constant(Ldk[j, :, :])*hI[i,j]))
                    tmp += (I*hH[i])
                    D_coll[i] = tmp
                    Dsum += tmp
                D = cp.hstack([D_coll[i]for i in range(s)])
                
                term1 = cp.square(cp.norm((Y - D @ X), 'fro'))
                term2 = cp.square(cp.norm(hI, 'fro')*lambda_)
                term3 = cp.square(cp.norm(hS, 'fro')*lambda_)
                term4 = cp.square(cp.norm(hH, 'fro')*lambda_)
                obj = cp.Minimize(term1 + term2 + term3 + term4)

            # Define the constraints
            constraints = [D_coll[i] >> 0 for i in range(s)] + \
                            [(cp.multiply(c, I) - D_coll[i]) >> 0 for i in range(s)] + \
                            [(Dsum - cp.multiply((c - epsilon), I)) >> 0, (cp.multiply((c + epsilon), I) - Dsum) >> 0]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.MOSEK, verbose=False)
            # Update the dictionary
            D = D.value

            # OMP Step
            dd = la.norm(D, axis=0)
            W = np.diag(1. / dd)
            Domp = D @ W
            X_train = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_train)
            X_test = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_test)
            # Normalize?
            X_train = W @ X_train
            X_test = W @ X_test

            # Error Updating
            error_train_norm = (1/m_train)* np.sum(la.norm(Y_train - (D @ X_train), axis=0)**2 /
                                    la.norm(Y_train, axis=0)**2)
            error_test_norm = (1/m_test)* np.sum(la.norm(Y_test - (D @ X_test), axis=0)**2 /
                                    la.norm(Y_test, axis=0)**2)

            # Error Storing
            if (error_train_norm < min_error_train_norm) and (abs(error_train_norm) > np.finfo(float).eps) and (abs(error_train_norm - min_error_train_norm) > tol):
                X_opt_train = X_train
                min_error_train_norm = error_train_norm

            if (error_test_norm < min_error_test_norm) and (abs(error_test_norm) > np.finfo(float).eps) and (abs(error_test_norm - min_error_test_norm) > tol):
                h_opt = h.value if dictionary_type in ["joint", "edge_laplacian"] else np.hstack([hI.value, hS.value, hH.value])
                D_opt = D
                X_opt_test = X_test
                min_error_test_norm = error_test_norm
                pat_iter = 0
                if verbose == 1:
                    print("New Best Test Error:", min_error_test_norm)
            else:
                pat_iter += 1

            iter_ += 1
    
    else:
        # Fourier Dictionary Benchmark
        L = Lu + Ld
        _, D_opt = sla.eigh(L)
        dd = la.norm(D_opt, axis=0)
        W = np.diag(1./dd)  
        D_opt = D_opt / la.norm(D_opt)
        Domp = D_opt@W
        X_opt_train = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp.real, col=x), axis=0, arr=Y_train)
        X_opt_test = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp.real, col=x), axis=0, arr=Y_test)
        X_opt_train = W @ X_opt_train
        X_opt_test = W @ X_opt_test
        # Error Updating
        min_error_train_norm = (1/m_train)* np.sum(la.norm(Y_train - (D_opt @ X_opt_train), axis=0)**2 /
                                la.norm(Y_train, axis=0)**2)
        min_error_test_norm = (1/m_test)* np.sum(la.norm(Y_test - (D_opt @ X_opt_test), axis=0)**2 /
                                la.norm(Y_test, axis=0)**2)
        h_opt = 0
        
    return min_error_train_norm, min_error_test_norm, h_opt, X_opt_test, X_opt_train