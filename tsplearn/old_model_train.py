import scipy.linalg as sla
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from .tsp_generation import *
from typing import Tuple


def sparse_transform(D, K0, Y_test, Y_train=None):

    dd = la.norm(D, axis=0)
    W = np.diag(1. / dd)
    Domp = D @ W
    X_test = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_test)
    # Normalization
    X_test = W @ X_test

    # Same for the training set
    if Y_train != None:
        X_train = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_train)
        X_train = W @ X_train

        return X_test, X_train
    
    return X_test


def nmse(D, X, Y, m):
    return (1/m)* np.sum(la.norm(Y - (D @ X), axis=0)**2 /la.norm(Y, axis=0)**2)


def _multiplier_search(*arrays, P, c, epsilon):
    is_okay = 0
    mult = 100
    tries = 0
    while is_okay==0:
        is_okay = 1
        h, c_try, _, tmp_sum_min, tmp_sum_max = generate_coeffs(arrays, P=P, mult=mult)
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


def init_dict(Lu: np.ndarray,
              Ld: np.ndarray, 
              P: int, 
              J: int, 
              Y_train: np.ndarray, 
              K0: int,
              dictionary_type: str, 
              c: float, 
              epsilon: float,
              h: np.ndarray = None, 
              only: str = "only_X") -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize the dictionary and the signal sparse representation for the alternating
    optimization algorithm.

    Args:
        Lu (np.ndarray): Upper Laplacian matrix
        Ld (np.ndarray): Lower Laplacian matrix
        P (int): Number of kernels (sub-dictionaries).
        J (int): Max order of the polynomial for the single sub-dictionary.
        Y_train (np.ndarray): Training data.
        K0 (int): Sparsity of the signal representation.
        dictionary_type (str): Type of dictionary.
        c (float): Boundary constant from the synthetic data generation process.
        epsilon (float): Boundary constant from the synthetic data generation process.
        only (str): Type of initialization. Can be one of: "only_X", "all", "only_D".

    Returns:
        Tuple[np.ndarray, np.ndarray, bool]: Initialized dictionary, initialized sparse representation, and discard flag value.
    """

    M = Lu.shape[0]
    D = np.zeros((M, M*P))
    X = np.zeros(Y_train.shape)
    X = np.tile(X, (P,1))
    discard = 1

    if (only == "only_D") or (only == "all"):
        
        while discard==1:

            if dictionary_type == "joint":
                Lj, lambda_max_j, lambda_min_j = compute_Lj_and_lambdaj(Lu + Ld, J)
                h, discard = _multiplier_search(lambda_max_j, lambda_min_j, P=P, c=c, epsilon=epsilon)
                D = generate_dictionary(h, P, Lj)

            elif dictionary_type == "edge_laplacian":
                Lj, lambda_max_j, lambda_min_j = compute_Lj_and_lambdaj(Ld, J)
                h, discard = _multiplier_search(lambda_max_j, lambda_min_j, P=P, c=c, epsilon=epsilon)
                D = generate_dictionary(h, P, Lj)

            elif dictionary_type == "separated":
                Luj, lambda_max_u_j, lambda_min_u_j = compute_Lj_and_lambdaj(Lu, J, separated=True)
                Ldj, lambda_max_d_j, lambda_min_d_j = compute_Lj_and_lambdaj(Ld, J, separated=True)
                h, discard = _multiplier_search(lambda_max_d_j, lambda_min_d_j, lambda_max_u_j, lambda_min_u_j, P=P, c=c, epsilon=epsilon)
                D = generate_dictionary(h, P, Luj, Ldj)
    
    if (only == "only_X" or only == "all"):
        
        if dictionary_type == "edge_laplacian":
            L = Ld
        else:
            L = Lu+Ld

        # If no prior info on the dictionary simply do the Fourier transform
        if h==None:
            _, Dx = sla.eig(L)
            dd = la.norm(Dx, axis=0)
            W = np.diag(1./dd)
            Dx = Dx / la.norm(Dx)  
            Domp = Dx@W
            X = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp.real, x), axis=0, arr=Y_train)
            X = np.tile(X, (P,1))
        # Otherwise use prior info about the dictionary to initialize the sparse representation
        else:
            if dictionary_type == "separated":
                Luj = np.array([la.matrix_power(Lu, i) for i in range(1, J + 1)])
                Ldj = np.array([la.matrix_power(Ld, i) for i in range(1, J + 1)])
                D = generate_dictionary(h, P, Luj, Ldj)
                X = sparse_transform(D, K0, Y_train)
            else: 
                Lj = np.array([la.matrix_power(L, i) for i in range(1, J + 1)])
                D = generate_dictionary(h, P, Lj)
                X = sparse_transform(D, K0, Y_train)                
        
    return D, X


def topological_dictionary_learn(Y_train: np.ndarray,
                                 Y_test: np.ndarray, 
                                 J: int, 
                                 M: int, 
                                 P: int,
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
        J (int): Max order of the polynomial for the single sub-dictionary.
        M (int): Number  of nodes ((k-1) order complex) in the data graph.
        P (int): Number of kernels (sub-dictionaries).
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
    min_error_train, min_error_test = 1e20, 1e20
    m_test, m_train = Y_test.shape[1], Y_train.shape[1]
    iter_, pat_iter = 1, 0

    if dictionary_type != "fourier":
        if dictionary_type=="joint":
            Lj, _, _ = compute_Lj_and_lambdaj(Lu + Ld, J)
        elif dictionary_type=="edge_laplacian":
            Lj, _, _ = compute_Lj_and_lambdaj(Ld, J)
        elif dictionary_type=="separated":
            Luj, _, _ = compute_Lj_and_lambdaj(Lu, J, separated=True)
            Ldj, _, _ = compute_Lj_and_lambdaj(Ld, J, separated=True)

        # Init the dictionary and the sparse representation 
        D_coll = [cp.Constant(D0[:,(M*i):(M*(i+1))]) for i in range(P)]
        Y = cp.Constant(Y_train)
        X_train = X0
        
        while pat_iter < patience and iter_ <= max_iter:
            
            # SDP Step
            # Init constants and parameters
            D_coll = [cp.Constant(np.zeros((M, M))) for i in range(P)] 
            Dsum = cp.Constant(np.zeros((M, M)))
            X = cp.Constant(X_train)
            I = cp.Constant(np.eye(M))
            
            # Define the objective function
            if dictionary_type != "separated":
                # Init the variables
                h = cp.Variable((P, J))
                hI = cp.Variable((P, 1))
                for i in range(0,P):
                    tmp =  cp.Constant(np.zeros((M, M)))
                    for j in range(0,J):
                        tmp += (cp.Constant(Lj[j, :, :]) * h[i,j])
                    tmp += (I*hI[i])
                    D_coll[i] = tmp
                    Dsum += tmp
                D = cp.hstack([D_coll[i]for i in range(P)])
                term1 = cp.square(cp.norm((Y - D @ X), 'fro'))
                term2 = cp.square(cp.norm(h, 'fro')*lambda_)
                term3 = cp.square(cp.norm(hI, 'fro')*lambda_)
                obj = cp.Minimize(term1 + term2 + term3)

            else:
                # Init the variables
                hI = cp.Variable((P, J))
                hS = cp.Variable((P, J))
                hH = cp.Variable((P, 1))
                for i in range(0,P):
                    tmp =  cp.Constant(np.zeros((M, M)))
                    for j in range(0,J):
                        tmp += ((cp.Constant(Luj[j, :, :])*hS[i,j]) + (cp.Constant(Ldj[j, :, :])*hI[i,j]))
                    tmp += (I*hH[i])
                    D_coll[i] = tmp
                    Dsum += tmp
                D = cp.hstack([D_coll[i]for i in range(P)])
                
                term1 = cp.square(cp.norm((Y - D @ X), 'fro'))
                term2 = cp.square(cp.norm(hI, 'fro')*lambda_)
                term3 = cp.square(cp.norm(hS, 'fro')*lambda_)
                term4 = cp.square(cp.norm(hH, 'fro')*lambda_)
                obj = cp.Minimize(term1 + term2 + term3 + term4)

            # Define the constraints
            constraints = [D_coll[i] >> 0 for i in range(P)] + \
                            [(cp.multiply(c, I) - D_coll[i]) >> 0 for i in range(P)] + \
                            [(Dsum - cp.multiply((c - epsilon), I)) >> 0, (cp.multiply((c + epsilon), I) - Dsum) >> 0]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.MOSEK, verbose=False)
            # Update the dictionary
            D = D.value

            # OMP Step

            # X_train,X_test = sparse_transform(D, K0, Y_test, Y_train)
            dd = la.norm(D, axis=0)
            W = np.diag(1. / dd)
            Domp = D @ W
            X_train = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_train)
            X_test = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_test)
            # Normalize?
            X_train = W @ X_train
            X_test = W @ X_test

            # Error Updating
            error_train = (1/m_train)* np.sum(la.norm(Y_train - (D @ X_train), axis=0)**2 /
                                    la.norm(Y_train, axis=0)**2)
            error_test = (1/m_test)* np.sum(la.norm(Y_test - (D @ X_test), axis=0)**2 /
                                    la.norm(Y_test, axis=0)**2)

            # Error Storing
            if (error_train < min_error_train) and (abs(error_train) > np.finfo(float).eps) and (abs(error_train - min_error_train) > tol):
                X_opt_train = X_train
                min_error_train = error_train

            if (error_test < min_error_test) and (abs(error_test) > np.finfo(float).eps) and (abs(error_test - min_error_test) > tol):
                h_opt = h.value if dictionary_type in ["joint", "edge_laplacian"] else np.hstack([hI.value, hS.value, hH.value])
                D_opt = D
                X_opt_test = X_test
                min_error_test = error_test
                pat_iter = 0
                if verbose == 1:
                    print("New Best Test Error:", min_error_test)
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
        min_error_train = (1/m_train)* np.sum(la.norm(Y_train - (D_opt @ X_opt_train), axis=0)**2 /
                                la.norm(Y_train, axis=0)**2)
        min_error_test = (1/m_test)* np.sum(la.norm(Y_test - (D_opt @ X_opt_test), axis=0)**2 /
                                la.norm(Y_test, axis=0)**2)
        h_opt = 0
        
    return min_error_train, min_error_test, h_opt, X_opt_test, X_opt_train, D_opt, Ldj, hist


