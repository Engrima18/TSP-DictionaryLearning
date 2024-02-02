import scipy.linalg as sla
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import data_preproc as dp

def initialize_dic(Lu, Ld, s, K, Y_train, K0, dictionary_type, c, epsilon, only):

    n = Lu.shape[0]
    D = np.zeros((n, n*s))
    X = np.zeros(Y_train.shape)
    X = np.tile(X, (s,1))
    discard = 0

    # maybe is better to create a wrapper
    def multiplier_search(*arrays, s=s):
        is_okay = 0
        mult = 100
        tries = 0
        while is_okay==0:
            is_okay = 1
            h, c_try, _, tmp_sum_min, tmp_sum_max = dp.generate_coeffs(arrays, s=s, mult=mult)
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
        # Joint Dictionary Model
        if dictionary_type == "joint":
            Lk, lambda_max_k, lambda_min_k = dp.compute_Lk_and_lambdak(Lu + Ld, K)
            h, discard = multiplier_search(lambda_max_k, lambda_min_k)
            D = dp.generate_dictionary(h, s, Lk)

        # Edge Laplacian Dictionary Model
        elif dictionary_type == "edge_laplacian":
            Lk, lambda_max_k, lambda_min_k = dp.compute_Lk_and_lambdak(Ld, K)
            h, discard = multiplier_search(lambda_max_k, lambda_min_k)
            D = dp.generate_dictionary(h, s, Lk)

        # Separated Dictionary Model
        elif dictionary_type == "separated":
            Luk, lambda_max_u_k, lambda_min_u_k = dp.compute_Lk_and_lambdak(Lu, K, separated=True)
            Ldk, lambda_max_d_k, lambda_min_d_k = dp.compute_Lk_and_lambdak(Ld, K, separated=True)
            h, discard = multiplier_search(lambda_max_d_k, lambda_min_d_k, lambda_max_u_k, lambda_min_u_k)
            D = dp.generate_dictionary(h, s, Luk, Ldk)
    
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
        X = np.apply_along_axis(lambda x: dp.get_omp_coeff(K0, Domp.real, x), axis=0, arr=Y_train)
        X = np.tile(X, (s,1))
        
    return D, X, discard


def topological_dictionary_learn(Y_train, Y_test, K, n, s, D0, X0, Lu, Ld, dictionary_type, c, epsilon, K0, lambda_=1e-3, max_iter=10, patience=10, tol=1e-7, verbose=0):

    # Define hyperparameters
    min_error_train_norm, min_error_test_norm = 1e20, 1e20
    m_test, m_train = Y_test.shape[1], Y_train.shape[1]
    iter_, pat_iter = 1, 0

    if dictionary_type != "fourier":
        if dictionary_type=="joint":
            Lk, _, _ = dp.compute_Lk_and_lambdak(Lu + Ld, K)
        elif dictionary_type=="edge_laplacian":
            Lk, _, _ = dp.compute_Lk_and_lambdak(Ld, K)
        elif dictionary_type=="separated":
            Luk, _, _ = dp.compute_Lk_and_lambdak(Lu, K, separated=True)
            Ldk, _, _ = dp.compute_Lk_and_lambdak(Ld, K, separated=True)

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
            X_train = np.apply_along_axis(lambda x: dp.get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_train)
            X_test = np.apply_along_axis(lambda x: dp.get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_test)
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
        X_opt_train = np.apply_along_axis(lambda x: dp.get_omp_coeff(K0, Domp=Domp.real, col=x), axis=0, arr=Y_train)
        X_opt_test = np.apply_along_axis(lambda x: dp.get_omp_coeff(K0, Domp=Domp.real, col=x), axis=0, arr=Y_test)
        X_opt_train = W @ X_opt_train
        X_opt_test = W @ X_opt_test
        # Error Updating
        min_error_train_norm = (1/m_train)* np.sum(la.norm(Y_train - (D_opt @ X_opt_train), axis=0)**2 /
                                la.norm(Y_train, axis=0)**2)
        min_error_test_norm = (1/m_test)* np.sum(la.norm(Y_test - (D_opt @ X_opt_test), axis=0)**2 /
                                la.norm(Y_test, axis=0)**2)
        h_opt = 0
        
    return min_error_train_norm, min_error_test_norm, h_opt, X_opt_test, X_opt_train