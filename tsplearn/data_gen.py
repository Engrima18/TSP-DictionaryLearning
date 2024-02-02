import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.linear_model import OrthogonalMatchingPursuit


def compute_Lk_and_lambdak(L, K, separated=False):
    lambdas, _ = eigs(L)
    lambdas[np.abs(lambdas) < np.finfo(float).eps] = 0
    lambda_max = np.max(lambdas).real
    lambda_min = np.min(lambdas).real
    Lk = np.array([la.matrix_power(L, i) for i in range(1, K + 1)])
    # for the "separated" implementation we need a different dimensionality
    if separated:
        lambda_max_k = lambda_max ** np.arange(1, K + 1)
        lambda_min_k = lambda_min ** np.arange(1, K + 1)
    else:
        lambda_max_k = lambda_max ** np.array(list(np.arange(1, K + 1))+[0])
        lambda_min_k = lambda_min ** np.array(list(np.arange(1, K + 1))+[0])
    return Lk, lambda_max_k, lambda_min_k


def generate_coeffs(*arrays, s, mult=10):
    """ 
    Select ad hoc parameters for synthetic data generation, randomly over
    an interval dependent on the max eigenvalues of the Laplacian(s)
    """

    # if passing four arguments (two for upper and two for lower laplacian eigevals)
    # it means that you are using dictionary_type="separated"
    if len(arrays)==2:
        lambda_max_k, lambda_min_k = arrays
        K = lambda_max_k.shape[0]
        h = mult / np.max(lambda_max_k) * np.random.rand(s, K)
        # For later sanity check in optimization phase 
        tmp_max_vec = h @ lambda_max_k # parallelize the code with simple matrix multiplications
        tmp_min_vec = h @ lambda_min_k
        c = np.max(tmp_max_vec)
        tmp_sum_max = np.sum(tmp_max_vec)
        tmp_sum_min = np.sum(tmp_min_vec)

        Delta_min = c - tmp_sum_min
        Delta_max = tmp_sum_max - c
        epsilon = (Delta_max - Delta_min) * np.random.rand() + Delta_min

    elif len(arrays)==4:
        lambda_max_u_k, lambda_min_u_k, lambda_max_d_k, lambda_min_d_k = arrays
        K = lambda_max_u_k.shape[0]
        hI = mult / np.max(lambda_max_d_k) * np.random.rand(s, K)
        hS = mult / np.max(lambda_max_u_k) * np.random.rand(s, K)
        hH = mult / np.min([np.max(lambda_max_u_k), np.max(lambda_max_d_k)]) * np.random.rand(s, 1)
        h = [hS, hI, hH]
        tmp_max_vec_S = (hS @ lambda_max_u_k).reshape(s,1)
        tmp_min_vec_S = (hS @ lambda_min_u_k).reshape(s,1)
        tmp_max_vec_I = (hI @ lambda_max_d_k).reshape(s,1)
        tmp_min_vec_I = (hI @ lambda_min_d_k).reshape(s,1)
        c = np.max(tmp_max_vec_I + tmp_max_vec_S + hH)
        tmp_sum_min = np.sum(tmp_min_vec_I + tmp_min_vec_S + hH)
        tmp_sum_max = np.sum(tmp_max_vec_I + tmp_max_vec_S + hH)
        Delta_min = c - tmp_sum_min
        Delta_max = tmp_sum_max - c
        epsilon = np.max([Delta_min, Delta_max])
    else:
        raise ValueError("Function accepts either 2 or 4 arrays! In case of 4 arrays are provided,\
                        the first 2 refer to upper laplacian and the other two to lower laplacian.")
    return h, c, epsilon, tmp_sum_min, tmp_sum_max


def generate_dictionary(h, s, *matrices):
    D = []
    # Check if upper and lower Laplacians are separately provided
    if len(matrices)==1:
        Lk = matrices[0]
        n = Lk.shape[-1]
        k = Lk.shape[0]

        for i in range(0,s):
            h_tmp = h[i,:-1].reshape(k,1,1)
            tmp = np.sum(h_tmp*Lk, axis=0) + h[i,-1]*np.eye(n,n)
            D.append(tmp)
    elif len(matrices)==2:
        Luk , Ldk = matrices
        n = Luk.shape[-1]
        k = Luk.shape[0]

        for i in range(0,s):
            hu = h[0][i].reshape(k,1,1)
            hd = h[1][i].reshape(k,1,1)
            hid = h[2][i]
            tmp = np.sum(hu*Luk + hd*Ldk, axis=0) + hid*np.eye(n,n)
            D.append(tmp)
    else:
        raise ValueError("Function accepts one vector and either 1 or 2 matrices.")
    D = np.hstack(tuple(D))
    return D


def create_ground_truth(Lu, Ld, m_train, m_test, s, K, K0, dictionary_type, sparsity_mode):

    if dictionary_type == "joint":
        Lk, lambda_max_k, lambda_min_k = compute_Lk_and_lambdak(Lu + Ld, K)
        h, c, epsilon, _, _ = generate_coeffs(lambda_max_k, lambda_min_k, s=s)
        D = generate_dictionary(h, s, Lk)

    elif dictionary_type == "edge_laplacian":
        Lk, lambda_max_k, lambda_min_k = compute_Lk_and_lambdak(Ld, K)
        h, c, epsilon, _, _ = generate_coeffs(lambda_max_k, lambda_min_k, s=s)
        D = generate_dictionary(h, s, Lk)

    elif dictionary_type == "separated":
        Luk, lambda_max_u_k, lambda_min_u_k = compute_Lk_and_lambdak(Lu, K, separated=True)
        Ldk, lambda_max_d_k, lambda_min_d_k = compute_Lk_and_lambdak(Ld, K, separated=True)
        h, c, epsilon, _, _ = generate_coeffs(lambda_max_u_k, lambda_min_u_k, lambda_max_d_k, lambda_min_d_k, s=s)
        D = generate_dictionary(h, s, Luk, Ldk)

    n = D.shape[0]

    # Signal Generation
    def create_column_vec(row,n, s):
        tmp = np.zeros(n*s)
        tmp[row['idxs']]=row['non_zero_coeff']
        return tmp

    m_total = m_train + m_test
    tmp = pd.DataFrame()

    if sparsity_mode == "max":
        tmp_K0 = np.random.choice(np.arange(1,K0+1), size=(m_total), replace=True)
    else:
        tmp_K0 = np.full((m_total,), K0)
    # sparsity coefficient for each column
    tmp['K0'] = tmp_K0
    # for each column get K0 indexes
    tmp['idxs'] = tmp.K0.apply(lambda x: np.random.choice(n*s, x, replace=False))
    # for each of the K0 row indexes in each column, sample K0 values
    tmp['non_zero_coeff'] = tmp.K0.apply(lambda x: np.random.randn(x))
    # create the column vectors with the desired characteristics
    tmp['column_vec'] = tmp.apply(lambda x: create_column_vec(x,n=n, s=s), axis=1)
    # finally derive the sparse signal representation matrix
    X = np.column_stack(tmp['column_vec'].values)

    all_data = D @ X
    X_train = X[:, :m_train]
    X_test = X[:, m_train:]
    train_Y = all_data[:, :m_train]
    test_Y = all_data[:, m_train:]

    return D, h, train_Y, test_Y, epsilon, c, X_train, X_test


def get_omp_coeff(K0, Domp, col):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K0)
    omp.fit(Domp, col)
    return omp.coef_

def verify_dic(D, Y_train, X_train_true, K0_max, acc_thresh):
    # OMP
    dd = la.norm(D, axis=0)
    W = np.diag(1. / dd)  
    Domp = D @ W
    for K0 in range(1, K0_max+1):
        idx = np.sum(np.abs(X_train_true) > 0, axis=0) == K0 
        try:
            tmp_train = Y_train[:, idx]
            X_true_tmp = X_train_true[:, idx]
            idx_group = np.abs(X_true_tmp) > 0
            X_tr = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp.real, x), axis=0, arr=tmp_train)
            idx_train = np.abs(X_tr) > 0
            acc = np.sum(np.sum(idx_group == idx_train, axis=0) == idx_group.shape[0])/idx_group.shape[1]
            if acc < acc_thresh:
                fin_acc = acc
                break
            else:
                fin_acc = acc
        except:
            fin_acc=0
    max_possible_sparsity = K0 - 1
    return max_possible_sparsity, fin_acc