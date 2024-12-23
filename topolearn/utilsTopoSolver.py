import scipy.linalg as sla
import numpy as np
import pandas as pd
import numpy.linalg as la
import cvxpy as cp
from .data_generation import get_omp_coeff


def indicator_matrix(row):
    tmp = row.sigma.copy()
    tmp[row.idx] = 0
    return np.diag(tmp)


def indicator_matrix_rev(row):
    tmp = row.sigma.copy()
    tmp[row.idx] = 1
    return np.diag(tmp)


def compute_Luj(row, b2, J):
    Lu = b2 @ row.sigma @ b2.T
    Luj = np.array([la.matrix_power(Lu, i) for i in range(1, J + 1)])
    return Luj


def split_coeffs(h, s, k, sep=False):
    h_tmp = h.value.flatten()
    # hH = h_tmp[:s,].reshape((s,1))
    # hS = h_tmp[s:s*(k+1),].reshape((s,k))
    # hI = h_tmp[s*(k+1):,].reshape((s,k))
    if sep:
        if s != 1 and k != 1:
            hH = h_tmp[np.arange(0, (s * (2 * k + 1)), (2 * k + 1))].reshape((s, 1))
            hS = h_tmp[
                np.hstack(
                    [[i, i + 1] for i in range(1, (s * (2 * k + 1)), (2 * k + 1))]
                )
            ].reshape((s, k))
            hI = h_tmp[
                np.hstack(
                    [[i, i + 1] for i in range((k + 1), (s * (2 * k + 1)), (2 * k + 1))]
                )
            ].reshape((s, k))
        else:
            hH = h_tmp[np.arange(0, (s * (2 * k)), (2 * k))]
            hS = h_tmp[
                np.hstack([[i, i + 1] for i in range(1, (s * (2 * k)), (2 * k))])
            ]
            hI = h_tmp[
                np.hstack([[i, i + 1] for i in range((k), (s * (2 * k)), (2 * k))])
            ]
        return [hH, hS, hI]
    if s != 1 and k != 1:
        hi = h_tmp[np.arange(0, (s * (k + 1)), (k + 1))].reshape((s, 1))
        h = h_tmp[
            np.hstack([[i, i + 1] for i in range(1, (s * (k + 1)), (k + 1))])
        ].reshape((s, k))
    else:
        hi = h_tmp[np.arange(0, (s * k), k)]
        h = h_tmp[np.hstack([[i, i + 1] for i in range(1, (s * k), k)])]
    return np.hstack([h, hi])


def rearrange_coeffs(h, J, P, sep=False):
    h_tmp = h.value.flatten()
    if sep:
        base_idxs = np.arange(0, (P * (2 * J + 1)), (2 * J + 1))
        hH = h_tmp[base_idxs].reshape((P, 1))
        hS = h_tmp[
            np.hstack([np.arange((i + 1), (i + 1 + J)) for i in base_idxs])
        ].reshape((P, J))
        hI = h_tmp[
            np.hstack([np.arange((i + 1 + J), (i + 1 + 2 * J)) for i in base_idxs])
        ].reshape((P, J))
        return [hH, hS, hI]
    base_idxs = np.arange(0, (P * (J + 1)), (J + 1))
    hi = h_tmp[base_idxs].reshape((P, 1))
    h = h_tmp[np.hstack([np.arange((i + 1), (i + 1 + J)) for i in base_idxs])].reshape(
        (P, J)
    )
    return np.hstack([h, hi])


def sparse_transform(D, K0, Y_te, Y_tr=None, fit_intercept=True):

    ep = np.finfo(float).eps  # to avoid some underflow problems
    dd = la.norm(D, axis=0) + ep
    W = np.diag(1.0 / dd)
    Domp = D @ W
    X_te = np.apply_along_axis(
        lambda x: get_omp_coeff(K0, Domp=Domp, col=x, fit_intercept=fit_intercept),
        axis=0,
        arr=Y_te,
    )
    # Normalization
    X_te = W @ X_te

    if np.all(Y_tr == None):

        return X_te

    # Same for the training set
    X_tr = np.apply_along_axis(
        lambda x: get_omp_coeff(K0, Domp=Domp, col=x, fit_intercept=fit_intercept),
        axis=0,
        arr=Y_tr,
    )
    X_tr = W @ X_tr

    return X_te, X_tr


def compute_vandermonde(L, k):

    def polynomial_exp(x, k):
        x = x ** np.arange(0, k + 1)
        return x

    eigenvalues, _ = sla.eig(L)
    idx = eigenvalues.argsort()
    tmp_df = pd.DataFrame({"Eigs": eigenvalues[idx]})
    tmp_df["Poly"] = tmp_df["Eigs"].apply(lambda x: polynomial_exp(x, k))
    B = np.vstack(tmp_df["Poly"].to_numpy())

    return B


def nmse(D, X, Y, m):
    return (1 / m) * np.sum(la.norm(Y - (D @ X), axis=0) ** 2 / la.norm(Y, axis=0) ** 2)
