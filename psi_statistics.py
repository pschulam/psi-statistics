from functools import partial

import autograd.numpy as np

from autograd.numpy import cholesky
from autograd.scipy import solve_triangular


solve_tril = partial(solve_triangular, lower=True)
ln_det_cholesky = lambda L: 2.0 * np.sum(np.log(np.diag(L)))


def rbf(X, Z, a, w):
    X = X[:, None, :]
    Z = Z[None, :, :]
    D = np.sum(np.square((X - Z) / w), axis=-1)
    K = a**2 * np.exp(-0.5 * D)
    return K


def rbf_psi0(m_x, C_x, a, w):
    return a**2


def rbf_psi1(m_x, C_x, Z, a, w):
    c0 = np.power(2.0 * np.pi, 0.5 * len(m_x)) * np.prod(w)
    return a**2 * c0 * mvn_pdf(Z, m_x, C_x + np.diag(w**2))


def rbf_psi2(m_x, C_x, Z, a, w):
    n, d = Z.shape
    c0 = np.power(2.0 * np.pi, d) * np.prod(w)**2
    B = np.kron(np.ones((2, 1)), np.eye(d))
    S = np.kron(np.ones((2, 2)), C_x) + np.kron(np.eye(2), np.diag(w**2))
    return a**2 * c0 * mvn_pdf(stack_pairs(Z), np.dot(B, m_x), S)


def mvn_pdf(X, m, C):
    """Compute the multivariate normal density function.

    This is an efficient vectorized implementation of the multivariate normal
    density. It assumes that the observed random variables run along the last
    axis of the array. The two most common examples are a single vector (1D
    array) and a matrix (2D array) where the vectors are packed into each row.
    The function will accept higher order arrays as well.

    """
    *shape, d = X.shape
    L = cholesky(C)
    X = X.reshape([-1, d]) - m
    A = solve_tril(L, X.T).T
    ln_Z = 0.5 * (d * np.log(2.0 * np.pi) + ln_det_cholesky(L))
    ln_E = 0.5 * np.sum(A**2, axis=-1)
    return np.exp(-ln_E - ln_Z).reshape(shape)


def stack_pairs(X):
    """Concatenate all pairs of vectors in the rows of a matrix."""
    X1 = np.kron([1, 0], X)
    X2 = np.kron([0, 1], X)
    return X1[:, None, :] + X2[None, :, :]
