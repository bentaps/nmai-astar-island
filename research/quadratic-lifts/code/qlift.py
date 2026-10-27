"""Small toolkit for experiments with quadratic lifts (MQAV-type) and Kahan's method."""
import numpy as np
from itertools import combinations_with_replacement


def kahan_step(F, JF, z, h):
    """Kahan's method for a QUADRATIC vector field F: z' = z + h (I - h/2 F'(z))^{-1} F(z)."""
    A = np.eye(len(z)) - 0.5 * h * JF(z)
    return z + h * np.linalg.solve(A, F(z))


def midpoint_step(F, JF, z, h, tol=1e-15, maxit=60):
    """Implicit midpoint rule, Newton iteration."""
    zn = z + h * F(z)
    for _ in range(maxit):
        zm = 0.5 * (z + zn)
        G = zn - z - h * F(zm)
        Jg = np.eye(len(z)) - 0.5 * h * JF(zm)
        dz = np.linalg.solve(Jg, -G)
        zn = zn + dz
        if np.max(np.abs(dz)) < tol * (1 + np.max(np.abs(zn))):
            break
    return zn


def run(step, z0, n, *args):
    Z = np.empty((n + 1, len(z0)))
    Z[0] = z0
    z = np.array(z0, float)
    for k in range(n):
        z = step(z, *args)
        Z[k + 1] = z
    return Z


def monomial_matrix(X, deg):
    """Rows: points; columns: all monomials of total degree <= deg in the columns of X."""
    N, d = X.shape
    cols, names = [np.ones(N)], [()]
    for k in range(1, deg + 1):
        for idx in combinations_with_replacement(range(d), k):
            cols.append(np.prod(X[:, idx], axis=1))
            names.append(idx)
    return np.array(cols).T, names


def algebraic_relation_spectrum(X, deg):
    """Singular values (normalised) of the monomial matrix: a sharp drop to ~1e-13 means
    the points lie EXACTLY on an algebraic variety of degree <= deg."""
    A, names = monomial_matrix(X, deg)
    A = A / np.linalg.norm(A, axis=0)
    s = np.linalg.svd(A, compute_uv=False)
    return s / s[0], names
