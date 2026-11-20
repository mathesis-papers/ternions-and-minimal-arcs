"""Shared numerics: width-three bridge F = sum_r M_r (x) N_r and the residual F^*F - I."""
import numpy as np
from scipy.optimize import least_squares

K = 3


def build_F(M, N):
    return sum(np.kron(M[r], N[r]) for r in range(M.shape[0]))


def unpack(x, a, b, c, d, k=K):
    nm, nn = k * a * b, k * d * c
    z = x[: nm + nn] + 1j * x[nm + nn:]
    return z[:nm].reshape(k, a, b), z[nm:].reshape(k, d, c)


def pack(M, N):
    z = np.concatenate([M.ravel(), N.ravel()])
    return np.concatenate([z.real, z.imag])


def residual_and_jac(x, a, b, c, d, k=K):
    M, N = unpack(x, a, b, c, d, k)
    F = build_F(M, N)
    R = F.conj().T @ F - np.eye(b * c)
    res = np.concatenate([R.real.ravel(), R.imag.ravel()])
    nm, nn = k * a * b, k * d * c
    ncomp = nm + nn
    J = np.empty((res.size, 2 * ncomp))
    # derivative of F with respect to each complex coordinate (holomorphic direction)
    col = 0
    dFs = []
    for r in range(k):
        for i in range(a):
            for j in range(b):
                E = np.zeros((a, b)); E[i, j] = 1.0
                dFs.append(np.kron(E, N[r]))
    for r in range(k):
        for p in range(d):
            for q in range(c):
                E = np.zeros((d, c)); E[p, q] = 1.0
                dFs.append(np.kron(M[r], E))
    FH = F.conj().T
    for idx, dF in enumerate(dFs):
        for part, scale in ((0, 1.0), (1, 1j)):
            dFp = scale * dF
            dR = dFp.conj().T @ F + FH @ dFp
            J[:, idx + part * ncomp] = np.concatenate([dR.real.ravel(), dR.imag.ravel()])
    return res, J


def solve(a, b, c, d, rng, k=K, x0=None, max_nfev=400):
    n = 2 * k * (a * b + d * c)
    if x0 is None:
        x0 = rng.standard_normal(n) * (1.0 / np.sqrt(2 * k * max(1, a * d))) ** 0.5
    cache = {}

    def f(x):
        r, J = residual_and_jac(x, a, b, c, d, k)
        cache['J'] = (x.copy(), J)
        return r

    def jac(x):
        xs, J = cache['J']
        if np.array_equal(xs, x):
            return J
        return residual_and_jac(x, a, b, c, d, k)[1]

    sol = least_squares(f, x0, jac=jac, method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=max_nfev)
    M, N = unpack(sol.x, a, b, c, d, k)
    F = build_F(M, N)
    err = np.linalg.norm(F.conj().T @ F - np.eye(b * c))
    return err, M, N
