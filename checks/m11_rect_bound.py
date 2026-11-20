"""Numerical checks for the hand proof of the first flat-spectrum obstruction
(Proposition m11:flat-obstruction): F^*F = lambda I, 3a < 2b  ==>  d >= 2c.

Usage: python3 check.py [--starts N] [--workers W]
Sections
  A  lemma tests on random instances (every step of the hand proof)
  B  full proof pipeline run on exact solutions found by optimisation
  C  counterexample search for 3a < 2b, d < 2c, and solvable controls
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import polar, schur, sqrtm, null_space

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m11_rect_flatcore import build_F, solve  # noqa: E402

TOL = 1e-8
SIG = np.diag([1.0, -1.0, 0.0]).astype(complex)
FAIL = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL.append(name)
    print(f"  [{tag}] {name} {detail}")


def crandn(rng, *shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def rand_unitary(rng, n):
    q, r = np.linalg.qr(crandn(rng, n, n))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def mix(B, X):
    return np.einsum('rs,sij->rij', B, X)


def phi(X, T):
    """Phi_X(T) = sum_{r,s} T_rs X_r^* X_s = S^*(T (x) I_a) S."""
    return np.einsum('rs,rji,sjk->ik', T, X.conj(), X)


def stack(X):
    k, a, b = X.shape
    return X.reshape(k * a, b)


def rank(A, tol=1e-8):
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())


def herm_basis(k):
    out = []
    for r in range(k):
        E = np.zeros((k, k), complex); E[r, r] = 1; out.append(E)
    for r in range(k):
        for s in range(r + 1, k):
            E = np.zeros((k, k), complex); E[r, s] = E[s, r] = 1; out.append(E)
            E = np.zeros((k, k), complex); E[r, s] = 1j; E[s, r] = -1j; out.append(E)
    return out


def hermitian_kernel(X, tol=1e-9):
    """Real basis of K_h = {T Hermitian : Phi_X(T) = 0}."""
    k = X.shape[0]
    basis = herm_basis(k)
    cols = []
    for E in basis:
        P = phi(X, E)
        cols.append(np.concatenate([P.real.ravel(), P.imag.ravel()]))
    A = np.array(cols).T
    scale = max(1.0, np.linalg.norm(A, 2))
    u, s, vh = np.linalg.svd(A)
    null = vh[(np.concatenate([s, np.zeros(vh.shape[0] - s.size)]) <= tol * scale)]
    return [sum(c * E for c, E in zip(v, basis)) for v in null]


def complex_kernel(X, tol=1e-9):
    k = X.shape[0]
    cols = []
    for r in range(k):
        for s in range(k):
            E = np.zeros((k, k), complex); E[r, s] = 1
            cols.append(phi(X, E).ravel())
    A = np.array(cols).T
    ns = null_space(A, rcond=tol)
    return [ns[:, j].reshape(k, k) for j in range(ns.shape[1])]


# ---------------------------------------------------------------------------
# The common neutral plane (Lemma rect-plane), implemented step by step as in the proof.
# ---------------------------------------------------------------------------
def common_plane(X, Kh, log=None, tol=1e-7):
    """Return (V, info): V 3x2 with orthonormal columns spanning a plane L with
    u^* T v = 0 for u, v in L and every T in span_R(Kh)."""
    info = {}
    a, b = X.shape[1], X.shape[2]
    Kh = [T for T in Kh if np.linalg.norm(T) > tol]
    if not Kh:
        info['case'] = 'K_h=0'
        return np.eye(3, 2, dtype=complex), info
    T = max(Kh, key=np.linalg.norm)
    T = T / np.linalg.norm(T)
    lam, W = np.linalg.eigh(T)
    ip, ineg = int(np.argmax(lam)), int(np.argmin(lam))
    iz = 3 - ip - ineg
    info['eigs'] = lam
    info['sig_ok'] = lam[ip] > tol and lam[ineg] < -tol and abs(lam[iz]) < tol
    B = W[:, [ip, ineg, iz]] @ np.diag([1 / np.sqrt(lam[ip]), 1 / np.sqrt(-lam[ineg]), 1.0])
    info['congruence_err'] = np.linalg.norm(B.conj().T @ T @ B - SIG)
    A = np.linalg.inv(B)
    Y = mix(A, X)
    info['gram_err'] = np.linalg.norm(Y[0].conj().T @ Y[0] - Y[1].conj().T @ Y[1])
    U = polar(Y[1] @ Y[0].conj().T)[0]
    info['unitary_err'] = np.linalg.norm(Y[1] - U @ Y[0]) + np.linalg.norm(U.conj().T @ U - np.eye(a))
    KY = [B.conj().T @ Q @ B for Q in Kh]
    info['KY_phi_err'] = max(np.linalg.norm(phi(Y, Q)) for Q in KY)
    info['T22_max'] = max(abs(Q[2, 2]) for Q in KY)
    rho = rank(Y[0])
    kerY0 = null_space(Y[0], rcond=1e-10)
    Rsp = Y[2] @ kerY0
    dimR = rank(Rsp)
    info['rank_X0'], info['dimR'] = rho, dimR
    info['dim_identity'] = (rho + dimR == b)
    info['half_rank'] = (a < 2 * rho)
    cross = max(max(abs(Q[2, 0]), abs(Q[2, 1])) for Q in KY)
    if cross <= tol:
        info['case'] = 'B1'
        Tf, Z = schur(U, output='complex')
        dvec = np.diag(Tf)
        Zm = np.array([[Q[0, 0] + Q[1, 1] + Q[0, 1] * dj + Q[1, 0] * np.conj(dj) for dj in dvec] for Q in KY])
        j = int(np.argmin(np.max(np.abs(Zm), axis=0)))
        info['z_min'] = float(np.max(np.abs(Zm[:, j])))
        w = dvec[j]
    else:
        info['case'] = 'B2'
        Q = max(KY, key=lambda Q: abs(Q[2, 0]) + abs(Q[2, 1]))
        BT = Q[2, 0] * np.eye(a) + Q[2, 1] * U
        Dbasis = np.linalg.svd(Y[0])[0][:, :rho]
        Rbasis = np.linalg.svd(Rsp)[0][:, :dimR]
        info['pairing_err'] = np.linalg.norm(Rbasis.conj().T @ BT @ Dbasis)
        sysm = np.block([[Dbasis, -Rbasis], [BT @ Dbasis, np.zeros((a, dimR))]])
        ns = null_space(sysm, rcond=1e-9)
        info['intersection_dim'] = ns.shape[1]
        info['intersection_lower_bound'] = rho + 2 * dimR - 2 * a
        v = Dbasis @ ns[:rho, 0]
        v = v / np.linalg.norm(v)
        w = -Q[2, 0] / Q[2, 1]
        info['eigvec_err'] = np.linalg.norm(U @ v - w * v)
        info['cross_err'] = max(abs(P[2, 0] + w * P[2, 1]) for P in KY)
    LY = np.array([[0, 1], [0, w], [1, 0]], dtype=complex)
    info['neutral_Y'] = max(np.linalg.norm(LY.conj().T @ Q @ LY) for Q in KY)
    V, _ = np.linalg.qr(B @ LY)
    info['neutral_X'] = max(np.linalg.norm(V.conj().T @ Q @ V) for Q in Kh)
    return V, info


# ---------------------------------------------------------------------------
# Section A: lemma tests on random instances
# ---------------------------------------------------------------------------
def section_A(rng):
    print("== Section A: lemma tests on random instances ==")
    # A1 rank/dimension lemma: if Phi(T)=0 and S injective then 2b + a rk T <= 2ka.
    # Instances with Phi(T)=0 built from a prescribed T: choose R inside a space where the
    # form (T (x) I) is neutral, then pick S with range R.
    worst = 0
    cnt = 0
    for trial in range(300):
        k = int(rng.integers(2, 4)); a = int(rng.integers(1, 4))
        t = int(rng.integers(1, k + 1))
        lam = np.concatenate([rng.uniform(0.5, 2, t) * rng.choice([-1, 1], t), np.zeros(k - t)])
        W = rand_unitary(rng, k)
        T = W @ np.diag(lam) @ W.conj().T
        Q = np.kron(T, np.eye(a))
        # maximal neutral subspace of the Hermitian form Q: pair + and - eigenvectors, add kernel
        ev, EV = np.linalg.eigh(Q)
        pos = [i for i in range(len(ev)) if ev[i] > 1e-9]
        neg = [i for i in range(len(ev)) if ev[i] < -1e-9]
        zer = [i for i in range(len(ev)) if abs(ev[i]) <= 1e-9]
        vecs = [EV[:, i] for i in zer]
        for ip, ineg in zip(pos, neg):
            vecs.append(EV[:, ip] / np.sqrt(ev[ip]) + EV[:, ineg] / np.sqrt(-ev[ineg]))
        if not vecs:
            continue
        Rmax = np.array(vecs).T
        bmax = Rmax.shape[1]
        b = int(rng.integers(1, bmax + 1))
        S = Rmax @ crandn(rng, bmax, b)
        X = S.reshape(k, a, b)
        assert np.linalg.norm(phi(X, T)) < 1e-9 and rank(S) == b
        slack = 2 * k * a - (2 * b + a * t)
        worst = min(worst, slack)
        cnt += 1
    check("A1 rank lemma 2b + a*rk(T) <= 2ka on Phi(T)=0 instances", worst >= 0,
          f"({cnt} instances, min slack {worst})")

    # A2 semidefinite lemma: X independent, T >= 0, Phi(T)=0 => T = 0; test the contrapositive
    # quantitatively: tr Phi(T) = sum_ij v_ij^* T v_ij > 0 for every nonzero PSD T.
    ok = True
    for trial in range(300):
        a, b = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        if a * b < 3:
            continue
        X = crandn(rng, 3, a, b)
        G = crandn(rng, 3, int(rng.integers(1, 4)))
        T = G @ G.conj().T
        ok &= np.trace(phi(X, T)).real > 1e-9
        # and the identity used in the proof: sum_ij v_ij^* T v_ij = tr Phi(T)
        v = X.reshape(3, a * b)
        ok &= abs(np.einsum('ri,rs,si->', v.conj(), T, v) - np.trace(phi(X, T))) < 1e-9
    check("A2 semidefinite lemma (tr Phi(T) > 0 for PSD T != 0, independent X)", ok)

    # A3 intersection lemma: dim D + dim R > 3a/2, R^* B D = 0 => nonzero v in D cap R cap ker B
    ok = True; tested = 0
    for trial in range(500):
        a = int(rng.integers(1, 7))
        dD = int(rng.integers(0, a + 1)); dR = int(rng.integers(0, a + 1))
        if not 3 * a < 2 * (dD + dR):
            continue
        D = np.linalg.qr(crandn(rng, a, a))[0][:, :dD]
        R = np.linalg.qr(crandn(rng, a, a))[0][:, :dR]
        PR = R @ R.conj().T
        PD = D @ D.conj().T
        B = (np.eye(a) - PR) @ crandn(rng, a, a) @ PD + crandn(rng, a, a) @ (np.eye(a) - PD)
        assert np.linalg.norm(R.conj().T @ B @ D) < 1e-9
        sysm = np.block([[D, -R], [B @ D, np.zeros((a, dR))]])
        ns = null_space(sysm, rcond=1e-10)
        ok &= ns.shape[1] >= dD + 2 * dR - 2 * a and ns.shape[1] >= 1
        tested += 1
    check("A3 intersection lemma (dim(D cap R cap ker B) >= dD + 2dR - 2a >= 1)", ok, f"({tested} instances)")

    # A4 equal Gram => unitary relation
    ok = True
    for trial in range(200):
        a, b = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        X0 = crandn(rng, a, b) @ np.diag(rng.integers(0, 2, b).astype(float))
        U0 = rand_unitary(rng, a)
        X1 = U0 @ X0
        U = polar(X1 @ X0.conj().T)[0]
        ok &= np.linalg.norm(X1 - U @ X0) < 1e-8 and np.linalg.norm(U.conj().T @ U - np.eye(a)) < 1e-8
    check("A4 X1^*X1 = X0^*X0 => X1 = U X0 with U unitary", ok)

    # A5 common plane lemma on synthetic triples with Sigma-type kernel element,
    # transported by a random coefficient change (tests the signature normalisation too)
    ok = True; cases = {}
    for trial in range(200):
        a = int(rng.integers(1, 5))
        b = int(rng.integers(3 * a // 2 + 1, 2 * a + 1)) if a > 1 else 2
        if not 3 * a < 2 * b:
            continue
        X0 = crandn(rng, a, b); U0 = rand_unitary(rng, a); X2 = crandn(rng, a, b)
        Xn = np.array([X0, U0 @ X0, X2])
        P = crandn(rng, 3, 3)
        X = mix(P, Xn)
        if rank(stack(X)) < b:
            continue
        Kh = hermitian_kernel(X)
        rkok = all(rank(T) <= 2 for T in Kh)
        V, info = common_plane(X, Kh)
        cases[info['case']] = cases.get(info['case'], 0) + 1
        ok &= rkok and info['neutral_X'] < 1e-6 and info.get('sig_ok', True) \
            and info.get('T22_max', 0) < 1e-6 and info.get('dim_identity', True) and info.get('half_rank', True)
    check("A5 common neutral plane on synthetic triples (Sigma in K_h after a random mix)", ok, f"cases={cases}")

    # A6 whitening and partial trace on random exact solutions mixed by a random coefficient change
    ok = True
    for trial in range(50):
        a, b, c = 2, 4, int(rng.integers(1, 3)); d = 3 * c
        Sst = np.linalg.qr(crandn(rng, 3 * a, b))[0]
        M = Sst.reshape(3, a, b)
        Nn = np.linalg.qr(crandn(rng, d, 3 * c))[0].reshape(d, 3, c).transpose(1, 0, 2)
        P = crandn(rng, 3, 3)
        M2 = mix(P, M); N2 = mix(np.linalg.inv(P.T), Nn)
        F = build_F(M2, N2)
        ok &= np.linalg.norm(F.conj().T @ F - np.eye(b * c)) < 1e-8
        Mw, Nw, rep = whiten(M2, N2, c)
        ok &= rep['mgram_err'] < 1e-8 and rep['recon_err'] < 1e-8
    check("A6 whitening: sum M'_r^*M'_r = I and F unchanged", ok)


def whiten(A, D, c):
    k = A.shape[0]
    H = np.array([[np.trace(D[r].conj().T @ D[s]) for s in range(k)] for r in range(k)]) / c
    B = sqrtm(H)
    C = np.linalg.inv(B.T)
    Mw, Nw = mix(B, A), mix(C, D)
    rep = {
        'H_min_eig': float(np.linalg.eigvalsh((H + H.conj().T) / 2).min()),
        'mgram_err': float(np.linalg.norm(sum(Mw[r].conj().T @ Mw[r] for r in range(k)) - np.eye(A.shape[2]))),
        'recon_err': float(np.linalg.norm(build_F(Mw, Nw) - build_F(A, D))),
    }
    return Mw, Nw, rep


def minimal_decomposition(F, a, b, c, d, tol=1e-7):
    R = F.reshape(a, d, b, c).transpose(0, 2, 1, 3).reshape(a * b, d * c)
    u, s, vh = np.linalg.svd(R)
    k = int((s > tol * s[0]).sum())
    A = np.array([(s[j] * u[:, j]).reshape(a, b) for j in range(k)])
    D = np.array([vh[j, :].reshape(d, c) for j in range(k)])
    return k, A, D


def pipeline(M, N, dims, verbose=False):
    """Run the hand proof on an exact solution and return the produced 2c-dimensional isometry."""
    a, b, c, d = dims
    F = build_F(M, N)
    lamb = np.real(np.trace(F.conj().T @ F)) / (b * c)
    G = F / np.sqrt(lamb)
    out = {'lambda': lamb, 'GG_err': np.linalg.norm(G.conj().T @ G - np.eye(b * c))}
    k, A, D = minimal_decomposition(G, a, b, c, d)
    out['k'] = k
    out['decomp_err'] = np.linalg.norm(build_F(A, D) - G)
    Mw, Nw, rep = whiten(A, D, c)
    out.update(rep)
    out['stack_iso_err'] = np.linalg.norm(stack(Mw).conj().T @ stack(Mw) - np.eye(b))
    out['b_le_ka'] = b <= k * a
    # block kernel: Phi(T^{lm}) = 0
    NN = np.array([[Nw[r].conj().T @ Nw[s] for s in range(k)] for r in range(k)])  # k,k,c,c
    errs = []
    Tlm = []
    for l in range(c):
        for m in range(c):
            T = NN[:, :, l, m] - (np.eye(k) * (1.0 if l == m else 0.0))
            Tlm.append(T)
            errs.append(np.linalg.norm(phi(Mw, T)))
    out['block_kernel_err'] = max(errs)
    if k == 2:
        out['Tlm_max'] = max(np.linalg.norm(T) for T in Tlm)
        Vt = np.eye(2, dtype=complex)
    else:
        Kh = hermitian_kernel(Mw)
        out['dimKh'] = len(Kh)
        out['Kh_rank_le_2'] = all(rank(T) <= 2 for T in Kh)
        Vt, info = common_plane(Mw, Kh)
        out['plane'] = info
        out['Tlm_neutral'] = max(np.linalg.norm(Vt.conj().T @ T @ Vt) for T in Tlm)
    Nt = mix(Vt.T, Nw)  # Ntilde_j = sum_r V_rj N_r
    W = np.concatenate([Nt[0], Nt[1]], axis=1)
    out['iso_err'] = np.linalg.norm(W.conj().T @ W - np.eye(2 * c))
    return out


def section_B(rng):
    print("== Section B: hand-proof pipeline on exact solutions (3a < 2b, d >= 2c) ==")
    dims_list = [(1, 2, 1, 2), (2, 4, 2, 4), (2, 4, 2, 5), (2, 4, 2, 6), (3, 5, 3, 6), (3, 5, 3, 7),
                 (3, 5, 2, 5), (1, 3, 1, 3), (2, 5, 1, 3), (3, 6, 2, 5)]
    for dims in dims_list:
        a, b, c, d = dims
        assert 3 * a < 2 * b
        got = 0; ks = {}; cases = {}; ok = True; worst = 0.0; dimKh = set()
        for s in range(8):
            try:
                err, M, N = solve(a, b, c, d, rng)
            except np.linalg.LinAlgError:
                continue
            if err > 1e-10:
                continue
            got += 1
            o = pipeline(M, N, dims)
            ks[o['k']] = ks.get(o['k'], 0) + 1
            errs = [o['GG_err'], o['decomp_err'], o['mgram_err'], o['recon_err'], o['stack_iso_err'],
                    o['block_kernel_err'], o['iso_err']]
            if o['k'] == 2:
                errs.append(o['Tlm_max'])
            else:
                p = o['plane']
                cases[p['case']] = cases.get(p['case'], 0) + 1
                dimKh.add(o['dimKh'])
                errs += [o['Tlm_neutral'], p.get('neutral_X', 0), p.get('T22_max', 0)]
                ok &= o['Kh_rank_le_2'] and p.get('sig_ok', True) and p.get('dim_identity', True) \
                    and p.get('half_rank', True)
                if p['case'] == 'B2':
                    errs += [p['eigvec_err'], p['cross_err'], p['pairing_err']]
                    ok &= p['intersection_dim'] >= max(1, p['intersection_lower_bound'])
            ok &= o['b_le_ka'] and o['H_min_eig'] > 0
            worst = max(worst, max(errs))
        check(f"B pipeline {dims}", ok and got > 0 and worst < 1e-6,
              f"solutions={got} k-counts={ks} plane-cases={cases} dimK_h={sorted(dimKh)} max-step-err={worst:.1e}")


# ---------------------------------------------------------------------------
# Section C: counterexample search
# ---------------------------------------------------------------------------
def _one(args):
    dims, seed = args
    rng = np.random.default_rng(seed)
    try:
        err, M, N = solve(*dims, rng, max_nfev=600)
    except np.linalg.LinAlgError:
        return None
    F = build_F(M, N)
    sv = np.linalg.svd(F, compute_uv=False)
    return err, float(sv.min()), float(sv.max())


def section_C(starts, workers):
    print("== Section C: counterexample search, minimise ||F^*F - I||_F over width-3 factors ==")
    print("   (rank<=3 realignments form a closed set, so a positive minimum excludes solutions)")
    obstructed = [(3, 5, 3, 5), (3, 5, 4, 7), (3, 5, 5, 9), (5, 8, 3, 5), (1, 2, 2, 3), (2, 4, 2, 3),
                  (2, 4, 3, 5), (1, 3, 1, 2), (2, 5, 2, 5)]
    controls = [(3, 5, 3, 6), (3, 5, 4, 8), (2, 4, 2, 4), (2, 3, 2, 3), (4, 6, 2, 3), (1, 3, 1, 3)]
    jobs = []
    for dims in obstructed + controls:
        n = starts if dims in obstructed else max(8, starts // 4)
        for s in range(n):
            jobs.append((dims, 1000 * hash(dims) % 100000 + s))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_one, jobs, chunksize=1))
    by = {}
    for (dims, _), r in zip(jobs, res):
        if r is not None:
            by.setdefault(dims, []).append(r)
    for dims in obstructed + controls:
        a, b, c, d = dims
        rs = by.get(dims, [])
        errs = sorted(r[0] for r in rs)
        cls = []
        cls.append('3a<2b' if 3 * a < 2 * b else '2b<=3a')
        cls.append('d<2c' if d < 2 * c else 'd>=2c')
        cls.append('ad>=bc' if a * d >= b * c else 'ad<bc(trivial)')
        expect_solvable = dims in controls
        if expect_solvable:
            ok = len(errs) > 0 and errs[0] < 1e-10
        else:
            ok = len(errs) > 0 and errs[0] > 1e-3
        check(f"C {dims} {' '.join(cls)} expect {'solution' if expect_solvable else 'NO solution'}", ok,
              f"starts={len(errs)} min={errs[0]:.6g} median={errs[len(errs)//2]:.6g}" if errs else "no runs")
    print(f"   section C wall time {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--starts', type=int, default=4)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    rng = np.random.default_rng(20261120)
    print(f"numpy {np.__version__}; seed 20261120; starts={args.starts}")
    print("cone point (3,5,3,5): 3a=9 < 2b=10; first bound would need d=5 >= 2c=6: false")
    section_A(rng)
    section_B(rng)
    section_C(args.starts, args.workers)
    print("== SUMMARY ==")
    print("all checks passed" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
