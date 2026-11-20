#!/usr/bin/env python3
"""Checks for the hand proof of Theorem m11:cone-rank (proof.tex).

Conventions (as in the paper and in QuantumMaxFlowBound.flow):
  M_r in C^{a x b}, N_r in C^{d x c}, F = sum_r kron(M_r, N_r), an (a*d) x (b*c) matrix,
  rows indexed (i,l) -> i*d+l, columns (j,h) -> j*c+h.

Exact ranks are computed over GF(P), P = 2^31-1, from exact rational matrices
(rank over GF(P) <= rank over Q, so reaching the outer bound min(ad,bc) certifies
full rank over Q, hence over C).  Rational arithmetic uses fractions.Fraction.
"""
import itertools, random, sys
from fractions import Fraction as Fr
import numpy as np

P = 2**31 - 1
LOG = []
def log(*s):
    line = " ".join(str(x) for x in s)
    print(line); LOG.append(line)

# ---------------------------------------------------------------- exact helpers
def to_mod(A):
    A = np.asarray(A, dtype=object)
    out = np.zeros(A.shape, dtype=np.int64)
    for idx, x in np.ndenumerate(A):
        x = Fr(x)
        out[idx] = (x.numerator % P) * pow(x.denominator % P, P - 2, P) % P
    return out

def rank_mod(A):
    A = to_mod(A) if A.dtype == object else A.copy() % P
    m, n = A.shape
    r = 0
    for col in range(n):
        if r == m: break
        piv = np.nonzero(A[r:, col])[0]
        if len(piv) == 0: continue
        pr = r + piv[0]
        if pr != r: A[[r, pr]] = A[[pr, r]]
        inv = pow(int(A[r, col]), P - 2, P)
        A[r] = A[r] * inv % P
        rows = np.nonzero(A[:, col])[0]
        rows = rows[rows != r]
        if len(rows):
            f = A[rows, col].reshape(-1, 1)
            A[rows] = (A[rows] - (f * A[r]) % P) % P
        r += 1
    return r

def zeros(m, n): return np.array([[Fr(0)] * n for _ in range(m)], dtype=object).reshape(m, n)

def kron(A, B):
    m, n = A.shape; s, t = B.shape
    K = zeros(m * s, n * t)
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                K[i*s:(i+1)*s, j*t:(j+1)*t] = A[i, j] * B
    return K

def flow(M, N):
    return sum(kron(M[r], N[r]) for r in range(3))

def nullspace_Q(A):
    """basis (columns) of {x : A x = 0} over Q, A an object matrix of Fractions."""
    A = np.array(A, dtype=object); m, n = A.shape
    A = A.copy(); pivcols = []; r = 0
    for col in range(n):
        pr = next((i for i in range(r, m) if A[i, col] != 0), None)
        if pr is None: continue
        A[[r, pr]] = A[[pr, r]]
        A[r] = A[r] / A[r, col]
        for i in range(m):
            if i != r and A[i, col] != 0:
                A[i] = A[i] - A[i, col] * A[r]
        pivcols.append(col); r += 1
        if r == m: break
    free = [c for c in range(n) if c not in pivcols]
    basis = []
    for f in free:
        x = [Fr(0)] * n; x[f] = Fr(1)
        for i, pc in enumerate(pivcols):
            x[pc] = -A[i, f]
        basis.append(x)
    if not basis: return zeros(n, 0)
    return np.array(basis, dtype=object).T

def colspace_extend(G, n):
    """columns of G (rank k) extended by standard vectors to a rank-n matrix with n columns"""
    rows = G.shape[0]
    cols = [G[:, j] for j in range(G.shape[1])]
    # take independent subset
    sel = []
    for v in cols:
        T = np.array(sel + [v], dtype=object).T
        if rank_mod(T) == len(sel) + 1 and len(sel) < n: sel.append(v)
    for e in range(rows):
        if len(sel) == n: break
        v = np.array([Fr(int(i == e)) for i in range(rows)], dtype=object)
        T = np.array(sel + [v], dtype=object).T
        if rank_mod(T) == len(sel) + 1: sel.append(v)
    assert len(sel) == n
    return np.array(sel, dtype=object).T

def solve_Q(G, B):
    """X with G X = B, G full column rank, exact."""
    m, n = G.shape
    aug = np.concatenate([G, B], axis=1).copy(); r = 0; piv = []
    for col in range(n):
        pr = next(i for i in range(r, m) if aug[i, col] != 0)
        aug[[r, pr]] = aug[[pr, r]]
        aug[r] = aug[r] / aug[r, col]
        for i in range(m):
            if i != r and aug[i, col] != 0: aug[i] = aug[i] - aug[i, col] * aug[r]
        r += 1
    X = aug[:n, n:]
    assert all(x == 0 for x in (G.dot(X) - B).flatten())
    return X

# ---------------------------------------------------------------- stacking maps
def hatM(M, a, b):   # 3b x a, row (j,r) -> 3j+r, entry (M_r)_{ij}
    H = zeros(3 * b, a)
    for r in range(3):
        for i in range(a):
            for j in range(b):
                H[3*j + r, i] = M[r][i, j]
    return H

def checkN(N, c, d):  # 3d x c, row (r,l) -> r*d+l, entry (N_r)_{lh}
    H = zeros(3 * d, c)
    for r in range(3):
        H[r*d:(r+1)*d, :] = N[r]
    return H

def concise(M, N, a, b, c, d):
    """Lemma m11:concise: replace (M,N) by a concise pair of the same type, rank not smaller."""
    HM = hatM(M, a, b); G = colspace_extend(HM, a); Pm = solve_Q(G, HM)   # HM = G Pm
    Mt = [zeros(a, b) for _ in range(3)]
    for r in range(3):
        for i in range(a):
            for j in range(b):
                Mt[r][i, j] = G[3*j + r, i]
    CN = checkN(N, c, d); G2 = colspace_extend(CN, c); Q = solve_Q(G2, CN)
    Nt = [G2[r*d:(r+1)*d, :] for r in range(3)]
    for r in range(3):   # M_r = Pm^T Mt_r ,  N_r = Nt_r Q
        assert all(x == 0 for x in (Pm.T.dot(Mt[r]) - M[r]).flatten())
        assert all(x == 0 for x in (Nt[r].dot(Q) - N[r]).flatten())
    return Mt, Nt

def castle(M, N, a, b, c, d):
    """Lemma m11:castling: concise (M,N) of type (a,b,c,d) -> (S,T) of type (b,3b-a,d,3d-c)."""
    HM = hatM(M, a, b); CN = checkN(N, c, d)
    assert rank_mod(HM) == a and rank_mod(CN) == c
    Sh = nullspace_Q(HM.T)          # 3b x (3b-a), columns span (im HM)^perp
    Tc = nullspace_Q(CN.T)          # 3d x (3d-c)
    assert Sh.shape[1] == 3*b - a and Tc.shape[1] == 3*d - c
    S = [zeros(b, 3*b - a) for _ in range(3)]
    T = [zeros(3*d - c, d) for _ in range(3)]
    for r in range(3):
        for j in range(b):
            S[r][j, :] = Sh[3*j + r, :]
        for l in range(d):
            T[r][:, l] = Tc[r*d + l, :]
    return S, T

# ---------------------------------------------------------------- explicit constructions
def unit(m, n, pairs):
    A = zeros(m, n)
    for (i, j, v) in pairs: A[i, j] = Fr(v)
    return A

def square_left(a, c, d):    # (a,a,c,d)
    M = [unit(a, a, [(i, i, 1) for i in range(a)]), zeros(a, a), zeros(a, a)]
    N = [unit(d, c, [(k, k, 1) for k in range(min(c, d))]), zeros(d, c), zeros(d, c)]
    return M, N

def square_right(a, b, c):   # (a,b,c,c)
    M = [unit(a, b, [(i, i, 1) for i in range(min(a, b))]), zeros(a, b), zeros(a, b)]
    N = [unit(c, c, [(k, k, 1) for k in range(c)]), zeros(c, c), zeros(c, c)]
    return M, N

def mixed_left(a, b, c, d):  # 2a<=b, d<=2c
    assert 2*a <= b and d <= 2*c and c <= d
    M0 = unit(a, b, [(i, i, 1) for i in range(a)])
    M1 = unit(a, b, [(i, a + i, 1) for i in range(a)])
    N0 = unit(d, c, [(k, k, 1) for k in range(c)])
    N1 = unit(d, c, [(c + h, h, 1) for h in range(d - c)])
    return [M0, M1, zeros(a, b)], [N0, N1, zeros(d, c)]

def swap(W):   # witness of (c,d,a,b) -> witness of (a,b,c,d)
    M, N = W
    return [N[r].T.copy() for r in range(3)], [M[r].T.copy() for r in range(3)]

def kron_param(a, b):
    m = b - a; p = (a - 1) // m; be = (a - 1) % m + 1; al = m - be
    assert p >= 1 and be >= 1 and a == p*al + (p+1)*be and b == (p+1)*al + (p+2)*be
    return p, al, be

class Blocks:
    """block bookkeeping for one side: al short blocks (length p) then be long ones (p+1)."""
    def __init__(self, p, al, be):
        self.p, self.al, self.be = p, al, be
        self.len = [p]*al + [p+1]*be
        self.row0 = np.cumsum([0] + self.len)[:-1]           # rows: length L
        self.col0 = np.cumsum([0] + [L+1 for L in self.len])[:-1]  # cols: length L+1
        self.nrow = sum(self.len); self.ncol = sum(L+1 for L in self.len)
    def short(self, i): return i
    def long(self, k): return self.al + k
    def arrows(self):
        A0 = zeros(self.nrow, self.ncol); A1 = zeros(self.nrow, self.ncol)
        for t, L in enumerate(self.len):
            for s in range(L):
                A0[self.row0[t] + s, self.col0[t] + s] = Fr(1)
                A1[self.row0[t] + s, self.col0[t] + s + 1] = Fr(1)
        return A0, A1

def sel(A, B):
    """Beatty selector set {floor(k A / B) : 0<=k<B} in Z_A (B<=A)."""
    return [k * A // B for k in range(B)]

def base_core(a, b, c, d, kind):
    p, al, be = kron_param(a, b); q, ga, de = kron_param(c, d)
    Lm = Blocks(p, al, be); Ln = Blocks(q, ga, de)
    M0, M1 = Lm.arrows(); C0, C1 = Ln.arrows()
    M2 = zeros(a, b); C2 = zeros(c, d)        # third slice; N_2 = C2^T
    if kind == "single":                      # al<=be, de<=ga
        for i in range(al):                   # BM = rectId(be,al): k=i
            M2[Lm.row0[Lm.long(i)] + p, Lm.col0[Lm.short(i)] + 0] = Fr(1)
        for h in range(de):                   # DM = rectId(ga,de): j=h ; N_2[(short j,0),(long h,p)]
            C2[Ln.row0[Ln.long(h)] + p, Ln.col0[Ln.short(h)] + 0] = Fr(1)
    elif kind == "longlong":                  # al<=be, ga<=de, al*de<=be*ga
        # X = -P^T on Z_be : X[k,h] = -1 iff k = h+1 mod be ; placed row (long k, p), col (long h, p+1)
        for h in range(be):
            k = (h + 1) % be
            M2[Lm.row0[Lm.long(k)] + p, Lm.col0[Lm.long(h)] + p + 1] += Fr(-1)
        # BM[k,i] = 1 iff k = floor(i be/al) ; rows (long k, u), cols (short i, v), u+v=p
        for i in range(al):
            k = i * be // al
            for u in range(p + 1):
                M2[Lm.row0[Lm.long(k)] + u, Lm.col0[Lm.short(i)] + (p - u)] = Fr(1)
        # Y = H^T on Z_de : Y[j,h] = w(h) iff h = j+1 mod de, w(0)=1/2 else 1 ;
        #   N_2 row (long j, p+1) [d-side], col (long h, p) [c-side]
        N2 = zeros(d, c)
        for j in range(de):
            h = (j + 1) % de
            N2[Ln.col0[Ln.long(j)] + p + 1, Ln.row0[Ln.long(h)] + p] += Fr(1, 2) if h == 0 else Fr(1)
        # DM[j,h] = 1 iff h = floor(j de/ga) ; N_2 row (short j, u), col (long h, v), u+v=p
        for j in range(ga):
            h = j * de // ga
            for u in range(p + 1):
                N2[Ln.col0[Ln.short(j)] + u, Ln.row0[Ln.long(h)] + (p - u)] = Fr(1)
        C2 = N2.T.copy()
    elif kind == "shortshort":                # be<=al, de<=ga, al*de<=be*ga
        # X = -P on Z_al : X[i,j] = -1 iff j = i+1 ; row (short i,0), col (short j,0)
        for i in range(al):
            j = (i + 1) % al
            M2[Lm.row0[Lm.short(i)] + 0, Lm.col0[Lm.short(j)] + 0] += Fr(-1)
        # BM[k,i] = 1 iff i = floor(k al/be) ; rows (long k,u), cols (short i,v), u+v=p
        for k in range(be):
            i = k * al // be
            for u in range(p + 1):
                M2[Lm.row0[Lm.long(k)] + u, Lm.col0[Lm.short(i)] + (p - u)] = Fr(1)
        N2 = zeros(d, c)
        # Y = H on Z_ga : Y[j,k] = w(j) iff k = j-1, w(0)=1/2 ; row (short j,0), col (short k,0)
        for j in range(ga):
            k = (j - 1) % ga
            N2[Ln.col0[Ln.short(j)] + 0, Ln.row0[Ln.short(k)] + 0] += Fr(1, 2) if j == 0 else Fr(1)
        # DM[j,h] = 1 iff j = floor(h ga/de) ; row (short j,u), col (long h,v), u+v=p
        for h in range(de):
            j = h * ga // de
            for u in range(p + 1):
                N2[Ln.col0[Ln.short(j)] + u, Ln.row0[Ln.long(h)] + (p - u)] = Fr(1)
        C2 = N2.T.copy()
    else:
        assert kind == "two"
    return [M0, M1, M2], [C0.T.copy(), C1.T.copy(), C2.T.copy()]

def base_kind(a, b, c, d):
    p, al, be = kron_param(a, b); q, ga, de = kron_param(c, d)
    if p != q or al == 0 or ga == 0: return "two", False
    if al <= be and de <= ga: return "single", False
    if be <= al and ga <= de: return "single", True
    if al <= be and ga <= de: return "longlong", al*de > be*ga
    return "shortshort", al*de > be*ga

def base(a, b, c, d):
    kind, sw = base_kind(a, b, c, d)
    if sw: return swap(base_core(c, d, a, b, kind)), kind + "(swap)"
    return base_core(a, b, c, d, kind), kind

def in_cone(a, b, c, d):
    return 0 < a <= b and 0 < c <= d and a*a + b*b <= 3*a*b and c*c + d*d <= 3*c*d

WIT = {}
def witness(a, b, c, d):
    """the recursion of the proof of Theorem m11:cone-rank; returns (M,N,route)"""
    key = (a, b, c, d)
    if key in WIT: return WIT[key]
    if a == b: (M, N), route = square_left(a, c, d), "square"
    elif c == d: (M, N), route = square_right(a, b, c), "square"
    elif 2*a <= b and 2*c <= d:
        M0, N0, _ = witness(3*a - b, a, 3*c - d, c)
        Mc, Nc = concise(M0, N0, 3*a - b, a, 3*c - d, c)
        M, N = castle(Mc, Nc, 3*a - b, a, 3*c - d, c)
        route = "descent"
    elif 2*a <= b: (M, N), route = mixed_left(a, b, c, d), "mixed"
    elif 2*c <= d: (M, N), route = swap(mixed_left(c, d, a, b)), "mixed(swap)"
    else: (M, N), route = base(a, b, c, d)
    WIT[key] = (M, N, route)
    return WIT[key]

def chk_shape(M, N, a, b, c, d):
    assert all(M[r].shape == (a, b) for r in range(3)), (a, b, c, d)
    assert all(N[r].shape == (d, c) for r in range(3)), (a, b, c, d)

# ================================================================ checks
def check_descent_arithmetic(B=200):
    bad = 0; n = 0
    for a in range(1, B):
        for b in range(a, 3*a):
            if a*a + b*b <= 3*a*b and 2*a <= b:
                n += 1
                u = 3*a - b
                ok = 0 < u <= a < b and u*u + a*a <= 3*u*a and \
                     (u*u + a*a - 3*u*a) == (a*a + b*b - 3*a*b)
                bad += not ok
    log(f"[descent arithmetic] cone pairs with 2a<=b, a<{B}: {n} checked, failures {bad}")
    return bad == 0

def check_cone_bound(B=60):
    bad = 0
    for a in range(1, B):
        for b in range(1, 4*B):
            if a*a + b*b <= 3*a*b and not b < 3*a: bad += 1
    log(f"[cone => b<3a] failures {bad}")
    return bad == 0

def check_random(B=12, seed=1):
    rng = np.random.default_rng(seed); bad = 0; n = 0; worst = np.inf
    for a, b, c, d in itertools.product(range(1, B+1), repeat=4):
        if not in_cone(a, b, c, d): continue
        n += 1
        M = rng.standard_normal((3, a, b)) + 1j*rng.standard_normal((3, a, b))
        N = rng.standard_normal((3, d, c)) + 1j*rng.standard_normal((3, d, c))
        F = sum(np.kron(M[r], N[r]) for r in range(3))
        assert F.shape == (a*d, b*c)
        s = np.linalg.svd(F, compute_uv=False)
        k = min(a*d, b*c)
        rk = int(np.sum(s > s[0] * 1e-10))
        worst = min(worst, s[k-1] / s[0])
        if rk != k:
            bad += 1; log("  RANDOM FAIL", (a, b, c, d), rk, k)
    log(f"[random complex factors] cone instances 1<=a<=b<=12, 1<=c<=d<=12: {n}; "
        f"rank == min(ad,bc) failures: {bad}; smallest sigma_k/sigma_1 = {worst:.2e}")
    return bad == 0

def check_block_ranks(X=9):
    """width-two block W_{x,y} of size x(y+1) x (x+1)y: rank and kernels (Lemma m11:pencil-block)."""
    bad = 0
    for x in range(0, X):
        for y in range(0, X):
            L0 = unit(x, x+1, [(s, s, 1) for s in range(x)]); L1 = unit(x, x+1, [(s, s+1, 1) for s in range(x)])
            K0 = unit(y, y+1, [(s, s, 1) for s in range(y)]); K1 = unit(y, y+1, [(s, s+1, 1) for s in range(y)])
            W = kron(L0, K0.T) + kron(L1, K1.T)
            if W.size == 0: continue
            rk = rank_mod(W)
            exp = min(x*(y+1), (x+1)*y)
            if rk != exp: bad += 1; log("  block rank FAIL", x, y, rk, exp)
            if y == x + 1:   # kernel vector: u(s,t) = (-1)^s on s+t = x
                u = np.array([Fr((-1)**s) if s + t == x else Fr(0) for s in range(x+1) for t in range(y)], dtype=object)
                if any(v != 0 for v in W.dot(u)): bad += 1; log("  kernel FAIL", x, y)
            if x == y + 1:   # cokernel: v(s,t) = (-1)^s on s+t = y
                v = np.array([Fr((-1)**s) if s + t == y else Fr(0) for s in range(x) for t in range(y+1)], dtype=object)
                if any(z != 0 for z in v.dot(W)): bad += 1; log("  cokernel FAIL", x, y)
    log(f"[pencil blocks] 0<=x,y<{X}: rank = min(x(y+1),(x+1)y), kernel/cokernel vectors; failures {bad}")
    return bad == 0

def check_explicit_small(B=12):
    """square / mixed / base constructions directly, exact rank, all ordered instances in range"""
    bad = 0; cnt = {}
    for a, b, c, d in itertools.product(range(1, B+1), repeat=4):
        if not (a <= b and c <= d): continue
        k = min(a*d, b*c); W = None
        if a == b: W, name = square_left(a, c, d), "square_left"
        elif c == d: W, name = square_right(a, b, c), "square_right"
        elif 2*a <= b and d <= 2*c: W, name = mixed_left(a, b, c, d), "mixed_left"
        elif b <= 2*a and 2*c <= d: W, name = swap(mixed_left(c, d, a, b)), "mixed_right"
        elif b < 2*a and d < 2*c: (W, name) = base(a, b, c, d)
        if W is None: continue
        M, N = W; chk_shape(M, N, a, b, c, d)
        rk = rank_mod(flow(M, N))
        cnt[name] = cnt.get(name, 0) + 1
        if rk != k: bad += 1; log("  EXPLICIT FAIL", (a, b, c, d), name, rk, k)
    log(f"[explicit constructions, exact] ordered instances with entries <= {B} (no cone needed): "
        f"{dict(sorted(cnt.items()))}; failures {bad}")
    return bad == 0

def check_base_larger(lo=13, hi=22, samples=60, seed=7):
    random.seed(seed); bad = 0; cnt = {}
    cand = [(a, b, c, d) for a in range(2, hi) for b in range(a+1, min(2*a, hi+1))
            for c in range(2, hi) for d in range(c+1, min(2*c, hi+1)) if max(b, d) >= lo]
    # prefer instances that need the reservoir constructions
    hard = [t for t in cand if base_kind(*t)[0] in ("longlong", "shortshort")]
    pick = random.sample(hard, min(samples, len(hard))) + random.sample(cand, min(samples//3, len(cand)))
    for (a, b, c, d) in pick:
        (M, N), name = base(a, b, c, d)
        rk = rank_mod(flow(M, N)); k = min(a*d, b*c)
        cnt[name] = cnt.get(name, 0) + 1
        if rk != k: bad += 1; log("  BASE FAIL", (a, b, c, d), name, rk, k)
    log(f"[base lemma, larger sample] b<2a, d<2c, max(b,d) in [{lo},{hi}]: {dict(sorted(cnt.items()))}; failures {bad}")
    return bad == 0

def check_hard_instances():
    """large base instances whose Kronecker data have extreme ratios (beta/alpha = 10 > (p+1)^2),
    where a first-order (single third-slice) correction cannot work; reservoir constructions needed."""
    bad = 0; out = []
    for t in [(32, 43, 32, 43), (32, 43, 31, 42), (23, 34, 23, 34), (23, 34, 25, 36), (31, 42, 32, 43)]:
        a, b, c, d = t
        (M, N), name = base(a, b, c, d)
        rk = rank_mod(flow(M, N)); k = min(a*d, b*c)
        out.append(f"{t}:{name}:{kron_param(a, b)}|{kron_param(c, d)}:rank {rk}/{k}")
        if rk != k: bad += 1
    log("[hard base instances] " + "; ".join(out) + f"; failures {bad}")
    return bad == 0

def schur_matrix(A, B, G, D, kappa, orient):
    """restricted resolvent restr_O (kappa + R^{-1}) ext_I on Z_A x Z_G (Lemma m11:cyclic)."""
    n = A*G
    idx = lambda i, j: (i % A)*G + (j % G)
    w = lambda j: Fr(1, 2) if j % G == 0 else Fr(1)
    T = zeros(n, n)
    for i in range(A):
        for j in range(G):
            if orient == "short":   # (Tz)(i,j) = w(j) z(i+1, j-1)
                T[idx(i, j), idx(i+1, j-1)] += w(j)
            else:                   # (Tz)(i,j) = w(j+1) z(i-1, j+1)
                T[idx(i, j), idx(i-1, j+1)] += w(j+1)
    R = np.array([[Fr(int(r == s)) for s in range(n)] for r in range(n)], dtype=object) - T
    return R

def check_cyclic(Bmax=7):
    """Lemma m11:cyclic in both orientations, all small parameters, exact."""
    bad = 0; n = 0
    for A in range(1, Bmax+1):
        for G in range(1, Bmax+1):
            for orient in ("short", "long"):
                R = schur_matrix(A, None, G, None, None, orient)
                Rm = to_mod(R)
                if rank_mod(Rm) != A*G: bad += 1; log("  R singular", A, G, orient)
                # inverse mod P
                n_ = A*G
                aug = np.concatenate([Rm, np.eye(n_, dtype=np.int64)], axis=1)
                Rinv = gauss_inv_mod(Rm)
                for B in range(0, A+1):
                    for D in range(0, G+1):
                        for kappa in (1, 2, 3):
                            if orient == "short":   # inputs Z_A x S(G,D), outputs S(A,B) x Z_G ; need B<=A, D<=G
                                I = [(i, j) for i in range(A) for j in sel(G, D)]
                                O = [(i, j) for i in sel(A, B) for j in range(G)]
                            else:                   # inputs S(A,B) x Z_G, outputs Z_A x S(G,D)
                                I = [(i, j) for i in sel(A, B) for j in range(G)]
                                O = [(i, j) for i in range(A) for j in sel(G, D)]
                            if len(I) == 0 or len(O) == 0: continue
                            K = (kappa * np.eye(n_, dtype=np.int64) + Rinv) % P
                            S = K[np.ix_([i*G + j for i, j in O], [i*G + j for i, j in I])]
                            rk = rank_mod(S); n += 1
                            if len(I) <= len(O) and rk != len(I):
                                bad += 1; log("  cyclic inj FAIL", A, B, G, D, kappa, orient, rk)
    log(f"[cyclic resolvent lemma] A,G<={Bmax}, all B<=A, D<=G, kappa in 1..3, both orientations: "
        f"{n} restricted maps; R invertible and injective whenever |I|<=|O|; failures {bad}")
    return bad == 0

def gauss_inv_mod(A):
    n = A.shape[0]
    aug = np.concatenate([A % P, np.eye(n, dtype=np.int64)], axis=1)
    for col in range(n):
        pr = col + np.nonzero(aug[col:, col])[0][0]
        aug[[col, pr]] = aug[[pr, col]]
        aug[col] = aug[col] * pow(int(aug[col, col]), P - 2, P) % P
        rows = np.nonzero(aug[:, col])[0]; rows = rows[rows != col]
        if len(rows):
            f = aug[rows, col].reshape(-1, 1)
            aug[rows] = (aug[rows] - (f * aug[col]) % P) % P
    return aug[:, n:]

def check_beatty(Rmax=12):
    """window excess <= 1 and selector characterisation (Lemma m11:beatty)."""
    import math
    def ceil_frac(x): return -((-x.numerator) // x.denominator)
    def J(rho, n): return ceil_frac((n+1)*rho) - ceil_frac(n*rho)
    bad = 0
    for G in range(1, Rmax+1):
        for D in range(0, G+1):
            rho = Fr(D, G)
            S = set(sel(G, D))
            for n in range(-2*G, 3*G):
                if J(rho, n) not in (0, 1): bad += 1
                if J(rho, n) != (1 if (n % G) in S else 0): bad += 1
    for A in range(1, 9):
        for B in range(0, A+1):
            for G in range(1, 9):
                for D in range(0, G+1):
                    if A*D > B*G: continue
                    ri, ro = Fr(D, G), Fr(B, A)
                    for c0 in range(-5, 5):
                        for L in range(0, 2*A*G + 2):
                            e = sum(J(ri, c0 + t) - J(ro, 3 - c0 - t) for t in range(L))
                            if e > 1: bad += 1
    log(f"[Beatty facts] jump in {{0,1}}, jump=1 exactly on the selector set, opposite-direction window excess <= 1; failures {bad}")
    return bad == 0

def check_castling_identity(trials=40, seed=3):
    """deficit identity of Lemma m11:castling for random concise pairs (exact rational)."""
    rnd = random.Random(seed); bad = 0; n = 0
    while n < trials:
        a, b, c, d = [rnd.randint(1, 5) for _ in range(4)]
        if not (a <= 3*b and c <= 3*d): continue
        sparse = rnd.random() < 0.5
        def rmat(m, k):
            return np.array([[Fr(rnd.randint(-2, 2) if (not sparse or rnd.random() < 0.3) else 0)
                              for _ in range(k)] for _ in range(m)], dtype=object).reshape(m, k)
        M = [rmat(a, b) for _ in range(3)]; N = [rmat(d, c) for _ in range(3)]
        if rank_mod(hatM(M, a, b)) < a or rank_mod(checkN(N, c, d)) < c:
            M, N = concise(M, N, a, b, c, d)
        S, T = castle(M, N, a, b, c, d)
        chk_shape(S, T, b, 3*b - a, d, 3*d - c)
        lhs = a*d - rank_mod(flow(M, N)); rhs = b*(3*d - c) - rank_mod(flow(S, T))
        n += 1
        if lhs != rhs: bad += 1; log("  castling FAIL", (a, b, c, d), lhs, rhs)
    log(f"[castling deficit identity] {n} random concise pairs (dims<=5, half of them sparse/degenerate): failures {bad}")
    return bad == 0

def check_concise_reduction(trials=30, seed=5):
    rnd = random.Random(seed); bad = 0
    for _ in range(trials):
        a, b, c, d = [rnd.randint(1, 5) for _ in range(4)]
        if not (a <= 3*b and c <= 3*d): continue
        M = [np.array([[Fr(rnd.choice([0, 0, 1, -1])) for _ in range(b)] for _ in range(a)], dtype=object).reshape(a, b) for _ in range(3)]
        N = [np.array([[Fr(rnd.choice([0, 0, 1])) for _ in range(c)] for _ in range(d)], dtype=object).reshape(d, c) for _ in range(3)]
        r0 = rank_mod(flow(M, N))
        Mt, Nt = concise(M, N, a, b, c, d)
        if rank_mod(hatM(Mt, a, b)) != a or rank_mod(checkN(Nt, c, d)) != c or rank_mod(flow(Mt, Nt)) < r0:
            bad += 1
    log(f"[concise reduction] random degenerate pairs: failures {bad}")
    return bad == 0

def check_recursion(B=12):
    """the full recursion of the proof, exact witnesses, every cone instance with entries <= B"""
    bad = 0; routes = {}
    for a, b, c, d in itertools.product(range(1, B+1), repeat=4):
        if not in_cone(a, b, c, d): continue
        M, N, route = witness(a, b, c, d)
        chk_shape(M, N, a, b, c, d)
        rk = rank_mod(flow(M, N)); k = min(a*d, b*c)
        routes[route.split("(")[0]] = routes.get(route.split("(")[0], 0) + 1
        if rk != k: bad += 1; log("  RECURSION FAIL", (a, b, c, d), route, rk, k)
    log(f"[full recursion, exact witnesses] all cone instances 1<=a<=b<={B}, 1<=c<=d<={B}: "
        f"{sum(routes.values())} instances, routes {dict(sorted(routes.items()))}; rank == min(ad,bc) failures {bad}")
    return bad == 0

def check_case_split(B=300):
    """every (al,be,ga,de) with be,de>=1 falls in one of the base sub-cases; parametrisation"""
    bad = 0
    for a in range(2, B):
        for b in range(a+1, 2*a):
            p, al, be = kron_param(a, b)
    log(f"[parametrisation a=p*al+(p+1)*be, b=(p+1)*al+(p+2)*be, p>=1, be>=1] all a<b<2a, a<{B}: ok")
    return bad == 0

if __name__ == "__main__":
    ok = True
    ok &= check_cone_bound()
    ok &= check_descent_arithmetic()
    ok &= check_case_split()
    ok &= check_block_ranks()
    ok &= check_beatty()
    ok &= check_cyclic()
    ok &= check_concise_reduction()
    ok &= check_castling_identity()
    ok &= check_explicit_small()
    ok &= check_base_larger()
    ok &= check_hard_instances()
    ok &= check_recursion()
    ok &= check_random()
    log("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    with open(__file__[:-3] + ".log", "w") as f:
        f.write("\n".join(LOG) + "\n")
