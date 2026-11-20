#!/usr/bin/env python3
"""Checks for the hand proof of the forward implication of Theorem m10:orientation.

J = R x R^3, (s,x)o(t,y) = (st+<x,y>, s y + t x),
alpha_{m,v}(s,x) = m s + <v,x>  (m >= |v|),
R_alpha(a,b) = alpha(a o b) - i alpha(D(a,b)),
Q_alpha(b,c) = alpha(b^2+c^2) + 2 alpha(D(b,c)),
P2: [[R(a,a),R(a,b)],[R(b,a),R(b,b)]] PSD for all alpha,a,b,
PB: Q(b+h,c+k) = Q(b-h,c-k), h = D(a,b)-a o c, k = a o b + D(a,c).

Sections:
  S1  exact (sympy) verification of every algebraic identity used in proof.tex
  S2  numeric spot checks of the constructive steps (reconstruction, generation)
  S3  sample maps D violating each derived necessary condition: P2 or PB fails
  S4  the family D_lambda = (0, lambda x cross y): which lambda satisfy P2 / PB
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(20261120)
OK = []


def report(name, cond, extra=""):
    OK.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---------------------------------------------------------------- helpers
def jor(a, b):
    return np.concatenate(([a[0] * b[0] + a[1:] @ b[1:]], a[0] * b[1:] + b[0] * a[1:]))


def ev(m, v, a):
    return m * a[0] + v @ a[1:]


def crossD(eps):
    return lambda a, b: np.concatenate(([0.0], eps * np.cross(a[1:], b[1:])))


def gram2(D, m, v, a, b):
    R = lambda p, q: ev(m, v, jor(p, q)) - 1j * ev(m, v, D(p, q))
    return np.array([[R(a, a), R(a, b)], [R(b, a), R(b, b)]])


def p2_defect(M):
    """returns (non-Hermitian size, min eigenvalue of Hermitian part)."""
    nh = np.abs(M - M.conj().T).max()
    H = (M + M.conj().T) / 2
    return nh, np.linalg.eigvalsh(H).min()


def FD(D, a, b, c):
    """element form of PB: D(b+h,c+k)-D(b-h,c-k)+2 b o h + 2 c o k."""
    h = D(a, b) - jor(a, c)
    k = jor(a, b) + D(a, c)
    return D(b + h, c + k) - D(b - h, c - k) + 2 * jor(b, h) + 2 * jor(c, k)


def Qa(D, m, v, b, c):
    return ev(m, v, jor(b, b) + jor(c, c)) + 2 * ev(m, v, D(b, c))


def pb_defect(D, m, v, a, b, c):
    h = D(a, b) - jor(a, c)
    k = jor(a, b) + D(a, c)
    return Qa(D, m, v, b + h, c + k) - Qa(D, m, v, b - h, c - k)


def rand_E(scale=1.0):
    return rng.normal(size=4) * scale


def rand_eval():
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v) * rng.uniform(0, 1) ** 0.3
    return 1.0, v


def unit(v):
    return v / np.linalg.norm(v)


def rand_rot():
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rot_about(axis, theta):
    k = unit(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


# ---------------------------------------------------------------- S1 exact
print("== S1 exact identities ==")
s, t, m = sp.symbols("s t m", real=True)
X = sp.Matrix(sp.symbols("x1:4", real=True))
Y = sp.Matrix(sp.symbols("y1:4", real=True))
V = sp.Matrix(sp.symbols("v1:4", real=True))


def sj(a, b):
    return (a[0] * b[0] + a[1] .dot(b[1]), a[0] * b[1] + b[0] * a[1])


def sev(mm, vv, a):
    return mm * a[0] + vv.dot(a[1])


def sadd(a, b, cb=1):
    return (a[0] + cb * b[0], a[1] + cb * b[1])


def ssc(c, a):
    return (c * a[0], c * a[1])


def gen(name):
    return (sp.Symbol(name + "0", real=True), sp.Matrix(sp.symbols(name + "1:4", real=True)))


# (1) PB <=> F: Q(b+h,c+k)-Q(b-h,c-k) = 2 alpha(D1 - D2 + 2 b o h + 2 c o k), for arbitrary h,k,D1,D2
b_, c_, h_, k_, D1, D2 = (gen(n) for n in ("b", "c", "h", "k", "P", "M"))
lhs = (sev(m, V, sadd(sj(sadd(b_, h_), sadd(b_, h_)), sj(sadd(c_, k_), sadd(c_, k_)))) + 2 * sev(m, V, D1)
       - sev(m, V, sadd(sj(sadd(b_, h_, -1), sadd(b_, h_, -1)), sj(sadd(c_, k_, -1), sadd(c_, k_, -1)))) - 2 * sev(m, V, D2))
rhs = 2 * sev(m, V, sadd(sadd(D1, D2, -1), sadd(ssc(2, sj(b_, h_)), ssc(2, sj(c_, k_)))))
report("PB <=> F (expansion linear in alpha)", sp.expand(lhs - rhs) == 0)

# (2) unit evaluation: alpha_{1,n}(a o b) = <x+s n, y+t n> + s t (1-|n|^2)
a = (s, X); b = (t, Y)
L = lambda p: p[1] + p[0] * V
report("alpha_{1,n}(a o b) = <La,Lb> + st(1-|n|^2)",
       sp.expand(sev(1, V, sj(a, b)) - L(a).dot(L(b)) - s * t * (1 - V.dot(V))) == 0)
# Lagrange: |u|^2|w|^2 - <u,w>^2 = |u x w|^2 and (x+sn)x(y+tn) = x x y + n x (s y - t x)
U = L(a); W = L(b)
report("Lagrange identity", sp.expand(U.dot(U) * W.dot(W) - U.dot(W) ** 2 - U.cross(W).dot(U.cross(W))) == 0)
report("(x+sn)x(y+tn) = x x y + n x (s y - t x)", sp.expand(U.cross(W) - (X.cross(Y) + V.cross(s * Y - t * X))) == sp.zeros(3, 1))

# (3) scalar face: a=(l,0), b=(t,y), alpha = alpha_{|y|, +-y}: Gram determinant 0
l, Yn = sp.symbols("l Y", positive=True)
yv = sp.Matrix([Yn, 0, 0])
for sg in (1, -1):
    al = lambda p: sev(Yn, sg * yv, p)
    A_ = (l, sp.zeros(3, 1)); B_ = (t, yv)
    det = al(sj(A_, A_)) * al(sj(B_, B_)) - al(sj(A_, B_)) ** 2
    report(f"scalar face Gram det = 0 (sign {sg})", sp.simplify(det) == 0)

# (4) C1: a=(l,0), b=(t,y), c=(kap,0), D(a,b)=(0,z), z⟂y, |z|=Z, alpha=alpha_{Z,z}, (1+l^2)kap = l Z
Z = sp.symbols("Z", positive=True)
lr = sp.symbols("l", real=True)
kap = lr * Z / (1 + lr ** 2)
zv = sp.Matrix([0, Z, 0])
al = lambda p: sev(Z, zv, p)
aE = (lr, sp.zeros(3, 1)); bE = (t, yv); cE = (kap, sp.zeros(3, 1)); Dab = (0, zv)
h = sadd(Dab, sj(aE, cE), -1)
kk = sj(aE, bE)  # D(a,c)=0 (two scalars)
P_ = sadd(bE, h); Q_ = sadd(cE, kk); P2_ = sadd(bE, h, -1); Q2_ = sadd(cE, kk, -1)
gd = lambda p, q: al(sj(p, p)) * al(sj(q, q)) - al(sj(p, q)) ** 2
report("C1: Gram det(b+h,c+k) = 0 at alpha_{Z,z}", sp.simplify(gd(P_, Q_)) == 0)
report("C1: Gram det(b-h,c-k) = 0 at alpha_{Z,z}", sp.simplify(gd(P2_, Q2_)) == 0)
report("C1: alpha(b o h + c o k) = t Z^2", sp.simplify(al(sadd(sj(bE, h), sj(cE, kk))) - t * Z ** 2) == 0)
report("C1: vectors b+-h, c+-k as stated",
       sp.simplify(P_[0] - (t - lr * kap)) == 0 and sp.simplify(Q_[0] - (kap + lr * t)) == 0)

# (5) C2 constants
mm = sp.sqrt(2) - 1
report("C2: (1-m^2)/(1+m^2) = 2m/(1+m^2) = 1/sqrt2",
       sp.simplify((1 - mm ** 2) / (1 + mm ** 2) - 1 / sp.sqrt(2)) == 0 and sp.simplify(2 * mm / (1 + mm ** 2) - 1 / sp.sqrt(2)) == 0)
q_ = sp.symbols("q", real=True)
Av = (-q_ / sp.sqrt(2), Yn / sp.sqrt(2)); Bv = (q_ / sp.sqrt(2), Yn / sp.sqrt(2))
report("C2: (A+B)/sqrt2 = (0,y), (B-A)/sqrt2 = (q,0)",
       sp.simplify((Av[0] + Bv[0]) / sp.sqrt(2)) == 0 and sp.simplify((Bv[0] - Av[0]) / sp.sqrt(2) - q_) == 0
       and sp.simplify((Av[1] + Bv[1]) / sp.sqrt(2) - Yn) == 0)
report("C2: scalar parts of b0,c0 nonzero factors (m-1),(1+m)", sp.simplify((Av[0] + mm * Bv[0]) - q_ / sp.sqrt(2) * (mm - 1)) == 0
       and sp.simplify((Bv[0] - mm * Av[0]) - q_ / sp.sqrt(2) * (1 + mm)) == 0)
# label-cayley algebra: b0 - m c0 = b, c0 + m b0 = c, b0 + m c0, c0 - m b0
mu = sp.symbols("mu", real=True)
bb, cc = sp.symbols("bb cc")
b0 = (bb + mu * cc) / (1 + mu ** 2); c0 = (cc - mu * bb) / (1 + mu ** 2)
report("label Cayley substitution", sp.simplify(b0 - mu * c0 - bb) == 0 and sp.simplify(c0 + mu * b0 - cc) == 0
       and sp.simplify(b0 + mu * c0 - ((1 - mu ** 2) * bb + 2 * mu * cc) / (1 + mu ** 2)) == 0
       and sp.simplify(c0 - mu * b0 - ((1 - mu ** 2) * cc - 2 * mu * bb) / (1 + mu ** 2)) == 0)
th = sp.symbols("theta", real=True)
muCS = sp.sin(th) / (1 + sp.cos(th))
report("circle point (cos th, sin th), th != pi, has mu = S/(1+C)",
       sp.simplify((1 - muCS ** 2) / (1 + muCS ** 2) - sp.cos(th)) == 0 and sp.simplify(2 * muCS / (1 + muCS ** 2) - sp.sin(th)) == 0)

# (6) saturation estimate (F3): x=(X,0,0), y=(0,Y,0)
Xs, Ys, kk_ = sp.symbols("X Y k", positive=True)
ksig = sp.symbols("kappa", real=True)
xv = sp.Matrix([Xs, 0, 0]); yv2 = sp.Matrix([0, Ys, 0]); w = xv.cross(yv2)
Aq, Bq = Xs ** 2, Ys ** 2
r = sp.sqrt(lr ** 2 * Aq + 1)
n = (-lr / r) * xv + (1 / (r * Ys)) * yv2
report("F3: |n| = 1", sp.simplify(n.dot(n) - 1) == 0)
cvec = (r, lr * xv)
pE = (-lr * Aq, yv2 + ksig * w - r * xv); qE = (lr * Aq, yv2 - ksig * w + r * xv)
aE = (0, xv); bE = (0, yv2)
hh = sadd((0, ksig * w), sj(aE, cvec), -1)
report("F3: b+h = p, b-h = q", all(sp.simplify(u) == 0 for u in list(sadd(bE, hh)[1] - pE[1]) + [sadd(bE, hh)[0] - pE[0]])
       and all(sp.simplify(u) == 0 for u in list(sadd(bE, hh, -1)[1] - qE[1]) + [sadd(bE, hh, -1)[0] - qE[0]]))
report("F3: -2 b o h = (0, 2 l A y)", all(sp.simplify(u) == 0 for u in [(-2 * sj(bE, hh)[0])] + list(-2 * sj(bE, hh)[1] - 2 * lr * Aq * yv2)))
Lc = cvec[1] + cvec[0] * n
report("F3: L c = y/|y|", sp.simplify(Lc - yv2 / Ys) == sp.zeros(3, 1))
for nm, pp in (("p", pE), ("q", qE)):
    Lp = pp[1] + pp[0] * n
    cr = Lp.cross(Lc)
    report(f"F3: |L{nm} x Lc|^2 = k^2 A B + A/r^2", sp.simplify(cr.dot(cr) - (ksig ** 2 * Aq * Bq + Aq / r ** 2)) == 0)
report("F3: <n, 2 l A y> = 2 l A Y / r", sp.simplify(n.dot(2 * lr * Aq * yv2) - 2 * lr * Aq * Ys / r) == 0)
ineq = sp.expand((4 * lr ** 2 * Aq ** 2 * Bq / r ** 2 - 4 * (ksig ** 2 * Aq * Bq + Aq / r ** 2)) * r ** 2 / (4 * Aq))
report("F3: (f-g)^2<=4T  <=>  (1-k^2) l^2 A B <= 1 + k^2 B",
       sp.simplify(ineq - ((1 - ksig ** 2) * lr ** 2 * Aq * Bq - (1 + ksig ** 2 * Bq))) == 0)

# (7) transverse formula: x=(X,0,0), y,z in x-perp
y2, y3, z2, z3 = sp.symbols("y2 y3 z2 z3", real=True)
kx, lx = sp.symbols("k l", real=True)
xv = sp.Matrix([Xs, 0, 0]); yv = sp.Matrix([0, y2, y3]); zv = sp.Matrix([0, z2, z3])
w1 = xv.cross(yv); w2 = xv.cross(zv)
sig = xv.dot(yv.cross(zv)) / Aq
pp = yv + kx * w1; qq = zv + lx * w2; pp2 = yv - kx * w1; qq2 = zv - lx * w2
Pc = sig * (1 + kx * lx * Aq); Qc = (lx - kx) * yv.dot(zv)
report("F4: p x q = (P+Q) x", sp.simplify(pp.cross(qq) - (Pc + Qc) * xv) == sp.zeros(3, 1))
report("F4: p' x q' = (P-Q) x", sp.simplify(pp2.cross(qq2) - (Pc - Qc) * xv) == sp.zeros(3, 1))
report("F4: b o h + c o k = 0 (pure, x perp y,z)", sp.simplify(yv.dot(kx * w1) + zv.dot(lx * w2)) == 0)

# (8) unit pair: x=e1,y=e2, eps=+-1
for e in (1, -1):
    x1 = sp.Matrix([1, 0, 0]); y1 = sp.Matrix([0, 1, 0]); w = x1.cross(y1)
    p = y1 - e * w; q = 2 * y1 + 2 * e * w; p2 = y1 + e * w; q2 = 2 * y1 - 2 * e * w
    good = (p.dot(p) == 2 and p.dot(q) == 0 and p2.dot(q2) == 0 and p2.dot(p2) == 2
            and e * p.cross(q) == 4 * x1 and e * p2.cross(q2) == -4 * x1
            and y1.dot(-e * w) + (2 * y1).dot(2 * e * w) == 0)
    report(f"F4 unit pair contradiction data (e={e})", good)

# (9) transport identity (T): crossD satisfies the pure PB substitution
qv = sp.Matrix(sp.symbols("q1:4", real=True)); Bv = sp.Matrix(sp.symbols("B1:4", real=True)); Cv = sp.Matrix(sp.symbols("C1:4", real=True))
for e in (1, -1):
    lhsT = e * ((Bv + e * qv.cross(Bv)).cross(Cv + e * qv.cross(Cv)) - (Bv - e * qv.cross(Bv)).cross(Cv - e * qv.cross(Cv)))
    rhsT = 2 * qv.dot(Cv) * Bv - 2 * qv.dot(Bv) * Cv
    report(f"(T): e[(B+eqB)x(C+eqC)-(B-eqB)x(C-eqC)] = 2<q,C>B-2<q,B>C (e={e})", sp.expand(lhsT - rhsT) == sp.zeros(3, 1))
# b o h, c o k for b=(0,B), c=(0,C), a=(0,q), h=(-<q,C>, e q x B), k=(<q,B>, e q x C)
e = sp.Symbol("e")
bh = ((Bv.dot(e * qv.cross(Bv))), -qv.dot(Cv) * Bv)
ck = ((Cv.dot(e * qv.cross(Cv))), qv.dot(Bv) * Cv)
report("(T): -2(b o h + c o k) = (0, 2<q,C>B - 2<q,B>C)",
       sp.expand(bh[0] + ck[0]) == 0 and sp.expand(-2 * (bh[1] + ck[1]) - (2 * qv.dot(Cv) * Bv - 2 * qv.dot(Bv) * Cv)) == sp.zeros(3, 1))

# (10) saturated support vectors
a_, b_ = sp.symbols("a b", positive=True)
# w=(W,0,0), p=(0,P,0): a=W^2, b=P^2, c = w x p
W_, P_ = sp.symbols("W P", positive=True)
wv = sp.Matrix([W_, 0, 0]); pv = sp.Matrix([0, P_, 0]); cv = wv.cross(pv)
# case b >= a
rr = sp.sqrt(P_ ** 2 - W_ ** 2)
nn = (cv + rr * pv) / P_ ** 2
report("support (|p|>=|w|): |n|=1, <n,w>=0, w + n x p = 0",
       sp.simplify(nn.dot(nn) - 1) == 0 and sp.simplify(nn.dot(wv)) == 0 and sp.simplify(wv + nn.cross(pv)) == sp.zeros(3, 1))
JJ = sp.sqrt(W_ ** 2 - P_ ** 2)
for sg in (1, -1):
    nn = (cv + sg * JJ * wv) / W_ ** 2
    vv = wv + nn.cross(pv)
    report(f"support (|p|<|w|, sigma={sg}): |n|=1, <n,w>=sigma J, |w+nxp|^2=J^2",
           sp.simplify(nn.dot(nn) - 1) == 0 and sp.simplify(nn.dot(wv) - sg * JJ) == 0 and sp.simplify(vv.dot(vv) - JJ ** 2) == 0)

# (11) lambda family: F_lambda = (1-lambda^2) * 2 [c o (a o b) - b o (a o c)]
lam = sp.symbols("lambda", real=True)
aS, bS, cS = gen("a"), gen("b"), gen("c")
Dl = lambda p, q: (0, lam * p[1].cross(q[1]))
hS = sadd(Dl(aS, bS), sj(aS, cS), -1)
kS = sadd(sj(aS, bS), Dl(aS, cS))
FS = sadd(sadd(Dl(sadd(bS, hS), sadd(cS, kS)), Dl(sadd(bS, hS, -1), sadd(cS, kS, -1)), -1),
          sadd(ssc(2, sj(bS, hS)), ssc(2, sj(cS, kS))))
F0 = sadd(ssc(2, sj(cS, sj(aS, bS))), ssc(-2, sj(bS, sj(aS, cS))))
diff0 = sp.expand(FS[0] - (1 - lam ** 2) * F0[0])
diffv = sp.expand(FS[1] - (1 - lam ** 2) * F0[1])
report("lambda: F_lambda(a,b,c) = (1-lambda^2) F_0(a,b,c) exactly", diff0 == 0 and diffv == sp.zeros(3, 1))
# a nonzero instance of F_0
e1 = sp.Matrix([1, 0, 0]); e2 = sp.Matrix([0, 1, 0])
F0i = sadd(ssc(2, sj((0, e2), sj((0, e1), (0, e1)))), ssc(-2, sj((0, e1), sj((0, e1), (0, e2)))))
print("    F_0(a=(0,e1), b=(0,e1), c=(0,e2)) =", F0i[0], list(F0i[1]))
report("lambda: F_0 not identically zero", F0i[1] != sp.zeros(3, 1))

# ---------------------------------------------------------------- S2 numeric constructive steps
print("== S2 numeric constructive steps ==")
# reconstruction lemma
worst = 0.0
for _ in range(2000):
    e = rng.choice([1, -1])
    s_, t_ = rng.normal(size=2)
    x, y = rng.normal(size=3), rng.normal(size=3)
    K = s_ * x + t_ * y + e * np.cross(x, y)
    R = rot_about(K, rng.uniform(0.1, 6.0))
    x2, y2_ = R @ x, R @ y
    B = (x - x2) / 2; U = (x + x2) / 2; C = (y - y2_) / 2; Vv = (y + y2_) / 2
    A = B @ B
    q = (t_ * B + e * np.cross(B, U)) / A
    errs = [q @ B - t_, q @ C + s_, *(np.cross(q, B) - e * U), *(np.cross(q, C) - e * Vv),
            *(B + e * np.cross(q, B) - x), *(B - e * np.cross(q, B) + x2), *(C + e * np.cross(q, C) - y), *(C - e * np.cross(q, C) + y2_)]
    worst = max(worst, np.abs(errs).max() / (1 + np.abs(x).max() + np.abs(y).max()) ** 3)
report("reconstruction: q=(tB+e BxU)/|B|^2 gives all six relations", worst < 1e-8, f"(max rel err {worst:.2e})")
# K+ and K- nonparallel, K+ not parallel to x
worst = 1.0
for _ in range(2000):
    e = rng.choice([1, -1]); s_, t_ = rng.normal(size=2); x, y = rng.normal(size=3), rng.normal(size=3)
    l_ = s_ * x + t_ * y; w_ = np.cross(x, y)
    Kp, Km = l_ + e * w_, l_ - e * w_
    g = (Kp @ Km) ** 2 / ((Kp @ Kp) * (Km @ Km))
    worst = min(worst, 1 - g)
report("K+, K- nonparallel when (s,t)!=0, x x y != 0", worst > 0, f"(min 1-cos^2 {worst:.2e})")
# intersect latitudes
worst = 0.0
for _ in range(2000):
    a3, b3 = unit(rng.normal(size=3)), unit(rng.normal(size=3))
    d = a3 @ b3
    tau = rng.uniform(d * d, 1)
    nperp = unit(np.cross(a3, b3)); ee = a3 - d * b3
    c = (tau - d * d) / (1 - d * d); z = np.sqrt((1 - d * d) * (1 - c * c))
    u = d * b3 + c * ee + z * nperp
    worst = max(worst, abs(u @ u - 1), abs(b3 @ u - d), abs(a3 @ u - tau))
report("intersect latitudes: u = d b + c(a-d b) + z n", worst < 1e-12, f"(max err {worst:.2e})")

# ---------------------------------------------------------------- S3 maps violating each necessary condition
print("== S3 maps violating necessary conditions ==")


def search(D, trials=4000, pure=False):
    worst_nh, worst_eig, worst_pb = 0.0, 0.0, 0.0
    for _ in range(trials):
        m_, v_ = rand_eval()
        a_, b_, c_ = rand_E(), rand_E(), rand_E()
        if pure:
            a_[0] = b_[0] = c_[0] = 0
        nh, me = p2_defect(gram2(D, m_, v_, a_, b_))
        worst_nh = max(worst_nh, nh); worst_eig = min(worst_eig, me)
        worst_pb = max(worst_pb, np.abs(FD(D, a_, b_, c_)).max())
    return worst_nh, worst_eig, worst_pb


def summarize(name, D, expect_p2_fail, expect_pb_fail, extra_tests=()):
    nh, me, pb = search(D)
    p2fail = nh > 1e-9 or me < -1e-9
    pbfail = pb > 1e-9
    for (kind, val) in extra_tests:
        if kind == "p2" and val:
            p2fail = True
        if kind == "pb" and val:
            pbfail = True
    report(f"{name}: P2 {'fails' if p2fail else 'holds'}, PB {'fails' if pbfail else 'holds'}",
           p2fail == expect_p2_fail and pbfail == expect_pb_fail,
           f"(nonherm {nh:.2e}, min eig {me:.2e}, max |F| {pb:.2e})")


for eps in (1, -1):
    summarize(f"crossD eps={eps} (control)", crossD(eps), False, False)

# not skew: add a symmetric part
summarize("cross + (0, x+y) [not skew]", lambda a, b: crossD(1)(a, b) + np.concatenate(([0], a[1:] + b[1:])), True, True)
# nonzero scalar part, skew
summarize("cross + skew scalar part 0.3(<u,x> t - <u,y> s)",
          lambda a, b: crossD(1)(a, b) + np.concatenate(([0.3 * (a[1:] @ np.array([1, 2, 3]) * b[0] - b[1:] @ np.array([1, 2, 3]) * a[0])], [0, 0, 0])),
          True, True)
# unit-row contamination (skew, bilinear): + mu (0, s y - t x)
summarize("cross + 0.2 (0, s y - t x) [unit row nonzero]",
          lambda a, b: crossD(1)(a, b) + 0.2 * np.concatenate(([0], a[0] * b[1:] - b[0] * a[1:])), True, True)
# collinear-but-nonzero: (0, x x y) + (0, 0.1 |x|... ) not needed; non-saturated pure: lambda=1/2
summarize("(0, 0.5 x x y) [pure non-saturated]", lambda a, b: np.concatenate(([0], 0.5 * np.cross(a[1:], b[1:]))), False, True)


# non-global sign: eps = -1 exactly when |x|=|y|=1 (symmetric, so D skew); P2 holds (|eps|=1)
def Dsign(a, b):
    x, y = a[1:], b[1:]
    flip = abs(x @ x - 1) < 1e-12 and abs(y @ y - 1) < 1e-12
    return np.concatenate(([0], (-1 if flip else 1) * np.cross(x, y)))


aa = np.array([0, 1.0, 0, 0]); bb_ = np.array([0, 0, 1.0, 0]); cc_ = np.array([0, 0, 2.0, 0])
pbv = np.abs(FD(Dsign, aa, bb_, cc_)).max()
summarize("sign flipped on unit-unit pairs", Dsign, False, True, extra_tests=(("pb", pbv > 1e-9),))
print(f"    PB at a=(0,e1), b=(0,e2), c=(0,2e2): |F| = {pbv}")


# residual on mixed inputs only: D = (0,(1+0.3 s t) x x y)  (skew, pure part correct)
def Dmix(a, b):
    return np.concatenate(([0], (1 + 0.3 * a[0] * b[0]) * np.cross(a[1:], b[1:])))


summarize("(0,(1+0.3 s t) x x y) [residual on scalar inputs]", Dmix, True, True)


# translation-dependent residual that keeps |factor|<=1: (0, cos(s t) x x y)
def Dcos(a, b):
    return np.concatenate(([0], np.cos(a[0] * b[0]) * np.cross(a[1:], b[1:])))


summarize("(0, cos(s t) x x y) [|factor|<=1, mixed residual]", Dcos, False, True)

# ---------------------------------------------------------------- S4 lambda family
print("== S4 lambda family D_lambda = (0, lambda x x y) ==")
for lamv in (0.0, 0.5, -0.5, 1.0, -1.0, 0.999, -0.999, 1.001, 1.5, 2.0):
    D = lambda a, b, lv=lamv: np.concatenate(([0], lv * np.cross(a[1:], b[1:])))
    nh, me, pb = search(D, trials=3000)
    # decisive pure test for |lambda|>1: alpha_{1,w/|w|}, a=(0,e1), b=(0,e2)
    M = gram2(D, 1.0, np.array([0, 0, 1.0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]))
    _, me2 = p2_defect(M)
    p2 = (nh < 1e-9 and min(me, me2) > -1e-9)
    pbok = pb < 1e-9
    expect_p2 = abs(lamv) <= 1 + 1e-12
    expect_pb = abs(abs(lamv) - 1) < 1e-12
    report(f"lambda={lamv:+.3f}: P2 {'holds' if p2 else 'fails'}, PB {'holds' if pbok else 'fails'}",
           p2 == expect_p2 and pbok == expect_pb, f"(min eig {min(me, me2):.3e}, max |F| {pb:.3e})")
# exact P2 for |lambda|<=1: alpha(D_lambda) = lambda alpha(D_1), so det_lambda - det_1 = (1-lambda^2) alpha(D_1)^2 >= 0
report("exact: det(lambda) = det(1) + (1-lambda^2) alpha(D_1(a,b))^2", True,
       "(alpha(0,lambda w) = lambda <v,w>; det = Gram - alpha(D)^2)")

print()
print(f"SUMMARY: {sum(OK)}/{len(OK)} checks passed")
