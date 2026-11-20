# Checks for Section 10E (sec:m13); vendored from the m13 module script.
#!/usr/bin/env python3
"""Finite checks for section.tex. NumPy is the only external dependency."""

from fractions import Fraction
from itertools import combinations, product
from math import gcd, isqrt

import numpy as np

SQRT5 = np.sqrt(5)
PHI = (1 + SQRT5) / 2
RHO = PHI - 1
PSI = -RHO


def close(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def fibonacci(n):
    values = [0, 1]
    while len(values) <= n:
        values.append(sum(values[-2:]))
    return values


def phi_power(k):
    """The exact coefficient pair of phi**k for k >= 0."""
    a, b = 1, 0
    for _ in range(k):
        a, b = b, a + b
    return a, b


def golden_sign(a, b):
    """Sign of a + b*phi, based on integer comparisons."""
    rational = 2 * a + b
    if b == 0:
        return (rational > 0) - (rational < 0)
    if b < 0:
        return -golden_sign(-a, -b)
    if rational >= 0:
        return 1
    return 1 if 5 * b * b > rational * rational else -1


def floor_phi(n):
    if n < 0:
        return -1 - floor_phi(-n)
    return (n + isqrt(5 * n * n)) // 2


def route(b):
    return floor_phi(b + 1) - b - 1, b


def legal_digits(value, fib):
    picked = []
    for k in range(len(fib) - 1, 1, -1):
        if fib[k] <= value:
            picked.append(k)
            value -= fib[k]
    assert value == 0
    return picked


def in_golden_window(pair, closed=True):
    a, b = pair
    # Compare (a+b)-b*phi to phi-2 and phi-1.
    lhs = golden_sign(a + b + 2, -b - 1)
    rhs = golden_sign(a + b + 1, -b - 1)
    return (lhs >= 0 if closed else lhs > 0) and rhs <= 0


def check_strip():
    close(np.linalg.det([[1, PHI], [1, PSI]]), -SQRT5)
    close([PHI - 2, PHI - 1, RHO + RHO**2], [-RHO**2, RHO, 1])
    pairs = [route(b) for b in range(-250, 251)]
    assert set(pairs) == {
        (a, b)
        for b in range(-250, 251)
        for a in range(-160, 161)
        if in_golden_window((a, b), closed=False)
    }
    assert all(in_golden_window(pair) for pair in pairs)
    assert in_golden_window((-1, -1))
    assert not in_golden_window((-1, -1), closed=False)
    assert [route(b) for b in (-2, -1, 0, 1)] == [(-1, -2), (0, -1), (0, 0), (1, 1)]
    points = sorted([a + b * PHI for a, b in pairs] + [-PHI**2])
    gaps = np.diff(points)
    assert set(np.round(gaps, 10)) == set(np.round([1, PHI, PHI**2], 10))
    assert np.count_nonzero(np.isclose(gaps, 1)) == 1
    close([np.min(gaps) / 2, np.max(gaps) / 2], [0.5, PHI**2 / 2])
    assert np.max(gaps) / 2 < 3

    for a, b in product(range(-12, 13), repeat=2):
        if (a, b) != (0, 0):
            norm = a * a + a * b - b * b
            close((a + b * PHI) * (a + b * PSI), norm)
            assert abs(norm) >= 1

    fib = fibonacci(22)
    expansions = []
    branches = [set(), set()]
    for value in range(5000):
        digits = legal_digits(value, fib)
        assert all(k - j >= 2 for k, j in zip(digits, digits[1:]))
        a = sum(phi_power(k)[0] for k in digits)
        b = sum(phi_power(k)[1] for k in digits)
        assert (a, b) == route(value)
        expansions.append(a + b * PHI)
        assert abs(a + b * PHI - SQRT5 * value) < 1
        if value:
            assert golden_sign(a + b, -b) == (-1) ** digits[-1]
        branches[0].add((b, a + b))
        branches[1].add((a + b + 1, a + 2 * b + 1))
    assert branches[0].isdisjoint(branches[1])
    assert set(map(route, range(5000))) <= branches[0] | branches[1]
    for branch in branches:
        assert all(route(b) == (a, b) for a, b in branch)

    closed = set(pairs) | {(-1, -1)}
    first = {(b, a + b) for a, b in closed}
    second = {(a + b + 1, a + 2 * b + 1) for a, b in closed}
    assert first & second == {(-1, -2)}  # -phi**3
    assert {pair for pair in closed if abs(pair[1]) <= 100} <= first | second

    for n in (0, 1, 2, 37, 4999):
        assert sum(0 <= x < expansions[n] for x in expansions) == n
    close(4999 / expansions[-1], 1 / SQRT5, atol=5e-5)
    close(expansions[2] - expansions[1], PHI)
    s = np.array([floor_phi(v + 1) - 1 for v in range(6000)])
    v = np.arange(len(s))
    np.testing.assert_array_equal(np.isclose(np.diff(expansions), PHI), np.diff(s[:5000]) == 1)
    first_fiber, second_fiber = 2 * s - 3 * v, 2 * v - s
    np.testing.assert_array_equal(second_fiber, np.floor((v + 1) / PHI**2))
    for a in range(1, 999):
        expected = np.arange(np.floor((a - 1) * PHI - 1) + 1, np.floor((a + 1) * PHI) + 1)
        np.testing.assert_array_equal(second_fiber[first_fiber == a], expected)
        assert len(expected) in (4, 5)
    assert np.count_nonzero(second_fiber == 0) == 2
    for b in range(1, 999):
        assert np.count_nonzero(second_fiber == b) in (2, 3)


def strip_count(window, start, length):
    lower, upper = window
    b = np.arange(np.floor((start - upper) / SQRT5), np.ceil((start + length - lower) / SQRT5) + 1)
    counts = 0
    for shift in range(1 + int(np.ceil(upper - lower))):
        a = np.ceil(lower - b * PSI) + shift
        inner = a + b * PSI
        outer = a + b * PHI
        counts += int(np.count_nonzero((lower <= inner) & (inner < upper) & (start <= outer) & (outer < start + length)))
    return counts


def check_rotation_and_density():
    n = np.arange(30000)
    for start in (-0.71, 0, 0.29, 0.99):
        phases = np.remainder(start + n * PHI, 1)
        for k in range(1, 13):
            ratio = np.exp(2j * np.pi * k * PHI)
            character = np.exp(2j * np.pi * k * phases).mean()
            predicted = np.exp(2j * np.pi * k * start) * (1 - ratio**len(n)) / (len(n) * (1 - ratio))
            close(character, predicted, atol=3e-10)
            assert abs(character) <= 2 / (len(n) * abs(1 - ratio)) + 1e-9
        close(np.mean((0.15 <= phases) & (phases < 0.37)), 0.22, atol=3e-4)
    for window in ((-RHO**2, RHO), (-0.35, -0.12), (-2.1, 3.3)):
        for start in (-721.72, 0, 873.43):
            close(strip_count(window, start, 30000) / 30000,
                  (window[1] - window[0]) / SQRT5, atol=3e-4)
    samples = np.cos(2 * np.pi * n * PHI)
    hits = np.flatnonzero((samples > 0.2) & (samples < 0.8))
    in_sample = hits[hits < 89]
    assert len(in_sample) == 21 and np.max(np.diff(in_sample)) == 7
    expected = (np.arccos(0.2) - np.arccos(0.8)) / np.pi
    close(len(hits) / len(n), expected, atol=3e-4)
    noisy = samples + 1 / (n + 1)
    noisy_hits = np.flatnonzero((noisy > 0.2) & (noisy < 0.8))
    assert np.max(np.diff(noisy_hits[noisy_hits > 100])) <= 11


def check_doubling():
    representatives = [Fraction(0), Fraction(1, 7), Fraction(1, 3), Fraction(2, 5)]
    periods = [1, 3, 2, 4]
    frequencies = [(1, 1), (Fraction(2, 3), Fraction(1, 3)), (Fraction(1, 2), 0),
                   (Fraction(1, 2), Fraction(1, 4))]
    for x, period, expected in zip(representatives, periods, frequencies):
        orbit = []
        y = x
        for i in range(period):
            assert y == (2**i * x) % 1
            orbit.append(y)
            y = (2 * y) % 1
        assert y == x
        observed = tuple(Fraction(sum(z < bound for z in orbit), period)
                         for bound in (Fraction(1, 2), Fraction(1, 4)))
        assert observed == expected
    for m in range(1, 30, 2):
        for e in range(5, 16):
            x = Fraction(m, 2**e)
            assert (2**(e - 1) * x) % 1 == Fraction(1, 2)
            assert (2**e * x) % 1 == 0
    for x, expected_step in ((0.1, 55), (np.sqrt(2) / 2, 53), (np.pi - 3, 48)):
        y, step = x, 0
        while y:
            y = (2 * y) % 1
            step += 1
        assert step == expected_step


def check_diophantine():
    fib = fibonacci(71)
    for n in range(70):
        assert fib[n] * fib[n + 2] - fib[n + 1]**2 == (-1)**(n + 1)
        assert phi_power(n + 1) == (fib[n], fib[n + 1])
    for k in range(1, 70):
        p, q = fib[k + 1], fib[k]
        assert gcd(p, q) == 1
        assert golden_sign(p, -q) == (-1)**k
        assert abs(p * p - p * q - q * q) == 1
        close(SQRT5 * q * RHO**k, 1 - (-1)**k * RHO**(2 * k), atol=2e-14)
    for q in range(1, 501):
        for p in range(floor_phi(q) - 2, floor_phi(q) + 4):
            eps = abs(p / q - PHI)
            bound = 2 / (q * q * (np.sqrt(5 + 4 / q**2) + SQRT5))
            assert eps >= bound - 5e-16
            assert bound > 1 / (SQRT5 * q * q + q)
            equal = abs(eps - bound) < 5e-16
            assert equal == (p / q > PHI and abs(p * p - p * q - q * q) == 1)
    assert 2 - PHI < 1 / SQRT5
    pi_points = np.sort(np.remainder(np.arange(114) * (np.pi - 3), 1))
    pi_gaps = np.diff(np.r_[pi_points, pi_points[0] + 1])
    close(np.min(pi_gaps), abs(113 * np.pi - 355))
    close(np.min(pi_gaps), 3.0e-5, atol=3e-7)
    assert round(np.max(pi_gaps) / np.min(pi_gaps)) == 294
    rotation = 2 - PHI
    shortest = (1, 0)
    best = (1, 0)
    argmin = 1
    for n in range(1, 2001):
        if n >= 2:
            m = n - 1
            p = floor_phi(m)
            # Minimize |m*phi - p| by comparing twice the fraction to 1.
            if golden_sign(-2 * p - 1, 2 * m) < 0:
                candidate = (-p, m)
            else:
                candidate = (p + 1, -m)
            if golden_sign(candidate[0] - shortest[0], candidate[1] - shortest[1]) < 0:
                shortest = candidate
        scaled = (n * shortest[0], n * shortest[1])
        if golden_sign(scaled[0] - best[0], scaled[1] - best[1]) < 0:
            best, argmin = scaled, n
        points = np.sort(np.remainder(np.arange(n) * rotation, 1))
        gaps = np.diff(np.r_[points, points[0] + 1])
        gap = np.min(gaps)
        assert np.max(gaps) <= PHI**2 * gap + 2e-12
        assert n * gap > 1 / SQRT5
        close(n * gap, scaled[0] + scaled[1] * PHI, atol=2e-9)
    assert argmin == 1598
    close(best[0] + best[1] * PHI, 0.44749, atol=5e-6)
    close((fib[31] + 1) * RHO**31, 1 / SQRT5, atol=5e-7)
    close(1 / rotation, PHI + 1)


def grid(depth):
    legal = (bits for bits in product((0, 1), repeat=depth)
             if all(not (a and b) for a, b in zip(bits, bits[1:])))
    weighted = [(sum(bit * RHO**j for j, bit in enumerate(bits, 1)), bits) for bits in legal]
    return sorted(weighted)


def advance(state):
    kind, t = state
    if kind == "S":
        return "L", t
    if t <= RHO:
        return "L", PHI * t
    return "S", PHI**2 * t - PHI


def arm(state):
    kind, t = state
    return min(t, 1 - t) * (RHO if kind == "S" else 1)


def check_survivor():
    fib = fibonacci(16)
    grids = [None] + [grid(q) for q in range(1, 12)]
    for q in range(1, 11):
        nodes = np.array([x for x, _ in grids[q]])
        refined = np.array([x for x, _ in grids[q + 1]])
        assert len(nodes) == fib[q + 2]
        scaled_prefix = [(a + b * PHI) / PHI**(q + 2)
                         for a, b in map(route, range(fib[q + 2]))]
        close(nodes, scaled_prefix)
        for (lo, bits), hi in zip(grids[q], nodes[1:]):
            kind = "S" if bits[-1] else "L"
            close(hi - lo, RHO**(q + bits[-1]))
            split = refined[(refined > lo + 1e-11) & (refined < hi - 1e-11)]
            close(split, [] if kind == "S" else [lo + RHO**(q + 1)])
            if q > 7:
                continue
            for t in (0, 0.2, 0.5, RHO, 0.8, 1):
                state = kind, t
                x = lo + t * (hi - lo)
                for r in range(q, 12):
                    grid_r = np.array([y for y, _ in grids[r]])
                    close(PHI**r * np.min(np.abs(grid_r - x)), arm(state))
                    state = advance(state)
    state = ("L", RHO**2 / 2)
    expected = [("L", RHO / 2), ("L", 0.5), ("L", PHI / 2), ("S", 0.5), ("L", 0.5)]
    for kind, t in expected:
        state = advance(state)
        assert state[0] == kind
        close(state[1], t)
    close([arm(s) for s in expected[1:4]], [0.5, RHO**2 / 2, RHO / 2])


def vectors(t):
    return np.array([(0, 1, t), (0, 1, -t), (1, t, 0),
                     (1, -t, 0), (t, 0, 1), (-t, 0, 1)])


def check_icosahedral_cut():
    outer = vectors(PHI)
    inner = vectors(PSI)
    close(outer.T @ outer, 2 * (1 + PHI**2) * np.eye(3))
    close(inner.T @ inner, 2 * (1 + PSI**2) * np.eye(3))
    close(outer.T @ inner, np.zeros((3, 3)))
    basis = np.hstack([outer, inner])
    close(abs(np.linalg.det(basis)), 40 * SQRT5)
    close(abs(np.linalg.det(basis)) / SQRT5**3, 8)
    lengths = np.linalg.norm(inner, axis=1)
    unit = inner / lengths[:, None]
    gram = unit @ unit.T
    close(np.abs(gram[np.triu_indices(6, 1)]), np.full(15, 1 / SQRT5))
    for i, j, k in combinations(range(6), 3):
        minor = gram[np.ix_([i, j, k], [i, j, k])]
        assert np.linalg.det(minor) > 0
        close(np.linalg.det(minor), 1 - 3 / 5 + 2 * minor[0, 1] * minor[1, 2] * minor[2, 0])
    close(np.sqrt((1 + 1 / SQRT5) / (1 - 1 / SQRT5)), PHI)
    vertices, edges = set(), set()
    normals, supports = [], []
    for i, j in combinations(range(6), 2):
        normal = np.cross(inner[i], inner[j])
        normal /= np.linalg.norm(normal)
        for direction in (normal, -normal):
            fixed = np.sign(inner @ direction)
            face = set()
            for a, b in product((-1, 1), repeat=2):
                signs = fixed.copy()
                signs[i], signs[j] = a, b
                face.add(tuple(signs))
            assert len(face) == 4
            vertices.update(face)
            edges.update(tuple(sorted((p, q))) for p, q in combinations(sorted(face), 2)
                         if sum(a != b for a, b in zip(p, q)) == 1)
            normals.append(direction)
            supports.append(sum(np.abs(inner @ direction)) / 2)
    assert len(normals) == 30 and len(vertices) == 32 and len(edges) == 60
    degree = {p: sum(p in edge for edge in edges) for p in vertices}
    assert list(degree.values()).count(5) == 12 and list(degree.values()).count(3) == 20
    for axis in inner:
        assert degree[tuple(np.sign(inner @ axis))] == 5
        assert degree[tuple(np.sign(-inner @ axis))] == 5
    assert degree[tuple(np.sign(inner @ np.ones(3)))] == 3
    close(2 * np.pi - 5 * np.pi / 3, np.pi / 3)
    projection = np.diag([1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(projection @ projection, projection)
    assert np.linalg.matrix_rank(projection) == 3

    # The facet inequalities check translation and the margin of a convex window.
    normals, supports = np.array(normals), np.array(supports)
    accepted = inner[0] / 3
    margin = np.min(supports - normals @ accepted)
    assert 0 < margin <= min(supports)
    assert margin >= min(supports) - np.linalg.norm(accepted) - 1e-12
    direction = normals[np.argmin(supports - normals @ accepted)]
    assert np.all(normals @ (accepted + 0.99 * margin * direction) < supports)
    assert np.any(normals @ (accepted + 1.01 * margin * direction) > supports)
    phases = np.array(list(product((-0.5, 0, 0.5), repeat=3)))
    prescribed = [np.zeros(3), inner[0]]
    test_region = [p for p in phases
                   if all(np.all(normals @ (z - p) <= supports + 1e-12) for z in prescribed)]
    assert 0 < len(test_region) < len(phases)
    assert all(np.all(normals @ (prescribed[1] - p) <= supports + 1e-12) for p in test_region)


def check_models():
    sequence = [{1, 2, 3, 4}, {1, 2, 3}, {1, 2}, {1, 2}]
    assert all(after <= before for before, after in zip(sequence, sequence[1:]))
    assert sum(before != after for before, after in zip(sequence, sequence[1:])) == 2
    close(np.dot([1, 0], [np.cos(0.03), np.sin(0.03)]), np.cos(0.03))
    sigma, r, beta = 10, 28, 8 / 3
    for x, y, z in ((0, 0, 0), (1, 2, 3), (-5, 6, 7)):
        jacobian = [[-sigma, sigma, 0], [r - z, -1, -x], [y, x, -beta]]
        close(np.trace(jacobian), -41 / 3)
    close(np.exp(-41 / 3), 1.1605334678748065e-6, atol=1e-10)
    assert round(2 + 0.9056 / 14.5723, 2) == 2.06


def main():
    checks = (
        check_strip, check_rotation_and_density, check_doubling, check_diophantine,
        check_survivor, check_icosahedral_cut, check_models,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"All {len(checks)} check groups passed.")


if __name__ == "__main__":
    main()
