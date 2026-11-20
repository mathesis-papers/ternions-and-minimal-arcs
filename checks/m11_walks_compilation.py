#!/usr/bin/env python3
"""Finite reproducibility checks for Sections 6B and 10D (sec:m11, sec:m11b); vendored from the m11 module script.
The module group that read the module's own source files was dropped."""
from itertools import combinations, product
from pathlib import Path
import re

import numpy as np


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
I = np.eye(2)
PAULI = np.array([X, Y, Z])
H = (X + Z) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])
RNG = np.random.default_rng(110)
EYE_ATOL = 3e-10


def close(actual, expected, atol=EYE_ATOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def norm(matrix):
    return np.linalg.norm(matrix, ord=2)


def adj(matrix):
    return matrix.conj().T


def rotation(axis, angle):
    return I * np.cos(angle / 2) - 1j * axis * np.sin(angle / 2)


def equatorial(phase):
    return np.cos(phase) * X + np.sin(phase) * Y


def hermitian_function(matrix, function):
    spectrum, vectors = np.linalg.eigh(matrix)
    return (vectors * function(spectrum)) @ adj(vectors)


def unitary_frame(rows, columns):
    entries = RNG.normal(size=(rows, columns)) + 1j * RNG.normal(size=(rows, columns))
    return np.linalg.qr(entries)[0]


def check_reflections():
    for n, a, b in [(8, 5, 4), (8, 2, 3), (5, 0, 2), (4, 2, 4)]:
        left, right = unitary_frame(n, a), unitary_frame(n, b)
        p, q = left @ adj(left), right @ adj(right)
        walk = (2 * p - np.eye(n)) @ (2 * q - np.eye(n))
        singular = np.linalg.svd(adj(left) @ right, compute_uv=False)
        k = sum(np.isclose(singular, 1))
        interior = singular[(singular > 1e-10) & (singular < 1 - 1e-10)]
        target = [1] * (n - a - b + 2 * k) + [-1] * (a + b - 2 * k - 2 * len(interior))
        for s in interior:
            angle = 2 * np.arccos(s)
            target += [np.exp(1j * angle), np.exp(-1j * angle)]
        # Traces of powers compare multisets without unstable complex sorting.
        spectrum = np.linalg.eigvals(walk)
        for j in range(1, n + 1):
            close(sum(spectrum ** j), sum(np.array(target) ** j))
    a = np.array([1.0, 0.0])
    b = np.array([0.4, np.sqrt(1 - 0.4 ** 2)])
    w = (2 * np.outer(a, a) - I) @ (2 * np.outer(b, b) - I)
    c = 2 * (a @ b) ** 2 - 1
    close(w @ w, 2 * c * w - I)

    weights = np.array([[1, 3, 2], [3, 1, 5], [2, 5, 4]])
    row_sum = weights.sum(axis=1)
    markov = weights / row_sum[:, None]
    pi = row_sum / row_sum.sum()
    frame = np.zeros((9, 3))
    for i in range(3):
        frame[3 * i:3 * i + 3, i] = np.sqrt(markov[i])
    swap_frame = frame.reshape(3, 3, 3).transpose(1, 0, 2).reshape(9, 3)
    close(adj(frame) @ frame, np.eye(3))
    close(adj(frame) @ swap_frame, np.sqrt(markov * markov.T))
    close(adj(frame) @ swap_frame, np.sqrt(pi[:, None] / pi) * markov)
    for data, expected in (
        ([[1, 1], [1, 1]], [1, 1, -1, -1]),
        ([[1, 1], [1, -1]], [1j, 1j, -1j, -1j]),
    ):
        cross = np.array(data) / 2
        complement = hermitian_function(I - cross.T @ cross, lambda x: np.sqrt(x.clip(0)))
        left = np.vstack((I, np.zeros_like(I)))
        right = np.vstack((cross, complement))
        walk = (2 * left @ adj(left) - np.eye(4)) @ (2 * right @ adj(right) - np.eye(4))
        for j in range(1, 5):
            close(np.trace(np.linalg.matrix_power(walk, j)), sum(np.array(expected) ** j))


def path_matrix(n, defect):
    edges = np.ones(2 * n - 1)
    edges[n - 1] = defect
    return np.diag(edges, 1) + np.diag(edges, -1)


def chebyshev(n, t):
    values = [1.0]
    previous = 0.0
    for _ in range(n):
        previous, current = values[-1], t * values[-1] - previous
        values.append(current)
    return np.array(values)


def check_defect():
    for n in (1, 2, 3, 5, 14, 24):
        for b in (0.5, 1.0, (n + 1) / n, np.sqrt(2), 1.7):
            matrix = path_matrix(n, b)
            values, vectors = np.linalg.eigh(matrix)
            spacing = np.diff(values)
            assert min(spacing) > 0
            close(values, -values[::-1])
            upper = sum(values > 2 + 1e-10)
            assert upper == int(b > (n + 1) / n)
            assert sum(np.isclose(values, 2, atol=1e-10, rtol=0)) == int(b == (n + 1) / n)
            weights = vectors[0, :] ** 2
            parity = np.sign(vectors[-1, :] * vectors[0, :])
            close(parity[:-1], -parity[1:])
            close(vectors[-1, :], parity * vectors[0, :])
            for t, sign, weight in zip(values, parity, weights):
                polynomials = chebyshev(n, t)
                close(polynomials[-1] / polynomials[-2], sign * b, atol=3e-6)
                close(1 / (2 * sum(polynomials[:-1] ** 2)), weight)
            close(weights.sum(), 1)
            average_limit = weights @ weights
            assert average_limit >= 1 / (2 * n) - 1e-13
            delta = min(spacing)
            if b == np.sqrt(2):
                assert delta >= 2 * b / ((1 + b) ** 2 * n ** 3) - 1e-13
            for epsilon in (0.03, 0.7, 1.4):
                tau = 4 * n / (epsilon * delta)
                frequencies = values[:, None] - values
                kernel = np.exp(-0.5j * tau * frequencies) * np.sinc(tau * frequencies / (2 * np.pi))
                transfer = (parity * weights) @ kernel @ (parity * weights)
                close(transfer.imag, 0)
                assert abs(transfer.real - average_limit) <= 2 * (1 - average_limit) / (tau * delta)
                assert transfer.real > (1 - epsilon) / (2 * n)
    b = np.sqrt(2)
    close(np.linalg.eigvalsh(path_matrix(1, b)), [-b, b])
    for n, value in [(64, 58.11), (128, 57.06), (256, 56.47)]:
        computed = n ** 3 * min(np.diff(np.linalg.eigvalsh(path_matrix(n, b))))
        close(round(computed, 2), value)
    close(round(4 * b * np.pi ** 2, 2), 55.83)


def walk_state(size, steps, coin, origin=0):
    state = np.zeros((2, size), dtype=complex)
    state[:, origin] = coin
    for _ in range(steps):
        coin_step = H @ state
        state = np.array([np.roll(coin_step[0], -1), np.roll(coin_step[1], 1)])
    return state


def check_coin():
    balanced = np.array([1, 1j]) / np.sqrt(2)
    for size, steps in product((1, 2, 9, 12), (0, 1, 3, 20)):
        reverse = -np.arange(size) % size
        state = walk_state(size, steps, balanced)
        probabilities = abs(state) ** 2
        close(probabilities[0], probabilities[1, reverse])
        close(probabilities.sum(), 1)
        coin = unitary_frame(2, 1).ravel()
        other = abs(walk_state(size, steps, coin)) ** 2
        close(probabilities[0] + probabilities[1, reverse], other[0] + other[1, reverse])
    for _ in range(60):
        a, b, u, v = RNG.random(4)
        first = np.array([[a, b * v], [b * u, 0]])
        equal = np.sqrt((u * u + v * v) / 2)
        second = np.array([[a, b * equal], [b * equal, 0]])
        assert norm(second) <= norm(first) + 1e-13
        close(np.linalg.norm(first, 'fro') ** 2, np.linalg.norm(second, 'fro') ** 2)
    for steps, width in ((50, 27.07), (60, 32.5)):
        coordinates = np.arange(-steps, steps + 1)
        probabilities = (abs(walk_state(len(coordinates), steps, balanced, steps)) ** 2).sum(axis=0)
        close(probabilities @ coordinates, 0)
        close(round(np.sqrt(probabilities @ (coordinates ** 2)), 2 if steps == 50 else 1), width)
        if steps == 50:
            close(coordinates[np.isclose(probabilities, probabilities.max())], [-34, 34])
        classical = np.array([1.0])
        for _ in range(steps):
            classical = np.convolve(classical, [0.5, 0, 0.5])
        close(classical @ coordinates ** 2, steps)
    real_coin = np.ones(2) / np.sqrt(2)
    real_probs = abs(walk_state(101, 50, real_coin, 50)) ** 2
    close(round(max(abs(real_probs[0] - real_probs[1, ::-1])), 3), 0.145)
    close(walk_state(5, 2, [1, 0])[1, 0], 0.5)


def round_distance(left, right):
    c = np.trace(adj(left) @ right).real / 2
    return np.arccos(np.clip(c, -1, 1))


def check_compile():
    for _ in range(35):
        ideal, actual = I.copy(), I.copy()
        angles, roots = [], []
        for _ in range(7):
            axis = RNG.normal(size=3)
            axis /= np.linalg.norm(axis)
            g = rotation(np.einsum('i,ijk->jk', axis, PAULI), RNG.uniform(-2, 2))
            h = g @ rotation(Z, RNG.uniform(-2, 2))
            ideal, actual = ideal @ g, actual @ h
            local = round_distance(g, h)
            angles.append(local)
            roots.append(np.sqrt(1 - np.cos(local)))
            angle = round_distance(ideal, actual)
            budget = min(np.pi, sum(angles))
            assert angle <= budget + 1e-12
            assert 1 - np.cos(angle) <= 1 - np.cos(budget) + 1e-12
            assert 1 - np.cos(budget) <= min(2, sum(roots) ** 2) + 1e-12
            assert sum(roots) ** 2 <= len(roots) * sum(np.square(roots)) + 1e-12
            assert norm(ideal - actual) <= np.sqrt(2) * sum(roots) + 1e-12
    a = 0.27
    local_error = rotation(Z, 2 * a)
    close(round_distance(I, local_error @ local_error), 2 * a)
    assert 1 - np.cos(2 * a) > 2 * (1 - np.cos(a))
    delta, budget = 0.8, 0.02
    for parameter in np.linspace(0, 1, 21):
        left = np.array([1.0, 0])
        right = np.array([np.cos(np.pi - delta), np.sin(np.pi - delta)])
        near_left = np.array([np.cos(budget), np.sin(budget)])
        near_right = np.array([np.cos(np.pi - delta + budget), np.sin(np.pi - delta + budget)])
        chord = (1 - parameter) * left + parameter * right
        near_chord = (1 - parameter) * near_left + parameter * near_right
        cosine = (chord @ near_chord) / (np.linalg.norm(chord) * np.linalg.norm(near_chord))
        assert np.arccos(np.clip(cosine, -1, 1)) <= np.pi * budget / np.sin(delta / 4)
    for p, n, queries in ((1, 0, 1), (1 / 4, 1, 3), (1 / 16, 3, 7), (1 / 64, 6, 13)):
        steps = int(np.ceil(np.pi / (4 * np.arcsin(np.sqrt(p))) - 0.5 - 1e-12))
        assert steps == n and queries == 2 * n + 1
        alpha = np.pi / (4 * n + 2)
        assert 0 < np.sin(alpha) ** 2 / p <= 1 + 1e-12
        close(np.sin((2 * n + 1) * alpha), 1)
        assert queries <= (np.pi / 2 + 2) / np.sqrt(p)
    close(round(np.log(5) / np.log(1.5), 2), 3.97)
    for scale in (0.02, 0.01):
        a, b = -1j * scale * X, -1j * scale * Z
        commutator = rotation(X, 2 * scale) @ rotation(Z, 2 * scale)
        commutator = commutator @ rotation(X, -2 * scale) @ rotation(Z, -2 * scale)
        assert norm(commutator - I - (a @ b - b @ a)) <= 10 * scale ** 3


def bloch_action(unitary):
    return np.array([[np.trace(p @ unitary @ q @ adj(unitary)).real / 2
                      for q in PAULI] for p in PAULI])


def check_orbits():
    lift_h, lift_t = 1j * H, np.exp(-1j * np.pi / 8) * T
    for gate, power in ((H, 2), (T, 8), (lift_h, 4), (lift_t, 16)):
        close(np.linalg.matrix_power(gate, power - 1), adj(gate))
    for gate in (lift_h, lift_t):
        close(np.linalg.det(gate), 1)
    ex, ey, ez = np.eye(3)
    rh, rt = bloch_action(H), bloch_action(T)
    close(rh @ ex, ez)
    close(rh @ ey, -ey)
    close(rt @ ex, (ex + ey) / np.sqrt(2))
    points = ez[None, :]
    target = [ez, ex, (ex + ey) / np.sqrt(2), ey, (ez - ey) / np.sqrt(2)]
    for length, count in enumerate((1, 2, 3, 5)):
        close(len(points), count)
        assert np.linalg.matrix_rank(points - points[0], tol=1e-9) == length
        for vertex in target[:count]:
            assert min(np.linalg.norm(points - vertex, axis=1)) < 1e-10
        if length < 3:
            points = np.unique(np.round(np.vstack((points, points @ rh.T, points @ rt.T)), 12), axis=0)
    assert np.all(points[:, [0, 2]] >= 0)
    close(1 - max(points @ ey), 0)
    close(1 - max(points @ (-ex)), 1)
    normal = np.ones(3) / np.sqrt(3)
    close(1 - max(np.vstack((np.eye(3), -np.eye(3))) @ normal), 1 - 1 / np.sqrt(3))



def hull_support_heights(points):
    """Incremental triangular hull, with an outward support check."""
    tetrahedron = [0]
    for idx in range(1, len(points)):
        candidates = points[tetrahedron + [idx]] - points[0]
        if np.linalg.matrix_rank(candidates, tol=1e-10) == len(tetrahedron):
            tetrahedron.append(idx)
        if len(tetrahedron) == 4:
            break
    assert len(tetrahedron) == 4
    interior = points[tetrahedron].mean(axis=0)

    def outward_normals(faces):
        vertices = points[faces]
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        assert np.all(lengths > 1e-12)
        signs = np.sign(np.sum(normals * (vertices[:, 0] - interior), axis=1))
        return normals * (signs / lengths)[:, None]

    faces = np.array(list(combinations(tetrahedron, 3)))
    for idx, point in enumerate(points):
        if idx in tetrahedron:
            continue
        normals = outward_normals(faces)
        visible = np.sum(normals * (point - points[faces[:, 0]]), axis=1) > 1e-10
        assert np.any(visible)
        front = faces[visible]
        edges = np.vstack((front[:, [0, 1]], front[:, [0, 2]], front[:, [1, 2]]))
        edges, incidence = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        horizon = edges[incidence == 1]
        faces = np.vstack((faces[~visible], np.column_stack((horizon, np.full(len(horizon), idx)))))
    normals = outward_normals(faces)
    heights = np.sum(normals * points[faces[:, 0]], axis=1)
    assert np.max(normals @ points.T - heights[:, None]) < 1e-10
    assert len(np.unique(faces)) == len(points)
    assert len(faces) == 2 * len(points) - 4
    assert min(heights) > 0
    return heights


def check_large_orbits():
    rh, rt = bloch_action(H), bloch_action(T)
    points = np.array([[0., 0., 1.]])
    for length in range(1, 21):
        layer = np.vstack((points, points @ rh.T, points @ rt.T))
        _, indices = np.unique(np.round(layer, 11), axis=0, return_index=True)
        points = layer[indices]
        if length in (6, 20):
            count, angle = {6: (19, 47.27), 20: (971, 9.10)}[length]
            assert len(points) == count
            height = min(hull_support_heights(points))
            close(round(np.rad2deg(np.arccos(height)), 2), angle)
        if length == 10:
            assert len(points) == 64
            theta, phi = np.deg2rad([110, 200])
            target = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            nearest = np.array([0., 0., 1.])
            for letter in reversed("HTHTTTTTH"):
                nearest = (rh if letter == "H" else rt) @ nearest
            height = max(points @ target)
            close(nearest @ target, height)
            close(round(height, 3), 0.956)
            close(round(np.rad2deg(np.arccos(height)), 2), 17.04)


def route_bridge(a, b, c, d, scale=1):
    left = np.zeros((3, a, b))
    right = np.zeros((3, d, c))
    for j in range(b):
        if j < a:
            left[0, j, j] = scale
        else:
            left[1, 2 * (j - a), j] = scale
            left[2, 2 * (j - a) + 1, j] = scale
    for k in range(c):
        right[0, k, k] = 1
        if k < c // 2:
            right[1, c + k, k] = 1
        else:
            right[2, c + k - c // 2, k] = 1
    return sum(np.kron(m, n) for m, n in zip(left, right))


def check_bridges():
    for a, b, c, d in product(range(1, 6), repeat=4):
        left = RNG.integers(-2, 3, size=(3, a, b))
        right = RNG.integers(-2, 3, size=(3, d, c))
        flow = sum(np.kron(m, n) for m, n in zip(left, right))
        k = np.linalg.matrix_rank(flow, tol=1e-8)
        assert np.linalg.matrix_rank(flow.T @ flow, tol=1e-7) == k
        assert k <= min(3 * a * c, a * d, b * c, 3 * b * d)
        left_cut = np.concatenate([np.kron(np.eye(a), n) for n in right], axis=1)
        right_cut = np.concatenate([np.kron(m, np.eye(c)) for m in left], axis=0)
        close(left_cut @ right_cut, flow)
        if a * a + b * b <= 3 * a * b and c * c + d * d <= 3 * c * d:
            assert max(a * d, b * c) < 3 * a * c
        close(np.log(min(3*a*c, a*d, b*c)), min(np.log(3)+np.log(a)+np.log(c),
                                               np.log(a)+np.log(d), np.log(b)+np.log(c)))
    for a, b, c, d in product(range(5), repeat=4):
        if 2 * b <= 3 * a and 3 * c <= 2 * d:
            flow = route_bridge(a, b, c, d, np.sqrt(2))
            close(flow.T @ flow, 2 * np.eye(b * c))
    for d, expected in ((2, 4), (3, 6), (4, 6)):
        assert (12, 2 * d, 6) == {2: (12, 4, 6), 3: (12, 6, 6), 4: (12, 8, 6)}[d]
        if d == 2:
            flow = np.kron(np.eye(2, 3), np.eye(2))
        else:
            flow = route_bridge(2, 3, 2, d)
        assert np.linalg.matrix_rank(flow) == expected
    odd = route_bridge(2, 3, 3, 5, np.sqrt(2))
    close(odd.T @ odd, 2 * np.eye(9))
    assert np.flatnonzero(~np.any(odd, axis=1)).tolist() == [4]
    assert min(75, 5, 5, 3) == 3 and min(75, 5, 5) == 5
    flow = np.diag([1, 1, 1, 0, 0])
    assert np.linalg.matrix_rank(flow) == min(3, 5, 5)
    # Exact small integer witness for attainable rank at the cone counterexample.
    m = RNG.integers(-2, 3, size=(3, 3, 5))
    n = RNG.integers(-2, 3, size=(3, 5, 3))
    flow = sum(np.kron(x, y) for x, y in zip(m, n))
    assert np.linalg.matrix_rank(flow) == 15
    assert 34 <= 45 and 3 * 3 < 2 * 5 and 5 < 2 * 3
    for weights in (np.ones(5) / 5, np.array([0.25, 0.75])):
        entropy = -sum(weights * np.log(weights))
        assert entropy <= np.log(len(weights)) + 1e-12
    # Two disjoint square coordinate blocks, plus their singleton complements.
    blocks = [(range(2), range(2)), ([0], [2]), ([1], [2]), ([2], [0]), ([2], [1]), ([2], [2])]
    maps = []
    for rows, cols in blocks:
        partial = np.zeros((3, 3))
        partial[list(rows), list(cols)] = 1
        maps.append(partial)
    unitary = sum(np.kron(partial, partial.T) for partial in maps)
    close(unitary.T @ unitary, np.eye(9))
    close(unitary @ unitary.T, np.eye(9))
    assert np.linalg.matrix_rank(unitary.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)) == len(blocks)


def density(vector):
    return np.outer(vector, np.conj(vector))


def expectation(observable, state):
    return np.trace(observable @ state).real


def check_gradient():
    theta, phi = 0.9, 0.4
    point = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    tangent = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    phase_tangent = np.array([-np.sin(phi), np.cos(phi), 0]) * np.sin(theta)
    st = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    rho = density(st)
    to_pauli = lambda xyz: np.einsum('i,ijk->jk', xyz, PAULI)
    close(rho, (I + to_pauli(point)) / 2)
    for velocity in (tangent, phase_tangent, tangent + phase_tangent, np.zeros(3)):
        speed = np.linalg.norm(velocity)
        t = 1e-4
        s = point * np.cos(t * speed) + t * np.sinc(t * speed / np.pi) * velocity
        comparison = (1 + point @ s) / 2
        born = np.trace(rho @ ((I + to_pauli(s)) / 2)).real
        close(born, comparison)
        close(2 * (1 - comparison) / t**2, speed**2 / 2, atol=7e-8)
        close((1 - born) / t**2, speed**2 / 4, atol=7e-8)
        log_derivative = to_pauli(velocity)
        close((rho @ log_derivative + log_derivative @ rho) / 2, log_derivative / 2)
        close(expectation(log_derivative @ log_derivative, rho), speed**2)
    coords = np.column_stack((tangent, phase_tangent, tangent))
    metric = coords.T @ coords / 4
    cost = coords.T @ np.array([0, 0, 1])
    solution = np.linalg.pinv(metric) @ cost
    close(metric @ solution, cost)
    close(coords @ solution, coords @ (solution + np.array([1, 0, -1])))
    for phi in np.linspace(0, 2*np.pi, 9):
        state = density(np.array([1, np.exp(1j*phi)]) / np.sqrt(2))
        close(np.diag(state), [0.5, 0.5])
    point = np.array([np.sqrt(3)/2, 0, 0.5])
    gradient = np.eye(3)[2] - point[2]*point
    close(gradient, [-np.sqrt(3)/4, 0, 0.75])
    close(gradient @ gradient, 0.75)
    step = point - gradient / 10
    close(step / np.linalg.norm(step), np.array([21*np.sqrt(3), 0, 17])/(2*np.sqrt(403)))
    assert (step / np.linalg.norm(step))[2] < 0.5
    for scale in (1, 4):
        time = np.linspace(0, 0.7, 4001)
        height = np.tanh(np.arctanh(point[2]) - scale*time)
        speed_sq = scale * (1 - height**2)
        close(np.trapezoid(speed_sq, time), height[0] - height[-1], atol=2e-7)

    for _ in range(15):
        generator = hermitian_function(to_pauli(RNG.normal(size=3)), np.sign)
        state = density(unitary_frame(2, 1).ravel())
        observable = to_pauli(RNG.normal(size=3))
        for theta, shift in ((0.13, 0.7), (np.pi/3, np.pi/2)):
            def evolved(angle):
                u = rotation(generator, angle)
                return u @ state @ adj(u)

            derivative = 1j * np.trace(evolved(theta) @ (generator @ observable - observable @ generator)) / 2
            difference = (expectation(observable, evolved(theta + shift)) -
                          expectation(observable, evolved(theta - shift))) / (2*np.sin(shift))
            close(derivative, difference)
            assert abs(expectation(generator, evolved(theta)) - expectation(generator, state)) < EYE_ATOL
    def sample(angle):
        return expectation(Z, density(rotation(Y, angle)[:, 0]))

    close([sample(np.pi/3), sample(5*np.pi/6), sample(-np.pi/6)],
          [0.5, -np.sqrt(3)/2, np.sqrt(3)/2])
    close((sample(5*np.pi/6) - sample(-np.pi/6)) / 2, -np.sqrt(3)/2)
    angle = np.pi / 4
    close(sample(2*(angle + np.pi/2)), sample(2*(angle - np.pi/2)))
    close(-2*np.sin(2*angle), -2)


def check_measurement():
    cz = np.diag([1, 1, 1, -1])
    plus = np.ones(2) / np.sqrt(2)
    rho = RNG.normal(size=(2, 2)) + 1j * RNG.normal(size=(2, 2))
    close(sum(np.trace(rho @ density(e)) for e in I), np.trace(rho))
    for phase in (0, np.pi/3, 1.9):
        for psi in (I[:, 0], unitary_frame(2, 1).ravel()):
            joint = (cz @ np.kron(psi, plus)).reshape(2, 2)
            for record in range(2):
                effect = np.array([1, (-1)**record * np.exp(1j*phase)]) / np.sqrt(2)
                branch = effect.conj() @ joint
                correction = np.linalg.matrix_power(X, record)
                close(branch, correction @ H @ np.diag([1, np.exp(-1j*phase)]) @ psi / np.sqrt(2))
                close(np.linalg.norm(branch)**2, 0.5)
                close(density(correction @ branch) * 2, density(H @ rotation(Z, -phase) @ psi))
                if phase == np.pi/3 and psi[1] == 0:
                    close(correction @ branch * np.sqrt(2), plus)
    d = 2**20
    close(1 / (d * 0.1**2), 100 / 2**20)
    close(np.mean(np.eye(5)[:, 0]**2), 1 / 5)
    close(sum(0.5**np.arange(1, 57)), 1)
    assert sum(0.5**np.arange(1, 57) >= 1/8) == 3
    close(1 / (1/8), 8)


def pulse_kernel(matrix, time):
    if time == 0:
        return matrix
    return hermitian_function(matrix, lambda x: np.expm1(1j*time*x)/(1j*time))


def sample_drive(omega, phase, t, envelope):
    fixed = envelope(t) * equatorial(phase) / 2
    fast = envelope(t) * equatorial(-(2*omega*t + phase)) / 2
    return fixed, fast


def check_pulses():
    plus = np.ones(2) / np.sqrt(2)
    for alpha, phase, theta in ((0.3, 1.2, 0.8), (np.pi/2, 0, np.pi/2)):
        clock = rotation(Z, alpha)
        close(clock @ rotation(equatorial(phase), theta) @ adj(clock),
              rotation(equatorial(phase + alpha), theta))
        close(rotation(equatorial(phase), theta) @ clock,
              clock @ rotation(equatorial(phase - alpha), theta))
    cumulative = 0
    desired, applied = I.copy(), I.copy()
    for z_angle, phi, theta in ((0.8, 0.4, 2), (-0.1, -0.5, 0.8), (0.4, 0, 0.6)):
        desired = rotation(equatorial(phi), theta) @ rotation(Z, z_angle) @ desired
        cumulative += z_angle
        applied = rotation(equatorial(phi - cumulative), theta) @ applied
        close(desired, rotation(Z, cumulative) @ applied)
        close(abs((desired @ plus)[0])**2, abs((applied @ plus)[0])**2)
        pulled = adj(rotation(Z, cumulative)) @ X @ rotation(Z, cumulative)
        close(expectation(X, density(desired @ plus)), expectation(pulled, density(applied @ plus)))
    state, effect = density(plus), X
    for steps in range(5):
        gate = np.linalg.matrix_power(rotation(Y, 0.7), steps)
        close(expectation(effect, gate @ state @ adj(gate)), expectation(adj(gate) @ effect @ gate, state))
    close(1/(5e9)*1e9, 0.2)
    close(25 / 0.2, 125)
    close(2*np.pi*10e6*25e-9, np.pi/2)
    close(abs(rotation(X, np.pi/2)[1, 0])**2, 0.5)
    close(abs((rotation(X, np.pi/2) @ rotation(Z, np.pi/2) @ plus)[0])**2, 1)
    close(abs((rotation(Y, np.pi/2) @ plus)[0])**2, 0)
    close(1.25 * (0.1**2), 0.0125)

    for omega in (14.0, 37.0):
        phase = 0.71
        envelope = lambda t: 1 + t
        bound = 65 / (16*omega)
        close((2*2 + 1) * (1 + 1.5*1.5) / (4*omega), bound)
        duration = 1
        intervals = 2400
        dt = duration / intervals
        propagator = I.copy()
        for k in range(intervals):
            mid = dt * (k + 0.5)
            fixed, fast = sample_drive(omega, phase, mid, envelope)
            frame = rotation(Z, -omega*mid)
            lab = omega*Z/2 + envelope(mid)*np.cos(omega*mid + phase)*X
            close(frame @ lab @ adj(frame) - omega*Z/2, fixed + fast)
            propagator = hermitian_function(fixed + fast, lambda x: np.exp(-1j*dt*x)) @ propagator
            if (k + 1) % 60 == 0:
                t = dt * (k + 1)
                average = rotation(equatorial(phase), t + t*t/2)
                assert norm(propagator - average) <= bound + 1e-5
                close(adj(propagator) @ propagator, I)
    check_matrix = np.array([[2, 0.7], [0.7, -1]])
    eig, vectors = np.linalg.eigh(check_matrix)
    for value, vector in zip(eig, vectors.T):
        close(hermitian_function(check_matrix, np.exp) @ vector, np.exp(value) * vector)
        evolved = hermitian_function(check_matrix, lambda x: np.exp(-0.7j*x)) @ vector
        close(density(evolved), density(vector))
    for _ in range(24):
        matrices = RNG.normal(size=(3, 4, 4))
        matrices = (matrices + matrices.transpose(0, 2, 1)) / 2
        limit = max(np.linalg.norm(m, 'fro') for m in matrices)
        a, b, c = matrices
        for t in (0, 0.01, 0.1, -0.1, 0.8):
            ka, kb, kc = (pulse_kernel(m, t) for m in matrices)
            assert abs(np.trace(ka @ kb @ kc).real - np.trace(a @ b @ c)) <= 1.25*limit**5*t*t + 1e-13
        close(pulse_kernel(a, 0), a)
        # Closed form against independently integrated matrix exponentials.
        nodes, weights = np.polynomial.legendre.leggauss(12)
        integrated = sum(w * (hermitian_function(a, lambda x: np.exp(0.05j*(node + 1)*x)) @ a)
                         for node, w in zip(nodes, weights)) / 2
        close(integrated, pulse_kernel(a, 0.1))


if __name__ == "__main__":
    for check in (check_reflections, check_defect, check_coin, check_compile, check_orbits,
                  check_large_orbits, check_bridges, check_gradient, check_measurement, check_pulses):
        check()
        print(f"PASS {check.__name__.removeprefix('check_')}")
    print("All ten m11 check groups passed.")
