# Checks for Section 0A (sec:m12); vendored from the m12 module script.
#!/usr/bin/env python3
"""Deterministic checks for section.tex. The standard library and NumPy suffice."""

from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
from math import isqrt, prod

import numpy as np


PI = np.pi
TOL = 2e-9
RNG = np.random.default_rng(1729)


def edge(u, v):
    return tuple(sorted((u, v)))


def darts(walk):
    return tuple(zip(walk, walk[1:] + walk[:1]))


def cycle_edges(walk):
    return {edge(u, v) for u, v in darts(walk)}


def adjacency(n, edges):
    result = [set() for _ in range(n)]
    for u, v in edges:
        result[u].add(v)
        result[v].add(u)
    return result


def simple_cycles(n, edges):
    neighbors = adjacency(n, edges)

    def visit(path):
        for nxt in sorted(neighbors[path[-1]]):
            if nxt == path[0] and len(path) >= 3 and path[1] < path[-1]:
                yield path
            if nxt > path[0] and nxt not in path:
                yield from visit(path + (nxt,))

    for start in range(n):
        yield from visit((start,))


def odd_costs(angles):
    angles = np.asarray(angles)
    for size in range(1, len(angles) + 1, 2):
        for subset in combinations(range(len(angles)), size):
            selected = angles[list(subset)].sum()
            cost = size * PI + angles.sum() - 2 * selected
            terms = [PI - t if i in subset else t for i, t in enumerate(angles)]
            np.testing.assert_allclose(cost, sum(terms), atol=TOL)
            yield cost


def fill_angles(n, entries):
    entries = dict(entries)
    for u, v in combinations(range(n), 2):
        if (u, v) in entries:
            continue
        distances = np.full((2 * n, 2 * n), np.inf)
        np.fill_diagonal(distances, 0)
        for (a, b), angle in entries.items():
            for s, t in product(range(2), repeat=2):
                i, j = 2 * a + s, 2 * b + t
                distances[i, j] = distances[j, i] = angle if s == t else PI - angle
        for i in range(2 * n):
            distances = np.minimum(distances, distances[:, i, None] + distances[None, i, :])
        assert np.all(distances[::2, 1::2].diagonal() >= PI - TOL)
        d0, d1 = distances[2 * u, 2 * v : 2 * v + 2]
        angle = max(0.0, PI - d1)
        assert -TOL <= angle <= PI + TOL
        assert angle <= d0 + TOL and PI - angle <= d1 + TOL
        entries[u, v] = angle

    result = np.zeros((n, n))
    for (u, v), angle in entries.items():
        result[u, v] = result[v, u] = angle
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_array_equal(np.diag(result), 0)
    assert np.all((0 <= result) & (result <= PI))
    for i, j, k in product(range(n), repeat=3):
        assert result[i, j] <= result[i, k] + result[k, j] + TOL
        assert result[i, j] + result[j, k] + result[k, i] <= 2 * PI + TOL
    return result


def vector_at_angle(first, second, alpha, beta):
    """Insert a vector at angle alpha to first and beta to second."""
    cosine = float(np.clip(first @ second, -1, 1))
    residual = second - cosine * first
    sine = np.linalg.norm(residual)
    if sine < TOL:
        transverse = np.eye(3)[np.argmin(np.abs(first))]
        transverse = transverse - (transverse @ first) * first
        transverse /= np.linalg.norm(transverse)
        assert abs(np.cos(beta) - cosine * np.cos(alpha)) < TOL
        return np.cos(alpha) * first + np.sin(alpha) * transverse
    transverse = residual / sine
    coefficient = (np.cos(beta) - cosine * np.cos(alpha)) / sine
    remaining = np.sin(alpha) ** 2 - coefficient ** 2
    assert remaining >= -TOL
    if abs(remaining) < TOL:
        remaining = 0
    return (np.cos(alpha) * first + coefficient * transverse
            + np.sqrt(remaining) * np.cross(first, transverse))


def realize(n, entries, earlier_neighbors):
    angles = fill_angles(n, entries)
    vectors = np.zeros((n, 3))
    for v, parents in enumerate(earlier_neighbors):
        assert all(u < v for u in parents)
        if not parents:
            vectors[v] = [1, 0, 0]
        elif len(parents) == 1:
            u, = parents
            vectors[v] = vector_at_angle(vectors[u], vectors[u], angles[v, u], angles[v, u])
        else:
            u, w = parents
            vectors[v] = vector_at_angle(vectors[u], vectors[w], angles[v, u], angles[v, w])
        for u in parents:
            np.testing.assert_allclose(vectors[v] @ vectors[u], np.cos(angles[v, u]), atol=TOL)
    gram = vectors @ vectors.T
    np.testing.assert_allclose(np.diag(gram), 1, atol=TOL)
    assert np.linalg.matrix_rank(gram, tol=TOL) <= 3
    assert np.all(np.linalg.eigvalsh(gram) >= -TOL)
    for (u, v), angle in entries.items():
        np.testing.assert_allclose(gram[u, v], np.cos(angle), atol=TOL)
    return gram


def normalized_vectors(n, dim):
    vectors = RNG.normal(size=(n, dim))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def check_gram():
    basis = np.eye(3)
    for n in (3, 4, 5, 6):
        neighbors = [(), (0,)] + [(0, v - 1) for v in range(2, n)]
        angles = [PI / 2] * 3 + [0] * (n - 3)
        vectors = basis[[0, 1, 2] + [0] * (n - 3)]
        np.testing.assert_allclose([vectors[u] @ vectors[v] for u, v in darts(tuple(range(n)))],
                                   np.cos(angles), atol=TOL)
        assert min(odd_costs(angles)) >= PI - TOL
        assert all(abs((np.dot(signs, angles) + PI) % (2 * PI) - PI) > TOL
                   for signs in product((-1, 1), repeat=n))
        invalid = [PI] + [0] * (n - 1)
        assert min(odd_costs(invalid)) == 0
        entries = dict(zip([edge(u, v) for u, v in darts(tuple(range(n)))], angles))
        completed = realize(n, entries, neighbors)
        assert np.linalg.matrix_rank(completed, tol=TOL) == 3
        if n == 3:
            np.testing.assert_allclose(completed, basis, atol=TOL)
        for _ in range(8):
            vectors = normalized_vectors(n, 5)
            entries = {edge(u, v): np.arccos(np.clip(vectors[u] @ vectors[v], -1, 1))
                       for u, v in darts(tuple(range(n)))}
            assert min(odd_costs(list(entries.values()))) >= PI - TOL
            realize(n, entries, neighbors)

    first, second = basis[[0, 1, 0, 1]], basis[[0, 1, 2, 1]]
    assert first[0] @ first[2] == 1 and second[0] @ second[2] == 0
    for u, v in darts((0, 1, 2, 3)):
        assert first[u] @ first[v] == second[u] @ second[v] == 0
    assert realize(2, {(0, 1): PI}, [(), (0,)])[0, 1] == -1
    realize(4, {(0, 1): PI, (2, 3): 0}, [(), (0,), (), (2,)])

    theta = [(PI / 3, PI / 3), (PI / 3, PI / 2), (PI / 2, PI / 2)]
    expected = [(0, 2 * PI / 3), (PI / 6, 5 * PI / 6), (0, PI)]
    for (a, b), interval in zip(theta, expected):
        np.testing.assert_allclose([abs(a - b), min(a + b, 2 * PI - a - b)], interval,
                                   atol=TOL)
        assert interval[0] <= PI / 2 <= interval[1]
        assert abs(basis[0] @ vector_at_angle(basis[0], basis[1], a, b) - np.cos(a)) < TOL
    entries = {(u, v): theta[v - 2][u] for u in range(2) for v in range(2, 5)}
    realize(5, entries, [(), (0,), (0, 1), (0, 1), (0, 1)])
    for _ in range(20):
        parents = [(), (0,), (0, 1)]
        edges = {(0, 1), (0, 2), (1, 2)}
        for v in range(3, 8):
            choices = sorted(edges)
            u, w = choices[RNG.integers(len(choices))]
            parents.append((u, w))
            edges.update(((u, v), (w, v)))
        vectors = normalized_vectors(8, 3)
        entries = {(u, v): np.arccos(np.clip(vectors[u] @ vectors[v], -1, 1))
                   for u, v in sorted(edges) if RNG.random() >= 0.25}
        for cycle in simple_cycles(8, entries):
            assert min(odd_costs([entries[edge(u, v)] for u, v in darts(cycle)])) >= PI - TOL
        realize(8, entries, parents)

    # The two-anchor interval is full when both anchor overlaps are zero.
    for x in (-1, -0.5, 0, 0.5, 1):
        vectors = np.array([[1, 0, 0], [0, 1, 0], [x, 0, np.sqrt(1 - x * x)]])
        np.testing.assert_allclose(np.diag(vectors @ vectors.T), 1)
        assert vectors[0] @ vectors[1] == vectors[1] @ vectors[2] == 0
        assert vectors[0] @ vectors[2] == x

    bad = 1.5 * np.eye(4) - 0.5 * np.ones((4, 4))
    np.testing.assert_allclose(bad @ np.ones(4), -0.5)
    for cycle in simple_cycles(4, combinations(range(4), 2)):
        costs = list(odd_costs([2 * PI / 3] * len(cycle)))
        assert min(costs) >= PI - TOL
        if len(cycle) == 3:
            np.testing.assert_allclose(costs[-1], PI)
    np.testing.assert_allclose(np.cos(fill_angles(4, {e: 2 * PI / 3
                                                    for e in combinations(range(4), 2)})), bad)

    for n, width in product(range(1, 8), repeat=2):
        edges = {(u, v) for u, v in combinations(range(n), 2) if v - u < width}
        for k in range(1, n + 1):
            for clique in combinations(range(n), k):
                if all(e in edges for e in combinations(clique, 2)):
                    assert max(clique) < min(clique) + width

    samples = np.linspace(0, 2 * PI, 101)
    for t in samples:
        x, y = np.cos(t), np.sin(t)
        projection = np.array([[1 + x, y], [y, 1 - x]]) / 2
        factor = np.array([np.cos(t / 2), np.sin(t / 2)])
        np.testing.assert_allclose(projection, np.outer(factor, factor), atol=TOL)
        np.testing.assert_allclose(projection.T @ projection, projection, atol=TOL)
        np.testing.assert_allclose(np.linalg.eigvalsh(projection), [0, 1], atol=TOL)
        np.testing.assert_allclose((factor[0] + 1j * factor[1]) ** 2, x + 1j * y, atol=TOL)
    np.testing.assert_allclose(np.exp(1j * samples[-1] / 2), -1, atol=TOL)

    leading = np.eye(5)
    leading[0, 1:] = leading[1:, 0] = 11 / 20
    determinants = [np.linalg.det(leading[:j, :j]) for j in range(1, 6)]
    np.testing.assert_allclose(determinants, 1 - np.arange(5) * (11 / 20) ** 2)
    assert min(determinants[:4]) > 0 > determinants[4]
    bad_leading = np.zeros((5, 5))
    bad_leading[:2, :2] = 1
    bad_leading[2:, 2:] = 1.75 * np.eye(3) - 0.75 * np.ones((3, 3))
    np.testing.assert_allclose([np.linalg.det(bad_leading[:j, :j]) for j in range(1, 6)],
                               [1, 0, 0, 0, 0])
    coeff = np.poly(-bad_leading)
    for k in range(1, 6):
        minors = [np.linalg.det(bad_leading[np.ix_(ix, ix)])
                  for ix in combinations(range(5), k)]
        np.testing.assert_allclose(coeff[k], sum(minors), atol=TOL)
    np.testing.assert_allclose(np.linalg.eigvalsh(bad_leading)[0], -0.5)
    potential = lambda x: (x * x - 1) ** 2
    assert potential(0) == 1 and potential(-1) == potential(1) == 0
    assert np.polyval(np.polyder([1, 0, -2, 0, 1]), 0) == 0


def binary_solve(matrix, target):
    augmented = np.column_stack((matrix, target)).astype(np.uint8)
    pivots = []
    for col in range(matrix.shape[1]):
        row = len(pivots)
        candidates = np.flatnonzero(augmented[row:, col])
        if not candidates.size:
            continue
        pivot = row + candidates[0]
        augmented[[row, pivot]] = augmented[[pivot, row]]
        for other in np.flatnonzero(augmented[:, col]):
            if other != row:
                augmented[other] ^= augmented[row]
        pivots.append(col)
    if np.any(augmented[len(pivots):, -1]):
        return len(pivots), None
    solution = np.zeros(matrix.shape[1], dtype=np.uint8)
    solution[pivots] = augmented[:len(pivots), -1]
    np.testing.assert_array_equal((matrix @ solution) % 2, target)
    return len(pivots), solution


def matchings(vertices, edges):
    if not vertices:
        yield ()
        return
    first = vertices[0]
    for other in vertices[1:]:
        if (first, other) in edges:
            rest = tuple(v for v in vertices if v not in (first, other))
            for submatching in matchings(rest, edges):
                yield ((first, other),) + submatching


def sign_of_listing(listing):
    inversions = sum(a > b for a, b in combinations(listing, 2))
    return (-1) ** inversions


def pfaffian(matrix):
    if len(matrix) == 0:
        return 1
    total = 0
    for j in range(1, len(matrix)):
        rest = [i for i in range(1, len(matrix)) if i != j]
        total += (-1) ** (j + 1) * matrix[0, j] * pfaffian(matrix[np.ix_(rest, rest)])
    return total


def opposite(walk, orientation):
    return sum((v, u) in orientation for u, v in darts(walk))


def rotation_faces(face_walks):
    rotation = {}
    for face in face_walks:
        ds = darts(face)
        for (u, v), nxt in zip(ds, ds[1:] + ds[:1]):
            assert (v, u) not in rotation
            rotation[v, u] = nxt

    for tail in {d[0] for d in rotation}:
        local = {d for d in rotation if d[0] == tail}
        start = min(local)
        current = rotation[start]
        seen = {start}
        while current not in seen:
            seen.add(current)
            current = rotation[current]
        assert current == start and seen == local

    unseen = set(rotation)
    orbits = []
    while unseen:
        start = min(unseen)
        current = start
        orbit = []
        while current in unseen:
            unseen.remove(current)
            u, v = current
            orbit.append(u)
            current = rotation[v, u]
        assert current == start
        orbits.append(tuple(orbit))
    assert Counter(frozenset(darts(f)) for f in orbits) == Counter(
        frozenset(darts(f)) for f in face_walks)
    return orbits


def check_rotation(n, supplied_faces, orientation=None):
    faces = rotation_faces(supplied_faces)
    face_of = {d: i for i, face in enumerate(faces) for d in darts(face)}
    edges = sorted({edge(u, v) for u, v in face_of})
    edge_set = set(edges)
    assert n - len(edges) + len(faces) == 2
    boundary = np.zeros((len(edges), len(faces)), dtype=np.uint8)
    for i, (u, v) in enumerate(edges):
        boundary[i, face_of[u, v]] ^= 1
        boundary[i, face_of[v, u]] ^= 1
    rank, _ = binary_solve(boundary, np.zeros(len(edges), dtype=np.uint8))
    assert rank == len(faces) - 1 == len(edges) - n + 1
    for omit in range(len(faces)):
        retained = [f for f in range(len(faces)) if f != omit]
        rank_other, _ = binary_solve(boundary[:, retained],
                                     np.zeros(len(edges), dtype=np.uint8))
        assert rank_other == rank

    initial = set(edges)
    initial_counts = np.array([opposite(f, initial) for f in faces], dtype=np.uint8)
    _, flips = binary_solve(boundary.T, 1 ^ (initial_counts % 2))
    assert (flips is not None) == (n % 2 == 0)
    if flips is not None and orientation is None:
        orientation = {(v, u) if flip else (u, v) for (u, v), flip in zip(edges, flips)}
    if orientation is not None:
        assert all(opposite(f, orientation) % 2 for f in faces)
        assert sum(opposite(f, orientation) for f in faces) == len(edges)

    dual_edges = [{face_of[u, v], face_of[v, u]} for u, v in edges]
    nice_count = 0
    for cycle in simple_cycles(n, edges):
        ce = cycle_edges(cycle)
        _, sides = binary_solve(boundary, np.array([e in ce for e in edges], dtype=np.uint8))
        assert sides is not None and set(sides) == {0, 1}
        cv = set(cycle)
        remaining = tuple(sorted(set(range(n)) - cv))
        complementary_matchings = list(matchings(remaining, edge_set))
        if len(cycle) % 2 == 0 and complementary_matchings:
            nice_count += 1
            if orientation is not None:
                assert opposite(cycle, orientation) % 2 == 1

        previous_vertices = set()
        for bit in (0, 1):
            faces_in_side = set(np.flatnonzero(sides == bit))
            reached_faces = {min(faces_in_side)}
            changed = True
            while changed:
                count = len(reached_faces)
                for ends in dual_edges:
                    if ends <= faces_in_side and ends & reached_faces:
                        reached_faces.update(ends)
                changed = len(reached_faces) != count
            assert reached_faces == faces_in_side
            selected_darts = {(u, v) for u, v in face_of
                              if face_of[u, v] in faces_in_side
                              and face_of[v, u] not in faces_in_side}
            same = selected_darts == set(darts(cycle))
            assert same or selected_darts == set(darts(cycle[::-1]))
            inner_edges = {(u, v) for u, v in edges
                           if face_of[u, v] in faces_in_side and face_of[v, u] in faces_in_side}
            inner_vertices = {u for u in remaining
                              if all(face_of[d] in faces_in_side for d in face_of if d[0] == u)}
            for u in remaining:
                labels = {sides[face_of[d]] for d in face_of if d[0] == u}
                assert len(labels) == 1
            assert not previous_vertices & inner_vertices
            assert not any(edge(u, v) in edge_set for u in previous_vertices for v in inner_vertices)
            previous_vertices |= inner_vertices
            vertices = cv | inner_vertices
            region_neighbors = adjacency(n, inner_edges | ce)
            reached = {cycle[0]}
            while True:
                expanded = reached | set().union(*(region_neighbors[u] for u in reached))
                if expanded == reached:
                    break
                reached = expanded
            assert reached == vertices
            assert len(vertices) - len(inner_edges | ce) + len(faces_in_side) == 1
            assert len(inner_vertices) - len(inner_edges) + len(faces_in_side) == 1
            if orientation is not None:
                traverse = cycle if same else cycle[::-1]
                assert opposite(traverse, orientation) % 2 == (1 + len(inner_vertices)) % 2
            for matching in complementary_matchings:
                restriction = {v for pair in matching if set(pair) <= inner_vertices for v in pair}
                assert restriction == inner_vertices and len(restriction) % 2 == 0
        assert previous_vertices == set(remaining)
    return edges, nice_count


def check_interlacing():
    vertices = tuple(range(6))
    edges = set(combinations(vertices, 2))
    pairings = list(matchings(vertices, edges))
    outputs = set()
    for first, second in product(pairings, repeat=2):
        s = dict(first + tuple((v, u) for u, v in first))
        t = dict(second + tuple((v, u) for u, v in second))
        unseen = set(vertices)
        q = [None] * len(vertices)
        while unseen:
            root = min(unseen)
            circuit = [root]
            nxt = s[root]
            while nxt != root:
                circuit.append(nxt)
                nxt = (s if len(circuit) % 2 else t)[nxt]
            assert len(circuit) % 2 == 0
            even = root
            for i in range(0, len(circuit), 2):
                assert circuit[i:i + 2] == [even, s[even]]
                even = t[s[even]]
            for i, (u, v) in enumerate(darts(tuple(circuit))):
                q[u] = v
                incoming = circuit[i - 1]
                assert (s[u], t[u]) == ((v, incoming) if i % 2 == 0 else (incoming, v))
            unseen.difference_update(circuit)
        outputs.add(tuple(q))
    assert len(outputs) == len(pairings) ** 2

    def even_orbits(q):
        unseen = set(vertices)
        while unseen:
            root = min(unseen)
            nxt, count = root, 0
            while nxt in unseen:
                unseen.remove(nxt)
                nxt, count = q[nxt], count + 1
            if count % 2:
                return False
        return True

    assert outputs == {q for q in permutations(vertices) if even_orbits(q)}
    for matching in pairings:
        listing = tuple(v for pair in matching for v in pair)
        crossings = sum(a < c < b < d for (a, b), (c, d) in combinations(matching, 2))
        nestings = sum(a < c < d < b for (a, b), (c, d) in combinations(matching, 2))
        assert sum(a > b for a, b in combinations(listing, 2)) == crossings + 2 * nestings
        assert sign_of_listing(listing) == (-1) ** crossings


def grid_faces(rows, cols):
    def label(r, c):
        return r * cols + c

    squares = [
        (label(r, c), label(r, c + 1), label(r + 1, c + 1), label(r + 1, c))
        for r in range(rows - 1) for c in range(cols - 1)
    ]
    outer = (tuple(label(r, 0) for r in range(rows))
             + tuple(label(rows - 1, c) for c in range(1, cols))
             + tuple(label(r, cols - 1) for r in range(rows - 2, -1, -1))
             + tuple(label(0, c) for c in range(cols - 2, 0, -1)))
    return squares + [outer]


def check_planar():
    check_interlacing()
    assert opposite((0, 1), {(0, 1)}) == opposite((0, 1), {(1, 0)}) == 1
    check_rotation(2, [(0, 1)])
    for state in product((0, 1), repeat=3):
        triangle = ((0, 1), (1, 2), (0, 2))
        orientation = {(v, u) if bit else (u, v) for (u, v), bit in zip(triangle, state)}
        parities = [opposite(w, orientation) for w in [(0, 1, 2), (0, 2, 1)]]
        assert sum(parities) == 3 and not all(value % 2 for value in parities)
    check_rotation(3, [(0, 1, 2), (0, 2, 1)])
    check_rotation(4, [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)])
    check_rotation(4, [(0, 1, 2, 0, 3), (0, 2, 1)])

    faces = grid_faces(2, 3)
    arrows = {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (4, 1), (2, 5)}
    assert [opposite(f, arrows) for f in faces] == [3, 1, 3]
    edges, _ = check_rotation(6, faces, arrows)
    listed = [((0, 3), (1, 4), (2, 5)), ((0, 1), (2, 5), (3, 4)),
              ((0, 3), (1, 2), (4, 5))]
    assert set(listed) == set(matchings(tuple(range(6)), set(edges)))
    coefficients = {}
    for matching in listed:
        sign = sign_of_listing(tuple(v for pair in matching for v in pair))
        coefficients[matching] = sign * (-1) ** sum(e[::-1] in arrows for e in matching)
    assert set(coefficients.values()) == {1}
    assert coefficients[listed[0]] * coefficients[listed[1]] == -(-1) ** opposite(faces[0], arrows)
    samples = [[1] * len(edges), list(range(-3, 4)), list(range(1, 8))]
    for weights in samples:
        by_edge = dict(zip(edges, weights))
        matrix = np.zeros((6, 6), dtype=object)
        for u, v in arrows:
            matrix[u, v] = by_edge[edge(u, v)]
            matrix[v, u] = -matrix[u, v]
        expected = sum(prod(by_edge[e] for e in m) for m in listed)
        assert pfaffian(matrix) == expected
        for modulus in (2, 6, 7):
            assert pfaffian(matrix % modulus) % modulus == expected % modulus
        np.testing.assert_allclose(np.linalg.det(matrix.astype(float)), expected ** 2, atol=TOL)
        denominator = 6
        exact_determinant = sum(sign_of_listing(p) * prod(matrix[i, j] for i, j in enumerate(p))
                                for p in permutations(range(6)))
        assert exact_determinant == expected ** 2
        assert Fraction(isqrt(exact_determinant), denominator ** 3) == Fraction(
            abs(expected), denominator ** 3)
    assert len(listed) == 3 and len(listed) ** 2 == 9
    assert pfaffian(np.zeros((0, 0), dtype=int)) == 1
    assert pfaffian(np.zeros((2, 2), dtype=int)) == 0
    for dimension in (2, 4, 6, 8):
        for _ in range(5):
            raw = RNG.integers(-4, 5, size=(dimension, dimension))
            matrix = raw - raw.T
            np.testing.assert_allclose(np.linalg.det(matrix), pfaffian(matrix) ** 2, atol=1e-6)

    faces = grid_faces(4, 4)
    arrows = {(4 * r + c, 4 * r + c + 1) for r in range(4) for c in range(3)}
    arrows |= {(4 * r + c, 4 * r + c + 4) if c % 2 == 0
               else (4 * r + c + 4, 4 * r + c) for r in range(3) for c in range(4)}
    assert len(faces) == 10 and opposite(faces[-1], arrows) == 3
    edges, nice_count = check_rotation(16, faces, arrows)
    example = (0, 1, 2, 3, 7, 11, 10, 9, 8, 4)
    assert nice_count > 1 and len(example) == 10 and opposite(example, arrows) == 7
    assert {v for face in faces[:6] for v in face} - set(example) == {5, 6}
    assert (5, 6) in edges and ((12, 13), (14, 15)) in list(matchings((12, 13, 14, 15), set(edges)))

    # Icosahedral faces, with each shared edge traversed in opposite directions.
    faces = []
    for i in range(5):
        u, unext, v, vprev, vnext = i, (i + 1) % 5, 5 + i, 5 + (i - 1) % 5, 5 + (i + 1) % 5
        faces.extend(((10, u, unext), (u, v, unext), (u, vprev, v), (11, vnext, v)))
    ordered_faces = rotation_faces(faces)
    face_of = {d: i for i, f in enumerate(ordered_faces) for d in darts(f)}
    hamiltonian = (10, 0, 5, 11, 6, 7, 8, 9, 4, 3, 2, 1)
    assert len(set(hamiltonian)) == 12 and len(faces) == 20
    assert all(d in face_of for d in darts(hamiltonian))
    on_cycle = cycle_edges(hamiltonian)
    dual_edges = [{face_of[u, v], face_of[v, u]} for u, v in face_of if edge(u, v) not in on_cycle]
    reached = {face_of[darts(hamiltonian)[0]]}
    while True:
        expanded = reached | set().union(*(ends for ends in dual_edges if ends & reached))
        if expanded == reached:
            break
        reached = expanded
    assert len(reached) == len(faces) - len(reached) == 10
    for side in (reached, set(range(20)) - reached):
        internal = {edge(u, v) for u, v in face_of if face_of[u, v] in side
                    and face_of[v, u] in side}
        assert 3 * len(side) == 12 + 2 * len(internal)
        assert len(side) - len(internal) == 1


def heights(n, arcs):
    neighbors = [0] * n
    for u, v in arcs:
        neighbors[u] |= 1 << v

    def continuation(v, used):
        able = neighbors[v] & ~used
        longest = 0
        while able:
            bit = able & -able
            able ^= bit
            longest = max(longest, continuation(bit.bit_length() - 1, used | bit))
        return 1 + longest

    return tuple(continuation(v, 1 << v) - 1 for v in range(n))


def census(n, edges, states):
    histogram = Counter()
    for choices in product(range(states), repeat=len(edges)):
        arcs = []
        for (u, v), choice in zip(edges, choices):
            if choice != 1:
                arcs.append((u, v))
            if choice != 0:
                arcs.append((v, u))
        histogram[1 + max(heights(n, arcs))] += 1
    return histogram


def check_coloring():
    c5 = tuple(sorted(cycle_edges((0, 1, 2, 3, 4))))
    c4 = tuple(combinations(range(4), 2))
    coloring = (0, 1, 0, 1, 2)
    arcs = [pair if coloring[pair[0]] < coloring[pair[1]] else pair[::-1] for pair in c5]
    assert (2, 3) in arcs and (3, 4) in arcs
    assert heights(5, arcs) == (1, 0, 2, 1, 0)
    cyclic = darts((0, 1, 2, 3, 4))
    assert heights(5, cyclic) == (4,) * 5
    assert heights(5, cyclic[:-1]) == (4, 3, 2, 1, 0)
    assert heights(3, darts((0, 1, 2))) == (2, 2, 2)

    ordinary5, ordinary4 = census(5, c5, 2), census(4, c4, 2)
    assert ordinary5 == {3: 10, 4: 10, 5: 12}
    assert ordinary4 == {4: 64}
    both5, both4 = census(5, c5, 3), census(4, c4, 3)
    assert min(both5) == 3 and both5[3] == 20 and both5.total() == 243
    assert both4 == {4: 729}
    petersen = (set(c5) | {edge(5 + v, 5 + (v + 2) % 5) for v in range(5)}
                | {(v, v + 5) for v in range(5)})
    assert len(petersen) == 15
    pet = census(10, sorted(petersen), 2)
    assert min(pet) == 3 and pet[3] == 80 and pet.total() == 32768
    for n, edges, histogram in ((5, c5, ordinary5), (4, c4, ordinary4)):
        for k in range(n + 1):
            colorable = any(all(colors[u] != colors[v] for u, v in edges)
                            for colors in product(range(k), repeat=n))
            assert colorable == (min(histogram) <= k)
    assert heights(0, ()) == ()


if __name__ == "__main__":
    check_gram()
    print("PASS Gram inequalities, rank bounds, examples, and principal minors")
    check_planar()
    print("PASS face spaces, cycle sides, alternating signs, and matching counts")
    check_coloring()
    print("PASS coloring equivalences and all orientation census values")
