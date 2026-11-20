# Checks for Section 0B (sec:m15); vendored from the m15 module script.
"""Deterministic finite checks for section.tex. Run with Python 3 and NumPy."""
import itertools

import numpy as np


RNG = np.random.default_rng(15112026)


def close(actual, expected, atol=2e-11, rtol=2e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def le(left, right):
    assert left <= right + 2e-11 * max(1, abs(right)), (left, right)


def bits(width, start=0, stop=None):
    if stop is None:
        stop = 1 << width
    words = np.arange(start, stop, dtype=np.uint64)
    return ((words[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)


def sign(parity):
    return 1 - 2 * (parity % 2).astype(np.int64)


def gf2_basis(matrix):
    reduced = matrix.copy()
    rank = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[rank:, column])
        if not candidates.size:
            continue
        pivot = rank + candidates[0]
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in np.flatnonzero(reduced[:, column]):
            if row != rank:
                reduced[row] ^= reduced[rank]
        rank += 1
    return reduced[:rank]


def grid(lengths, periodic):
    """Return cells and boundaries for a product of labelled intervals or circles."""
    dimension = len(lengths)
    shape = lengths if periodic else tuple(length + 1 for length in lengths)
    cells = []
    for degree in range(dimension + 1):
        cells.append([
            (vertex, axes)
            for axes in itertools.combinations(range(dimension), degree)
            for vertex in np.ndindex(shape)
            if periodic or all(vertex[axis] < lengths[axis] for axis in axes)
        ])
    maps = [None]
    for degree in range(1, dimension + 1):
        index = {cell: i for i, cell in enumerate(cells[degree - 1])}
        boundary = np.zeros((len(index), len(cells[degree])), dtype=np.uint8)
        for column, (vertex, axes) in enumerate(cells[degree]):
            for axis in axes:
                shifted = list(vertex)
                shifted[axis] = (shifted[axis] + 1) % shape[axis]
                face_axes = tuple(value for value in axes if value != axis)
                boundary[index[vertex, face_axes], column] ^= 1
                boundary[index[tuple(shifted), face_axes], column] ^= 1
        maps.append(boundary)
    return cells, maps


def check_grid():
    for m, n in ((1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (2, 3), (3, 3)):
        q = m * n
        cells, (_, edge_boundary, face_boundary) = grid((m, n), True)
        entries = cells[1]
        d, D = edge_boundary.T, face_boundary.T
        row = np.array([axis == (1,) and v[0] == 0 for v, axis in entries], dtype=np.uint8)
        col = np.array([axis == (0,) and v[1] == 0 for v, axis in entries], dtype=np.uint8)
        seam_h = np.array([axis == (1,) and v[1] == n-1 for v, axis in entries], dtype=np.uint8)
        seam_v = np.array([axis == (0,) and v[0] == m-1 for v, axis in entries], dtype=np.uint8)
        periods = np.array([row, col])
        seams = np.array([seam_h, seam_v])
        assert not (D @ d % 2).any()
        assert not (D @ seams.T % 2).any()
        assert not (periods @ d % 2).any()
        np.testing.assert_array_equal(periods @ seams.T % 2, np.eye(2))
        assert gf2_basis(D).shape[0] == q - 1
        assert gf2_basis(d).shape[0] == q - 1
        relations = bits(q)[np.all(bits(q) @ D % 2 == 0, axis=1)]
        np.testing.assert_array_equal(relations, [np.zeros(q), np.ones(q)])

        labels = bits(2*q)
        plaques = labels @ D.T % 2
        nonflat = np.any(plaques, axis=1)
        flat = labels[~nonflat]
        true_exact = flat[np.all(flat @ periods.T % 2 == 0, axis=1)]
        potential = bits(q - 1)
        potential = np.column_stack((np.zeros(len(potential), dtype=np.uint8), potential))
        gradients = potential @ d.T % 2
        encode = 2 ** np.arange(2*q)
        np.testing.assert_array_equal(np.sort(true_exact @ encode), np.sort(gradients @ encode))
        assert len(flat) == 2**(q+1) and len(labels) == 2**(2*q)
        assert len(true_exact) == 2**(q-1)
        sectors = flat @ periods.T % 2
        for sector in bits(2):
            assert np.count_nonzero(np.all(sectors == sector, axis=1)) == 2**(q-1)
        orbit_image = np.column_stack((plaques, labels @ periods.T % 2))
        orbit_key = orbit_image @ (2 ** np.arange(q+2))
        counts = np.bincount(orbit_key)
        assert np.count_nonzero(counts) == 2**(q+1)
        assert np.all(counts[counts > 0] == 2**(q-1))
        assert np.all(plaques.sum(axis=1) % 2 == 0)
        for boundary in (row, col, row ^ col):
            assert not (edge_boundary @ boundary % 2).any()
            assert np.any(seams @ boundary % 2)

    cells, (_, _, face_boundary) = grid((3, 3), True)
    y = bits(len(cells[1]))
    plaquette_sign = sign(y @ face_boundary)
    picked = np.array([v[0] < 2 and v[1] < 2 for v, _ in cells[2]])
    cycle = face_boundary @ picked % 2
    beta = 1.3
    t = np.tanh(beta)
    weight = np.exp(beta * plaquette_sign.sum(axis=1))
    direct = np.sum(sign(y @ cycle) * weight) / weight.sum()
    close(direct, (t**4 + t**5)/(1+t**9))
    assert round(direct, 6) == 0.813439 and round(t**4, 6) == 0.551405
    close(weight.sum(), 2**18 * np.cosh(beta)**9 * (1+t**9))
    inhomogeneous = np.linspace(-0.53, 1.3, 9)
    weight = np.exp(plaquette_sign @ inhomogeneous)
    t_p = np.tanh(inhomogeneous)
    close(np.sum(sign(y @ cycle) * weight) / weight.sum(),
          (np.prod(t_p[picked]) + np.prod(t_p[~picked])) / (1 + np.prod(t_p)))
    close(sign(y @ cycle).mean(), 0)
    for size in (9, 16, 25, 100):
        observed = (t**4+t**(size-4))/(1+t**size)
        le(t**4, observed)
        le(observed, 2*t**4)
        close(observed-t**4, t**(size-4)*(1-t**8)/(1+t**size))
    frozen = np.exp(80 * (plaquette_sign.sum(axis=1) - 9))
    close(frozen[~np.all(plaquette_sign == 1, axis=1)].sum()/frozen.sum(), 0)

    cells, (_, _, face_boundary) = grid((3, 3), False)
    chain = np.ones(len(cells[2]), dtype=np.uint8)
    without_centre = [i for i, (v, _) in enumerate(cells[2]) if v != (1, 1)]
    fillings = bits(len(without_centre))
    perimeter = face_boundary @ chain % 2
    assert not np.any(np.all(fillings @ face_boundary[:, without_centre].T % 2 == perimeter, axis=1))
    assert gf2_basis(face_boundary.T).shape[0] == 9
    print("Periodic lattice ranks, counts, seams and exact Wilson averages")


def check_parity():
    for edges, faces in ((0, 0), (0, 3), (3, 0), (2, 3), (4, 5), (6, 6)):
        for _ in range(8):
            D = RNG.integers(0, 2, size=(faces, edges), dtype=np.uint8)
            y, sets = bits(edges), bits(faces)
            weights = RNG.integers(-2, 4, size=faces)
            boltzmann = np.prod(1 + sign(y @ D.T) * weights, axis=1)
            product = np.prod(np.where(sets, weights, 1), axis=1)
            for c in y:
                lhs = np.sum(sign(y @ c) * boltzmann)
                rhs = (2**edges)*product[np.all(sets @ D % 2 == c, axis=1)].sum()
                assert lhs == rhs
                for modulus in (2, 4, 6, 11):
                    modular_lhs = int((sign(y @ c) * np.prod(
                        (1 + sign(y @ D.T) * weights) % modulus, axis=1)).sum()) % modulus
                    assert modular_lhs == int(rhs) % modulus
    chars = sign(bits(5) @ bits(5).T)
    np.testing.assert_array_equal(chars @ chars.T, 32*np.eye(32))
    print("Parity orthogonality and expansion over integers and residue rings")


def star(matrix):
    return np.swapaxes(matrix.conj(), -1, -2)


def f2(matrix):
    return float(np.vdot(matrix, matrix).real)


def weighted(matrix, weight):
    return np.trace(star(matrix) @ matrix @ weight).real


def unitary(n):
    return np.linalg.qr(RNG.normal(size=(n, n)) + 1j*RNG.normal(size=(n, n)))[0]


def comm(a, b):
    return a @ b - b @ a


def check_weighted():
    for n in range(1, 7):
        eye = np.eye(n)
        for trial in range(24):
            a = RNG.normal(size=(n, n)) + 1j*RNG.normal(size=(n, n))
            b = RNG.normal(size=(n, n)) + 1j*RNG.normal(size=(n, n))
            u = unitary(n)
            vals = np.exp(RNG.uniform(-3, 3, n))
            w = (u*vals) @ star(u)
            p = u[:, :trial % (n + 1)] @ star(u[:, :trial % (n + 1)])
            left = comm(a, b)
            a2, b2, bp = f2(a), f2(b), f2(a @ p)
            operator = np.kron(eye, a) - np.kron(a.T, eye)
            gap = 2*a2 - np.linalg.norm(operator, 2)**2
            le(0, gap)
            gap = max(0, gap)
            c = RNG.normal() + 1j*RNG.normal()
            le(np.linalg.norm(a-c*eye, 2)**2, a2+2*abs(c)**2+2*abs(c)*np.sqrt(gap))
            le(f2(left @ p), (a2+2*bp+2*np.sqrt(bp*gap))*b2)
            close(weighted(a, eye), a2)
            le(weighted(left, w), (1+max(vals)/min(vals))*weighted(a, w)*b2)
            q = star(b) @ a + a @ star(b)
            rho = (star(b) @ b + b @ star(b))/(2*b2)
            k = (star(a) @ a + a @ star(a))/2
            close(f2(left)+f2(q), 4*b2*np.trace(k @ rho))
            close(np.trace(q @ b), 2*b2*np.trace(a @ rho))
            le(f2(left), 4*b2*(np.trace(k @ rho).real-abs(np.trace(a @ rho))**2))
            z = b @ p
            h = p @ a @ p
            pencil = a @ star(a) @ z - a @ z @ star(h) - star(a) @ z @ h
            close(comm(a, comm(star(a), z)) @ p, pencil + z @ p @ star(a) @ a @ p)
            le(f2(comm(star(a), z)), (a2+2*bp+2*np.sqrt(bp*gap))*f2(z))
            s = trial / 5
            two = eye + s*p
            defect = (2+s)*weighted(a, two)*b2 - weighted(left, two)
            le((np.sqrt(gap)-s*np.sqrt(bp))**2*b2, defect)
        if n == 1:
            close(comm(a, b), 0)
            continue
        for lo, hi in ((1, 1), (0.1, 4), (2, 17)):
            u = unitary(n)
            w = (u*np.linspace(lo, hi, n)) @ star(u)
            a = np.outer(u[:, 0], u[:, 0].conj())/lo - np.outer(u[:, -1], u[:, -1].conj())/hi
            b = np.outer(u[:, 0], u[:, -1].conj())
            close(weighted(comm(a, b), w)/(weighted(a, w)*f2(b)), 1+hi/lo)
    print("Weighted commutators, proof identities, projected bounds and sharpness")


def deficit(u):
    return f2(u-np.eye(u.shape[0]))/2


def check_local():
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.diag([1, -1])
    eye = np.eye(2)
    for theta in (np.pi/3, 0.1, 0.001):
        u = np.cos(theta/2)*eye - 1j*np.sin(theta/2)*pauli_x
        v = np.cos(theta/2)*eye - 1j*np.sin(theta/2)*pauli_y
        h = u @ v @ star(u) @ star(v)
        close(comm(u, v), -2j*np.sin(theta/2)**2*pauli_z)
        close(deficit(u), 4*np.sin(theta/4)**2)
        close(deficit(h)/(4*deficit(u)*deficit(v)), np.cos(theta/4)**4)
        if theta == np.pi/3:
            close(deficit(h)/2, 1/8)
            assert round(np.degrees(2*np.arccos(7/8)), 2) == 57.91

    for n in range(1, 6):
        ident = np.eye(n)
        for _ in range(20):
            u, v, uu, vv = [unitary(n) for _ in range(4)]
            h = u @ v @ star(u) @ star(v)
            p = u @ v @ star(uu) @ star(vv)
            close(h-ident, comm(u-ident, v-ident) @ star(v @ u))
            close(f2(u @ v-v @ u), 2*deficit(h))
            le(deficit(h), 4*deficit(u)*deficit(v))
            le(np.sqrt(deficit(u @ v)), np.sqrt(deficit(u))+np.sqrt(deficit(v)))
            le(np.linalg.norm(h-ident, 2), 2*np.linalg.norm(u-ident, 2)*np.linalg.norm(v-ident, 2))
            le(np.sqrt(f2(p-ident)),
               np.sqrt(2*f2(u-ident)*f2(v-ident))+np.sqrt(f2(uu-u))+np.sqrt(f2(vv-v)))
            even, odd = ident-(p+star(p))/2, (p-star(p))/2
            le(0, np.linalg.eigvalsh(even)[0])
            close(np.trace(even), deficit(p))
            close(f2(even)+f2(odd), 2*deficit(p))
            norm_even = np.linalg.norm(even, 2)
            close(norm_even, np.linalg.norm(p-ident, 2)**2/2)
            le((1-norm_even/2)*deficit(p), f2(odd)/2)
            le(f2(odd)/2, deficit(p))
            chords = [item-ident for item in (u, v, star(uu), star(vv))]
            first = sum(chords)
            second = sum(chords[i] @ chords[j] for i, j in itertools.combinations(range(4), 2))
            sizes = [np.linalg.norm(x, 2) for x in chords]
            bound = sum(np.prod(s) for k in (3, 4) for s in itertools.combinations(sizes, k))
            le(np.linalg.norm(p-ident-first-second, 2), bound)
        identity = ident
        h0 = unitary(n)
        theta = np.linspace(-0.15, 0.15, n)
        u = np.diag(np.exp(1j*theta))
        h = h0
        for depth in range(1, 7):
            h = u @ h @ star(u) @ star(h)
            le(deficit(h), (4*deficit(u))**depth*deficit(h0))
        assert np.linalg.norm(h-identity) < 0.01

    # Flux of 1+x+x^2 on a square centred at the origin.
    side = np.array([0.25, 0.125, 0.0625])
    centre_flux = side**2 + side**4/12
    centre_error = np.abs(2*np.sin(centre_flux/2)**2 - side**4/2)
    close(centre_error[:-1]/centre_error[1:], 64, atol=3)
    print("Unitary commutator, spin, plaquette and centred-curvature estimates")


def check_cube_and_staples():
    for _ in range(16):
        X, Y, Z = [np.array([[unitary(3) for _ in range(2)] for _ in range(2)]) for _ in range(3)]
        f = [
            Z[0,0] @ Y[0,1] @ star(Z[0,1]) @ star(Y[0,0]),
            Y[0,0] @ Z[0,1] @ X[1,1] @ star(Z[1,1]) @ star(X[1,0]) @ star(Y[0,0]),
            Y[0,0] @ X[1,0] @ star(Y[1,0]) @ star(X[0,0]),
            X[0,0] @ Y[1,0] @ Z[1,1] @ star(Y[1,1]) @ star(Z[1,0]) @ star(X[0,0]),
            X[0,0] @ Z[1,0] @ star(X[0,1]) @ star(Z[0,0]),
            Z[0,0] @ X[0,1] @ Y[1,1] @ star(X[1,1]) @ star(Y[0,1]) @ star(Z[0,0]),
        ]
        product = np.linalg.multi_dot(f)
        close(product, np.eye(3))
        sizes = np.sqrt([deficit(p) for p in f])
        for i, size in enumerate(sizes):
            others = np.delete(sizes, i)
            le(size**2, others.sum()**2)
            le(others.sum()**2, 5*f2(others))
    flux = np.full(6, np.pi/3)
    close(flux.sum()/(2*np.pi), 1)
    close(np.max(1-np.cos(flux)), 0.5)
    le(4/3, np.sum(1-np.cos(flux)))
    small = np.array([1, *([-1/5]*5)]) * 0.001
    values = 2*np.sin(small/2)**2
    close(values[0]/(5*np.sum(values[1:])), 1, atol=1e-6)

    for length in (2, 3):
        n, beta = 3, 2.3
        links = np.array([unitary(n) for _ in range(2*length**2)]).reshape(length,length,2,n,n)

        def action(data):
            result = 0
            for i, j in np.ndindex(length, length):
                ip, jp = (i+1) % length, (j+1) % length
                p = data[i,j,0] @ data[ip,j,1] @ star(data[i,jp,0]) @ star(data[i,j,1])
                result += beta*deficit(p)/n
            return result

        flow = np.zeros((length,length,n,n), dtype=complex)
        original = action(links)
        for i, j, axis in np.ndindex(length,length,2):
            vertex = (i, j)
            shift = np.eye(2, dtype=int)
            end = tuple((np.array(vertex) + shift[axis]) % length)
            side = 1-axis
            above = tuple((np.array(vertex) + shift[side]) % length)
            below = tuple((np.array(vertex) - shift[side]) % length)
            end_below = tuple((np.array(end) - shift[side]) % length)
            return_path = (
                links[end][side] @ star(links[above][axis]) @ star(links[vertex][side])
                + star(links[end_below][side]) @ star(links[below][axis]) @ links[below][side])
            u = links[i,j,axis]
            local, incoming = u @ return_path, return_path @ u
            odd = (local-star(local))/2
            flow[vertex] += odd
            flow[end] -= (incoming-star(incoming))/2
            v = unitary(n)
            modified = links.copy()
            modified[i,j,axis] = v @ u
            close(action(modified)-original, -beta/n*np.trace((v-np.eye(n)) @ local).real)
            generator = unitary(n)
            generator = (generator-star(generator))/2
            eigenvalues, frame = np.linalg.eigh(-1j*generator)
            actions = []
            for s in (1e-4, -1e-4):
                modified[i,j,axis] = (frame*np.exp(1j*s*eigenvalues)) @ star(frame) @ u
                actions.append(action(modified))
            close((actions[0]-actions[1])/2e-4, -beta/n*np.trace(generator @ odd).real, atol=1e-8)
        close(flow, 0)
    print("Cube cancellation, magnetic-charge threshold and staple variation")


def character_data(name):
    if name == "S3":
        permutations = list(itertools.permutations(range(3)))
        index = {perm: j for j, perm in enumerate(permutations)}
        table = np.array([[index[tuple(p[i] for i in q)] for q in permutations] for p in permutations])
        standard = [sum(i == value for i, value in enumerate(p))-1 for p in permutations]
        parity = [1 if i else -1 for i in standard]
        characters = np.array([np.ones(6), standard, parity])
    else:
        order = int(name)
        table = np.add.outer(np.arange(order), np.arange(order)) % order
        characters = np.exp(2j*np.pi*np.outer(np.arange(order), np.arange(order))/order)
    inverse = np.argmax(table == 0, axis=1)
    return table, inverse, characters


def finite_group_coefficients(chars, beta, representation=1):
    dimensions = chars[:, 0].real
    weight = np.exp(beta*chars[representation].real/dimensions[representation])
    coefficients = (chars.conj() @ weight / (dimensions * len(weight))).real
    multiplicities = np.rint(
        (chars*chars[representation]) @ chars.conj().T / len(weight)
    ).real.astype(int)
    return dimensions, weight, coefficients, multiplicities


def group_configs(order, width):
    return np.indices((order,)*width, dtype=np.uint8).reshape(width, -1).T


def check_group_torus():
    for name in ("2", "3", "S3"):
        table, inv, chars = character_data(name)
        order = len(table)
        dim, phi, f, nrf = finite_group_coefficients(chars, 0.8)
        close(chars @ chars.conj().T/order, np.eye(len(chars)))
        if name == "3":
            close(f[1]/f[0], (np.exp(0.8)-np.exp(-0.4))/(np.exp(0.8)+2*np.exp(-0.4)))
        if name == "S3":
            assert round(-np.log(f[1]/f[0]), 4) == 1.4403
            assert round(-np.log(f[2]/f[0]), 4) == 2.4508
        for m, n in ((2, 2), (2, 3)):
            y = group_configs(order, m*n+1)
            h = np.zeros((m, n, len(y)), dtype=np.uint8)
            v = np.zeros_like(h)
            # Tree gauge sets ordinary vertical links and the first row's ordinary horizontal links to I.
            h[0,-1] = y[:, 0]
            h[1:] = y[:, 1:1+(m-1)*n].T.reshape(m-1, n, -1)
            v[-1] = y[:, 1+(m-1)*n:].T
            plaquettes = table[table[table[h, np.roll(v, -1, axis=1)],
                                      inv[np.roll(h, -1, axis=0)]], inv[v]]
            weights = np.prod(phi[plaquettes], axis=(0, 1))
            close(weights.mean(), np.sum(f**(m*n)))
            for area in range(1, n):
                loop = np.zeros(len(y), dtype=np.uint8)
                for j in range(area):
                    loop = table[loop, h[0,j]]
                loop = table[loop, v[0,area]]
                for j in reversed(range(area)):
                    loop = table[loop, inv[h[1,j]]]
                loop = table[loop, inv[v[0,0]]]
                direct = np.sum(weights * chars[1, loop]/dim[1])/weights.sum()
                expected = np.sum(
                    nrf*(dim[:, None]/dim[None, :])/dim[1]
                    * f[:, None]**area * f[None, :]**(m*n-area)
                )/np.sum(f**(m*n))
                close(direct, expected)
        for length in (1, 2):
            configs = group_configs(order, length)
            volume = len(configs)
            left = configs[:, None, :]
            right = configs[None, :, :]
            projection = np.zeros((volume, volume))
            for g in configs:
                shifted = table[table[g, configs], inv[np.roll(g, -1)]]
                columns = shifted @ order**np.arange(length - 1, -1, -1)
                projection[np.arange(volume), columns] += 1/volume
            kernel = np.prod(phi[table[right, inv[left]]], axis=2)/volume
            q = kernel @ projection
            close(q, projection @ kernel)
            close(q, q.T)
            close(projection @ projection, projection)
            expected_spectrum = np.sort(np.r_[f**length, np.zeros(volume-len(f))])
            close(np.linalg.eigvalsh(q), expected_spectrum)
            holonomies = np.zeros(volume, dtype=np.uint8)
            for j in range(length):
                holonomies = table[holonomies, configs[:, j]]
            operator = np.diag(chars[1, holonomies])
            for time in (2, 3, 4):
                partition = np.trace(np.linalg.matrix_power(q, time))
                close(partition, np.sum(f**(length*time)))
                for step in range(time+1):
                    first = np.linalg.matrix_power(q, step)
                    second = np.linalg.matrix_power(q, time-step)
                    direct = np.trace(operator @ first @ operator.conj().T @ second)/partition
                    expected = np.sum(nrf**2 * f[:, None]**(length*(time-step))
                                      * f[None, :]**(length*step))/partition
                    close(direct, expected)
    for beta in (-1.5, -0.7, 0, 0.8):
        _, _, f, _ = finite_group_coefficients(character_data("2")[2], beta)
        close(f, [np.cosh(beta), np.sinh(beta)])
        le(0, np.min(f**2))
        if beta < 0:
            assert f[0]*f[1] < 0

    # Compare compact-group Haar quadrature and cosine-integral ratios.
    theta = (np.arange(16384)+0.5)*np.pi/16384
    cos = np.cos(theta)
    for beta in (0.5, 1, 2, 3, 4, 6):
        weight = np.exp(beta*cos)
        integral = [np.mean(weight*np.cos(j*theta)) for j in range(3)]
        close(integral[0]-integral[2], 2*integral[1]/beta)
        close(np.sum(weight*cos*np.sin(theta)**2)/np.sum(weight*np.sin(theta)**2),
              integral[2]/integral[1])
        assert 0 < integral[1]/integral[0] < 1
        assert 0 < integral[2]/integral[1] < 1

    for name in ("2", "3", "5", "S3"):
        table, inv, chars = character_data(name)
        dim = chars[:, 0].real
        deficit_values = 1-chars[1].real/dim[1]
        minimum = np.min(deficit_values[1:])
        nearest = np.flatnonzero(np.isclose(deficit_values, minimum))
        delta = len(nearest)*np.eye(len(table))
        for g in nearest:
            delta -= np.eye(len(table))[table[inv[g]]]
        beta = 32
        lam = np.exp(-beta*minimum)
        weights = np.exp(-beta*deficit_values)
        scaled_generator = np.zeros_like(delta)
        for g in range(1, len(table)):
            scaled_generator += (weights[g]/lam)*(np.eye(len(table))-np.eye(len(table))[table[inv[g]]])
        scaled_generator /= weights.sum()
        error = np.linalg.norm(scaled_generator-delta, 2)
        assert error < 3e-2, (name, error)
        spectrum = len(nearest)-chars[:, nearest].sum(axis=1).real/dim
        expanded = np.repeat(spectrum, dim.astype(int)**2)
        close(np.linalg.eigvalsh(delta), np.sort(expanded))
        if name == "S3":
            close(spectrum, [0, 3, 6])
            close(sorted(deficit_values), [0, 1, 1, 1, 1.5, 1.5])
        if name == "5":
            beta = 24
            weights = np.exp(-beta*deficit_values)
            gap = np.sum(weights*(1-chars[1].real))/weights.sum()
            leading = np.exp(-beta*minimum)*spectrum[1]
            close(gap, 8.676304e-8, atol=6e-15)
            close(leading, 8.676305e-8, atol=6e-15)
    print("Planar Haar ratios, finite-group torus integrals, transfer and kinetic spectrum")


def check_duality():
    for lengths, periodic in (((1, 1, 1), False), ((2, 2, 1), False),
                              ((2, 2, 2), False), ((2, 2, 2), True)):
        cells, (_, b1, b2, b3) = grid(lengths, periodic)
        e, p, q = len(cells[1]), len(cells[2]), len(cells[3])
        r2, r3 = len(gf2_basis(b2)), len(gf2_basis(b3))
        assert not (b1 @ b2 % 2).any() and not (b2 @ b3 % 2).any()
        assert r3 == q-int(periodic)
        assert p-r2-r3 == 3*periodic
        spin = bits(q)
        seams = np.array([[set(axes) != set(range(3)) and
                           axis not in axes and v[axis] == 0
                           for v, axes in cells[2]] for axis in range(3)], dtype=np.uint8)
        if periodic:
            assert (e, p) == (3*q, 3*q)
            assert not (b2 @ seams.T % 2).any()
            closed = ((spin @ b3.T)[None, :, :] + (bits(3) @ seams)[:, None, :]) % 2
            unique = np.unique(closed.reshape(-1, p), axis=0)
            assert len(unique)*2 == closed.shape[0]*closed.shape[1]
        else:
            closed = (spin @ b3.T % 2)[None, :, :]
        # Independent cut-set encoding of unsatisfied dual Ising bonds.
        bond_values = np.zeros_like(closed)
        for face in range(p):
            neighbours = np.flatnonzero(b3[face])
            bond_values[:, :, face] = spin[:, neighbours].sum(axis=1)[None, :] % 2
        if periodic:
            bond_values ^= (bits(3) @ seams % 2)[:, None, :]
        np.testing.assert_array_equal(closed, bond_values)
        if periodic:
            # Enumerate all frustration patterns via an independent basis of the link map.
            basis = gf2_basis(b2)
            syndromes = bits(len(basis)) @ basis % 2
            face = 0
            degree = syndromes.sum(axis=1)
            assert np.sum(degree == 2) == 0
            assert np.sum((degree == 4) & (syndromes[:, face] == 1)) == 7
            s0 = np.eye(p, dtype=np.uint8)[face]
            for beta in (0.3, 0.761, 1.4):
                t = np.tanh(beta)
                k = -np.log(t)/2
                close(np.tanh(k), np.exp(-2*beta))
                gauge_weights = np.exp(beta*(p-2*degree.astype(int)))
                ising = np.exp(k*(p-2*closed.sum(axis=2).astype(int)))
                factor = 2**(e-1)*np.cosh(beta)**p*np.exp(-k*p)
                z = 2**(e-r2)*gauge_weights.sum()
                close(z, factor*ising.sum(), rtol=1e-13)
                flipped = np.exp(k*(p-2*(closed ^ s0).sum(axis=2).astype(int)))
                close(np.sum(sign(syndromes @ s0)*gauge_weights)/gauge_weights.sum(),
                      flipped.sum()/ising.sum())
                le(ising[0].sum(), ising.sum())
                le(ising.sum(), 8*ising[0].sum())
                if beta == 0.761:
                    assert round(2*ising[0].sum()/ising.sum(), 3) == 0.493
        else:
            for level in range(lengths[2]+1):
                surface = np.array([axes == (0, 1) and v[2] == level
                                    for v, axes in cells[2]], dtype=np.uint8)
                area = int(surface.sum())
                sizes = (closed[0] ^ surface).sum(axis=1)
                assert sizes.min() == area and np.count_nonzero(sizes == area) == 1
                assert np.count_nonzero(sizes == area+4) == area*(1+(0 < level < lengths[2]))
                assert np.all((sizes - area) % 2 == 0)
                if q == 1:
                    np.testing.assert_array_equal(np.sort(sizes), [1, 5])
                    np.testing.assert_array_equal(closed[0].sum(axis=1), [0, 6])
                    configs = bits(e)
                    beta = 0.8
                    weights = np.exp(beta*sign(configs @ b2).sum(axis=1))
                    close(np.sum(sign(configs @ b2 @ surface)*weights)/weights.sum(),
                          (np.tanh(beta)+np.tanh(beta)**5)/(1+np.tanh(beta)**6))
                    k = -np.log(np.tanh(beta))/2
                    spin_weights = np.exp(k*(p-2*closed.sum(axis=2).astype(int)))
                    close(weights.sum(), 2**e*np.cosh(beta)**p*np.exp(-p*k)*spin_weights.sum())

    # Classify four-edge excitations on the 3^3 torus by enumerating even dual walks of length four.
    cells, (_, _, b2, b3) = grid((3, 3, 3), True)
    dual = b3.T
    adj = [np.flatnonzero(dual[i]) for i in range(dual.shape[0])]
    endpoints = [np.flatnonzero(row) for row in dual.T]
    representatives = np.array([[axis not in axes and v[axis] == 0 for v, axes in cells[2]]
                                for axis in range(3)], dtype=np.uint8)
    four_cycles = set()
    for v0 in range(len(adj)):
        for e1 in adj[v0]:
            v1 = next(v for v in endpoints[e1] if v != v0)
            for e2 in adj[v1]:
                v2 = next(v for v in endpoints[e2] if v != v1)
                if v2 == v0 or e2 == e1:
                    continue
                for e3 in adj[v2]:
                    v3 = next(v for v in endpoints[e3] if v != v2)
                    if v3 in (v0, v1) or e3 in (e1, e2):
                        continue
                    for e4 in adj[v3]:
                        if e4 not in (e1, e2, e3) and v0 in endpoints[e4]:
                            four_cycles.add(tuple(sorted((e1, e2, e3, e4))))
    cycles = np.array(sorted(four_cycles))
    zero_winding = cycles[np.all(representatives[:, cycles].sum(axis=2) % 2 == 0, axis=0)]
    assert len(zero_winding) == 81
    assert np.count_nonzero(np.any(zero_winding == 0, axis=1)) == 4
    print("Cubical homology, strong coupling, twisted Ising duality and perimeter coefficients")



def check_local_gauge_source():
    # Triangle, with one vertex gauge reversing the signs of the first two links.
    labels = sign(bits(3))
    local_path = labels[:, 0]*labels[:, 2]
    beta = 0.9
    base = np.exp(beta*np.prod(labels, axis=1))
    close(np.dot(local_path, base)/base.sum(), 0)
    local_flip = np.array([-1, -1, 1])
    for source in (-1, -0.3, -0.01, 0, 0.01, 0.4, 1.2):
        perturbed = base*np.exp(source*labels.sum(axis=1))
        changed = base*np.exp(source*(labels*local_flip).sum(axis=1))
        close(perturbed.sum(), changed.sum())
        close(np.dot(local_path, perturbed), -np.dot(local_path, changed))
        le(abs(np.dot(local_path, perturbed)/perturbed.sum()), np.expm1(4*abs(source)))
    print("Local gauge averaging and the volume-independent source bound")


def check_rectangular_and_compact():
    rows, columns, n = 3, 4, 3
    horizontal = [[unitary(n) for _ in range(rows+1)] for _ in range(columns)]
    vertical = [[unitary(n) for _ in range(rows)] for _ in range(columns+1)]
    result = np.eye(n, dtype=complex)
    bottom = [np.eye(n)]
    for links in horizontal:
        bottom.append(bottom[-1] @ links[0])
    accumulated = []
    for i in reversed(range(columns)):
        access = bottom[i]
        for j in range(rows):
            face = (horizontal[i][j] @ vertical[i+1][j]
                    @ star(horizontal[i][j+1]) @ star(vertical[i][j]))
            result = result @ access @ face @ star(access)
            accumulated.append(deficit(face))
            access = access @ vertical[i][j]
    boundary = bottom[-1].copy()
    for link in vertical[-1]:
        boundary = boundary @ link
    for links in reversed(horizontal):
        boundary = boundary @ star(links[-1])
    for link in reversed(vertical[0]):
        boundary = boundary @ star(link)
    close(result, boundary)
    le(np.sqrt(deficit(result)), np.sqrt(accumulated).sum())
    le(deficit(result), rows*columns*np.sum(accumulated))
    eta = 1e-5
    close((np.sin(rows*columns*eta/2)/np.sin(eta/2))**2/(rows*columns),
          rows*columns, atol=3e-8)

    theta = (np.arange(16384)+0.5)*np.pi/16384
    cosine = np.cos(theta)
    for beta in (30., 60., 120., 240.):
        weight = np.exp(beta*(cosine-1))
        moments_circle = np.array([np.mean(weight*np.cos(r*theta)) for r in range(6)])
        spin_weight = weight*np.sin(theta)**2
        close(np.mean(weight*(1-np.cos(2*theta))), 2*moments_circle[1]/beta)
        for r in range(5):
            ratio = moments_circle[r]/moments_circle[0]
            assert abs(-np.log(ratio)-r*r/(2*beta)) < 8/beta**2
        for dimension in range(1, 5):
            spin = (dimension-1)/2
            normalized_character = np.sin(dimension*theta)/(dimension*np.sin(theta))
            ratio = np.sum(spin_weight*normalized_character)/spin_weight.sum()
            close(ratio, moments_circle[dimension]/moments_circle[1])
            assert abs(-np.log(ratio)-2*spin*(spin+1)/beta) < 8/beta**2
    weights = np.exp(0.9*(cosine-1))
    fourier = np.array([np.mean(weights*np.cos(r*theta)) for r in range(12)])
    indices = np.abs(np.arange(-10, 11))
    f, shifted = fourier[indices], fourier[np.abs(np.arange(-10, 11)+1)]
    for volume in (9, 16, 100):
        f_normalized, shifted_normalized = f/f[10], shifted/f[10]
        value = np.sum(f_normalized**(volume-4)*shifted_normalized**4)/np.sum(f_normalized**volume)
        if volume == 100:
            close(value, (fourier[1]/fourier[0])**4)
    # At zero coupling the scalar action is constant although an individual staple is odd.
    link, first_return, second_return = 1j, 1, 1
    odd = (link*(first_return+second_return)
           - np.conj(link*(first_return+second_return)))/2
    close(odd, 2j)
    assert 0*deficit(np.array([[link]])) == 0
    print("Ordered rectangular boundary, circle torus limit and compact kinetic coefficients")


if __name__ == "__main__":
    check_grid()
    check_parity()
    check_local_gauge_source()
    check_weighted()
    check_local()
    check_cube_and_staples()
    check_group_torus()
    check_duality()
    check_rectangular_and_compact()
    print("All m15 checks passed.")
