# Checks for Section 10C (sec:m9); vendored from the m9 module script.
#!/usr/bin/env python3
"""Finite regression checks for the arithmetic section. NumPy is the only dependency."""
from itertools import combinations, product
from fractions import Fraction
from math import comb, gcd, isqrt

import numpy as np


def close(actual, expected, *, atol=2e-11, rtol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def legal_words(order, length):
    return ["".join(bits) for bits in product("01", repeat=length)
            if "1" * order not in "".join(bits)]


def bonacci(order):
    return max(np.roots([1] + [-1] * order).real)


def fibs(length):
    values = [0, 1]
    for _ in range(length - 2):
        values.append(sum(values[-2:]))
    return values[:length]


def circular_gaps(step, count):
    points = np.sort(np.mod(step * np.arange(count), 1))
    return np.diff(np.r_[points, points[0] + 1])


def check_circle():
    angle = np.arange(32768) / 32768
    scalar = np.cos(2 * np.pi * angle)
    evolved = scalar.copy()
    for n in range(8):
        close(evolved, np.cos((2**n) * 2 * np.pi * angle), atol=2e-11)
        evolved = 2 * evolved**2 - 1
    x = np.linspace(0, 1, 1025)
    close(np.cos(np.pi * (1 - abs(1 - 2*x))), 2*np.cos(np.pi*x)**2 - 1)
    close(np.mean(scalar <= x[::32, None] * 2 - 1, axis=1),
          1 - np.arccos(x[::32] * 2 - 1)/np.pi, atol=1/32768)
    close([np.mean(scalar**n) for n in (1, 2, 4)], [0, 1/2, 3/8])
    for n in range(1, 12):
        codes = np.floor(angle * 2**n).astype(int)
        assert np.all(np.bincount(codes) == 32768 // 2**n)
        tent = x.copy()
        for _ in range(n):
            tent = 1 - abs(1 - 2*tent)
        if n <= 10:
            close(tent[::2**(10-n)], np.arange(2**n + 1) % 2)
    for alpha in [(5**0.5-1)/2, 2**0.5-1, np.pi-3, np.e-2]:
        for count in range(2, 90):
            gaps = circular_gaps(alpha, count)
            close(sum(gaps), 1)
            lengths = np.unique(np.round(gaps, 10))
            assert len(lengths) <= 3
            if len(lengths) == 3:
                close(lengths[-1], sum(lengths[:2]), atol=2e-10)
    alpha = (5**0.5-1)/2
    fib = fibs(13)
    for n in range(3, 12):
        close(fib[n]*alpha-fib[n-1], (-1)**(n-1)*alpha**n)
        for j in range(fib[n-1] + 1):
            expected = np.repeat([alpha**(n-2), alpha**(n-1), alpha**n],
                                 [fib[n-1]-j, fib[n-2]+j, j])
            close(np.sort(circular_gaps(alpha, fib[n] + j)), np.sort(expected))


def check_digits_parry():
    roots = [bonacci(order) for order in range(2, 19)]
    assert np.all(np.diff(roots) > 0)
    close(roots[:3], [1.618034, 1.839287, 1.927562], atol=5e-7, rtol=0)
    close((2-roots[-1]) * 2**18, 1, atol=4e-5)
    for order in range(2, 10):
        roots_all = np.roots([1] + [-1] * order)
        beta = roots[order - 2]
        assert 1 < beta < 2
        other = roots_all[abs(roots_all-beta) > 1e-10]
        assert len(other) == order-1 and np.max(abs(other)) < 1
        close(2-beta, beta**(-order))
        incidence = np.zeros((order, order))
        incidence[0] = 1
        incidence[1:, :-1] = np.eye(order-1)
        close(np.poly(incidence), [1] + [-1] * order)
        assert np.all(np.linalg.matrix_power(incidence, 2*order) > 0)
        p = 1/beta
        durations = p**np.arange(1, order+1)
        w = np.cumsum(durations)[::-1]
        normalizer = np.arange(1, order+1) @ durations
        pi = p**np.arange(order)*w/normalizer
        transition = np.zeros((order, order))
        signed = np.zeros((2*order, 2*order))
        for i in range(order):
            transition[i, 0] = p/w[i]
            if i+1 < order:
                transition[i, i+1] = p*w[i+1]/w[i]
            for sign in range(2):
                signed[sign*order+i, (1-sign)*order] = transition[i, 0]
                if i+1 < order:
                    signed[sign*order+i, sign*order+i+1] = transition[i, i+1]
        close(sum(durations), 1)
        close(sum(pi), 1)
        close(transition.sum(axis=1), 1)
        close(pi @ transition, pi)
        close(np.tile(pi/2, 2) @ signed, np.tile(pi/2, 2))
        assert np.min(np.linalg.matrix_power(signed, 4*order)) > 0
        cube = np.linalg.matrix_power(signed, 3)
        assert np.min(cube[:, [0, order]]) >= 1/8-1e-14
        for length in range(1, 10):
            close(pi @ transition[:, 0] * transition[0, 0]**(length-1),
                  p**(length-1)/normalizer)
        close(transition[0, 0], p)
        close((transition @ transition)[0, 0], 2*p*p)
        close(-durations @ np.log(durations), normalizer*np.log(beta))
        law = np.arange(1, order+1, dtype=float)
        law /= sum(law)
        close(law @ np.log(law/durations),
              law @ np.log(law) + np.log(beta)*(law @ np.arange(1, order+1)))
        counts = [len(legal_words(order, n)) for n in range(10)]
        for n in range(10):
            words = legal_words(order, n)
            if n < order:
                assert counts[n] == 2**n
            else:
                assert counts[n] == sum(counts[n-order:n])
            int_values = [sum(int(bit)*counts[i] for i, bit in enumerate(word[::-1]))
                          for word in words]
            assert int_values == list(range(counts[n]))
            endpoints = [sum(int(bit)*p**i for i, bit in enumerate(word, 1))
                         for word in words]
            suffixes = np.array([len(word)-len(word.rstrip("1")) for word in words])
            gaps = np.diff(np.r_[endpoints, 1])
            close(gaps, p**n*w[suffixes])
            assert len(np.unique(suffixes[:-1])) == min(order, n)
            assert len(np.unique(suffixes)) == min(order, n+1)
            for word, start, gap in zip(words, endpoints, gaps):
                remainder = start + gap/2
                probability = 1
                state = 0
                for bit in word:
                    digit = int(beta*remainder)
                    assert digit == int(bit)
                    remainder = beta*remainder-digit
                    next_state = state+1 if digit else 0
                    probability *= transition[state, next_state]
                    state = next_state
                close(probability, p**n*w[state])
        if order == 2:
            assert counts == fibs(12)[2:]
            observations = np.array([0., 0.5, 0.55, 2.2, 13.8])
            for level in range(1, 10):
                reconstructed = np.floor(beta**level*observations)/beta**level
                assert np.all(reconstructed <= observations)
                assert np.all(observations-reconstructed < beta**(-level))
            for n in range(10):
                for x in (0, 0.1, -0.3, 0.95):
                    close(sum(x**i for i in range(counts[n])),
                          (1-x**counts[n])/(1-x))
            close(p*p/(1-p*p), p)
            close([2*np.cos(2*np.pi/5), 2*np.cos(np.pi/5)], [p, beta])
        if order == 3:
            assert counts[:5] == [1, 2, 4, 7, 13]


def apply_substitution(word):
    images = {"A": "AB", "B": "AC", "C": "A"}
    return "".join(images[letter] for letter in word)


def substitute(word, levels):
    for _ in range(levels):
        word = apply_substitution(word)
    return word


def check_windows():
    roots = np.roots([1, -1, -1, -1])
    beta = roots[np.argmax(roots.real)].real
    z = roots[np.argmax(roots.imag)]
    r = abs(z)
    close(z, -0.419643 + 0.606291j, atol=5e-7, rtol=0)
    close(r, 0.737353, atol=5e-7, rtol=0)
    close(np.angle(z, deg=True), 124.69, atol=0.005, rtol=0)
    close(2*z.real, 1-beta)
    close(r**2, 1/beta)
    close(sum(r**(2*np.arange(1, 4))), 1)
    close(r**6+r**4+r**2, 1)
    close(beta*r**2, 1)
    assert 0.73735270 < r < 0.73735271
    lambda_boundary = max(np.roots([1, 0, 0, -2, -1]).real)
    close(2*np.log(lambda_boundary)/np.log(beta), 1.09336, atol=5e-6, rtol=0)
    p = (5**0.5-1)/2
    for actual, expected in zip(np.sort(-p*np.array([-1, 1/p])), [-1, p]):
        close(actual, expected)
    close(1+p*p*np.array([-1, 1/p]), [p, 1/p])
    letter_value = dict(zip("ABC", [1, z-1, z*z-z-1]))
    close([sum(letter_value[c] for c in apply_substitution(letter))
           for letter in "ABC"], [z*letter_value[c] for c in "ABC"])
    close(np.array([beta*beta, beta, 1]) @ list(letter_value.values()), 0)
    for length in range(9):
        words = legal_words(3, length)
        values = [sum(int(bit)*z**i for i, bit in enumerate(word)) for word in words]
        prefixes = np.r_[0, np.cumsum([letter_value[c] for c in
                                      substitute("A", length)])[:-1]]
        close(np.sort_complex(np.round(prefixes, 10)),
              np.sort_complex(np.round(values, 10)), atol=2e-10)
        assert max(abs(np.array(values))) <= 1/(1-r)
        if length >= 3:
            for word, value in zip(words, values):
                zeros = word.index("0")
                offset = sum(z**j for j in range(zeros))
                remainder = word[zeros+1:]
                close(value, offset + z**(zeros+1) *
                      sum(int(bit)*z**i for i, bit in enumerate(remainder)))
    for a, b, c in product(range(-2, 3), repeat=3):
        close(sum(a+b*x+c*x*x for x in roots), 3*a+b+3*c)


def valuation(value, prime):
    count = 0
    while value % prime == 0:
        value //= prime
        count += 1
    return count


def check_completions():
    matrices = [(a, b, c, d) for a, b, c, d in product(range(2), repeat=4)
                if a*d != b*c]
    assert len(matrices) == 6
    assert {(a, b, c, d) for a, b, c, d in matrices
            if (d-a)**2 + 4*b*c == 5} == {(1, 1, 1, 0), (0, 1, 1, 1)}
    for pair in ((-1, 5), (6, 30), (3, 51), (4, 21)):
        x, y = pair
        for p in (2, 3, 5, 7):
            depth = valuation(x-y, p)
            assert all(x % p**k == y % p**k for k in range(depth+1))
            assert x % p**(depth+1) != y % p**(depth+1)
            for z in range(-20, 22):
                def distance(a, b):
                    return Fraction(1, p**valuation(a-b, p)) if a != b else 0
                assert distance(x, z) <= max(distance(x, y), distance(y, z))
            for k in range(1, 6):
                unit = 11
                inverse = pow(unit, -1, p**k)
                residue = (x*inverse) % p**k
                assert (residue*unit-x) % p**k == 0
    for value in (Fraction(-35, 72), Fraction(225, 14), Fraction(1, 3), Fraction(6)):
        factor = abs(value)
        for p in (2, 3, 5, 7, 11):
            exponent = valuation(value.numerator, p)-valuation(value.denominator, p)
            factor *= Fraction(p)**(-exponent)
        assert factor == 1
    for moduli in [(4, 9), (5, 7, 8), (2, 3, 5, 7)]:
        count = int(np.prod(moduli))
        assert len({tuple(x % m for m in moduli) for x in range(count)}) == count
    for a in ((1,)*7, (2, 1, 3, 2, 1), (4, 2, 1, 1), (1, 3, 2, 1)):
        q = [0, 1]
        for digit in a:
            q.append(digit*q[-1] + q[-2])
        values = []
        for digits in product(range(a[0]), *(range(d+1) for d in a[1:])):
            if all(digits[i] < a[i] or digits[i-1] == 0 for i in range(1, len(a))):
                values.append(sum(v*w for v, w in zip(digits, q[1:])))
        assert sorted(values) == list(range(q[-1]))
    size = 0
    for _ in range(5):
        size = 2**size
    assert size == 65536
    diagonal = np.array([[0, 1], [0, 0]])
    assert np.all(np.any(diagonal != 1-np.diag(diagonal), axis=1))
    complex_turn = np.array([[0, -1], [1, 0]])
    close(complex_turn @ complex_turn, -np.eye(2))
    for labels in ([0, 0], [0, 1, 0, 2, 1], [0, 1, 2, 3]):
        counts = np.bincount(labels)
        n = len(labels)
        separates = sum(a != b for a in labels for b in labels)
        assert separates == n*n-counts @ counts
        close(separates/(n*(n-1)), (1-(counts/n) @ (counts/n))*n/(n-1))


def prime_table(limit):
    sieve = np.ones(limit+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, isqrt(limit)+1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.flatnonzero(sieve)


def check_euler():
    primes = prime_table(100000)
    selected = primes[primes <= 5]
    close(np.prod(1/(1-selected.astype(float)**-2)), 25/16)
    missing = [n for n in range(1, 15)
               if any(n % p == 0 for p in primes[3:6])]
    assert missing == [7, 11, 13, 14]
    assert [n*n for n in missing] == [49, 121, 169, 196]
    close(np.pi**2/6-25/16, 0.082434, atol=5e-7, rtol=0)
    integers = np.arange(1, 100001, dtype=float)
    assert abs(sum(integers**-2)-np.pi**2/6) < 1/len(integers)
    assert primes[np.searchsorted(primes, 31)+1] == 37
    assert int(np.prod(primes[primes <= 13]))+1 == 30031 == 59*509
    for ratio, period in zip([3**12/2**19, 81/80, 5*3**8/2**15, 128/125],
                             [463.67, 505.79, 5567.67, 264.93]):
        close(2*np.pi/np.log(ratio), period, atol=0.005, rtol=0)
    mu = np.ones(len(integers)+1, dtype=int)
    for pp in primes:
        p = int(pp)
        mu[p::p] *= -1
        mu[p*p::p*p] = 0
    assert mu[1:11].tolist() == [1, -1, -1, 0, -1, 1, -1, 0, 0, 1]
    assert sum(mu[1:11]) == -1
    for n in range(1, 200):
        assert sum(mu[d] for d in range(1, n+1) if n % d == 0) == (n == 1)
    for s in [2, 2.3+2j, 3.1-4j]:
        z = np.exp(-s*np.log(primes))
        partial = np.sum(integers**(-s))
        logarithms = -np.log1p(-z)
        assert np.all(abs(logarithms.imag - z.imag) <=
                      abs(z)**2/(1-abs(z)) + 1e-15)
        close(np.exp(logarithms), 1/(1-z))
        close(np.exp(np.sum(logarithms)), partial, atol=1.1e-5)
        close(np.exp(-np.sum(logarithms)), mu[1:] @ integers**(-s), atol=1.1e-5)
        close(np.sum(np.log(primes)*z/(1-z)),
              np.sum(np.log(integers)*integers**(-s))/partial, atol=1.4e-4)
        cutoff = np.prod(1/(1-z[:3]))
        q = 7.
        assert abs(partial-cutoff) <= q**(-s.real)+q**(1-s.real)/(s.real-1)
        if np.isreal(s):
            assert (partial-cutoff).real > q**(-s) and np.all(abs(logarithms.imag) < 1e-14)
        close(np.prod(np.exp(1j*logarithms.imag)), np.exp(1j*sum(logarithms.imag)))
        alternating = np.sum(integers[::2]**(-s)-integers[1::2]**(-s))
        close(alternating, (1-2**(1-s))*partial, atol=1.1e-5)
    at_one = np.sum(1/integers[::2]-1/integers[1::2])
    close(at_one, np.log(2), atol=1/len(integers))
    for s in [0.25, 0.5+2j, 1.2-3j]:
        odds = integers[::2]
        pairs = odds**(-s)-(odds+1)**(-s)
        for m in [1, 7, 20, 100, 1000]:
            assert abs(sum(pairs[m-1:])) <= abs(s)*(2*m-1)**(-s.real)/s.real
    def staircase(x, midpoint=False):
        return sum(np.log(p)*(0.5 if midpoint and p**j == x else 1)
                   for p in (2, 3, 5, 7) for j in range(1, 4) if p**j <= x)
    close(staircase(8, True), np.log(420*np.sqrt(2)))
    close(staircase(9.5), np.log(2520))
    partition_rows = [
        [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42],
        [1, 2, 5, 10, 20, 36, 65, 110, 185, 300, 481],
        [1, 1, 3, 6, 13, 24, 48, 86, 160, 282, 500],
    ]
    for colors, expected in zip([lambda n: 1, lambda n: 2, lambda n: n], partition_rows):
        coefficients = np.array([1]+[0]*10)
        for n in range(1, 11):
            factor = np.zeros(11, dtype=int)
            for occupation in range(10//n+1):
                factor[occupation*n] = comb(colors(n)+occupation-1, occupation)
            coefficients = np.convolve(coefficients, factor)[:11]
        assert coefficients.tolist() == expected


def check_lattices():
    J = np.array([[0, 1], [1, 1]])
    identity = np.eye(2, dtype=int)
    assert np.array_equal(J @ J, J+identity)
    for t, p, q in product(range(-5, 6), repeat=3):
        eisenstein = p*p-(2*t+1)*p*q+(t*t+t+1)*q*q
        x = p-(t+1)*q
        assert eisenstein == x*x+x*q+q*q
        assert (2*t+1)**2-4*(t*t+t+1) == -3
    assert 8**2-5*8*3+7*3**2 == (-1)**2-3+3**2 == 7
    assert 2**2+2-1 == 5 and 5**2-5 == 20
    unit_examples = set()
    for n in range(-16, 17):
        g = np.rint(np.linalg.matrix_power(J if n >= 0 else J-identity, abs(n))).astype(int)
        trace = int(np.trace(g))
        y = int(g[0, 1])
        assert trace**2-5*y*y == 4*(-1)**(n % 2)
        unit_examples.update([(trace, y), (-trace, -y)])
    for x, y in product(range(-100, 101), repeat=2):
        if abs(x*x-5*y*y) == 4:
            assert (x-y) % 2 == 0
            assert (x, y) in unit_examples
    fib = fibs(8)
    assert [(2 if n == 0 else 2*fib[n-1]+fib[n], fib[n]) for n in range(6)] == [
        (2, 0), (1, 1), (3, 1), (4, 2), (7, 3), (11, 5)]
    for a, b in product(range(-8, 9), repeat=2):
        g = a*identity+b*J
        norm = a*a+a*b-b*b
        close(np.linalg.det(g), norm)
        assert np.trace(g) == 2*a+b and (2*a+b)**2-5*b*b == 4*norm
    norms = np.sum(np.array(list(product(range(-2, 3), repeat=4)))**2, axis=1)
    assert np.bincount(norms)[1:5].tolist() == [8, 24, 32, 24]
    assert gcd(2, 3) == 1


def entropy(values):
    positive = values[values > 1e-14]
    return -positive @ np.log(positive)


def check_cuts():
    for order in range(2, 7):
        for m, n in product(range(6), repeat=2):
            left, right = legal_words(order, m), legal_words(order, n)
            amplitude = np.array([["1"*order not in x+y for y in right] for x in left])
            total = np.count_nonzero(amplitude)
            probability = total/(len(left)*len(right))
            reduced = (amplitude.astype(float) @ amplitude.astype(float).T)/total
            spectrum = np.linalg.eigvalsh(reduced)
            u, v = min(m, order-1), min(n, order-1)
            expected_rank = 1+max(0, u+v-order+1)
            assert sum(spectrum > 1e-11) == expected_rank
            A = np.bincount([len(x)-len(x.rstrip("1")) for x in left])
            B = np.bincount([len(y)-len(y.lstrip("1")) for y in right])
            kernel = np.sqrt(A[:, None]*B[None, :]/total)
            kernel *= np.arange(len(A))[:, None]+np.arange(len(B)) < order
            close(np.linalg.eigvalsh(kernel @ kernel.T)[-expected_rank:],
                  spectrum[-expected_rank:])
            d = (order-1)/2**order
            assert 1-probability <= d+1e-12
            assert entropy(spectrum) <= entropy(np.array([d, 1-d]))+d*np.log(order-1)+1e-12
        p = 1/bonacci(order)
        counts = [2**j for j in range(order)]
        for j in range(order, 101):
            counts.append(sum(counts[j-order:j]))
        M = sum(j*p**j for j in range(1, order+1))
        close(counts[100]*p**100, 1/(p*M))
        i, j = np.indices((order, order))
        limit = (i+j < order)*p**((i+j+1)/2)/np.sqrt(M)
        finite = (i+j < order)*np.sqrt(np.array(
            [[counts[39-u]*counts[39-v]/counts[80] for v in range(order)]
             for u in range(order)]))
        close(finite, limit)
        spectrum = np.linalg.eigvalsh(limit @ limit.T)
        close(sum(spectrum), 1)
        assert 0 < entropy(spectrum) < np.log(order)
    for matrix, eigenvalues in [([[1, 1], [1, 1]], [0, 1]),
                                 ([[1, 1], [1, -1]], [1/2, 1/2])]:
        amplitude = np.array(matrix)/2
        close(amplitude**2, 1/4)
        close(np.linalg.eigvalsh(amplitude @ amplitude.T), eigenvalues)
    close(np.linalg.eigvalsh(np.array([[2, 1], [1, 1]])/3),
          [(3-np.sqrt(5))/6, (3+np.sqrt(5))/6])
    left, right = legal_words(3, 4), legal_words(3, 5)
    assert [sum(x.startswith(str(b)) for x in left) for b in (0, 1)] == [7, 6]
    assert [sum(y.startswith(str(b)) for y in right) for b in (0, 1)] == [13, 11]
    a = [sum(x.startswith("1") and len(x)-len(x.rstrip("1")) < s for x in left)
         for s in range(3)]
    n = [sum(y.startswith("1") and len(y)-len(y.lstrip("1")) < 3-s for y in right)
         for s in range(3)]
    assert a == [0, 3, 5] and n == [11, 7, 0]
    assert [min(7, 13//2), min(6, *[x+y for x, y in zip(a, n)])] == [6, 5]
    for ranks in product(range(1, 7), range(1, 6)):
        for bit, dimension, rank in zip("01", (2, 1), ranks):
            rows = [x for x in left if x.startswith(bit)]
            cols = [x for x in right if x.startswith(bit)]
            rows = sorted(rows, key=lambda x: len(x)-len(x.rstrip("1")))[:rank]
            vectors = np.zeros((dimension, rank, len(cols)))
            used = set()
            for r, x in reversed(list(enumerate(rows))):
                candidates = [i for i, y in enumerate(cols) if "111" not in x+y and i not in used]
                assert len(candidates) >= dimension
                for j, i in enumerate(candidates[:dimension]):
                    vectors[j, r, i] = 1/np.sqrt(rank)
                    used.add(i)
            for i, j in product(range(dimension), repeat=2):
                close(vectors[i] @ vectors[j].T, np.eye(rank)/rank if i == j else 0)


def check_parent_chain():
    for K in range(2, 5):
        ell = 2*K-1
        for N in range(1, 8):
            words = ["".join(bits) for bits in product("01", repeat=N)]
            index = {word: i for i, word in enumerate(words)}
            H = np.zeros((len(words), len(words)))
            for word, i in index.items():
                H[i, i] += sum(word[j:j+K] == "1"*K for j in range(N-K+1))
                for site in range(N):
                    blocked = any(word[j:site]+word[site+1:j+K] == "1"*(K-1)
                                  for j in range(max(0, site-K+1), min(site, N-K)+1))
                    if not blocked:
                        flipped = word[:site] + ("1" if word[site] == "0" else "0") + word[site+1:]
                        H[i, i] += 1
                        H[i, index[flipped]] -= 1
            ground = np.array([("1"*K not in word) for word in words], dtype=float)
            close(H @ ground, 0)
            spectrum = np.linalg.eigvalsh(H)
            close(spectrum[0], 0)
            assert spectrum[1] >= 2/(ell*ell*2**ell)-1e-10
            if N < K:
                close(spectrum[1], 2)
            if N == K == 2:
                close(spectrum, [0, 1, 1, 3])
    assert [substitute(letter, 4) for letter in "ABC"] == [
        "ABACABAABACAB", "ABACABAABAC", "ABACABA"]
    charge = lambda word: sum((-1)**i * (2-"ABC".index(letter))
                              for i, letter in enumerate(word))
    witness = ""
    for j in range(4):
        images = [substitute(letter, 4*j) for letter in "ABC"]
        assert [charge(word) for word in images] == [
            2, 2-(-1)**j, 2-(2*j+2)*(-1)**j]
        if j:
            witness = str(images[2])+witness
        assert images[0].startswith(witness)
        assert charge(witness) == 1-(-1)**j*(j*j+3*j+1)
        assert abs(charge(witness)) >= j*j and len(witness) <= 13**j
    for level in range(1, 10):
        word = substitute("A", level)
        N = len(word)//2*2
        for rho in (0.45, 0.73, 0.97):
            hopping = rho**np.array([2-"ABC".index(c) for c in word[:N-1]])
            diag, sub = hopping[::2], hopping[1::2]
            L = np.diag(diag)-np.diag(sub, -1)
            inverse = np.linalg.inv(L)
            delta = np.linalg.svd(L, compute_uv=False)[-1]
            assert delta > 0
            close(delta*np.linalg.norm(inverse, ord=2), 1, atol=1e-7)
            if level >= 4:
                k = 4*((level-4)//4)
                start = len(substitute("ABAC", k))
                m = len(substitute("ABA", k))
                q = (m-1)//2
                assert start % 2 == 0 and m >= 3 and m % 2 == 1
                assert start+2*m <= N <= 128*m
                assert word[start:start+m] == word[start+m:start+2*m]
                a = start//2
                pair_sum = 0
                for l in range(q):
                    for d in range(q-l):
                        first = inverse[a+l+d+1, a+l]
                        second = inverse[a+q+l+d+1, a+q+l+1]
                        close(first*second, 1/(hopping[start+2*l]*hopping[start+2*(l+d+1)]),
                              atol=1e-8)
                        pair_sum += first+second
                assert q*(q+1) <= pair_sum*(1+1e-11)
                assert delta*N <= 73728 and delta*q*(q+1) <= N
    rng = np.random.default_rng(641)
    for N in range(1, 8):
        basis = np.linalg.qr(rng.normal(size=(N, N))+1j*rng.normal(size=(N, N)))[0]
        for k in range(N+1):
            amplitudes = {tuple(rows): np.linalg.det(basis[list(rows), :k])
                          for rows in combinations(range(N), k)}
            close(sum(abs(z)**2 for z in amplitudes.values()), 1)
            cut = N//2
            left = [X for size in range(cut+1) for X in combinations(range(cut), size)]
            right = [Y for size in range(N-cut+1) for Y in combinations(range(cut, N), size)]
            state = np.array([[amplitudes.get(X+Y, 0) for Y in right] for X in left])
            reduced = state @ state.conj().T
            close(np.trace(reduced), 1)
            for i, j in product(range(len(left)), repeat=2):
                close(reduced[i, j], sum(amplitudes.get(left[i]+Y, 0) *
                                         np.conj(amplitudes.get(left[j]+Y, 0))
                                         for Y in right))


def check_slater_and_localization():
    def slater_matrix(V, left, right, *, grouped=False):
        left_sets = [p for d in range(len(left)+1) for p in combinations(left, d)]
        right_sets = [p for d in range(len(right)+1) for p in combinations(right, d)]
        out = np.zeros((len(left_sets), len(right_sets)), dtype=complex)
        for row, x in enumerate(left_sets):
            for col, y in enumerate(right_sets):
                if len(x+y) == V.shape[1]:
                    sites = x+y if grouped else sorted(x+y)
                    out[row, col] = np.linalg.det(V[list(sites), :])
        return out

    rng = np.random.default_rng(721)
    size = 7
    basis = np.linalg.qr(rng.normal(size=(size, size)))[0]
    for left in [(0, 1, 2), (2, 3, 4), (0, 1, 5, 6), (0, 2, 4, 6)]:
        right = tuple(i for i in range(size) if i not in left)
        for filling in [0, 1, 2, 3, size-1, size]:
            V = basis[:, :filling]
            left_V = V[list(left)]
            params = np.linalg.eigvalsh(left_V @ left_V.T)
            expected = sorted(np.prod([p if occupied else 1-p
                                       for p, occupied in zip(params, bits)])
                              for bits in product((0, 1), repeat=len(params)))
            grouped_amplitude = slater_matrix(V, left, right, grouped=True)
            close(np.linalg.eigvalsh(grouped_amplitude @ grouped_amplitude.conj().T), expected)
            if left != (0, 2, 4, 6) or min(filling, size-filling) <= 1:
                ordered_amplitude = slater_matrix(V, left, right)
                close(np.linalg.eigvalsh(ordered_amplitude @ ordered_amplitude.conj().T), expected)
    for copies in range(1, 4):
        # Each consecutive four-site block contributes one interlaced pair.
        size = 4*copies
        V = np.zeros((size, 2*copies))
        localized = V.copy()
        for j in range(copies):
            V[[4*j, 4*j+2], 2*j] = 1/np.sqrt(2)
            V[[4*j+1, 4*j+3], 2*j+1] = 1/np.sqrt(2)
            localized[[4*j, 4*j+2], 2*j] = 1/np.sqrt(2)
            localized[4*j+3, 2*j+1] = 1
        left = tuple(range(0, size, 2))
        right = tuple(range(1, size, 2))
        close(V[list(left)], localized[list(left)])
        amplitude = slater_matrix(V, left, right)
        product_amplitude = slater_matrix(localized, left, right)
        spectrum = np.linalg.eigvalsh(amplitude @ amplitude.conj().T)
        product_spectrum = np.linalg.eigvalsh(product_amplitude @ product_amplitude.conj().T)
        close(spectrum[-2**copies:], 2.**(-copies))
        close(spectrum[:-2**copies], 0)
        close(product_spectrum[-1], 1)
        close(entropy(spectrum), copies*np.log(2))

    dim = 43
    matrix = (np.diag(rng.normal(size=dim))
              + np.diag(rng.normal(size=dim-1), 1)
              + np.diag(rng.normal(size=dim-1), -1))
    vector = rng.normal(size=dim)
    sub_sup = np.max(abs(np.diag(matrix, 1))) + np.max(abs(np.diag(matrix, -1)))
    for length in range(1, 15):
        weight = np.sin(np.pi*np.arange(1, length+1)/(length+1))
        scale = 2*np.sin(np.pi/(2*(length+1)))
        close(np.linalg.norm(np.diff(np.r_[0, weight, 0])), scale*np.linalg.norm(weight))
        window_array = np.zeros((dim+length-1, dim))
        for row, start in enumerate(range(1-length, dim)):
            for offset, value in enumerate(weight):
                if 0 <= start+offset < dim:
                    window_array[row, start+offset] = value
        localized = window_array*vector
        close(np.linalg.norm(localized), np.linalg.norm(weight)*np.linalg.norm(vector))
        assert np.linalg.norm(localized @ matrix.T) <= np.linalg.norm(weight)*(
            np.linalg.norm(matrix @ vector)+scale*sub_sup*np.linalg.norm(vector))


if __name__ == "__main__":
    checks = [
        check_circle, check_digits_parry, check_windows, check_completions,
        check_euler, check_lattices, check_cuts, check_parent_chain,
        check_slater_and_localization,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print(f"All {len(checks)} check groups passed.")
