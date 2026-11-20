# Checks for Section 10F (sec:m14); vendored from the m14 module script.
#!/usr/bin/env python3
"""Exact finite checks and numerical regression cases for section.tex."""

from collections import Counter
from fractions import Fraction
from math import gamma, gcd, lcm, log, prod, sqrt

import numpy as np
from numpy.testing import assert_allclose


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def fibonacci(length):
    values = [0, 1]
    while len(values) < length:
        values.append(values[-1] + values[-2])
    return values


def accelerated_eta(s, count=72):
    """Euler transform of the alternating sum, using ordinary differences."""
    differences = np.arange(1, count + 1, dtype=float) ** (-s)
    result = 0j
    for k in range(count):
        result += differences[0] * 2.0 ** (-k - 1)
        differences = differences[:-1] - differences[1:]
    return result


def xi(s):
    """Numerical Mellin integral of the pole-subtracted theta kernel."""
    t = np.linspace(1.0, 18.0, 34001)
    n = np.arange(1, 8)[:, None]
    f = np.exp(-np.pi * n**2 * t).sum(axis=0)
    integrand = f * (t ** (s / 2) + t ** ((1 - s) / 2)) / t
    integral = np.trapezoid(integrand, t)
    return (1 + s * (s - 1) * integral) / 2


def check_completion():
    assert_allclose([xi(0), xi(1)], [0.5, 0.5], atol=1e-15)
    for s in (0.6 + 2j, -4.1 + 0.5j, 0.5 + 14j, 3.0 - 1j):
        assert_allclose(xi(1 - s), xi(s), atol=1e-12)
        assert_allclose(xi(s.conjugate()), xi(s).conjugate(), atol=1e-12)
        assert_allclose(xi(1 - s.conjugate()), xi(s).conjugate(), atol=1e-12)
    require(abs(xi(0.5 + 14j).imag) < 1e-14, "xi is real on the critical line")

    even_values = (np.pi**2 / 6, np.pi**4 / 90, np.pi**6 / 945)
    negative_values = (-1 / 12, 1 / 120, -1 / 252)
    completed_values = (np.pi / 6, np.pi**2 / 15, 2 * np.pi**3 / 63)
    for k, (zeta_even, zeta_negative, expected) in enumerate(
        zip(even_values, negative_values, completed_values), start=1
    ):
        for s, zeta_value in ((2 * k, zeta_even), (1 - 2 * k, zeta_negative)):
            gamma_factor = np.pi ** (-s / 2) * gamma(s / 2)
            assert_allclose(s * (s - 1) * gamma_factor * zeta_value / 2, expected)
            assert_allclose(xi(s), expected, atol=6e-7)
        s = 2 * k
        assert_allclose(accelerated_eta(s) / (1 - 2.0 ** (1 - s)), zeta_even)
    assert_allclose(xi(1e-4) * 2 / (1e-4 - 1) /
                    (1e-4 * np.pi ** (-1e-4 / 2) * gamma(1e-4 / 2)),
                    -0.5, atol=1e-4)

    assert_allclose(accelerated_eta(1), np.log(2), atol=1e-14)
    require(0 * 42 != accelerated_eta(1), "the pointwise totalized product fails")
    assert_allclose((1 - 2.0 ** (-1e-7)) / 1e-7, np.log(2), atol=1e-8)
    n = np.arange(1, 10002)
    signs = np.where(n % 2 == 1, 1, -1)
    partial = np.cumsum(signs / np.sqrt(n))
    assert_allclose(partial[:4], [1, 0.292893, 0.870243, 0.370243], atol=5e-7)
    eta_half = accelerated_eta(0.5).real
    assert_allclose(eta_half, 0.604899, atol=5e-7)
    assert_allclose(eta_half / (1 - sqrt(2)), -1.460355, atol=5e-7)
    require(np.all(np.diff(partial[1::2]) > 0), "even partial sums increase")
    require(np.all(np.diff(partial[::2]) < 0), "odd partial sums decrease")
    require(partial[-2] < eta_half < partial[-1], "alternating bounds at 1/2")
    for s in (0.1, 0.5, 0.9, 1.1, 1.5, 2.0):
        eta = accelerated_eta(s).real
        require(eta >= 1 - 2.0 ** (-s) > 0, "positive eta")
        zeta = eta / (1 - 2.0 ** (1 - s))
        require((zeta > 0) == (s > 1), "real zeta sign")
    s = 2.5 + 1.25j
    series = (np.arange(1, 20001, dtype=float) ** (-s)).sum()
    assert_allclose(accelerated_eta(s), (1 - 2 ** (1 - s)) * series, atol=3e-7)
    sigma = 1.5
    u = np.linspace(-8, 8, 33)
    v = np.linspace(-4*np.pi, 4*np.pi, 201)
    assert_allclose(3 + 4*np.cos(v) + np.cos(2*v), 2*(1 + np.cos(v))**2,
                    atol=1e-14)
    zeta = lambda s: accelerated_eta(s) / (1 - 2**(1 - s))
    require(all(abs(zeta(sigma))**3 * abs(zeta(sigma + 1j*t))**4 *
                abs(zeta(sigma + 2j*t)) >= 1 for t in u),
            "boundary argument product at an absolutely convergent abscissa")


def check_orbits():
    for beta in (0.25, 0.5, 0.6, 0.75):
        s = complex(beta, 14)
        orbit = {s, s.conjugate(), 1 - s, 1 - s.conjugate()}
        require(len(orbit) == (2 if beta == 0.5 else 4), "orbit cardinality")
        if beta != 0.5:
            require(sum(z.real > 0.5 and z.imag > 0 for z in orbit) == 1,
                    "unique upper right orbit point")
        cayley = (s - 1) / s
        assert_allclose(abs(s - 1)**2 - abs(s)**2, 1 - 2 * s.real, atol=1e-12)
        assert_allclose(cayley * s / (s - 1), 1)
        require(np.isclose(np.log(abs(cayley)), 0) == (beta == 0.5),
                "Cayley unit circle")
    delta, ordinate = 0.25, 14

    def quartic(s):
        z = s - 0.5
        return ((z - delta)**2 + ordinate**2) * ((z + delta)**2 + ordinate**2)

    roots = [0.5 + sign * delta + sign_im * ordinate * 1j
             for sign in (-1, 1) for sign_im in (-1, 1)]
    require(len(set(roots)) == 4 and all(quartic(z) == 0 for z in roots),
            "off-line quartet for the symmetric polynomial")
    s = 0.3 + 2j
    assert_allclose(quartic(s), quartic(1 - s))
    assert_allclose(quartic(s.conjugate()), quartic(s).conjugate())


def primes_through(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for prime in range(2, int(sqrt(limit)) + 1):
        if sieve[prime]:
            sieve[prime * prime::prime] = False
    return np.flatnonzero(sieve)


def check_prime_powers():
    primes = primes_through(10**6)
    require(len(primes) == 78498, "prime count at 10^6")
    assert_allclose(np.searchsorted(primes, 10, side="right"), 4)
    small = np.zeros(11)
    psi_large = 0.0
    theta_large = 0.0
    for prime in map(int, primes):
        weight = np.log(prime)
        theta_large += weight
        power = prime
        while power <= 10**6:
            psi_large += weight
            if power <= 10:
                small[power] = weight
            power *= prime
    assert_allclose(psi_large / 10**6, 0.999587, atol=5e-7, rtol=0)
    require(0 <= psi_large - theta_large <= 10**3 * np.log(10**6),
            "prime-power tail")
    staircase = np.cumsum(small)
    assert_allclose(staircase[1:], [np.log(lcm(*range(1, n + 1)))
                                   for n in range(1, 11)], atol=1e-14)
    assert_allclose([staircase[10], (staircase[7] + staircase[8]) / 2, staircase[9]],
                    np.log([2520, 420 * sqrt(2), 2520]), atol=1e-14)
    assert_allclose(small[[4, 8, 9]], np.log([2, 2, 3]), atol=1e-14)
    require(np.all(staircase[2:] <= np.arange(2, 11) * np.log(np.arange(2, 11))),
            "elementary upper bound on log lcm")
    # Compare the absolutely convergent prime-power and Dirichlet readings.
    s = 3.5
    n = np.arange(1, 5001, dtype=float)
    von_mangoldt_sum = sum(np.log(p) / (p**s - 1) for p in map(int, primes[:5000]))
    assert_allclose(von_mangoldt_sum, np.sum(np.log(n) * n**(-s)) / np.sum(n**(-s)),
                    atol=1e-8)
    # Numerical scope of the unconditional PNT here is the defining finite sums.


def log_series_bounds(x):
    """Exact lower and upper bounds from the positive atanh series on [1, 2]."""
    u = (x - 1) / (x + 1)
    partial = 2 * sum((u ** (2 * k + 1) / (2 * k + 1) for k in range(12)),
                      Fraction(0))
    tail = 2 * u**25 / (25 * (1 - u*u))
    return partial, partial + tail


LOG_TWO = log_series_bounds(Fraction(2))
LOG_SCALE = 10**12


def log_bounds(x):
    x = Fraction(x)
    require(x >= 1, "log enclosures used only on [1, infinity)")
    scale = 0
    while x >= 2:
        x /= 2
        scale += 1
    lo, hi = log_series_bounds(x)
    lo += scale * LOG_TWO[0]
    hi += scale * LOG_TWO[1]
    lower = (lo * LOG_SCALE).__floor__()
    upper = (hi * LOG_SCALE).__ceil__()
    return Fraction(lower, LOG_SCALE), Fraction(upper, LOG_SCALE)


def loglog_bounds(n):
    lower, upper = log_bounds(n)
    return log_bounds(lower)[0], log_bounds(upper)[1]


def check_robin():
    gamma_multiple_lower = Fraction(89, 50)
    gamma_multiple_upper = Fraction(179, 100)
    harmonic = sum((Fraction(1, j) for j in range(1, 1025)), Fraction(0))
    k_log_lower, k_log_upper = log_bounds(1024)
    require(log_bounds(gamma_multiple_lower)[1] <
            harmonic - k_log_upper - Fraction(1, 1024), "lower bound on exp(gamma)")
    require(harmonic - k_log_lower < log_bounds(gamma_multiple_upper)[0],
            "upper bound on exp(gamma)")
    primes = list(map(int, primes_through(23)))
    require(primes == [2, 3, 5, 7, 11, 13, 17, 19, 23], "first nine primes")
    primorial = prod(primes)
    require(primorial == 223092870, "ninth primorial")

    values = [0] * 6000
    for divisor in range(1, len(values)):
        for multiple in range(divisor, len(values), divisor):
            values[multiple] += divisor
    require(5040 == 2**4 * 3**2 * 5 * 7, "threshold factorization")
    threshold_ratio = Fraction(values[5040], 5040)
    require(threshold_ratio == Fraction(403, 105), "divisor ratio at 5040")
    require(loglog_bounds(5040)[1] < Fraction(2144, 1000), "upper loglog bound")
    require(gamma_multiple_upper * Fraction(2144, 1000) < threshold_ratio,
            "failure at 5040")
    require(loglog_bounds(primorial)[0] > Fraction(29, 10), "lower loglog bound")
    require(prod(Fraction(p + 1, p) for p in primes) <
            gamma_multiple_lower * Fraction(29, 10), "success at the ninth primorial")

    # Independent finite reduction of Robin's inequality for at most eight primes.
    # Sort exponents onto the first k prime factors. For p < q, a >= b,
    # the factor S_p(a)/S_p(b) >= S_q(a)/S_q(b), where S_p(e)=sum(p**-j).
    for p, q in zip(primes, primes[1:]):
        for a in range(1, 7):
            for b in range(a + 1):
                p_short = sum(Fraction(1, p**j) for j in range(b + 1))
                q_short = sum(Fraction(1, q**j) for j in range(b + 1))
                p_long = sum(Fraction(1, p**j) for j in range(a + 1))
                q_long = sum(Fraction(1, q**j) for j in range(a + 1))
                require(p_long * q_short >= q_long * p_short, "exponent exchange")

    def lower_robin(n):
        return gamma_multiple_lower * loglog_bounds(n)[0]

    for n in range(5041, 6000):
        require(Fraction(values[n], n) < lower_robin(n), f"small Robin case {n}")
    prefix_maximum = max(Fraction(values[n], n) for n in range(1, 5041))
    require(prefix_maximum == threshold_ratio < lower_robin(6000),
            "compressed integer at or below 5040 for an input at or above 6000")

    cutoff = 10**12
    saturation = prod(Fraction(p, p - 1) for p in primes[:8])
    require(saturation < lower_robin(cutoff), "saturated eight-prime bound")

    def compressed_numbers(position, n, ratio, previous_exponent):
        yield n, ratio
        if position == 8:
            return
        p = primes[position]
        power = 1
        factor = Fraction(1)
        for exponent in range(1, previous_exponent + 1):
            power *= p
            if n * power >= cutoff:
                break
            factor += Fraction(1, power)
            yield from compressed_numbers(position + 1, n * power,
                                          ratio * factor, exponent)

    compressed_count = 0
    for n, ratio in compressed_numbers(0, 1, Fraction(1), cutoff.bit_length()):
        if n > 5040:
            require(ratio < lower_robin(n), f"compressed Robin case {n}")
            compressed_count += 1
    require(compressed_count == 4204, "all compressed cases")
    print(f"  Robin: 959 small cases and {compressed_count} compressed cases")

    delta = Fraction(1, 10)
    d = 14 + 10
    m0 = 100 * d**2 / delta**2
    m1 = 500 * d**4 / delta**2
    require((d, m0, m1, 4 * m0) == (24, 5760000, 16588800000, 23040000),
            "conditional scale arithmetic")
    log_b0 = 4 * (28 * m1 + 20)**2
    assert_allclose(float(log_b0), 8.62990e23, rtol=1e-6)
    require(4 * m0 < 28 * m1, "conditional log window order")
    beta = Fraction(1, 2) + delta
    require(beta**2 + 16 * beta == Fraction(33, 4) + 17 * delta + delta**2,
            "exponent expansion")
    require(4000 * m0 * d**2 < (17 * delta * m0)**2 / 2,
            "scalar domination in the cited peak proof")

    # Use an ordinary direct sum for U at 10 to check its normalization.
    powers = (2, 3, 4, 5, 7, 8, 9)
    bases = (2, 3, 2, 5, 7, 2, 3)
    moment = sum(np.log(p) / power for p, power in zip(bases, powers))
    left = 10 * moment - np.log(2520) - 10 * np.log(10) + 10 * (1 + np.euler_gamma)
    right = sum((10 / power - 1) * np.log(p) for p, power in zip(bases, powers))
    right += 10 * (1 + np.euler_gamma - np.log(10))
    assert_allclose(left, right, atol=1e-12)
    assert_allclose(moment, staircase_moment(10, primes), atol=1e-14)
    for t in (10000, 10001, 15000, 40000):
        check_mertens_test(t)


def staircase_moment(x, primes):
    """Finite partial summation using the exact staircase breakpoints."""
    jumps = sorted((p**k, np.log(p)) for p in primes
                   for k in range(1, x.bit_length()) if p**k <= x)
    level = 0.0
    integral = 0.0
    previous = 1
    for power, jump in jumps:
        integral += level * (1/previous - 1/power)
        level += jump
        previous = power
    integral += level * (1/previous - 1/x)
    return level/x + integral


def check_mertens_test(t):
    primes = primes_through(t)
    require(8 * 9 * 5 * 7 * 11 == 27720 > 5040, "lcm lower bound")
    n = 1
    psi = 0.0
    factors = []
    for p in map(int, primes):
        power = p
        exponent = 1
        while power*p <= t:
            power *= p
            exponent += 1
        n *= power
        psi += exponent * log(p)
        factors.append(p**(-exponent-1))
    assert_allclose(psi, log(n), rtol=1e-14)
    require(n % 27720 == 0 and psi > 1, "prime powers in the lcm")
    loss = -np.log1p(-np.asarray(factors)).sum()
    require(max(factors) < 1/t and loss <= 4/np.sqrt(t), "truncated Euler factor loss")
    tangent = np.log(np.log(t)) + (psi - t)/(t*np.log(t))
    require(np.log(np.log(psi)) <= tangent, "concavity at the prime power staircase")
    e3 = np.log1p(-1/primes).sum() + np.log(np.log(t)) + np.euler_gamma
    log_ratio = -loss - np.log1p(-1/primes).sum()
    require(log_ratio >= -4/np.sqrt(t) - e3 + np.log(np.log(t)) + np.euler_gamma,
            "Mertens test lower estimate")


def zeckendorf(n, fib):
    support = []
    for k in range(len(fib) - 1, 1, -1):
        if fib[k] <= n:
            support.append(k)
            n -= fib[k]
    require(n == 0, "Fibonacci table covers the integer")
    require(all(a - b >= 2 for a, b in zip(support, support[1:])),
            "greedy support is canonical")
    return support


def legal_words(h, entering=0):
    if h == 0:
        yield ""
        return
    for word in legal_words(h - 1, 0):
        yield "0" + word
    if entering == 0:
        for word in legal_words(h - 1, 1):
            yield "1" + word


def check_carries():
    fib = fibonacci(90)
    r = (sqrt(5) - 1) / 2
    assert_allclose(1 - r*r, r)
    for k in range(1, 26):
        assert_allclose(fib[k] * r, fib[k - 1] - (-r)**k)
    require(r**2 + r**4 > r**2, "support {2,4} refutes the symmetric bound")
    for m in range(4, 18):
        for q in range(25):
            support = list(range(m, m + 2*q + 1, 2))
            n = fib[m] + sum(fib[k] for k in support)
            require(zeckendorf(n, fib) == [m + 2*q + 1, m - 2],
                    "sharp support for alternating inputs")
            require(2*fib[m] == fib[m + 1] + fib[m - 2], "double-weight identity")
            require(m + 2*q + 1 - (m - 2) == 2*q + 3, "support gap")
            for radius in (1, 2, 7):
                rounds = (2*q + radius) // radius
                require((rounds - 1)*radius < 2*q + 1 <= rounds*radius,
                        "minimum radius propagation time")
    require(sum(fib[k] for k in (16, 14, 12, 10, 8)) == 1584 and fib[8] == 21,
            "specified carry input")
    require(1584 + 21 == 1605 == fib[17] + fib[6], "specified carry output")

    for m in range(4, 15):
        for bits in legal_words(18 - m):
            original = [m + k for k, bit in enumerate(bits) if bit == "1"]
            n = sum(fib[k] for k in original)
            old_error = sum((-r)**k for k in original)
            for j in range(m - 1, 21):
                canonical = zeckendorf(n + fib[j], fib)
                ell = canonical[-1]
                raw_error = old_error + (-r)**j
                error = sum((-r)**k for k in canonical)
                require(ell >= m - 2, "carry barrier")
                require(abs(raw_error) < r**(m - 2), "raw conjugate bound")
                require(-r*r < error < r, "canonical asymmetric enclosure")
                assert_allclose(error, raw_error, atol=1e-14, rtol=0)
                require(r**(ell + 1) < abs(error), "least-index lower bound")


BLOCK = "00010101001000"


def count_avoiders(max_length):
    # Keep the longest suffix that is a proper prefix of the forbidden block.
    suffixes = {BLOCK[:k] for k in range(len(BLOCK))}
    counts = Counter({(0, ""): 1})
    yield sum(counts.values())
    for _ in range(max_length):
        next_counts = Counter()
        for (previous, suffix), multiplicity in counts.items():
            for bit in (0, 1):
                if previous == bit == 1:
                    continue
                candidate = suffix + str(bit)
                if candidate == BLOCK:
                    continue
                next_suffix = max((p for p in suffixes if candidate.endswith(p)), key=len)
                next_counts[bit, next_suffix] += multiplicity
        counts = next_counts
        yield sum(counts.values())


def quadratic_product(left, right):
    """Multiply pairs representing a+b*phi, using phi**2=phi+1."""
    a, b = left
    c, d = right
    return a*c + b*d, a*d + b*c + b*d


def quadratic_nonnegative(value):
    a, b = value
    # Twice the value is (2*a+b) + b*sqrt(5). Square only opposite signs.
    rational = 2*a + b
    if b >= 0:
        return rational >= 0 or 5*b*b >= rational*rational
    return rational >= 0 and rational*rational >= 5*b*b


def check_avoidance():
    phi = (1 + sqrt(5)) / 2
    fib = fibonacci(205)
    require(len(BLOCK) == 14 and BLOCK[0] == BLOCK[-1] == "0", "block syntax")
    weight = (phi, 1)
    for entering in (0, 1):
        for h in range(15):
            words = list(legal_words(h, entering))
            assert_allclose(sum(weight[int(w[-1])] if w else weight[entering]
                                for w in words), phi**h * weight[entering])
        require(BLOCK in words, "forbidden block is legal from both entering states")
        remaining = sum(weight[int(w[-1])] for w in words if w != BLOCK)
        assert_allclose(remaining, phi**14 * weight[entering] - phi)
        require(remaining <= phi**14 * weight[entering] * (1 - phi**-14) + 1e-9,
                "conditional weighted bound")

    counts = list(count_avoiders(200))
    require((counts[0], counts[13], counts[14]) == (1, 610, 986),
            "short avoidance examples")
    for h in range(19):
        words = list(legal_words(h))
        require(len(words) == fib[h + 2], "unrestricted count")
        require(counts[h] == sum(BLOCK not in w for w in words), "suffix automaton count")

    # Evaluate phi**(r+1) * (phi**14-1)**q exactly without floating-point overflow.
    phi_powers = [(1, 0)]
    for _ in range(14):
        phi_powers.append(quadratic_product(phi_powers[-1], (0, 1)))
    a, b = phi_powers[14]
    block_factor = (a - 1, b)
    aligned_factor = (1, 0)
    for h, n in enumerate(counts):
        q, remainder = divmod(h, 14)
        if q and remainder == 0:
            aligned_factor = quadratic_product(aligned_factor, block_factor)
        a, b = quadratic_product(phi_powers[remainder + 1], aligned_factor)
        require(n > 0 and quadratic_nonnegative((a - n, b)), "uniform avoidance bound")
    capacity = np.log2(phi)
    capped_capacity = capacity + np.log2(1 - phi**-14) / 14
    require(capped_capacity < capacity, "strict entropy improvement")


def check_crt():
    a = np.diag([1, 2])
    zero = np.zeros((2, 2), dtype=np.int64)
    nilpotent = np.array([[0, 7], [0, 0]], dtype=np.int64)
    for j in range(9):
        require(np.trace(np.linalg.matrix_power(nilpotent, j)) ==
                np.trace(np.linalg.matrix_power(zero, j)) == (2 if j == 0 else 0),
                "equal power traces for different matrices")
    trace = np.trace(np.linalg.matrix_power(a, 1))
    require(trace == 3 and [y for y in range(-4, 5) if y % 11 == trace] == [3],
            "bounded trace recovery")
    require(all(y % 11 != 5 for y in range(-4, 5)), "a missing class")
    for bound in range(1, 21):
        values = range(-(bound - 1), bound)
        for modulus in range(1, 2*bound + 5):
            images = [y % modulus for y in values]
            require((len(set(images)) == len(images)) == (modulus >= 2*bound - 1),
                    "sharp integral modulus")
            if modulus <= 2*bound - 2:
                x = -(bound - 1)
                y = x + modulus
                require(y in values and x != y and x % modulus == y % modulus,
                        "sharpness witness")
    moduli = (3, 5, 7)
    require(all(gcd(p, q) == 1 for i, p in enumerate(moduli) for q in moduli[i+1:]),
            "coprime test moduli")
    modulus = prod(moduli)
    for x in (-52, -3, 0, 27, 52):
        residues = tuple(x % p for p in moduli)
        require([y for y in range(-(modulus//2), modulus//2 + 1)
                 if tuple(y % p for p in moduli) == residues] == [x],
                "agreement of separate and assembled residues")


def check_charge():
    for charge, angle in ((2, 0), (1, 0), (0, 0), (0, np.pi), (-2, 0.7)):
        z = lambda m: np.exp((charge + 1j*angle) / m)
        assert_allclose(np.log(abs(z(1))), charge, atol=1e-14)
        for m in range(1, 20):
            assert_allclose(m * np.log(abs(z(m))), charge, atol=1e-13)
            phase = z(m) / abs(z(m))
            assert_allclose(abs(phase), 1, atol=1e-14)
            assert_allclose(np.exp(charge/m) * phase, z(m), atol=1e-14)
            for n in (1, 3, 8):
                assert_allclose(z(m*n)**n, z(m), atol=1e-13)
                assert_allclose((z(m*n)/abs(z(m*n)))**n, phase, atol=1e-13)
    assert_allclose(np.exp(1j*np.pi), -1, atol=1e-14)
    for n in (1, 2, 5):
        require(0**n == 0 and abs(0) != 1, "zero-thread counterexample")


def main():
    for check in (check_completion, check_orbits, check_prime_powers, check_robin,
                  check_carries, check_avoidance, check_crt, check_charge):
        check()
        print(f"PASS {check.__name__}")
    print("PASS m14")


if __name__ == "__main__":
    main()
