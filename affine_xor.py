"""Explicit affine decompositions: U_f for f = XOR_j 1_{A_j}, A_j affine subspaces (stdlib only).

Each term 1_{A_j} is the indicator of {x : row_i . x = b_i for i = 1..r_j}.  A term with r_j <= ell
is implemented exactly: a CNOT basis change brings its r_j constraint values into r_j data
coordinates and one r_j-controlled NOT (negative controls) flips the target.  A term with
r_j > ell is replaced by the GKZ test: ell random combinations of its constraints, conditioned on
linear independence, so it fires on A_j always and outside A_j with probability at most 2^-ell.
With ell = ceil(log2(4J/eps)) the pointwise miss probability is at most J 2^-ell <= eps/4, so by
the pointwise-miss lemma (Appendix A) the mixed channel is within diamond distance eps of U_f.  All multi-controlled
NOTs borrow idle data qubits (toffoli_circuits.mct_borrowed_refined), so no clean ancilla and no
measurement are used when d >= ceil(3 min(r_j, ell) / 2) for every term.

This script (1) checks f = XOR of the term indicators, (2) computes the exact pointwise miss
probability max_x Pr[g(x) != f(x)] over the sampling distribution, (3) builds sampled branches and
verifies them exactly by simulation, and (4) reports T-counts, also for sizes too large to simulate.
It also checks the degree-criterion remark (Appendix B): (5) for every f on n = 3 and n = 4 bits, the coefficient of x_1...x_n in the
algebraic normal form equals |f^-1(1)| mod 2, which is the sign of det U_f, and (6) every sampled branch with
ell < d implements an oracle with an even number of accepting inputs, so det U_g = 1, whatever the parity of f.

Run:  python3 -I affine_xor.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toffoli_circuits import (check_permutation, complete_basis, dagger, gf2_rank,
                              linear_map_circuit, mct_borrowed_refined, tcount)


def par(v):
    return bin(v).count('1') & 1


def mct_t(m):
    """T-count of mct_borrowed_refined with m controls (verified in toffoli_circuits.py)."""
    return 0 if m <= 1 else 7 if m == 2 else 8 * m - 8


def ell_of(J, eps):
    return math.ceil(math.log2(4 * J / eps))


def term_circuit(cons, d, tgt):
    """Flip tgt iff row.x = b for all (row, b) in cons; rows independent over F_2^d.
    Borrows idle data coordinates for the multi-controlled NOT."""
    r = len(cons)
    rows = [row for row, _ in cons]
    assert gf2_rank(rows) == r
    data = list(range(d))
    lin = linear_map_circuit(complete_basis(rows, d), data)
    # after lin, coordinate i < r holds row_i . x; the term fires iff it equals b_i, i.e. we need
    # coordinate i XOR b_i XOR 1 = 1 on every control.
    flip = [('x', i) for i, (_, b) in enumerate(cons) if b == 0]
    core = mct_borrowed_refined(data[:r], tgt, data[r:])
    return lin + flip + core + flip + dagger(lin)


def sample_term(cons, ell, rng):
    """GKZ test for a term with r > ell constraints: ell independent random combinations."""
    r = len(cons)
    while True:
        S = [rng.randrange(1, 1 << r) for _ in range(ell)]
        if gf2_rank(S) == ell:
            break
    out = []
    for s in S:
        row = b = 0
        for i in range(r):
            if s >> i & 1:
                row ^= cons[i][0]; b ^= cons[i][1]
        out.append((row, b))
    return out


def miss_prob(r, ell):
    """Pr[all ell independent random combinations vanish on a fixed nonzero v in F_2^r]."""
    p = 1.0
    for i in range(ell):
        p *= (2 ** (r - 1) - 2 ** i) / (2 ** r - 2 ** i)
    return p


class Decomposition:
    def __init__(self, d, terms, name):
        self.d, self.terms, self.name = d, terms, name   # terms: list of constraint lists

    def indicator(self, j, x):
        return all(par(row & x) == b for row, b in self.terms[j])

    def f(self, x):
        return sum(self.indicator(j, x) for j in range(len(self.terms))) & 1

    def tcount(self, eps):
        ell = ell_of(len(self.terms), eps)
        return sum(mct_t(min(len(t), ell)) for t in self.terms), ell

    def borrow_ok(self, eps):
        ell = ell_of(len(self.terms), eps)
        return all(self.d - min(len(t), ell) >= max(-(-(min(len(t), ell) - 2) // 2), 0) for t in self.terms)

    def exact_miss(self, eps):
        """max_x Pr[g(x) != f(x)]: only sampled terms can err, and only outside their subspace."""
        J = len(self.terms); ell = ell_of(J, eps); worst = 0.0
        for x in range(1 << self.d):
            prod = 1.0
            for j, t in enumerate(self.terms):
                if len(t) > ell and not self.indicator(j, x):
                    prod *= 1 - 2 * miss_prob(len(t), ell)
            worst = max(worst, (1 - prod) / 2)
        return worst

    def branch(self, eps, rng):
        """One sampled branch: circuit and the Boolean function g it implements."""
        J = len(self.terms); ell = ell_of(J, eps); d = self.d
        used = [t if len(t) <= ell else sample_term(t, ell, rng) for t in self.terms]
        gates = []
        for t in used:
            gates += term_circuit(t, d, d)

        def g(x):
            return sum(all(par(row & x) == b for row, b in t) for t in used) & 1
        return gates, g


def spectral_norm(f, d):
    vals = [1 - 2 * 0] * 0
    F = [f(x) for x in range(1 << d)]
    # Walsh-Hadamard transform of the {0,1}-valued f, normalized: fhat(S) = 2^-d sum f(x) chi_S(x)
    a = [float(v) for v in F]
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            for j in range(i, i + h):
                a[j], a[j + h] = a[j] + a[j + h], a[j] - a[j + h]
        h *= 2
    return sum(abs(v) for v in a) / (1 << d)


# ---------------------------------------------------------------- example families
def marked_set(d, points):
    """f(x) = [x in points]: disjoint union of points, each a codimension-d subspace."""
    return Decomposition(d, [[(1 << i, p >> i & 1) for i in range(d)] for p in points],
                         f"marked set, d={d}, J={len(points)}")


def sparse_anf(d, monomials):
    """f = XOR of monomials prod_{i in S} x_i: each monomial is {x_i = 1, i in S}."""
    return Decomposition(d, [[(1 << i, 1) for i in S] for S in monomials],
                         f"sparse ANF, d={d}, J={len(monomials)}")


def random_pdt(d, depth, rng):
    """Random parity decision tree; accepting leaves give a disjoint affine decomposition."""
    leaves = []

    def grow(cons, k):
        if k == depth:
            if rng.random() < 0.5:
                leaves.append(list(cons))
            return
        rows = [r for r, _ in cons]
        while True:
            q = rng.randrange(1, 1 << d)
            if gf2_rank(rows + [q]) == len(rows) + 1:
                break
        for b in (0, 1):
            grow(cons + [(q, b)], k + 1)
    grow([], 0)
    return Decomposition(d, leaves, f"random parity decision tree, d={d}, depth={depth}, J={len(leaves)}")


def check(dec, eps, rng, branches=2):
    J = len(dec.terms); T, ell = dec.tcount(eps)
    q = dec.exact_miss(eps)
    L = spectral_norm(dec.f, dec.d)
    print(f"{dec.name}: eps={eps:g} ell={ell} codims={sorted(len(t) for t in dec.terms)}")
    print(f"  T={T}  clean ancillae=0 (borrowing ok: {dec.borrow_ok(eps)})  exact miss q={q:.3e} <= eps/4={eps/4:.3e}: {q <= eps/4}")
    print(f"  spectral norm L={L:.3f}; GKZ Fourier sampler would draw k=ceil(8L^2 ln(8/eps))={math.ceil(8*L*L*math.log(8/eps))} parities")
    assert q <= eps / 4 and dec.borrow_ok(eps)
    for b in range(branches):
        gates, g = dec.branch(eps, rng)
        assert tcount(gates) == T
        d = dec.d
        ok, msg = check_permutation(gates, d + 1, lambda s: s ^ (1 << d) if g(s & ((1 << d) - 1)) else s)
        weight = sum(g(x) for x in range(1 << d))
        print(f"  branch {b}: exact simulation over all {1 << (d + 1)} basis inputs: {ok} {'' if ok else msg};"
              f" |g^-1(1)| = {weight} (even)")
        assert ok
        if ell < d:
            assert weight % 2 == 0


def top_anf_coefficient(fv, n):
    """Coefficient of x_1...x_n in the ANF of f (Moebius transform over the subset lattice)."""
    a = list(fv)
    for i in range(n):
        for x in range(1 << n):
            if x >> i & 1:
                a[x] ^= a[x ^ (1 << i)]
    return a[(1 << n) - 1]


def check_degree_criterion():
    for n in (3, 4):
        full = 0
        for code in range(1 << (1 << n)):
            fv = [(code >> x) & 1 for x in range(1 << n)]
            top = top_anf_coefficient(fv, n)
            assert top == sum(fv) % 2
            full += top
        print(f"n={n}: all {1 << (1 << n)} functions: deg f = n exactly when |f^-1(1)| is odd ({full} of them)")


def main():
    rng = random.Random(7)
    print("== degree criterion of the degree-criterion remark (Appendix B): det U_f = (-1)^|f^-1(1)| = -1 exactly when deg f = n")
    check_degree_criterion()
    pts = [3, 77, 500]
    dec = marked_set(10, pts)
    print(f"marked set of 3 points on d=10: |f^-1(1)| = {sum(dec.f(x) for x in range(1 << 10))} (odd, deg f = 10)")
    print("== small instances: exact miss probability and exact branch simulation")
    pts = rng.sample(range(1 << 10), 3)
    check(marked_set(10, pts), 0.2, rng)           # large eps so that terms are sampled at d=10
    check(sparse_anf(10, [[0, 1, 2, 3, 4, 5, 6, 9], [2, 5, 7, 8], [1, 8]]), 0.25, rng)
    pdt = random_pdt(8, 4, rng)
    check(pdt, 1e-2, rng)
    print("\n== T-counts at scale (no simulation): marked-set oracle, d = 100, eps = 1e-6")
    print("   term-by-term exact alternatives need an extra qubit (a point leaves no idle input): KG24 Table 1 gives")
    print("   8d-12 per point with one clean ancilla and a measurement, 16d-32 with one external borrowed qubit")
    d = 100
    for J in (1, 4, 16, 64):
        dec = Decomposition(d, [[(1 << i, 0) for i in range(d)]] * J, "")
        T, ell = dec.tcount(1e-6)
        assert dec.borrow_ok(1e-6)
        print(f"  J={J}: ell={ell}  randomized T={T} (0 extra qubits, no measurement)"
              f"  vs exact J*(8d-12)={J * (8 * d - 12)} (1 clean + measurement) or J*(16d-32)={J * (16 * d - 32)} (1 borrowed)")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
