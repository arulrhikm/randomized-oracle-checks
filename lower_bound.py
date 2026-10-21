"""Lower bounds for mixed implementations via non-adaptive randomized parity decision trees:
the OR proposition for Toff_n (a-d) and the distance-from-affine bound with its two-sided law (e, f).

Original description of (a)-(d):

GKZ (arXiv 2510.07223, Theorem 23) prove T^mixed_eps(U_f) >= RPDT^na_eps(f) - 1, even with ancillae in an
arbitrary advice state, and Toff_n is Clifford-equivalent to U_{OR_m} with m = n - 1.  For OR_m put
    r_q = (2^(m-q) - 1) / (2^m - 1),   0 <= q <= m,
the fraction of nonzero inputs in a subspace of codimension q.  The paper's the OR query proposition (Appendix C) states
    min{q : r_q <= eps/(1-eps)}  <=  RPDT_eps(OR_m)  <=  min{q : r_q <= eps}.
This script checks the claims numerically:
  (a) brute force for m <= 5: every subspace V of codimension q contains at least 2^(m-q) - 1 nonzero
      points (the Yao lower bound), and a uniformly random codimension-q subspace contains a fixed
      nonzero point with probability exactly r_q (the upper bound);
  (b) the two bounds differ by at most one for eps <= 1/2;
  (c) the closed form RPDT_eps(OR_m) >= min{m, log2((1-eps)/eps)} - 1;
  (d) the corollary: the ancilla-free branches of Theorem 1 use at most 8 L + 40 T gates,
      where L is any lower bound on T^mixed_eps(Toff_n) of the form min{n-1, log2((1-eps)/eps)} - 2,
      whenever n >= ceil(3 ell / 2).
Run:  python3 -I lower_bound.py
"""
import itertools
import math
from fractions import Fraction


def r(m, q):
    return Fraction(2 ** (m - q) - 1, 2 ** m - 1)


def q_low(m, eps):
    a = Fraction(eps) / (1 - Fraction(eps))
    return min(q for q in range(m + 1) if r(m, q) <= a)


def q_up(m, eps):
    return min(q for q in range(m + 1) if r(m, q) <= Fraction(eps))


def rank(rows):
    rows = list(rows); rk = 0
    for bit in reversed(range(max((x.bit_length() for x in rows), default=0))):
        piv = next((i for i in range(rk, len(rows)) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[rk], rows[piv] = rows[piv], rows[rk]
        for i in range(len(rows)):
            if i != rk and rows[i] >> bit & 1:
                rows[i] ^= rows[rk]
        rk += 1
    return rk


def subspaces_by_codim(m):
    """All subspaces of F_2^m, as frozensets of vectors, keyed by codimension (kernels of query sets)."""
    out = {}
    for q in range(m + 1):
        seen = set()
        for queries in itertools.combinations(range(1, 1 << m), q):
            if rank(queries) != q:
                continue
            V = frozenset(x for x in range(1 << m) if all(bin(s & x).count('1') % 2 == 0 for s in queries))
            seen.add(V)
        out[q] = seen
    return out


def brute_force(m):
    subs = subspaces_by_codim(m)
    for q, Vs in subs.items():
        assert all(len(V) == 2 ** (m - q) for V in Vs)
        # (Yao) the leaf containing 0 is V itself and holds 2^(m-q) - 1 nonzero inputs
        assert all(len(V) - 1 >= 2 ** (m - q) - 1 for V in Vs)
        # (upper bound) a uniform random codim-q subspace contains a fixed nonzero x with probability r_q
        for x in range(1, 1 << m):
            frac = Fraction(sum(1 for V in Vs if x in V), len(Vs))
            assert frac == r(m, q), (m, q, x, frac)
    return {q: len(Vs) for q, Vs in subs.items()}


def affine_functions(n):
    for u in range(1 << n):
        for c in (0, 1):
            yield u, c


def yao_error(e_vals, n, V):
    """Least mu-error of a deterministic tree that is constant on the cosets of V, for e with
    support E, where mu puts 1/2 uniformly on E and 1/2 uniformly on its complement."""
    E = [x for x in range(1 << n) if e_vals[x]]
    Ec = (1 << n) - len(E)
    seen, err = set(), Fraction(0)
    for x0 in range(1 << n):
        if x0 in seen:
            continue
        coset = [x0 ^ v for v in V]
        seen.update(coset)
        inside = sum(1 for x in coset if e_vals[x])
        cost1 = Fraction(len(coset) - inside, 2 * Ec)   # output 1: wrong on C minus E
        cost0 = Fraction(inside, 2 * len(E))            # output 0: wrong on C cap E
        err += min(cost1, cost0)
    return err


def check_distance_bound(n, functions):
    """(e) for each non-affine f and each affine a with delta = Pr[f != a] <= 1/2, every depth-q tree for
    e = f xor a has mu-error at least (2^-q - delta)/2; this is the Yao step of the distance-from-affine bound (Appendix C)."""
    subs = subspaces_by_codim(n)
    checked = 0
    for fv in functions:
        affs = [[(bin(u & x).count('1') + c) % 2 for x in range(1 << n)] for u, c in affine_functions(n)]
        if any(fv == a for a in affs):
            continue
        for a in affs:
            e = [fv[x] ^ a[x] for x in range(1 << n)]
            delta = Fraction(sum(e), 1 << n)
            if delta > Fraction(1, 2):
                continue
            for q, Vs in subs.items():
                bound = (Fraction(1, 2 ** q) - delta) / 2
                for V in Vs:
                    assert yao_error(e, n, sorted(V)) >= bound, (fv, a, q)
                checked += 1
    return checked


def code_membership_example(r, eps):
    """f = a xor 1_A with A of codimension r: lower and upper bounds of the two-sided distance corollary (Appendix C)."""
    delta = Fraction(1, 2 ** r)
    lower = min(r, math.log2(1 / float(eps)) - 1) - 3
    ell = math.ceil(math.log2(4 / float(eps)))
    m = min(r, ell)
    upper = 0 if m <= 1 else 7 if m == 2 else 8 * m - 8
    cap = 8 * math.log2(1 / max(float(delta), float(eps))) + 16
    return lower, upper, cap


def main():
    print("== (a) brute force over all subspaces")
    for m in range(1, 6):
        counts = brute_force(m)
        print(f"m={m}: subspaces by codimension {counts}; Yao and random-subspace claims hold")
    print("\n== (b), (c) bounds on RPDT_eps(OR_m)")
    eps_grid = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 10), Fraction(1, 1000), Fraction(1, 10 ** 6), Fraction(1, 10 ** 10)]
    worst_gap = 0
    for m in range(1, 65):
        for e in eps_grid:
            lo, up = q_low(m, e), q_up(m, e)
            assert lo <= up <= lo + 1, (m, e, lo, up)
            worst_gap = max(worst_gap, up - lo)
            closed = min(m, math.log2((1 - e) / e)) - 1
            assert lo >= closed - 1e-12, (m, e, lo, closed)
    for m, e in [(99, Fraction(1, 10 ** 6)), (99, Fraction(1, 10 ** 10)), (49, Fraction(1, 10 ** 10))]:
        print(f"m={m} eps={float(e):g}: {q_low(m, e)} <= RPDT <= {q_up(m, e)};"
              f"  T_mixed(Toff_{m + 1}) >= {q_low(m, e) - 1}")
    print(f"max upper - lower over the grid: {worst_gap}")
    print("\n== (d) ancilla-free upper bound vs the lower bound, n >= ceil(3 ell / 2)")
    for e in [Fraction(1, 10 ** 3), Fraction(1, 10 ** 6), Fraction(1, 10 ** 10), Fraction(1, 10 ** 15)]:
        ell = math.ceil(math.log2(4 / e))
        n = -(-3 * ell // 2)
        upper = 8 * ell - 8
        L = q_low(n - 1, e) - 1
        assert upper <= 8 * L + 40
        print(f"eps={float(e):g}: ell={ell}, n={n}: upper {upper}, lower {L}, ratio {upper / L:.2f}, 8L+40={8 * L + 40}")
    print("\n== (e) distance from affine: Yao step, every non-affine f on n=3, sample on n=4")
    import random
    n3 = [[(fv >> x) & 1 for x in range(8)] for fv in range(256)]
    print(f"n=3: {check_distance_bound(3, n3)} (f, a, q) triples checked over all trees")
    rng = random.Random(5)
    n4 = [[rng.randrange(2) for _ in range(16)] for _ in range(60)]
    n4 += [[1 if x in S else 0 for x in range(16)] for S in ({3}, {1, 6}, {0, 5, 10}, {2, 3, 7, 11})]
    print(f"n=4: {check_distance_bound(4, n4)} (f, a, q) triples checked over all trees")
    print("\n== (f) two-sided law for f = a xor 1_A, A of codimension r")
    for r in (8, 20, 36, 60):
        for eps in (1e-3, 1e-6, 1e-10):
            lo, up, cap = code_membership_example(r, eps)
            assert lo <= up <= cap and up <= 8 * lo + 48
            print(f"r={r:2d} eps={eps:g}: lower {lo:6.2f}  construction {up:4d}  cap 8log(1/max(2^-r,eps))+16 = {cap:6.1f}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
