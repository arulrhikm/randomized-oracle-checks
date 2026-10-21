"""Brute-force the Yao bound of the random-set theorem (Appendix C) for tiny n.

For E subset F_2^n and mu_E = 1/2 unif(E) + 1/2 unif(E^c), the least mu_E-error
of a deterministic non-adaptive parity tree whose queries span a subspace
W = V^perp of dimension q is  err(V) = 1/2 sum_C min(|C cap E|/|E|, |C minus E|/|E^c|)
(cosets C of V).  If min_V err(V) > eps then RPDT_eps(1_E) >= q+1 (Yao).
We also report eta(q) = max_C | |C cap E|/(p|C|) - 1 |  with p = |E|/2^n.

Families: (i) random E (each point w.p. 2^-k);  (ii) the explicit
Maiorana-McFarland set E = {(y,z) in F_{2^m}^2 : pi(yz) = 0}, pi = first k bits;
(iii) an affine subspace of codimension k (structured; err drops to 0 at q=k).
"""
import random
import sys

PAR = None


def parity_table(n):
    return [bin(x).count("1") & 1 for x in range(1 << n)]


def subspaces(n, q):
    """Yield bases (lists of ints) of all q-dim subspaces of F_2^n, each once (RREF)."""
    def rec(start, rows_left, pivots):
        if rows_left == 0:
            yield from fill(sorted(pivots, reverse=True))
            return
        for p in range(start, -1, -1):
            yield from rec(p - 1, rows_left - 1, pivots + [p])

    def fill(piv):
        pivset = set(piv)
        free = []
        for p in piv:
            free.append([b for b in range(p) if b not in pivset])
        def build(i, acc):
            if i == len(piv):
                yield list(acc)
                return
            fb = free[i]
            for mask in range(1 << len(fb)):
                v = 1 << piv[i]
                for j, b in enumerate(fb):
                    if mask >> j & 1:
                        v |= 1 << b
                acc.append(v)
                yield from build(i + 1, acc)
                acc.pop()
        yield from build(0, [])

    yield from rec(n - 1, q, [])


def analyse(n, E, qmax):
    N = 1 << n
    inE = [0] * N
    for x in E:
        inE[x] = 1
    nE = len(E)
    nC = N - nE
    p = nE / N
    out = []
    for q in range(0, qmax + 1):
        best = 1.0
        eta = 0.0
        count = 0
        for basis in subspaces(n, q):
            count += 1
            cE = [0] * (1 << q)
            cA = [0] * (1 << q)
            for x in range(N):
                lab = 0
                for i, r in enumerate(basis):
                    lab |= PAR[r & x] << i
                cA[lab] += 1
                cE[lab] += inE[x]
            err = 0.5 * sum(min(cE[c] / nE, (cA[c] - cE[c]) / nC) for c in range(1 << q))
            if err < best:
                best = err
            for c in range(1 << q):
                eta = max(eta, abs(cE[c] / (p * cA[c]) - 1))
        out.append((q, count, best, eta))
    return out


# GF(2^m) arithmetic
POLY = {2: 0b111, 3: 0b1011, 4: 0b10011}


def gf_mul(a, b, m):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> m & 1:
            a ^= POLY[m]
    return r


def mm_set(m, k):
    n = 2 * m
    E = []
    for y in range(1 << m):
        for z in range(1 << m):
            w = gf_mul(y, z, m)
            if w & ((1 << k) - 1) == 0:
                E.append(y | (z << m))
    return n, E


def mm_bias(m, k, qmax):
    """max over cosets of codim q of |sum_{x in C} (-1)^{lambda.pi(yz)}|/|C|, over lambda != 0."""
    n = 2 * m
    N = 1 << n
    res = []
    vals = []
    for x in range(N):
        y, z = x & ((1 << m) - 1), x >> m
        vals.append(gf_mul(y, z, m) & ((1 << k) - 1))
    for q in range(qmax + 1):
        worst = 0.0
        for basis in subspaces(n, q):
            for lam in range(1, 1 << k):
                s = [0] * (1 << q)
                cnt = [0] * (1 << q)
                for x in range(N):
                    lab = 0
                    for i, r in enumerate(basis):
                        lab |= PAR[r & x] << i
                    s[lab] += 1 - 2 * PAR[lam & vals[x]]
                    cnt[lab] += 1
                worst = max(worst, max(abs(s[c]) / cnt[c] for c in range(1 << q)))
        res.append((q, worst, 2.0 ** (q - n / 2)))
    return res


if __name__ == "__main__":
    random.seed(20261014)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    qmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    PAR = parity_table(n)
    print(f"n={n}: least mu_E-error over all deterministic trees with q queries (Yao value)")
    for k in [1, 2, 3]:
        for trial in range(2):
            E = [x for x in range(1 << n) if random.random() < 2.0 ** -k]
            print(f"random E, k={k}, |E|={len(E)} (trial {trial}):")
            for q, cnt, best, eta in analyse(n, E, qmax):
                print(f"   q={q} #subspaces={cnt:<7} min err={best:.4f}  eta={eta:.3f}")
    for k in [1, 2, 3]:
        E = [x for x in range(1 << n) if (x & ((1 << k) - 1)) == ((1 << k) - 1)]
        print(f"affine subspace, codim k={k}, |E|={len(E)}:")
        for q, cnt, best, eta in analyse(n, E, min(qmax, k)):
            print(f"   q={q} min err={best:.4f}")
    if n % 2 == 0 and n // 2 in POLY:
        m = n // 2
        for k in [1, 2]:
            _, E = mm_set(m, k)
            print(f"explicit MM set m={m}, k={k}, |E|={len(E)} (2^(n-k)={2**(n-k)}):")
            for q, cnt, best, eta in analyse(n, E, qmax):
                print(f"   q={q} min err={best:.4f}  eta={eta:.3f}  (bound eta<=2^(k+q-n/2)={2.0**(k+q-n/2):.3f})")
        print("bent-component bias check (max |coset average| vs 2^(q-n/2)):")
        for q, worst, bound in mm_bias(m, 2, min(qmax, 3)):
            print(f"   q={q} max bias={worst:.4f} bound={bound:.4f}  ok={worst <= bound + 1e-12}")
