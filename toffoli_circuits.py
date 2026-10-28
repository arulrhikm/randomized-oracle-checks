"""Build and verify the randomized Toffoli branch with borrowed idle qubits (stdlib only).

The branch of the Toffoli branch construction (Appendix B) spends all of its T gates on one ell-controlled NOT.  Here that
gate is Maslov's V-chain [Maslov 2016, Prop. 5], in the preliminary form with m - 2 borrowed qubits and
the refined form with ceil((m - 2)/2), the one the branch uses:
a special-form Toffoli on the target, a relative-phase Toffoli in the middle and 2-T
relative-phase pieces along the chain, for 8m - 8 T gates in total.  The borrowed qubits are idle
input coordinates of the branch, in an arbitrary (possibly entangled) state, returned exactly.

Verification is exact.  Amplitudes are kept in Z[w], w = exp(i pi/4), with one global 1/sqrt(2)
power, so no floating point is involved.  A circuit passes if every computational-basis input is
sent to the intended basis state with one common phase; that is equality of unitaries up to a
global phase, for every state of the borrowed qubits.

Run:  python3 -I toffoli_circuits.py
"""
import itertools
import random

# ---------------------------------------------------------------- gates and exact simulation
# A gate is a tuple: ('x', q) ('h', q) ('t', q) ('tdg', q) ('cx', c, t).
# Z[w] element: 4 integer coefficients of 1, w, w^2, w^3 with w^4 = -1.


def _wmul(a, k):
    """Multiply a Z[w] element by w^k."""
    k %= 8
    out = list(a)
    for _ in range(k):
        out = [-out[3], out[0], out[1], out[2]]
    return tuple(out)


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _neg(a):
    return tuple(-x for x in a)


ONE = (1, 0, 0, 0)


def run(gates, basis):
    """Apply gates to basis state |basis> (an int). Returns (state dict, number of H)."""
    state = {basis: ONE}
    nh = 0
    for g in gates:
        op = g[0]
        if op == 'x':
            m = 1 << g[1]
            state = {b ^ m: a for b, a in state.items()}
        elif op == 'cx':
            c, t = 1 << g[1], 1 << g[2]
            state = {(b ^ t) if b & c else b: a for b, a in state.items()}
        elif op in ('t', 'tdg'):
            m, k = 1 << g[1], (1 if op == 't' else -1)
            state = {b: (_wmul(a, k) if b & m else a) for b, a in state.items()}
        elif op == 'h':
            m = 1 << g[1]
            new = {}
            for b, a in state.items():
                b0 = b & ~m
                for tgt, sign in ((b0, 1), (b0 | m, -1 if b & m else 1)):
                    v = a if sign == 1 else _neg(a)
                    new[tgt] = _add(new.get(tgt, (0, 0, 0, 0)), v)
            state = {b: a for b, a in new.items() if any(a)}
            nh += 1
        else:
            raise ValueError(op)
    return state, nh


def dagger(gates):
    inv = {'t': 'tdg', 'tdg': 't'}
    return [(inv.get(g[0], g[0]),) + g[1:] for g in reversed(gates)]


def tcount(gates):
    return sum(1 for g in gates if g[0] in ('t', 'tdg'))


def phase_poly(terms):
    """Diagonal gate prod_j w^{k_j * parity(S_j)} built from CNOT ladders.
    terms: list of (k, [qubits]) with k in {+1,-1}; the parity lands on the last qubit."""
    out = []
    for k, qs in terms:
        tgt = qs[-1]
        lad = [('cx', q, tgt) for q in qs[:-1]]
        out += lad + [('t' if k == 1 else 'tdg', tgt)] + list(reversed(lad))
    return out


# ---------------------------------------------------------------- Toffoli gadgets
def toffoli(a, b, c):
    """Exact Toffoli, 7 T: H_c CCZ H_c with CCZ = w^{a+b+c-(a^b)-(a^c)-(b^c)+(a^b^c)}."""
    return [('h', c)] + phase_poly([(1, [a]), (1, [b]), (1, [c]), (-1, [a, b]), (-1, [a, c]),
                                    (-1, [b, c]), (1, [a, b, c])]) + [('h', c)]


def rtl(a, b, c):
    """Relative-phase Toffoli, 4 T (Margolus form): c ^= a&b up to a diagonal phase on (a,b)."""
    return [('h', c), ('t', c), ('cx', b, c), ('tdg', c), ('cx', a, c), ('t', c), ('cx', b, c),
            ('tdg', c), ('h', c)]


def rts(a, b, c):
    """First half of rtl, 2 T: rtl with the trailing V(b,c) removed [Maslov, Prop. 5]."""
    return rtl(a, b, c)[:5]


def srts(a, b, c):
    """Special-form Toffoli, 4 T: Toffoli(a,b;c) = srts(a,b,c) . V(a,c), where V(a,c) =
    H_c w^{a+c-(a^c)} H_c carries the three T gates that touch only a and c."""
    return [('h', c)] + phase_poly([(1, [b]), (-1, [a, b]), (-1, [b, c]), (1, [a, b, c])]) + [('h', c)]


def mct_borrowed(ctrl, tgt, anc):
    """C^mX on controls ctrl -> tgt, borrowing len(anc) >= m - 2 qubits in any state.
    Maslov 2016 Prop. 5 (preliminary form with m - 2 borrowed qubits); T-count 8m - 8 for m >= 3."""
    m = len(ctrl)
    if m == 0:
        return [('x', tgt)]
    if m == 1:
        return [('cx', ctrl[0], tgt)]
    if m == 2:
        return toffoli(ctrl[0], ctrl[1], tgt)
    assert len(anc) >= m - 2, "needs m-2 borrowed qubits"
    c = lambda i: ctrl[i - 1]          # c(1..m)
    a = lambda i: anc[i - 1]           # a(1..m-2)
    item2 = []
    for k in range(1, m - 2):          # k = 1..m-3
        item2 += rts(a(m - 2 - k), c(m - k), a(m - 1 - k))
    item4 = []
    for k in range(1, m - 2):
        item4 += dagger(rts(a(k), c(k + 2), a(k + 1)))
    g = []
    g += srts(c(m), a(m - 2), tgt)                       # 1
    g += item2                                           # 2
    g += rtl(c(1), c(2), a(1))                           # 3
    g += item4                                           # 4
    g += dagger(srts(c(m), a(m - 2), tgt))               # 5
    g += item2                                           # 6
    g += dagger(rtl(c(1), c(2), a(1)))                   # 7
    g += item4                                           # 8
    return g


def rt4l(x, y, z, t):
    """Relative-phase Toffoli-4, 8 T, 6 CNOT, 4 H: t ^= x&y&z up to a diagonal phase on (x,y,z)."""
    return [('h', t), ('t', t), ('cx', z, t), ('tdg', t), ('h', t),
            ('cx', x, t), ('t', t), ('cx', y, t), ('tdg', t), ('cx', x, t), ('t', t), ('cx', y, t), ('tdg', t),
            ('h', t), ('t', t), ('cx', z, t), ('tdg', t), ('h', t)]


def rt4s(x, y, z, t):
    """First ten gates of rt4l, 4 T: rt4l with the trailing V(y,z,t) removed [Maslov, Prop. 5]."""
    return rt4l(x, y, z, t)[:10]


def mct_borrowed_refined(ctrl, tgt, anc):
    """C^mX borrowing only ceil((m-2)/2) qubits in any state, T-count 8m - 8 for m >= 3.
    Maslov 2016 Prop. 5, refined form: the middle Toffoli absorbs three controls (rt4l) and each
    chain step absorbs two (rt4s), with at most one single-control step (rts)."""
    m = len(ctrl)
    if m <= 3:
        return mct_borrowed(ctrl, tgt, anc)
    p = -(-(m - 2) // 2)
    assert len(anc) >= p, "needs ceil((m-2)/2) borrowed qubits"
    a = anc[:p]
    rest = ctrl[3:m - 1]                       # controls between the middle gate and c_m
    groups = [rest[i:i + 2] for i in range(0, len(rest), 2)]
    assert len(groups) == p - 1
    steps = []                                  # step j: a[j+1] ^= a[j] & group_j
    for j, gr in enumerate(groups):
        if len(gr) == 2:
            steps.append((rt4s(a[j], gr[0], gr[1], a[j + 1])))
        else:
            steps.append((rts(a[j], gr[0], a[j + 1])))
    desc = [g for s in reversed(steps) for g in s]
    asc = [g for s in steps for g in dagger(s)]
    mid = rt4l(ctrl[0], ctrl[1], ctrl[2], a[0])
    top = srts(ctrl[m - 1], a[p - 1], tgt)
    return top + desc + mid + asc + dagger(top) + desc + dagger(mid) + asc


def verify_mct_refined(m):
    ctrl = list(range(m)); tgt = m
    p = max(-(-(m - 2) // 2), 0)
    anc = list(range(m + 1, m + 1 + p))
    nq = m + 1 + p
    gates = mct_borrowed_refined(ctrl, tgt, anc)

    def f(b):
        return b ^ (1 << tgt) if all((b >> q) & 1 for q in ctrl) else b
    ok, msg = check_permutation(gates, nq, f)
    return ok, msg, tcount(gates), nq, p


# ---------------------------------------------------------------- verification helpers
def check_permutation(gates, nqubits, target_map):
    """True iff gates == (basis permutation target_map) up to one global phase."""
    ref = None
    for b in range(1 << nqubits):
        st, _ = run(gates, b)
        if len(st) != 1:
            return False, f"input {b}: output not a basis state"
        (out, amp), = st.items()
        if out != target_map(b):
            return False, f"input {b}: wrong output {out} != {target_map(b)}"
        if ref is None:
            ref = amp
        elif amp != ref:
            return False, f"input {b}: relative phase differs"
    return True, "ok"


def verify_mct(m):
    ctrl = list(range(m))
    tgt = m
    anc = list(range(m + 1, m + 1 + max(m - 2, 0)))
    nq = m + 1 + len(anc)
    gates = mct_borrowed(ctrl, tgt, anc)

    def f(b):
        allc = all((b >> q) & 1 for q in ctrl)
        return b ^ (1 << tgt) if allc else b
    ok, msg = check_permutation(gates, nq, f)
    return ok, msg, tcount(gates), nq


# ---------------------------------------------------------------- the randomized branch
def gf2_rank(rows):
    rows = list(rows); r = 0
    for bit in reversed(range(max((x.bit_length() for x in rows), default=0))):
        piv = next((i for i in range(r, len(rows)) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] >> bit & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def complete_basis(rows, d):
    """Extend independent rows (ints over F_2^d) to a basis; returns the full list."""
    basis = list(rows)
    for i in range(d):
        if gf2_rank(basis + [1 << i]) > len(basis):
            basis.append(1 << i)
    return basis


def linear_map_circuit(A, qubits):
    """CNOT circuit mapping register y (on qubits) to A y in place, A invertible over F_2.
    Row i of A is an int whose bit j is A[i][j].  Gaussian elimination: build the circuit for A^{-1}
    by reducing A to I with row operations, then reverse."""
    d = len(A)
    M = list(A)
    ops = []                     # row ops r_i ^= r_j, recorded as (i, j)
    for col in range(d):
        piv = next(i for i in range(col, d) if M[i] >> col & 1)
        if piv != col:           # swap via three xors
            for (i, j) in ((col, piv), (piv, col), (col, piv)):
                M[i] ^= M[j]; ops.append((i, j))
        for i in range(d):
            if i != col and M[i] >> col & 1:
                M[i] ^= M[col]; ops.append((i, col))
    # ops turn A into I: E_k...E_1 A = I, so A = E_1...E_k (each E self-inverse).
    # Acting on the state y -> A y: apply E_k first ... E_1 last.  Row op r_i ^= r_j on the
    # matrix corresponds to y_i ^= y_j on the register.
    return [('cx', qubits[j], qubits[i]) for (i, j) in reversed(ops)]


def toffoli_branch(n, rows, borrow=True):
    """Branch of the Toffoli branch construction (Appendix B) for Toff_n (n-1 controls on qubits 0..n-2, target n-1).
    rows: ell linearly independent vectors a_j in F_2^{n-1} (ints).  Implements U_g with
    g(x) = 1 iff a_j . (x xor 1) = 0 for all j.  Returns (gates, nqubits, clean_ancillae)."""
    d, ell = n - 1, len(rows)
    data = list(range(d)); tgt = d
    A = complete_basis(rows, d)
    lin = linear_map_circuit(A, data)
    idle = data[ell:]
    if ell == 0:
        core, extra = [('x', tgt)], []
    elif len(idle) >= max(-(-(ell - 2) // 2), 0):
        core, extra = mct_borrowed_refined(data[:ell], tgt, idle), []
    else:
        raise ValueError("too few idle qubits for the borrowed-qubit gate")
    neg = [('x', q) for q in data[:ell]]
    xs = [('x', q) for q in data]
    gates = xs + lin + neg + core + neg + dagger(lin) + xs
    return gates, n, 0


def verify_branch(n, ell, rng):
    d = n - 1
    while True:
        rows = [rng.randrange(1, 1 << d) for _ in range(ell)]
        if gf2_rank(rows) == ell:
            break
    gates, nq, _ = toffoli_branch(n, rows)

    def f(b):
        x = b & ((1 << d) - 1)
        y = x ^ ((1 << d) - 1)
        fire = all(bin(r & y).count('1') % 2 == 0 for r in rows)
        return b ^ (1 << d) if fire else b
    ok, msg = check_permutation(gates, nq, f)
    return ok, msg, tcount(gates)


# ---------------------------------------------------------------- determinants
def gate_det_exponent(g, n):
    """det of a k-qubit gate g acting on n qubits, as w^e with w = exp(i pi/4); returns e mod 8.
    det(G (x) I) = det(G)^(2^(n-k)), with det T = w, det T^dag = w^-1, det H = det X = -1 = w^4,
    and det CNOT = -1 (one transposition, |10> <-> |11>)."""
    k = 2 if g[0] == 'cx' else 1
    base = {'t': 1, 'tdg': -1, 'h': 4, 'x': 4, 'cx': 4}[g[0]]
    return (base * pow(2, n - k, 8)) % 8


def det_exponent(gates, n):
    return sum(gate_det_exponent(g, n) for g in gates) % 8


def perm_sign(perm, size):
    seen, sign = [False] * size, 1
    for i in range(size):
        if not seen[i]:
            j, length = i, 0
            while not seen[j]:
                seen[j] = True; j = perm(j); length += 1
            if length % 2 == 0:
                sign = -sign
    return sign


def determinant_checks():
    """(i) every Clifford+T gate on n >= 4 qubits has determinant 1, so every circuit does;
    (ii) Toff_n is a permutation with one transposition, determinant -1;
    (iii) each built branch is c * P with sign(P) = +1 when ell <= n-2."""
    for n in (4, 5, 8):
        assert all(gate_det_exponent(g, n) == 0 for g in [('t', 0), ('tdg', 0), ('h', 0), ('x', 0), ('cx', 0, 1)])
    for n in (4, 6, 9):
        d = n - 1
        assert perm_sign(lambda b: b ^ (1 << d) if (b & ((1 << d) - 1)) == (1 << d) - 1 else b, 1 << n) == -1
    rng = random.Random(11)
    out = []
    for n, ell in [(5, 2), (6, 3), (7, 3), (8, 4), (9, 5), (10, 6)]:
        d = n - 1
        while True:
            rows = [rng.randrange(1, 1 << d) for _ in range(ell)]
            if gf2_rank(rows) == ell:
                break
        gates, nq, _ = toffoli_branch(n, rows)

        def f(b):
            y = (b & ((1 << d) - 1)) ^ ((1 << d) - 1)
            fire = all(bin(r & y).count('1') % 2 == 0 for r in rows)
            return b ^ (1 << d) if fire else b
        sgn = perm_sign(f, 1 << n)
        ones = sum(1 for x in range(1 << d) if f(x) != x)
        out.append((n, ell, ones, sgn, det_exponent(gates, n)))
        assert sgn == 1 and ones == 2 ** (d - ell) and det_exponent(gates, n) == 0
    return out


# ---------------------------------------------------------------- resource counts
def ell_of(eps):
    import math
    return math.ceil(math.log2(4 / eps))


def branch_tcount(n, eps):
    """T-count of the branch built by toffoli_branch (by construction, counting gates)."""
    ell = ell_of(eps)
    if ell > n - 1:
        return None                     # sampling does not apply; implement Toff_n exactly
    d = n - 1
    rows = [1 << i for i in range(ell)]  # T-count does not depend on the sampled rows
    if d - ell < max(-(-(ell - 2) // 2), 0):
        return None
    gates, _, _ = toffoli_branch(n, rows)
    return tcount(gates)


def main():
    print("== C^mX with m-2 borrowed qubits (Maslov Prop. 5), exact check over all inputs")
    for m in range(1, 7):
        ok, msg, t, nq = verify_mct(m)
        expect = {1: 0, 2: 7}.get(m, 8 * m - 8)
        print(f"m={m}: qubits={nq} T={t} (expected {expect}) exact={ok} {'' if ok else msg}")
        assert ok and t == expect
    print("\n== refined form: C^mX with ceil((m-2)/2) borrowed qubits, exact check over all inputs")
    for m in range(3, 9):
        ok, msg, t, nq, p = verify_mct_refined(m)
        print(f"m={m}: qubits={nq} borrowed={p} T={t} (expected {8 * m - 8}) exact={ok} {'' if ok else msg}")
        assert ok and t == 8 * m - 8
    print("\n== full randomized branches (refined gate), random sampled rows, exact check over all inputs")
    rng = random.Random(2026)
    for n, ell in [(5, 2), (6, 3), (7, 3), (8, 4), (9, 4), (9, 5), (10, 6), (11, 7)]:
        for trial in range(2):
            ok, msg, t = verify_branch(n, ell, rng)
            print(f"n={n} ell={ell} trial={trial}: T={t} exact={ok} {'' if ok else msg}")
            assert ok
    print("\n== determinants: Clifford+T gates on n >= 4 qubits have det 1; Toff_n has det -1")
    for n, ell, ones, sgn, e in determinant_checks():
        print(f"n={n} ell={ell}: branch flips the target on {ones} = 2^(n-1-ell) inputs, permutation sign {sgn:+d}, circuit det exponent {e} (det = w^{e})")
    print("\n== T-count of the borrowed-qubit branch (counted from the built circuit)")
    for n in (10, 50, 100):
        row = []
        for eps in (1e-3, 1e-6, 1e-10):
            row.append(branch_tcount(n, eps))
        print(f"n={n}: ell={[ell_of(e) for e in (1e-3, 1e-6, 1e-10)]} T={row}  (None: condition n >= ceil(3 ell / 2) fails)")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
