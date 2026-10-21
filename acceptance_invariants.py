# Checks the acceptance-count lemma (Appendix C): the acceptance function q(x) = Pr[target reads 1] of a Clifford+T circuit
# U = C R(P_t)...R(P_1) on inputs |x>|0>|0^a> depends only on the invariants of (Q_0, P_t, ..., P_1),
# where Q_0 = C^dag Z_target C: the phases sigma_F, the matrix (w_F . v_G mod 2), the kernel of the
# X-parts and the Z-parts on the input qubits.  Every Hermitian Pauli Q_0 != +-I is C^dag Z C for
# some Clifford C, so ranging over all Q_0 covers all circuits.  Also prints the bounds of
# the random-set T-count theorem (Appendix C) at eps = 1/4.
import random, math, cmath, itertools
random.seed(2026)
C8, S8 = math.cos(math.pi / 8), math.sin(math.pi / 8)

def apply_pauli(P, psi, N):
    s, v, w = P  # P = s * i^{|v&w|} X(v) Z(w), s = +-1
    ph = s * (1j) ** (bin(v & w).count('1') % 4)
    out = [0j] * (1 << N)
    for z, a in enumerate(psi):
        if a:
            out[z ^ v] += ph * (-1) ** bin(w & z).count('1') * a
    return out

def acceptance(Q0, Ps, n, N):
    q = []
    for x in range(1 << n):
        psi = [0j] * (1 << N); psi[x] = 1  # qubits 0..n-1 inputs, n target, rest ancillae: basis index = x
        for P in Ps:  # Ps = [P_1, ..., P_t], applied P_1 first
            Ppsi = apply_pauli(P, psi, N)
            psi = [C8 * a - 1j * S8 * b for a, b in zip(psi, Ppsi)]
        Qpsi = apply_pauli(Q0, psi, N)
        e = sum((a.conjugate() * b) for a, b in zip(psi, Qpsi)).real
        q.append(round((1 - e) / 2, 9))
    return tuple(q)

def invariant(Q0, Ps, n):
    F = [Q0] + Ps
    sig = tuple((s * (1j) ** (bin(v & w).count('1') % 4)) for s, v, w in F)
    sig = tuple((round(z.real), round(z.imag)) for z in sig)
    B = tuple(bin(F[i][2] & F[j][1]).count('1') % 2 for i in range(len(F)) for j in range(len(F)) if i != j)
    ker = []
    for S in range(1 << len(F)):
        acc = 0
        for i in range(len(F)):
            if S >> i & 1: acc ^= F[i][1]
        if acc == 0: ker.append(S)
    mask = (1 << n) - 1
    Win = tuple(f[2] & mask for f in F)
    return (sig, B, tuple(ker), Win)

def rand_pauli(N, nonid=False):
    while True:
        P = (random.choice((1, -1)), random.getrandbits(N), random.getrandbits(N))
        if not nonid or P[1] or P[2]: return P

def check(n, t, amax, samples):
    seen, worlds = {}, 0
    for _ in range(samples):
        a = random.randint(0, amax); N = n + 1 + a
        Q0 = rand_pauli(N, nonid=True); Ps = [rand_pauli(N) for _ in range(t)]
        key = invariant(Q0, Ps, n); q = acceptance(Q0, Ps, n, N)
        if key in seen:
            worlds += 1
            assert seen[key] == q, (n, t, key)
        else:
            seen[key] = q
    return len(seen), worlds

for n, t, amax, samples in [(1, 1, 2, 3000), (2, 1, 1, 3000), (1, 2, 1, 3000), (2, 2, 1, 2000)]:
    distinct, collisions = check(n, t, amax, samples)
    bound = (t + 1) * (2 * t + n + 3)
    print(f"n={n} t={t} a<={amax}: {distinct} invariant classes, {collisions} repeated classes agree; "
          f"log2 classes {math.log2(distinct):.1f} <= (t+1)(2t+n+3) = {bound}")

def min_t(n, k, a):
    # least t with m (t+1) min{2t+n+3, 2(n+a)+3} >= 2^(n-k) - n, m = 8(n+1)  (a = None: any number of clean ancillae)
    m = 8 * (n + 1); rhs = 2 ** (n - k) - n
    f = lambda t: m * (t + 1) * (2 * t + n + 3 if a is None else min(2 * t + n + 3, 2 * (n + a) + 3))
    lo, hi = 0, 1
    while f(hi) < rhs: hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) >= rhs: hi = mid
        else: lo = mid + 1
    return lo

print("the random-set T-count theorem (Appendix C) at eps=1/4 (least t not excluded):")
for n, k in [(40, 4), (60, 10), (80, 20), (100, 30)]:
    print(f"n={n} k={k}: any clean ancillae t>={min_t(n, k, None)}, no ancilla t>={min_t(n, k, 0)}, "
          f"decision trees n-k-2log(n+1)-7 = {n - k - math.ceil(2 * math.log2(n + 1)) - 7}")
