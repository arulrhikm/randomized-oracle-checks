"""T-counts for the n-qubit Toffoli gate Toff_n (m = n - 1 controls), for Table 1.

Every formula is quoted from its source; nothing here is fitted or estimated.
  KG24 = Khattar and Gidney, arXiv 2407.17966, Table 1 (Sec. 5.4: one dirty ancilla; Sec. 5.2: one clean ancilla)
  KASD26 = Kole, Assaad, Schnittka, Drechsler, arXiv 2605.18169, Eq. (8) and Table I
  KSD26 = Kole, Schnittka, Drechsler, arXiv 2605.18159, Eqs. (3)-(4) and Table I
  GKZ = Gosset, Kothari, Zhang, arXiv 2510.07223, Theorem 1 (ell = ceil(log2(1/eps)) + 2 parities)
Our branch is one ell-controlled NOT after a CNOT basis change (Theorem 1), costing 8 ell - 8
with ceil((ell - 2)/2) borrowed idle inputs when n >= ceil(3 ell / 2) (Maslov 2016, Prop. 5), with
ell = ceil(log2(4/eps)), the same ell as GKZ.  For ell >= n - 1 we implement Toff_n exactly instead.
Run: python3 toffoli_table.py   (prints the LaTeX body rows)
"""
import math

NS = (50, 100)
EPSS = (1e-3, 1e-6, 1e-10)


def ell(eps):
    return math.ceil(math.log2(4 / eps))


def kg_dirty(m):      # KG24 Table 1, one dirty ancilla, no measurement: T = 16m - 32
    return 16 * m - 32


def kg_clean(m):      # KG24 Table 1, one clean ancilla, measurement-based uncomputation: T = 8m - 12
    return 8 * m - 12


def maslov_borrowed(m):  # Maslov 2016, Prop. 5 on m + 1 qubits (ceil((m-2)/2) borrowed qubits): T = 8m - 8;
    return 8 * m - 8       # built and verified exactly in toffoli_circuits.py


def zb_dirty(m, k):      # Zindorf-Bose arXiv 2404.02279, Thm 2 with Thm 10: k borrowed qubits,
    assert m >= 5 and 1 <= k <= (m - 3) // 2   # 1 <= k <= floor((m-3)/2): T = 16m - 16k - 16
    return 16 * m - 16 * k - 16


def ours_unitary(n, eps):
    """Branch of Theorem 1 / Theorem 1; returns (T, clean ancillae)."""
    l, m = ell(eps), n - 1
    if l >= 3 and n >= -(-3 * l // 2):
        return maslov_borrowed(l), 0   # borrow ceil((l - 2)/2) idle data coordinates
    if l <= n - 2:
        k = n - 1 - l                  # idle data coordinates, 1 <= k <= floor((l-3)/2) here
        return (zb_dirty(l, k) if l >= 5 else kg_dirty(l)), 0   # k = 1 gives KG24's 16l - 32
    return kg_dirty(m), 1              # exact Toff_n; the clean ancilla serves as the work qubit


def ours_measured(n, eps):
    """Same branch with the KG24 one-clean-ancilla gate (uses one mid-circuit measurement)."""
    l, m = ell(eps), n - 1
    return (kg_clean(l) if l <= n - 2 else kg_clean(m)), 1


def gkz_clean_unitary(eps):   # GKZ's sampler with KG24's one-clean-ancilla gate without measurement:
    return 8 * ell(eps) - 9    # 8m - 9 as counted in KASD26 Eq. (12) (also Maslov 2016 Prop. 4, clean)


def gkz_clean_measured(eps):  # GKZ's sampler with KASD26's measured one-clean-ancilla gate, 8m - 13
    return 8 * ell(eps) - 13


def gkz_and_ladder(eps):      # GKZ's sampler with the measured AND ladder (Jones 2013, Gidney 2018),
    return 4 * ell(eps) - 1    # m - 2 clean ancillae, 4m - 1 as counted in KSD26


ROWS = [
    # label, clean ancillae (per n), measurement, exact, T(n, eps)
    ("This work, unitary branches", lambda n: "0", "no", "no", lambda n, e: ours_unitary(n, e)[0]),
    ("GKZ sampler, one clean ancilla [GKZ, KG24, KASD26]", lambda n: "1", "no", "no", lambda n, e: gkz_clean_unitary(e)),
    ("GKZ sampler, one clean ancilla, measured [GKZ, KASD26]", lambda n: "1", "yes", "no", lambda n, e: gkz_clean_measured(e)),
    ("GKZ sampler, ell-2 clean ancillae, measured [GKZ, Jones13, Gidney18]", lambda n: "l-2", "yes", "no", lambda n, e: gkz_and_ladder(e)),
    ("One borrowed qubit [ZB24, KG24]", lambda n: "0", "no", "yes", lambda n, e: kg_dirty(n - 1)),
    ("Maslov, ceil((m-2)/2) borrowed qubits [Maslov16]", lambda n: "0", "no", "yes", lambda n, e: maslov_borrowed(n - 1)),
    ("One clean ancilla [KG24, KASD26 Eq. 12]", lambda n: "1", "no", "yes", lambda n, e: 8 * (n - 1) - 9),
    ("One clean ancilla, measured [KASD26]", lambda n: "1", "yes", "yes", lambda n, e: 8 * (n - 1) - 13),
    ("$m-2$ clean ancillae, measured [Jones13, Gidney18, KSD26]", lambda n: "m-2", "yes", "yes", lambda n, e: 4 * (n - 1) - 1),
]


def main():
    print("% eps columns:", ", ".join(f"{e:g} (ell={ell(e)})" for e in EPSS))
    for label, anc, meas, exact, T in ROWS:
        cells = []
        for n in NS:
            vals = [T(n, e) for e in EPSS]
            cells.append("/".join(str(v) for v in vals) if len(set(vals)) > 1 else str(vals[0]))
        print(f"{label} & {anc(NS[0])} & {meas} & {exact} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
