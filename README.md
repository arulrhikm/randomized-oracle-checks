# randomized-oracle-checks

Scripts that check the constructions and numerical claims in *Ancilla-free randomized Toffoli gates and the T-count of Boolean oracles* (A. R. Mazumder, [arXiv:2610.38189](https://arxiv.org/abs/2610.38189)).

Python 3.9 or later, standard library only.

```
python3 run_all.py            # runs every check (about 20 s) and compares with expected/
python3 toffoli_circuits.py   # or run a single check
```

| Script | Checks |
|---|---|
| `toffoli_circuits.py` | Builds one branch of the randomized Toffoli gate (Theorem 1) with borrowed qubits and verifies it exactly, including the count of 8ℓ−8 T gates. |
| `toffoli_table.py` | The T-counts in Table 1. |
| `affine_xor.py` | The ancilla-free construction for exclusive-ors of affine-subspace indicators (Theorem 2) and the degree criterion for exact oracles. |
| `lower_bound.py` | The lower bound for the Toffoli gate (Theorem 3), the factor-of-eight comparison and the distance-from-affine bound. |
| `oracle_restrict.py` | The restriction to a random subspace used for Theorem 4. |
| `acceptance_invariants.py` | The acceptance-count lemma and the random-set T-count bounds (Theorem 5). |
| `unstructured_brute.py` | The random-set decision-tree bound by brute force at n = 8 (uses `unstructured_lib.py`). |
| `unstructured_union_bound.py` | The union-bound inequality in the random-set theorem. |

Each script prints its results, and `expected/` holds the output it should produce.

## Citing

```bibtex
@misc{mazumder2026ancillafree,
  author        = {Mazumder, Arul Rhik},
  title         = {Ancilla-free randomized {Toffoli} gates and the {$T$}-count of {Boolean} oracles},
  year          = {2026},
  eprint        = {2610.38189},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```
