"""Run every check and compare its output with expected/<name>.txt."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = ["toffoli_circuits", "toffoli_table", "affine_xor", "lower_bound",
          "oracle_restrict", "acceptance_invariants", "unstructured_brute", "unstructured_union_bound"]

failed = 0
for name in CHECKS:
    start = time.time()
    out = subprocess.run([sys.executable, os.path.join(HERE, name + ".py")],
                         capture_output=True, text=True, cwd=HERE)
    with open(os.path.join(HERE, "expected", name + ".txt")) as f:
        ok = out.returncode == 0 and out.stdout == f.read()
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:26s} {time.time() - start:5.1f}s")
print("all checks passed" if not failed else f"{failed} check(s) failed")
sys.exit(1 if failed else 0)
