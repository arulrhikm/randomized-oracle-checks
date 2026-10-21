"""Random-set lower bound by brute force for n=8, many sets at once.
Computes coset labels once per subspace (incrementally, by linearity) and evaluates all sets."""
import random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unstructured_lib import subspaces, mm_set

def run(n, sets, qmax):
    N = 1 << n
    info = []
    for name, E in sets:
        inE = [0]*N
        for x in E: inE[x] = 1
        info.append((name, inE, len(E), N-len(E)))
    res = {name: [] for name, _ in sets}
    for q in range(qmax+1):
        best = {name: 1.0 for name, _ in sets}
        for basis in subspaces(n, q):
            unit = [sum((bin(r >> b & 1).count("1")) << i for i, r in enumerate(basis)) for b in range(n)]
            lab = [0]*N
            for x in range(1, N):
                low = (x & -x).bit_length()-1
                lab[x] = lab[x & (x-1)] ^ unit[low]
            cA = [0]*(1<<q)
            for x in range(N): cA[lab[x]] += 1
            for name, inE, nE, nC in info:
                cE = [0]*(1<<q)
                for x in range(N):
                    if inE[x]: cE[lab[x]] += 1
                err = 0.5*sum(min(cE[c]/nE, (cA[c]-cE[c])/nC) for c in range(1<<q))
                if err < best[name]: best[name] = err
        for name in best: res[name].append(best[name])
    return res

if __name__ == "__main__":
    random.seed(8)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7; qmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    sets = []
    for k in [1,2,3]:
        for t in range(2):
            E = [x for x in range(1<<n) if random.random() < 2.0**-k]
            sets.append((f"random k={k} t={t} |E|={len(E)}", E))
    for k in [1,2,3]:
        sets.append((f"affine codim {k}", [x for x in range(1<<n) if x & ((1<<k)-1) == (1<<k)-1]))
    if n == 8:
        for k in [1,2]:
            _, E = mm_set(4, k); sets.append((f"explicit MM m=4 k={k} |E|={len(E)}", E))
    res = run(n, sets, qmax)
    print(f"n={n}: least mu_E-error of a deterministic tree with q queries, q=0..{qmax}")
    for name, v in res.items():
        print(f"  {name:<34}", " ".join(f"{e:.4f}" for e in v))
