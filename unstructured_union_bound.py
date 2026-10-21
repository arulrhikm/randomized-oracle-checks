"""Check the union-bound inequality (*) of the random-set theorem (Appendix C).

Theorem condition (eta = 1/5, Chernoff 2exp(-eta^2 mu/3) = 2exp(-2^s/75)):
  failure <= 4*2^{q(n-q)} * 2^q * 2*exp(-2^s/75),  q = n-k-s.
We report, for each (n,k), the least s making failure <= 2^{-n}, both
  (a) via the exact Gaussian binomial count and exact log, and
  (b) via the sufficient closed form 2^s >= 52*(q(k+s+1)+n+3),
and compare with s0 = ceil(2 log2(n+1)) + 6 and with log2(n(k+log2 n)).
"""
import math


def gauss_binom(n, q):
    num = 1
    den = 1
    for i in range(q):
        num *= (2 ** (n - i) - 1)
        den *= (2 ** (i + 1) - 1)
    return num // den


def log2_gauss_binom(n, q):
    f = lambda a: a + math.log2(-math.expm1(-a * math.log(2)))  # log2(2^a - 1)
    return sum(f(n - i) - f(i + 1) for i in range(q))


def log2_fail_exact(n, k, s):
    q = n - k - s
    if q < 0:
        return None
    return log2_gauss_binom(n, q) + q + 1 - (2 ** s / 75) / math.log(2)


def least_s_exact(n, k):
    for s in range(0, n - k + 1):
        v = log2_fail_exact(n, k, s)
        if v is not None and v <= -n:
            return s
    return None


def least_s_closed(n, k):
    for s in range(0, n - k + 1):
        q = n - k - s
        if 2 ** s >= 52 * (q * (k + s + 1) + n + 3):
            return s
    return None


if __name__ == "__main__":
    print("n    k   s_exact s_closed s0=ceil(2lg(n+1))+6  lg(n(k+lg n))   q=n-k-s_exact")
    for n in [64, 128, 256, 512, 1024, 4096]:
        for k in [1, 2, 8, n // 4, n // 2]:
            se = least_s_exact(n, k)
            sc = least_s_closed(n, k)
            s0 = math.ceil(2 * math.log2(n + 1)) + 6
            ref = math.log2(n * (k + math.log2(n)))
            q = None if se is None else n - k - se
            print(f"{n:<5}{k:<5}{str(se):<8}{str(sc):<9}{s0:<21}{ref:<14.1f}{q}")
    # verify closed-form choice s0 satisfies the closed condition for all k, 2<=n<=5000
    bad = []
    for n in range(2, 1001):
        s0 = math.ceil(2 * math.log2(n + 1)) + 6
        for k in range(1, n - s0 + 1):
            q = n - k - s0
            if not 2 ** s0 >= 52 * (q * (k + s0 + 1) + n + 3):
                bad.append((n, k))
    print("closed-form s0 failures for 2<=n<=1000:", bad[:10], len(bad))
    # verify closed condition implies the exact bound (sanity of the 52 constant)
    worst = -1e9
    for n in list(range(2, 80)) + [100, 150, 200, 300]:
        for k in range(1, n):
            sc = least_s_closed(n, k)
            if sc is None:
                continue
            worst = max(worst, log2_fail_exact(n, k, sc) + n)
    print("max over n<=300 (sampled) of log2(failure)+n at closed-form s (must be <=0):", round(worst, 3))
