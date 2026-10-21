# Restriction lemma: Z subset F_2^m, 0 in Z, |Z|-1 <= delta(2^m-1)  =>  exists subspace W, dim w=floor(log2(1/delta)),
# with W cap Z = {0}.  Then g restricted to W is OR_w (in coordinates of W).  Also worst-case (adversarial) Z check.
import random, math, itertools
random.seed(2026)
def rand_subspace(m,w):
    while True:
        B=[random.getrandbits(m) for _ in range(w)]
        span={0}
        for b in B: span|={s^b for s in span}
        if len(span)==2**w: return B,span
def check(m,delta,trials=200,adv=False):
    w=math.floor(math.log2(1/delta)); K=math.floor(delta*(2**m-1))
    worst=0
    for _ in range(trials):
        if adv:  # adversarial: Z\{0} = nonzero part of a random subspace of size close to K+1 plus random fill
            d=int(math.log2(K+1)); _,S=rand_subspace(m,d); Z=set(S)
            while len(Z)<K+1: Z.add(random.getrandbits(m))
        else:
            Z={0}|set(random.sample(range(1,2**m),K))
        ok=False
        for t in range(2000):
            B,W=rand_subspace(m,w)
            if not (W & Z) - {0}:
                ok=True
                # restricted function on coordinates c in F_2^w: g(sum c_i B_i) = 0 iff c=0
                for c in range(2**w):
                    v=0
                    for i in range(w):
                        if c>>i&1: v^=B[i]
                    assert (v in Z) == (c==0)
                worst=max(worst,t+1); break
        assert ok, (m,delta)
    return w,K,worst
for m,delta in [(8,1/16),(10,1/8),(10,1/33),(12,1/64),(12,1/100)]:
    for adv in (False,True):
        w,K,worst=check(m,delta,adv=adv)
        print(f"m={m} delta={delta:.4f} |Z|-1={K} w={w} adversarial={adv}: W found in all trials (max tries {worst})")
