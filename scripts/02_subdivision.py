"""
Reproduces every number in Section 3.5.1-3.5.2 of cdz_paper.tex
(two-parameter family, subdivision identity, Proposition 3.18, Remark 3.19).

Run:  python3 verify_subdivision.py        Requires: networkx
"""
from math import gcd, sqrt
from collections import Counter
import networkx as nx

R2, R3, R6 = sqrt(2), sqrt(3), sqrt(6)

def w(a, b):
    return gcd(a, b) / sqrt(a * b)

def cdz(G):
    d = dict(G.degree())
    return sum(w(d[u], d[v]) for u, v in G.edges())

def sigma(a, b):
    """Change in CDZ when an edge joining degrees a and b is subdivided (Lemma 3.17)."""
    return w(a, 2) + w(2, b) - w(a, b)

def chemical_trees(n):
    return [G for G in nx.nonisomorphic_trees(n) if max(dict(G.degree()).values()) <= 4]

def subdivide(G, u, v):
    H = G.copy(); H.remove_edge(u, v); z = max(H) + 1
    H.add_edge(u, z); H.add_edge(z, v)
    return H

def internal(G):
    d = dict(G.degree())
    return sorted((d[v] for v in G if d[v] > 1), reverse=True)

def check(label, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    return ok

def main():
    ok = True
    mu = {n: min(cdz(G) for G in chemical_trees(n)) for n in range(4, 18)}

    print("Lemma 3.17  CDZ(T_xy) = CDZ(T) + sigma(d_x, d_y)")
    pairs = bad = 0
    for n in range(2, 13):
        for G in chemical_trees(n):
            d = dict(G.degree())
            for u, v in G.edges():
                pairs += 1
                bad += abs(cdz(subdivide(G, u, v)) - cdz(G) - sigma(d[u], d[v])) > 1e-12
    ok &= check(f"{pairs} (tree, edge) pairs with n <= 12, {bad} failures", bad == 0)

    print("\nEquation (2): subdivision costs")
    exact = {(1, 3): 1/R2 + 1/R6 - 1/R3, (1, 4): R2 - 0.5, (3, 3): 2/R6 - 1,
             (3, 4): 1/R2 + 1/R6 - 1/(2*R3), (4, 4): R2 - 1}
    for e, val in exact.items():
        ok &= check(f"sigma{e} = {sigma(*e):+.5f}", abs(sigma(*e) - val) < 1e-12)
    neg = [(a, b) for a in (1, 3, 4) for b in (1, 3, 4) if a <= b and sigma(a, b) < 0]
    ok &= check(f"negative costs among degrees {{1,3,4}}: {neg}", neg == [(3, 3)])

    print("\nSection 3.5.1: (3,4)-tree family reproduces Table 1")
    def fam(p, q):
        return (p+q-1)/(2*R3) + (2*p-q+1)/R3 + (3*q-p+1)/2
    def realisable(p, q):
        if (p, q) in ((1, 0), (0, 1)): return True
        return p >= 1 and q >= 1 and 2*p-q+1 >= 0 and 3*q-p+1 >= 0
    hit = []
    for n in range(4, 18):
        vals = [fam(p, (n-2-2*p)//3) for p in range(n) if (n-2-2*p) >= 0
                and (n-2-2*p) % 3 == 0 and realisable(p, (n-2-2*p)//3)]
        if vals and abs(min(vals) - mu[n]) < 1e-9: hit.append(n)
    ok &= check(f"reproduced orders {hit}", hit == [4, 5, 7, 9, 10, 12, 13, 14, 15, 17])

    print("\nProposition 3.18: the three trees with internal degrees 4,3,2")
    T7 = min(chemical_trees(7), key=cdz)
    ok &= check(f"CDZ(T*_7) = 3/2 + 5*sqrt3/6 = {cdz(T7):.5f}", abs(cdz(T7) - (1.5 + 5*R3/6)) < 1e-12)
    d7 = dict(T7.degree()); seen = {}
    for u, v in T7.edges():
        t = tuple(sorted((d7[u], d7[v])))
        seen.setdefault(t, cdz(subdivide(T7, u, v)))
    trees432 = sorted(cdz(G) for G in chemical_trees(8) if internal(G) == [4, 3, 2])
    ok &= check(f"exactly three trees: {[round(x, 5) for x in trees432]}", len(trees432) == 3)
    ok &= check("they are the subdivisions of T*_7 at (1,3), (3,4), (1,4)",
                sorted(seen.values()) == trees432)
    ok &= check(f"432 = 3/2+sqrt3/2+sqrt2/2+sqrt6/6 = {seen[(1,3)]:.5f}",
                abs(seen[(1, 3)] - (1.5 + R3/2 + R2/2 + R6/6)) < 1e-12)
    ok &= check(f"423 = 3/2+2sqrt3/3+sqrt2/2+sqrt6/6 = {seen[(3,4)]:.5f}",
                abs(seen[(3, 4)] - (1.5 + 2*R3/3 + R2/2 + R6/6)) < 1e-12)
    ok &= check(f"342 = 1+5sqrt3/6+sqrt2 = {seen[(1,4)]:.5f}",
                abs(seen[(1, 4)] - (1 + 5*R3/6 + R2)) < 1e-12)
    ok &= check("423 - 432 = 1/(2 sqrt3) = w(3,4)",
                abs(seen[(3, 4)] - seen[(1, 3)] - w(3, 4)) < 1e-12)
    ok &= check("432 is the global minimiser on 8 vertices", abs(seen[(1, 3)] - mu[8]) < 1e-12)

    print("\nRemark 3.19")
    s13, s33 = sigma(1, 3), sigma(3, 3)
    ok &= check("mu_8  = mu_7  + sigma(1,3)", abs(mu[8] - mu[7] - s13) < 1e-9)
    ok &= check("mu_11 = mu_10 + sigma(1,3)", abs(mu[11] - mu[10] - s13) < 1e-9)
    ok &= check(f"mu_6  = sqrt3 + 2 sigma(1,3) = {R3 + 2*s13:.5f}", abs(mu[6] - (R3 + 2*s13)) < 1e-9)
    ok &= check(f"mu_15 = 4 + 4/sqrt3 = {mu[15]:.5f}", abs(mu[15] - (4 + 4/R3)) < 1e-9)
    ok &= check(f"mu_16 = 9/2 + 5sqrt3/6 + sqrt6/3 = {mu[16]:.5f}",
                abs(mu[16] - (4.5 + 5*R3/6 + R6/3)) < 1e-9)
    # the 15-vertex tree whose (3,3) edge is subdivided
    best = None
    for G in chemical_trees(15):
        d = dict(G.degree())
        if any(d[v] == 2 for v in G): continue
        for u, v in G.edges():
            if (d[u], d[v]) == (3, 3):
                val = cdz(G)
                if best is None or val < best: best = val
    ok &= check(f"15-vertex tree with a (3,3) edge: 11/2 + 5sqrt3/6 = {best:.5f}",
                abs(best - (5.5 + 5*R3/6)) < 1e-9)
    ok &= check(f"its subdivision gives mu_16", abs(best + s33 - mu[16]) < 1e-9)
    ok &= check(f"premium (3-sqrt3)/2 = {best - mu[15]:.5f}", abs(best - mu[15] - (3-R3)/2) < 1e-9)
    ok &= check(f"mu_15 + sigma(1,3) = {mu[15] + s13:.5f} > mu_16; margin {mu[15]+s13-mu[16]:.5f}",
                mu[15] + s13 > mu[16])

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))

if __name__ == "__main__":
    main()
