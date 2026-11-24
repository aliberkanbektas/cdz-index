"""
The local analysis of Section 3.5.3 and Appendix A (Proposition 3.20).

Over all chemical trees with 5 <= n <= 16 that contain an edge uv with
d_u = d_v >= 3 (that is, an edge of type (3,3) or (4,4)), this script checks

  (i)   that some single leaf relocation strictly decreases CDZ;
  (ii)  which of them contain one of the five configurations of Appendix A
        (Propositions A.6-A.10) with a strictly negative change, which is
        part (b) of Proposition 3.20: every tree containing an edge of type
        (3,3) does, while 373 trees whose only equal-degree edges are of
        type (4,4) do not;
  (iii) the smallest improvement achieved by the best leaf relocation.

A leaf relocation deletes the edge px joining a leaf x to its neighbour p and
adds tx for some t not in {x, p} with d_t <= 3 (so the result is again a
chemical tree).  Every change is computed by direct evaluation of CDZ on the
new tree, not from the formulas of Appendix A, so this script is an
independent check of those formulas as well.

Run time: about one minute.
"""

import os
import sys
from math import gcd, sqrt

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TOL = 1e-12
W = {(a, b): gcd(a, b) / sqrt(a * b) for a in range(1, 6) for b in range(1, 6)}
PAPER_COUNT = 10818
PAPER_UNCOVERED = 373
PAPER_MARGIN = 1.5 + sqrt(6) / 3 - sqrt(2) - sqrt(3) / 2   # 0.03626
CASE2_MARGIN = 2 / sqrt(3) - 1 / sqrt(2) - 1 / sqrt(6)       # 0.03935


def total(edges, deg):
    return sum(W[deg[u], deg[v]] for u, v in edges)


def move_delta(edges, deg, base, x, p, t):
    """Change in CDZ when leaf x is moved from p to t."""
    d2 = dict(deg)
    d2[p] -= 1
    d2[t] += 1
    new_edges = [e for e in edges if set(e) != {p, x}] + [(t, x)]
    return total(new_edges, d2) - base


def five_configurations(T, deg, leaves):
    """Yield (case, x, p, t) for every instance of the five configurations."""
    for a, b in T.edges():
        if deg[a] != deg[b]:
            continue
        for u, v in ((a, b), (b, a)):
            if deg[u] == 3:
                # Case 1: p of degree 2 carrying a leaf x, p != u; move x to u
                for p in T:
                    if deg[p] == 2 and p != u:
                        for x in T[p]:
                            if x in leaves and x != u:
                                yield 1, x, p, u
                # Case 2: x a leaf of u, t != x a leaf whose neighbour has degree 3
                for x in T[u]:
                    if x in leaves:
                        for t in leaves:
                            if t != x and deg[next(iter(T[t]))] == 3:
                                yield 2, x, u, t
                # Case 4: p in N(u) of degree 4, x a leaf of p; move x to u
                for p in T[u]:
                    if deg[p] == 4:
                        for x in T[p]:
                            if x in leaves:
                                yield 4, x, p, u
                # Case 5: x a leaf of v, t in N(v) of degree 2; move x to t
                for x in T[v]:
                    if x in leaves:
                        for t in T[v]:
                            if deg[t] == 2:
                                yield 5, x, v, t
            if deg[u] == 4:
                # Case 3: p != t of degree 2 carrying a leaf x, t in N(u) of degree 2
                for t in T[u]:
                    if deg[t] != 2:
                        continue
                    for p in T:
                        if deg[p] == 2 and p != t:
                            for x in T[p]:
                                if x in leaves:
                                    yield 3, x, p, t


def main():
    ok = True
    n_trees = n_leaf_ok = n_five_ok = 0
    n33 = n33_ok = 0
    uncovered_types = set()
    worst_best_leaf = float("inf")
    case_use = {k: 0 for k in range(1, 6)}
    print(f"{'n':>3}{'chem':>7}{'with (3,3)/(4,4)':>18}{'leaf move':>11}{'five configs':>14}")
    for n in range(5, 17):
        chem = [T for T in nx.nonisomorphic_trees(n) if max(dict(T.degree()).values()) <= 4]
        cnt = leaf_ok = five_ok = 0
        for T in chem:
            deg = dict(T.degree())
            if not any(deg[u] == deg[v] and deg[u] >= 3 for u, v in T.edges()):
                continue
            cnt += 1
            edges = list(T.edges())
            base = total(edges, deg)
            leaves = {v for v in T if deg[v] == 1}

            best = 0.0
            for x in leaves:
                p = next(iter(T[x]))
                for t in T:
                    if t in (x, p) or deg[t] > 3:
                        continue
                    best = min(best, move_delta(edges, deg, base, x, p, t))
            if best < -TOL:
                leaf_ok += 1
                worst_best_leaf = min(worst_best_leaf, -best)

            has33 = any(deg[u] == deg[v] == 3 for u, v in T.edges())
            n33 += has33
            covered = False
            for case, x, p, t in five_configurations(T, deg, leaves):
                if move_delta(edges, deg, base, x, p, t) < -TOL:
                    five_ok += 1
                    case_use[case] += 1
                    covered = True
                    break
            n33_ok += has33 and covered
            if not covered:
                uncovered_types.add("(3,3)" if has33 else "(4,4) only")
        n_trees += cnt
        n_leaf_ok += leaf_ok
        n_five_ok += five_ok
        print(f"{n:3d}{len(chem):7d}{cnt:18d}{leaf_ok:11d}{five_ok:14d}")

    print("-" * 53)
    print(f"{'all':>3}{'':7s}{n_trees:18d}{n_leaf_ok:11d}{n_five_ok:14d}")
    print(f"\nfirst configuration found, by case: {case_use}")

    c1 = n_trees == PAPER_COUNT
    c2 = n_leaf_ok == n_trees
    c3 = n33_ok == n33 and n_trees - n_five_ok == PAPER_UNCOVERED and uncovered_types == {"(4,4) only"}
    c4 = abs(worst_best_leaf - PAPER_MARGIN) < 1e-9
    ok = c1 and c2 and c3 and c4
    print(f"\n  [{'OK' if c1 else 'FAIL'}] trees with a (3,3) or (4,4) edge: {n_trees} (paper: {PAPER_COUNT})")
    print(f"  [{'OK' if c2 else 'FAIL'}] an improving leaf relocation exists in every one")
    print(f"  [{'OK' if c3 else 'FAIL'}] Proposition 3.20(b): all {n33} trees with a (3,3) edge contain a "
          f"configuration with negative change; the {n_trees - n_five_ok} that do not all have only "
          f"(4,4) equal-degree edges (paper: {PAPER_UNCOVERED})")
    print(f"  [{'OK' if c4 else 'FAIL'}] smallest best improvement = {worst_best_leaf:.5f} "
          f"= 3/2 + sqrt6/3 - sqrt2 - sqrt3/2; the guaranteed margin of Case 2 is {CASE2_MARGIN:.5f}")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
