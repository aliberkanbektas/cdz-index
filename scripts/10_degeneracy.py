"""
Degeneracy (discriminative power) of CDZ and the comparison indices.
=====================================================================

Uses the SAME index set as the other comparison tables in the paper
(06_correlation_trees.py, 07_octane_properties.py), so that
CDZ is measured against one fixed reference set throughout.

Konstantinova degeneracy of an index TI on a family F of N graphs:

        D(TI) = ( N - #{ distinct values TI(G) : G in F } ) / N

i.e. the fraction of structures the index fails to separate. Lower is better;
1 - D is the fraction of the family that the index does distinguish.

For a graph G = (V, E), u and v adjacent, d_u and d_v their degrees:

CDZ(G)      Degree-atomic number commensurability index (this work), reduced
            to its homogeneous-Z form (all families here are carbon skeletons,
            so Z cancels; Proposition 2.4 of the paper):
                CDZ(G) = sum_{uv in E}  gcd(d_u, d_v) / sqrt(d_u * d_v)

R(G)        Randic index (classical):
                R(G) = sum_{uv in E}  1 / sqrt(d_u * d_v)

GA(G)       Geometric-arithmetic index:
                GA(G) = sum_{uv in E}  2*sqrt(d_u*d_v) / (d_u + d_v)

ABC(G)      Atom-bond connectivity index:
                ABC(G) = sum_{uv in E}  sqrt((d_u + d_v - 2) / (d_u * d_v))

SO2(G)      Second Sombor index:
                SO2(G) = sum_{uv in E}  |d_u^2 - d_v^2| / (d_u^2 + d_v^2)

M2(G)       Second Zagreb index:
                M2(G) = sum_{uv in E}  d_u * d_v

SDD(G)      Symmetric division deg index:
                SDD(G) = sum_{uv in E}  d_u/d_v + d_v/d_u

SO(G)       Sombor index:
                SO(G) = sum_{uv in E}  sqrt(d_u^2 + d_v^2)

R_alpha(G)  General Randic index, exponent on the PRODUCT of degrees:
                R_alpha(G) = sum_{uv in E}  (d_u * d_v)^alpha
            R_{1/4} -> alpha = 1/4 ; R_{1/2} -> alpha = 1/2

chi_alpha(G) General sum-connectivity index, exponent on the SUM of degrees:
                chi_alpha(G) = sum_{uv in E}  (d_u + d_v)^alpha
            chi_{-1} -> alpha = -1 ; chi_{1/4} -> alpha = 1/4

Graph families:
  Octanes / Nonanes / Decanes -- the alkane isomers C_nH_{2n+2} for n = 8,9,10,
      as hydrogen-suppressed carbon skeletons: trees on n vertices with maximum
      degree at most 4 (carbon valence). Sizes 18 / 35 / 75 (OEIS A000602).
  n vertices -- all non-isomorphic trees on n vertices, n = 6..13.
      Sizes 6/11/23/47/106/235/551/1301 (OEIS A000055).
Both family sizes are asserted at run time, so a wrong filter or a networkx
change cannot silently produce a wrong table.

On the clustering tolerance
----------------------------
An absolute tolerance makes degeneracy scale-dependent in principle: scaling an
index by 1000 widens the gaps between its values and could lower the reported
degeneracy, though TI and 1000*TI carry identical information. The tolerance is
therefore justified against the data rather than assumed: check_tolerance()
reports, for every index and family, the smallest positive gap between distinct
values. As long as TOL sits far below that, the counts are free of both
floating-point noise and scale artefacts.

What this script does
----------------------
1. Builds each graph family and asserts its size against the known counts.
2. Computes every index for every graph in every family.
3. Reports the degeneracy table, the "percent distinguished" table (1 - D),
   and the tolerance safety margin.
"""

from math import gcd, sqrt
import networkx as nx

TOL = 1e-9

# ---------------------------------------------------------------------
# 1. Indices -- same set as the other comparison tables
# ---------------------------------------------------------------------

def deg(G):
    return dict(G.degree())

def CDZ(G):
    d = deg(G)
    return sum(gcd(d[u], d[v]) / sqrt(d[u] * d[v]) for u, v in G.edges())

def R(G):
    d = deg(G)
    return sum(1 / sqrt(d[u] * d[v]) for u, v in G.edges())

def GA(G):
    d = deg(G)
    return sum(2 * sqrt(d[u] * d[v]) / (d[u] + d[v]) for u, v in G.edges())

def ABC(G):
    d = deg(G)
    return sum(sqrt((d[u] + d[v] - 2) / (d[u] * d[v])) for u, v in G.edges())

def SO2(G):
    d = deg(G)
    return sum(abs(d[u] ** 2 - d[v] ** 2) / (d[u] ** 2 + d[v] ** 2) for u, v in G.edges())

def M2(G):
    d = deg(G)
    return sum(d[u] * d[v] for u, v in G.edges())

def SDD(G):
    d = deg(G)
    return sum(d[u] / d[v] + d[v] / d[u] for u, v in G.edges())

def SO(G):
    d = deg(G)
    return sum(sqrt(d[u] ** 2 + d[v] ** 2) for u, v in G.edges())

def chi(alpha):
    """General sum-connectivity index chi_alpha: exponent on (d_u + d_v)."""
    def f(G):
        d = deg(G)
        return sum((d[u] + d[v]) ** alpha for u, v in G.edges())
    return f

def R_alpha(alpha):
    """General Randic index R_alpha: exponent on (d_u * d_v)."""
    def f(G):
        d = deg(G)
        return sum((d[u] * d[v]) ** alpha for u, v in G.edges())
    return f

INDICES = [
    ("CDZ",     CDZ),
    ("R",       R),
    ("GA",      GA),
    ("ABC",     ABC),
    ("SO2",     SO2),
    ("M2",      M2),
    ("SDD",     SDD),
    ("SO",      SO),
    ("R_1/4",   R_alpha(0.25)),
    ("R_1/2",   R_alpha(0.5)),
    ("chi_-1",  chi(-1)),
    ("chi_1/4", chi(0.25)),
]

# ---------------------------------------------------------------------
# 2. Degeneracy
# ---------------------------------------------------------------------

def degeneracy(values, tol=TOL):
    """
    (N - number of distinct values) / N.

    Values are sorted; each new value is compared against the current cluster
    representative. The representative is not updated within a cluster, so a
    cluster's width stays bounded by tol and no chaining can occur.
    """
    distinct = []
    for x in sorted(values):
        if not distinct or abs(x - distinct[-1]) > tol:
            distinct.append(x)
    return (len(values) - len(distinct)) / len(values)


def min_positive_gap(values, floor=1e-13):
    """Smallest gap between distinct values; used to justify TOL."""
    v = sorted(values)
    gaps = [b - a for a, b in zip(v, v[1:]) if b - a > floor]
    return min(gaps) if gaps else float("inf")


# ---------------------------------------------------------------------
# 3. Graph families (sizes asserted against OEIS counts)
# ---------------------------------------------------------------------

TREE_COUNTS = {6: 6, 7: 11, 8: 23, 9: 47, 10: 106, 11: 235, 12: 551, 13: 1301}
ALKANE_COUNTS = {8: 18, 9: 35, 10: 75}


def all_trees(n):
    T = list(nx.nonisomorphic_trees(n))
    assert len(T) == TREE_COUNTS[n], f"n={n}: got {len(T)}, expected {TREE_COUNTS[n]}"
    return T


def alkane_isomers(n):
    """Chemical trees on n vertices: maximum degree at most 4."""
    T = [G for G in nx.nonisomorphic_trees(n)
         if max(dict(G.degree()).values()) <= 4]
    assert len(T) == ALKANE_COUNTS[n], f"n={n}: got {len(T)}, expected {ALKANE_COUNTS[n]}"
    return T


# ---------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------

def table(title, families, transform=None, suffix=""):
    print(f"\n{title}\n")
    names = [n for n, _ in INDICES]
    w, w0 = 10, 14
    print(f"{'Family':<{w0}}{'N':>5}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (w0 + 5 + w * len(names)))
    for label, graphs in families:
        row = f"{label:<{w0}}{len(graphs):>5}"
        for _, f in INDICES:
            d = degeneracy([f(G) for G in graphs])
            val = transform(d) if transform else d
            row += f"{val:>{w-len(suffix)}.5f}{suffix}" if not suffix \
                else f"{val:>{w-1}.1f}{suffix}"
        print(row)


def check_tolerance(families):
    print("\nTolerance check: smallest positive gap between distinct values\n")
    names = [n for n, _ in INDICES]
    w, w0 = 10, 14
    print(f"{'Family':<{w0}}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (w0 + w * len(names)))
    worst = float("inf")
    for label, graphs in families:
        row = f"{label:<{w0}}"
        for _, f in INDICES:
            g = min_positive_gap([f(G) for G in graphs])
            worst = min(worst, g)
            row += f"{g:>{w}.1e}" if g != float("inf") else f"{'-':>{w}}"
        print(row)
    print(f"\nworst case {worst:.2e}   TOL = {TOL:.0e}   margin {worst/TOL:,.0f}x"
          f"   -> {'SAFE' if worst > 100 * TOL else 'UNSAFE'}")


if __name__ == "__main__":
    families = [
        ("Octanes", alkane_isomers(8)),
        ("Nonanes", alkane_isomers(9)),
        ("Decanes", alkane_isomers(10)),
    ] + [(f"{n} vertices", all_trees(n)) for n in range(6, 14)]
    print("Family sizes verified against OEIS A000055 / A000602.  OK")

    table("Table. Degeneracy values of the indices for octanes, nonanes, decanes,\n"
          "and trees with 6, 7, ..., 13 vertices.", families)
    table("Percentage of the family distinguished (1 - D):", families,
          transform=lambda d: 100 * (1 - d), suffix="%")
    check_tolerance(families)
