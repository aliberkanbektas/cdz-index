"""
Correlation between CDZ and well-known topological indices.
=============================================================

For a graph G = (V, E), u and v adjacent, d_u and d_v their degrees:

CDZ(G)      Degree-atomic number commensurability index (this work), reduced
            to its homogeneous-Z form (Proposition 2.4 of the paper):
                CDZ(G) = sum_{uv in E}  gcd(d_u, d_v) / sqrt(d_u * d_v)

R(G)        Randic index (classical, alpha = -1/2 of the general family below):
                R(G) = sum_{uv in E}  1 / sqrt(d_u * d_v)
            Exact sandwich with CDZ: R(G) <= CDZ(G) <= Delta*R(G) (Theorem 2.5).

GA(G)       Geometric-arithmetic index:
                GA(G) = sum_{uv in E}  2*sqrt(d_u*d_v) / (d_u + d_v)
            Exact sandwich with CDZ: GA(G)/Delta <= CDZ(G) <= GA(G) (Theorem 2.6).

ABC(G)      Atom-bond connectivity index:
                ABC(G) = sum_{uv in E}  sqrt((d_u + d_v - 2) / (d_u * d_v))
            Included because it is used alongside R(G) in the discrimination
            discussion (Section 5) to show CDZ separates constitutionally
            distinct molecules that ABC and R cannot.

SO2(G)      Second Sombor index:
                SO2(G) = sum_{uv in E}  |d_u^2 - d_v^2| / (d_u^2 + d_v^2)
            CDZ's empirically closest relative among all indices tested; kept
            in the comparison so the "not a repackaging of a known index"
            claim is checked against its hardest case, not its easiest.

M2(G)       Second Zagreb index:
                M2(G) = sum_{uv in E}  d_u * d_v

SDD(G)      Symmetric division deg index:
                SDD(G) = sum_{uv in E}  d_u/d_v + d_v/d_u

SO(G)       Sombor index:
                SO(G) = sum_{uv in E}  sqrt(d_u^2 + d_v^2)

R_alpha(G)  General Randic index, exponent applied to the PRODUCT of degrees:
                R_alpha(G) = sum_{uv in E}  (d_u * d_v)^alpha
            R_{1/4}  -> alpha = 1/4
            R_{1/2}  -> alpha = 1/2 (equals the "reciprocal Randic" index)
            The classical Randic index R(G) above is the case alpha = -1/2.

chi_alpha(G) General sum-connectivity index, exponent applied to the SUM of
            degrees (not the product -- this is what distinguishes it from
            R_alpha):
                chi_alpha(G) = sum_{uv in E}  (d_u + d_v)^alpha
            chi_{-1}   -> alpha = -1
            chi_{1/4}  -> alpha = 1/4

R_alpha and chi_alpha with these four specific exponents are the same four
columns used in the IJMC paper's correlation table for the geometric-quadratic
index GQ, reproduced here so that CDZ is checked against exactly the same
comparison set.

What this script does
----------------------
1. Builds every non-isomorphic tree on n=10 vertices (106 trees; swap the
   commented line below to restrict to decane isomers, i.e. chemical trees
   with maximum degree <= 4, 75 trees -- the family used in the HJMS/IJMC
   correlation tables this comparison is modelled on).
2. Computes CDZ(G) and each listed index for every tree in the family.
3. Reports the absolute Pearson correlation |R| between CDZ and each index.

A high |R| would indicate CDZ is close to being a rescaling of that index; the
result (all values between 0.70 and 0.87, well below e.g. GQ-GA's published
0.999) supports the claim that CDZ is not a repackaging of any single existing
index.
"""

import networkx as nx
from math import gcd, sqrt
from scipy.stats import pearsonr

def deg(G):
    return dict(G.degree())

def CDZ(G):
    d = deg(G)
    return sum(gcd(d[u], d[v]) / sqrt(d[u] * d[v]) for u, v in G.edges())

# --- classical indices ------------------------------------------------

def M2(G):
    d = deg(G)
    return sum(d[u] * d[v] for u, v in G.edges())

def GA(G):
    d = deg(G)
    return sum(2 * sqrt(d[u] * d[v]) / (d[u] + d[v]) for u, v in G.edges())

def chi(alpha):
    """General sum-connectivity index chi_alpha: exponent on (d_u + d_v)."""
    def f(G):
        d = deg(G)
        return sum((d[u] + d[v]) ** alpha for u, v in G.edges())
    return f

def R(alpha):
    """General Randic index R_alpha: exponent on (d_u * d_v)."""
    def f(G):
        d = deg(G)
        return sum((d[u] * d[v]) ** alpha for u, v in G.edges())
    return f

def SDD(G):
    d = deg(G)
    return sum(d[u] / d[v] + d[v] / d[u] for u, v in G.edges())

def SO(G):
    d = deg(G)
    return sum(sqrt(d[u] ** 2 + d[v] ** 2) for u, v in G.edges())

def ABC(G):
    d = deg(G)
    return sum(sqrt((d[u] + d[v] - 2) / (d[u] * d[v])) for u, v in G.edges())

def SO2(G):
    d = deg(G)
    return sum(abs(d[u] ** 2 - d[v] ** 2) / (d[u] ** 2 + d[v] ** 2) for u, v in G.edges())

INDICES = [
    ("R(G)",       R(-0.5)),    # classical Randic index (alpha = -1/2)
    ("GA(G)",      GA),
    ("ABC(G)",     ABC),
    ("SO2(G)",     SO2),
    ("M2(G)",      M2),
    ("SDD(G)",     SDD),
    ("SO(G)",      SO),
    ("R_1/4(G)",   R(0.25)),    # general Randic, alpha = 1/4
    ("R_1/2(G)",   R(0.5)),     # general Randic, alpha = 1/2
    ("chi_-1(G)",  chi(-1)),    # general sum-connectivity, alpha = -1
    ("chi_1/4(G)", chi(0.25)),  # general sum-connectivity, alpha = 1/4
]

# graph family: all trees on n=10 vertices (swap for decane isomers with
# max degree <= 4 by uncommenting the filter below)
n = 10
graphs = list(nx.nonisomorphic_trees(n))
# graphs = [G for G in graphs if max(dict(G.degree()).values()) <= 4]  # decanes

cdz_vals = [CDZ(G) for G in graphs]

print(f"Correlation between CDZ(G) and well-known topological indices  (n={n}, N={len(graphs)})\n")
for name, f in INDICES:
    r, _ = pearsonr(cdz_vals, [f(G) for G in graphs])
    print(f"  {name:12s} |R| = {abs(r):.5f}")
