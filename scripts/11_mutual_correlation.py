"""
Mutual Pearson correlation matrix among the comparison indices (decane isomers).
==================================================================================

Reproduces the format of Table 4 in:
  M. S. Oz, I. Gutman, "More geometric studies of vertex-degree-based graph
  indices - tangent Sombor index", Hacet. J. Math. Stat. 54(6) (2025) 2195-2205,

but for the paper's own fixed comparison set (the same 12 indices used
throughout: 06_correlation_trees.py, 07_octane_properties.py,
10_degeneracy.py), computed over the decane isomers rather than a
single fixed pair.

For a graph G = (V, E), u and v adjacent, d_u and d_v their degrees:

CDZ(G)      Degree-atomic number commensurability index (this work), reduced
            to its homogeneous-Z form (decanes are pure carbon; Proposition
            2.4 of the paper):
                CDZ(G) = sum_{uv in E}  gcd(d_u, d_v) / sqrt(d_u * d_v)

R(G)        Randic index (classical):            sum 1/sqrt(d_u*d_v)
GA(G)       Geometric-arithmetic index:           sum 2*sqrt(d_u*d_v)/(d_u+d_v)
ABC(G)      Atom-bond connectivity index:         sum sqrt((d_u+d_v-2)/(d_u*d_v))
SO2(G)      Second Sombor index:                  sum |d_u^2-d_v^2|/(d_u^2+d_v^2)
M2(G)       Second Zagreb index:                  sum d_u*d_v
SDD(G)      Symmetric division deg index:         sum d_u/d_v + d_v/d_u
SO(G)       Sombor index:                         sum sqrt(d_u^2+d_v^2)
R_alpha(G)  General Randic index (product form):  sum (d_u*d_v)^alpha
            R_{1/4} -> alpha=1/4 ; R_{1/2} -> alpha=1/2
chi_alpha(G) General sum-connectivity index (sum form): sum (d_u+d_v)^alpha
            chi_{-1} -> alpha=-1 ; chi_{1/4} -> alpha=1/4

Graph family: the 75 decane isomers (C10H22), i.e. all non-isomorphic trees
on 10 vertices with maximum degree at most 4 (OEIS A000602 gives 75 for n=10).

Validation
----------
Before computing the paper's own matrix, this script first reproduces the
published Table 4 above using ITS index set (Tan, M2, Alb, R_{-1}, SDD, ISDD,
SO) on the same 75 decane isomers, and checks every one of the 21 published
entries to within 3e-3. All 21 match exactly (0 deviations), confirming both
the decane-isomer construction and the correlation methodology before it is
applied to the paper's own index set.

What this script does
----------------------
1. Builds the 75 decane isomers as trees on 10 vertices, max degree <= 4.
2. Validates the pipeline against the published Table 4 (Tan-paper index set).
3. Computes the upper-triangular |Pearson r| matrix for the paper's 12-index
   comparison set and prints it in the same layout.
"""

from math import gcd, sqrt
import networkx as nx
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# 1. Graph family: decane isomers
# ---------------------------------------------------------------------

def deg(G):
    return dict(G.degree())

def decane_isomers():
    T = [G for G in nx.nonisomorphic_trees(10)
         if max(dict(G.degree()).values()) <= 4]
    assert len(T) == 75, f"expected 75 decane isomers, got {len(T)}"
    return T

# ---------------------------------------------------------------------
# 2. Validation index set (Oz & Gutman, Table 4)
# ---------------------------------------------------------------------

def Tan(G):
    d = deg(G)
    return sum(abs(d[u] ** 2 - d[v] ** 2) / (2 * d[u] * d[v]) for u, v in G.edges())

def M2(G):
    d = deg(G)
    return sum(d[u] * d[v] for u, v in G.edges())

def Alb(G):
    d = deg(G)
    return sum(abs(d[u] - d[v]) for u, v in G.edges())

def R_minus1(G):
    d = deg(G)
    return sum(1 / (d[u] * d[v]) for u, v in G.edges())

def SDD(G):
    d = deg(G)
    return sum(d[u] / d[v] + d[v] / d[u] for u, v in G.edges())

def ISDD(G):
    d = deg(G)
    return sum(d[u] * d[v] / (d[u] ** 2 + d[v] ** 2) for u, v in G.edges())

def SO(G):
    d = deg(G)
    return sum(sqrt(d[u] ** 2 + d[v] ** 2) for u, v in G.edges())

VALIDATION_INDICES = [
    ("M2", M2), ("Alb", Alb), ("R_-1", R_minus1),
    ("SDD", SDD), ("ISDD", ISDD), ("SO", SO), ("Tan", Tan),
]

PUBLISHED = {
    ("M2", "Alb"): 0.80512, ("M2", "R_-1"): 0.61860, ("M2", "SDD"): 0.77272,
    ("M2", "ISDD"): 0.80627, ("M2", "SO"): 0.88741, ("M2", "Tan"): 0.81078,
    ("Alb", "R_-1"): 0.88125, ("Alb", "SDD"): 0.96100, ("Alb", "ISDD"): 0.97774,
    ("Alb", "SO"): 0.97241, ("Alb", "Tan"): 0.98900,
    ("R_-1", "SDD"): 0.96341, ("R_-1", "ISDD"): 0.94506, ("R_-1", "SO"): 0.88862,
    ("R_-1", "Tan"): 0.92961,
    ("SDD", "ISDD"): 0.98810, ("SDD", "SO"): 0.97547, ("SDD", "Tan"): 0.98513,
    ("ISDD", "SO"): 0.98093, ("ISDD", "Tan"): 0.99619,
    ("SO", "Tan"): 0.98291,
}


def validate(graphs, tol=3e-3):
    vals = {name: [f(G) for G in graphs] for name, f in VALIDATION_INDICES}
    bad = 0
    for (a, b), pub in PUBLISHED.items():
        r = abs(pearsonr(vals[a], vals[b])[0])
        if abs(r - pub) > tol:
            bad += 1
            print(f"  DEVIATION {a}-{b}: computed={r:.5f} published={pub:.5f}")
    print(f"Validation against published Table 4: {len(PUBLISHED) - bad}/"
          f"{len(PUBLISHED)} entries match.  {'OK' if bad == 0 else 'CHECK NEEDED'}")


# ---------------------------------------------------------------------
# 3. The paper's own comparison set
# ---------------------------------------------------------------------

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

def chi(alpha):
    def f(G):
        d = deg(G)
        return sum((d[u] + d[v]) ** alpha for u, v in G.edges())
    return f

def R_alpha(alpha):
    def f(G):
        d = deg(G)
        return sum((d[u] * d[v]) ** alpha for u, v in G.edges())
    return f

PAPER_INDICES = [
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


def print_matrix(indices, graphs, title):
    vals = {name: [f(G) for G in graphs] for name, f in indices}
    names = [n for n, _ in indices]
    w, w0 = 8, 8
    print(f"\n{title}\n")
    print(f"{'':<{w0}}" + "".join(f"{n:>{w}}" for n in names))
    for i, a in enumerate(names):
        row = f"{a:<{w0}}"
        for j, b in enumerate(names):
            if j < i:
                row += " " * w
            else:
                row += f"{abs(pearsonr(vals[a], vals[b])[0]):>{w}.5f}"
        print(row)


# ---------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------

decanes = decane_isomers()
print(f"Decane isomers: {len(decanes)} (verified against OEIS A000602).")

validate(decanes)

print_matrix(PAPER_INDICES, decanes,
             "Mutual |Pearson r| among the paper's comparison indices "
             f"(decane isomers, N={len(decanes)})")
