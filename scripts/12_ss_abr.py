"""
Structure sensitivity (SS) and abruptness (Abr), Furtula-Gutman-Dehmer (2013).
=================================================================================

Implements the definitions

        SS(TI, G)  = (1/|S(G)|) * sum_{H in S(G)}  |TI(H) - TI(G)| / TI(G)
        Abr(TI, G) = max_{H in S(G)}                |TI(H) - TI(G)| / TI(G)

where S(G) is the set of graphs "structurally similar" to G -- here taken to
be the trees obtainable from G by relocating a single edge (deleting one edge
and reconnecting the two resulting components at a different pair of
vertices), keeping the vertex count fixed. A family-level value for trees of
order n is then

        SS(TI, n)  = average over G in the family of SS(TI, G)
        Abr(TI, n) = average over G in the family of Abr(TI, G)

VALIDATION: this exact combination (single-edge-relocation neighbourhood,
denominator TI(G) rather than a symmetric average, SS aggregated as the
average of per-graph averages, Abr aggregated as the average of per-graph
maxima) reproduces every one of the 16 published (SS, Abr) pairs for SO1 over
n = 6..13 in Oz & Gutman (Hacet. J. Math. Stat. 54(6), 2025, Table 3) exactly,
and both entries for SO1, SO2, SO and Tan at n = 6, 7, 8 (24 values total).
Earlier attempts in this project used a symmetric denominator
(TI(G)+TI(H))/2 and a double maximum for Abr; neither matches the published
table. The definitions above are the ones that do.

Ratio = SS(TI, n) / Abr(TI, n), also reported in the published table (its
third row per index), needs no separate computation: it is simply the ratio
of the two quantities above. Validated the same way (all 8 SO1 values and all
3 SO2/SO/Tan values match). A ratio close to 1 indicates a smooth index whose
value changes gradually under small structural perturbations; a low ratio
indicates occasional large jumps (abrupt behaviour) alongside many small ones.

For a graph G = (V, E), u and v adjacent, d_u and d_v their degrees:

CDZ(G)   Degree-atomic number commensurability index (this work), reduced
         to its homogeneous-Z form:
             CDZ(G) = sum_{uv in E}  gcd(d_u, d_v) / sqrt(d_u * d_v)

R(G)     Randic index:           sum 1/sqrt(d_u*d_v)
GA(G)    Geometric-arithmetic:   sum 2*sqrt(d_u*d_v)/(d_u+d_v)
ABC(G)   Atom-bond connectivity: sum sqrt((d_u+d_v-2)/(d_u*d_v))
SO2(G)   Second Sombor index:    sum |d_u^2-d_v^2|/(d_u^2+d_v^2)
M2(G)    Second Zagreb index:    sum d_u*d_v
SDD(G)   Symmetric division deg: sum d_u/d_v + d_v/d_u
SO(G)    Sombor index:           sum sqrt(d_u^2+d_v^2)
R_alpha, chi_alpha: general Randic / sum-connectivity families, as in the
         paper's other comparison tables.

What this script does
----------------------
1. Validates the SS/Abr pipeline against the published SO1 column for all
   n = 6..13, and against SO1/SO2/SO/Tan for n = 6,7,8 (24 checks total).
2. Computes SS(TI, n) and Abr(TI, n) for CDZ and the paper's comparison
   indices, for trees on n = 6..13 vertices.

Note: SS/Abr require TI(G) != 0 for every G in the family (it appears in the
denominator). All indices used here are strictly positive on any graph with
at least one edge, so this holds throughout for n >= 2.
"""

from math import gcd, sqrt
import statistics
import networkx as nx

# ---------------------------------------------------------------------
# 1. Indices
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

def Tan(G):
    d = deg(G)
    return sum(abs(d[u] ** 2 - d[v] ** 2) / (2 * d[u] * d[v]) for u, v in G.edges())

def SO1(G):
    d = deg(G)
    return 0.5 * sum(abs(d[u] ** 2 - d[v] ** 2) for u, v in G.edges())

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

# ---------------------------------------------------------------------
# 2. Structural neighbourhood S(G): single-edge-relocation trees
# ---------------------------------------------------------------------

def canonical_form(G):
    """AHU canonical string for an unrooted tree, for isomorphism testing."""
    c = nx.center(G)
    def enc(root, parent):
        return "(" + "".join(sorted(enc(v, root) for v in G[root] if v != parent)) + ")"
    if len(c) == 1:
        return enc(c[0], None)
    a, b = c
    return "|".join(sorted([enc(a, b), enc(b, a)]))


def relocation_neighbours(G):
    """
    Canonical forms of every tree obtainable from G by deleting one edge and
    reconnecting the two resulting components at a different vertex pair
    (i.e. S(G), the "structurally similar" graphs of Furtula-Gutman-Dehmer).
    """
    out = set()
    for e in list(G.edges()):
        H = G.copy()
        H.remove_edge(*e)
        A, B = list(nx.connected_components(H))
        for x in A:
            for y in B:
                if {x, y} == {e[0], e[1]}:
                    continue
                H.add_edge(x, y)
                out.add(canonical_form(H))
                H.remove_edge(x, y)
    return out


# ---------------------------------------------------------------------
# 3. SS and Abr
# ---------------------------------------------------------------------

def ss_abr_per_graph(values, neighbour_lists):
    """
    values[i]           = TI(G_i)
    neighbour_lists[i]  = list of indices j (j != i) with G_j in S(G_i)

    Returns (SS_i, Abr_i) for every graph i in the family.
    """
    ss, abr = [], []
    for i, nbrs in enumerate(neighbour_lists):
        ratios = [abs(values[j] - values[i]) / values[i] for j in nbrs]
        ss.append(statistics.fmean(ratios))
        abr.append(max(ratios))
    return ss, abr


def ss_abr_family(trees, f, neighbour_lists=None):
    """
    SS(TI, n) and Abr(TI, n): averages, over the family, of the per-graph SS_i
    and Abr_i. Pass a precomputed neighbour_lists (see build_neighbour_lists)
    to avoid recomputing the single-edge-relocation neighbourhood for every
    index -- that part is index-independent and the expensive part for large n.
    """
    if neighbour_lists is None:
        neighbour_lists = build_neighbour_lists(trees)
    values = [f(G) for G in trees]
    ss_i, abr_i = ss_abr_per_graph(values, neighbour_lists)
    return statistics.fmean(ss_i), statistics.fmean(abr_i)


def build_neighbour_lists(trees):
    """Single-edge-relocation neighbour index-lists, shared across all indices for this family."""
    cid = {canonical_form(G): i for i, G in enumerate(trees)}
    neighbour_lists = []
    for i, G in enumerate(trees):
        idxs = [cid[c] for c in relocation_neighbours(G) if cid[c] != i]
        neighbour_lists.append(idxs)
    return neighbour_lists


# ---------------------------------------------------------------------
# 4. Validation against Oz & Gutman, Table 3
# ---------------------------------------------------------------------

PUBLISHED_SO1 = {
    6: (1.07494, 1.61806), 7: (0.81580, 1.35036), 8: (0.60135, 1.15274),
    9: (0.48720, 1.01002), 10: (0.39877, 0.91481), 11: (0.33955, 0.83478),
    12: (0.29656, 0.77716), 13: (0.26419, 0.72689),
}
PUBLISHED_OTHERS = {
    "SO2": {6: (0.45051, 0.64811), 7: (0.35102, 0.56479), 8: (0.26478, 0.50563)},
    "SO":  {6: (0.18346, 0.28034), 7: (0.15089, 0.25486), 8: (0.12455, 0.23615)},
    "Tan": {6: (0.68751, 1.02777), 7: (0.52638, 0.85421), 8: (0.40056, 0.73494)},
}


def validate(tol=1e-4):
    print("Validating against Oz & Gutman (2025), Table 3 (SS, Abr, and Ratio = SS/Abr):")
    bad = 0
    cache = {}
    for n in range(6, 9):  # SO2/SO/Tan published only for n=6,7,8; SO1 needs 6..13 separately below
        pass
    for n in range(6, 14):
        T = list(nx.nonisomorphic_trees(n))
        nl = build_neighbour_lists(T)
        cache[n] = (T, nl)
        ss, abr = ss_abr_family(T, SO1, nl)
        ratio = ss / abr
        pub_ss, pub_abr = PUBLISHED_SO1[n]
        pub_ratio = pub_ss / pub_abr
        if abs(ss - pub_ss) > tol or abs(abr - pub_abr) > tol or abs(ratio - pub_ratio) > tol:
            bad += 1
            print(f"  DEVIATION SO1 n={n}: SS={ss:.5f}(pub {pub_ss}) Abr={abr:.5f}(pub {pub_abr}) "
                  f"Ratio={ratio:.5f}(pub {pub_ratio:.5f})")
    for name, f in [("SO2", SO2), ("SO", SO), ("Tan", Tan)]:
        for n in (6, 7, 8):
            T, nl = cache[n]
            ss, abr = ss_abr_family(T, f, nl)
            ratio = ss / abr
            pub_ss, pub_abr = PUBLISHED_OTHERS[name][n]
            pub_ratio = pub_ss / pub_abr
            if abs(ss - pub_ss) > tol or abs(abr - pub_abr) > tol or abs(ratio - pub_ratio) > tol:
                bad += 1
                print(f"  DEVIATION {name} n={n}: SS={ss:.5f}(pub {pub_ss}) Abr={abr:.5f}(pub {pub_abr}) "
                      f"Ratio={ratio:.5f}(pub {pub_ratio:.5f})")
    total = 8 + 3 * 3
    print(f"  {total - bad}/{total} checks passed (SS, Abr, and Ratio all verified).  "
          f"{'OK' if bad == 0 else 'CHECK NEEDED'}\n")
    return cache


# ---------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    outfile = sys.argv[1] if len(sys.argv) > 1 else None
    n_range = range(int(sys.argv[2]), int(sys.argv[3]) + 1) if len(sys.argv) > 3 else range(6, 14)
    out = open(outfile, "a") if outfile else sys.stdout

    if 6 in n_range:
        cache_valid = validate()
        print("SS(TI, n), Abr(TI, n), and Ratio = SS/Abr for the paper's comparison")
        print("indices. A ratio near 1 means smooth (gradual) response to structural")
        print("change; a low ratio means occasional large jumps.\n", file=out, flush=True)

    names = [nm for nm, _ in PAPER_INDICES]
    for n in n_range:
        T = list(nx.nonisomorphic_trees(n))
        nl = build_neighbour_lists(T)
        ss_row, abr_row, ratio_row = [], [], []
        for name, f in PAPER_INDICES:
            ss, abr = ss_abr_family(T, f, nl)
            ss_row.append(ss)
            abr_row.append(abr)
            ratio_row.append(ss / abr)
        print(f"n={n} SS   : " + "  ".join(f"{nm}={v:.5f}" for nm, v in zip(names, ss_row)), file=out, flush=True)
        print(f"n={n} Abr  : " + "  ".join(f"{nm}={v:.5f}" for nm, v in zip(names, abr_row)), file=out, flush=True)
        print(f"n={n} Ratio: " + "  ".join(f"{nm}={v:.5f}" for nm, v in zip(names, ratio_row)), file=out, flush=True)
    if outfile:
        out.close()
