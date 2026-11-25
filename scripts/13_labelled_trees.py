"""
Heteroatom-labelled degeneracy: the structural discrimination test.
=====================================================================

Reproduces Table 10 of the paper: the fraction of C/N/O-labelled trees that
each index can tell apart.  The table counts only labellings that respect the
valences of the three elements (degree at most 4 at C, 3 at N, 2 at O); the
counts over all labellings, chemically possible or not, are printed as well.

The point of the test
----------------------
Every classical vertex-degree-based index (Randić R, GA, ABC, Sombor SO,
second Zagreb M2, SDD, ...) ignores atom labels.  So if two
molecules have the same hydrogen-suppressed graph shape but different
heteroatom placement -- ethanol (C-C-O) and dimethyl ether (C-O-C), say,
both of which are the 3-vertex path P3 -- every one of those indices assigns
them THE SAME VALUE, by construction. They are blind to heteroatom placement.

CDZ is not, because its vertex weight a_u = d_u * Z_u carries the atomic
number. This script quantifies how big that difference is, by brute-force
enumeration rather than by statistical correlation: for each n, it builds
every non-isomorphic C/N/O labelling of every tree on n vertices, then counts
how many distinct values each index produces.

    CDZ(G)     = sum_{uv in E} gcd(a_u, a_v) / sqrt(a_u * a_v),  a_u = d_u * Z_u
    CDZ_hom(G) = the same formula with every Z_u fixed to 6

CDZ_hom is the control: it is CDZ stripped of exactly the atomic-number
information and nothing else. If CDZ_hom scores like the classical indices
(it does -- identically), then any advantage CDZ shows comes from the atomic
number weighting specifically, not from the gcd structure.

What counts as "the same structure"
------------------------------------
Two labelled trees are the same iff there is an isomorphism between them that
PRESERVES the atomic-number labels. Node-attributed Weisfeiler-Lehman hashing
groups candidates cheaply; because WL hashing can (rarely) collide, every
hash collision is confirmed with an exact isomorphism test via
nx.is_isomorphic with a node_match on Z. This matters: skipping the exact
check would silently merge distinct structures and inflate every index's
apparent degeneracy.

Degeneracy, as elsewhere in this project, is the Konstantinova measure:

    D(TI) = (N - #{distinct values of TI}) / N

so the fraction of structures an index DOES distinguish is 1 - D, which is
what the paper's table reports as a percentage.

Self-check
----------
Ethanol vs dimethyl ether is verified explicitly at startup as the smallest
instance of the phenomenon: the two must be indistinguishable to every
classical index and distinguishable by CDZ. If that assertion ever fails,
something is wrong with the index implementations and the whole table is
suspect.

Requirements: networkx
"""

from itertools import product
from math import gcd, sqrt

import networkx as nx

TOL = 1e-9
ALPHABET = [6, 7, 8]          # C, N, O
CARBON = 6
MAX_DEGREE = {6: 4, 7: 3, 8: 2}   # valence of C, N, O


def respects_valence(G):
    return all(G.degree(v) <= MAX_DEGREE[G.nodes[v]["Z"]] for v in G)


# ---------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------

def cdz(G, use_Z=True):
    """CDZ with real atomic numbers (use_Z=True) or all-carbon (use_Z=False)."""
    d = dict(G.degree())
    a = {v: d[v] * (G.nodes[v]["Z"] if use_Z else CARBON) for v in G}
    return sum(gcd(a[u], a[v]) / sqrt(a[u] * a[v]) for u, v in G.edges())


def R(G):
    d = dict(G.degree())
    return sum(1 / sqrt(d[u] * d[v]) for u, v in G.edges())


def GA(G):
    d = dict(G.degree())
    return sum(2 * sqrt(d[u] * d[v]) / (d[u] + d[v]) for u, v in G.edges())


def ABC(G):
    d = dict(G.degree())
    return sum(sqrt((d[u] + d[v] - 2) / (d[u] * d[v])) for u, v in G.edges())


def SO(G):
    d = dict(G.degree())
    return sum(sqrt(d[u] ** 2 + d[v] ** 2) for u, v in G.edges())


def M2(G):
    d = dict(G.degree())
    return sum(d[u] * d[v] for u, v in G.edges())


def SDD(G):
    d = dict(G.degree())
    return sum(d[u] / d[v] + d[v] / d[u] for u, v in G.edges())


INDICES = [
    ("CDZ",     lambda G: cdz(G, True)),
    ("CDZ_hom", lambda G: cdz(G, False)),
    ("R",       R),
    ("GA",      GA),
    ("ABC",     ABC),
    ("SO",      SO),
    ("M2",      M2),
    ("SDD",     SDD),
]


# ---------------------------------------------------------------------
# Enumeration of Z-labelled trees up to label-preserving isomorphism
# ---------------------------------------------------------------------

def _wl_hash(G):
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="Z", iterations=5)


def _same_labelled(G, H):
    return nx.is_isomorphic(G, H, node_match=lambda a, b: a["Z"] == b["Z"])


def generate_labelled_trees(n, alphabet=ALPHABET):
    """
    Every non-isomorphic Z-labelled tree on n vertices, labels drawn from
    `alphabet`. WL-hash buckets first, then an exact label-preserving
    isomorphism test inside each bucket (WL hashes can collide).
    """
    buckets = {}
    for T in nx.nonisomorphic_trees(n):
        nodes = sorted(T.nodes())
        for labels in product(alphabet, repeat=n):
            G = T.copy()
            for v, z in zip(nodes, labels):
                G.nodes[v]["Z"] = z
            h = _wl_hash(G)
            bucket = buckets.setdefault(h, [])
            if any(_same_labelled(G, K) for K in bucket):
                continue
            bucket.append(G)
    return [G for bucket in buckets.values() for G in bucket]


# ---------------------------------------------------------------------
# Degeneracy
# ---------------------------------------------------------------------

def degeneracy(values, tol=TOL):
    distinct = []
    for x in sorted(values):
        if not distinct or abs(x - distinct[-1]) > tol:
            distinct.append(x)
    return (len(values) - len(distinct)) / len(values)


# ---------------------------------------------------------------------
# Self-check: ethanol vs dimethyl ether
# ---------------------------------------------------------------------

def self_check():
    """Both are P3; ethanol has O terminal, dimethyl ether has O central."""
    def p3(z0, z1, z2):
        G = nx.path_graph(3)
        for v, z in zip([0, 1, 2], [z0, z1, z2]):
            G.nodes[v]["Z"] = z
        return G

    ethanol = p3(6, 6, 8)      # C-C-O
    dme = p3(6, 8, 6)          # C-O-C

    for name, f in INDICES:
        if name == "CDZ":
            continue
        assert abs(f(ethanol) - f(dme)) < TOL, \
            f"{name} unexpectedly distinguishes ethanol from dimethyl ether"
    assert abs(cdz(ethanol) - cdz(dme)) > TOL, \
        "CDZ fails to distinguish ethanol from dimethyl ether"

    print("self-check: ethanol vs dimethyl ether")
    print(f"   CDZ:     {cdz(ethanol):.4f} vs {cdz(dme):.4f}   -> distinguished")
    print(f"   R, GA, ABC, SO, M2, SDD, CDZ_hom: identical values -> not distinguished")
    print("   OK\n")


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def table(rows, title):
    names = [nm for nm, _ in INDICES]
    print(title)
    print(f"{'n':>3}{'#structures':>13}  " + "".join(f"{nm:>10}" for nm in names))
    print("-" * (16 + 10 * len(names)))
    for n, N, degs, _ in rows:
        print(f"{n:>3}{N:>13}  " + "".join(f"{d:>10.5f}" for d in degs))
    print(f"\n{'n':>3}{'#structures':>13}{'CDZ':>8}{'CDZ_hom':>9}{'R':>6}{'GA':>5}{'ABC':>5}"
          f"{'SO':>5}{'M2':>5}{'SDD':>5}   (number of distinct values)")
    for n, N, _, nd in rows:
        print(f"{n:>3}{N:>13}{nd[0]:>8}{nd[1]:>9}" + "".join(f"{k:>{w}}" for k, w in zip(nd[2:], (6, 5, 5, 5, 5, 5))))
    print(f"\n{'n':>3}{'#structures':>13}{'CDZ':>10}{'classical':>12}{'CDZ_hom':>10}{'ratio':>8}")
    print("-" * 56)
    for n, N, degs, _ in rows:
        cdz_pct = 100 * (1 - degs[0])
        cls_pct = 100 * (1 - degs[2])          # R, which attains the ceiling
        hom_pct = 100 * (1 - degs[1])
        print(f"{n:>3}{N:>13}{cdz_pct:>9.1f}%{cls_pct:>11.1f}%{hom_pct:>9.1f}%{cdz_pct/cls_pct:>7.0f}x")
    print()


def n_distinct(values, tol=TOL):
    return round(len(values) * (1 - degeneracy(values, tol)))


if __name__ == "__main__":
    self_check()
    print(f"alphabet: C({ALPHABET[0]}), N({ALPHABET[1]}), O({ALPHABET[2]})\n")

    valid_rows, all_rows, ceilings = [], [], []
    for n in range(4, 8):
        graphs = generate_labelled_trees(n)
        valid = [G for G in graphs if respects_valence(G)]
        for target, gs in ((all_rows, graphs), (valid_rows, valid)):
            vals = [[f(G) for G in gs] for _, f in INDICES]
            target.append((n, len(gs), [degeneracy(v) for v in vals], [n_distinct(v) for v in vals]))
        ceilings.append(sum(1 for T in nx.nonisomorphic_trees(n) if max(d for _, d in T.degree()) <= 4))

    table(valid_rows, "TABLE 10 -- labellings respecting valence (C <= 4, N <= 3, O <= 2)")
    print("ceiling for a vertex-degree-based index = number of unlabelled trees with max degree <= 4: "
          + ", ".join(f"n={n}: {c}" for n, c in zip(range(4, 8), ceilings)) + "\n")
    table(all_rows, "for comparison -- all C/N/O labellings, valence not imposed")
