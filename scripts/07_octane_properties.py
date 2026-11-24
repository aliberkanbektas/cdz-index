"""
Correlation between topological indices and octane thermodynamic properties.
==============================================================================

Uses the SAME index set as the structural comparison table
(06_correlation_trees.py), so that every table in the paper compares
CDZ against a single fixed reference set rather than a different set per
table.

For a graph G = (V, E), u and v adjacent, d_u and d_v their degrees:

CDZ(G)      Degree-atomic number commensurability index (this work), reduced
            to its homogeneous-Z form (Proposition 2.4 of the paper):
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

Chemical properties (18 octane isomers, i.e. trees on 8 vertices with maximum
degree <= 4, one per structural isomer of C8H18):

AcenFac     Acentric factor                              (dimensionless)
DHvap       Standard enthalpy of vaporization, 25 C       (kcal/mol)
Entropy     Entropy                                       (J/mol.K)
Hvap        Enthalpy of vaporization, 25 C                (kJ/mol)
BoilingP    Normal boiling point                          (K)
Enthalpy    Enthalpy of formation, gas phase               (kJ/mol)

AcenFac, DHvap, Entropy and Hvap have been checked to reproduce, for SO/SDD/Tan,
the correlations published in Oz & Gutman (Hacet. J. Math. Stat. 54(6), 2025,
Table 1) to 3 decimal places; BoilingP and Enthalpy come from a different
source (NIST WebBook) and should be re-verified against a primary source
before publication.

What this script does
----------------------
1. Builds the hydrogen-suppressed carbon skeleton of each of the 18 octane
   isomers from its SMILES string, and checks all 18 are distinct trees on
   8 vertices with maximum degree <= 4.
2. Computes CDZ and each of the eleven comparison indices for every isomer.
3. Reports |Pearson r| between each index and each property.
"""

import itertools
from math import gcd, sqrt
import networkx as nx
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# 1. Octane isomers: SMILES and validated property data
#    (AcenFac, DHvap, Entropy, Hvap, BoilingP[K], Enthalpy[kJ/mol])
# ---------------------------------------------------------------------

OCTANES = [
    ("n-octane",                  "CCCCCCCC",         (0.397898, 9.915, 111.67, 73.19, 398.82, -208.75)),
    ("2-methylheptane",           "CC(C)CCCCC",       (0.377916, 9.484, 109.84, 70.30, 390.80, -215.30)),
    ("3-methylheptane",           "CCC(C)CCCC",       (0.371002, 9.521, 111.26, 71.30, 392.08, -212.50)),
    ("4-methylheptane",           "CCCC(C)CCC",       (0.371504, 9.483, 109.32, 70.91, 390.86, -211.90)),
    ("3-ethylhexane",             "CCC(CC)CCC",       (0.362472, 9.476, 109.43, 71.70, 391.69, -210.70)),
    ("2,2-dimethylhexane",        "CC(C)(C)CCCC",     (0.339426, 8.915, 103.42, 67.70, 379.99, -224.60)),
    ("2,3-dimethylhexane",        "CC(C)C(C)CCC",     (0.348671, 9.272, 108.02, 70.20, 388.76, -213.80)),
    ("2,4-dimethylhexane",        "CC(C)CC(C)CC",     (0.344223, 9.029, 106.98, 68.50, 382.58, -219.20)),
    ("2,5-dimethylhexane",        "CC(C)CCC(C)C",     (0.356830, 9.051, 105.72, 68.60, 382.26, -222.50)),
    ("3,3-dimethylhexane",        "CCC(C)(C)CCC",     (0.322596, 8.973, 104.74, 68.50, 385.12, -220.00)),
    ("3,4-dimethylhexane",        "CCC(C)C(C)CC",     (0.340345, 9.316, 106.59, 70.20, 390.88, -212.80)),
    ("3-ethyl-2-methylpentane",   "CC(C)C(CC)CC",     (0.332433, 9.209, 106.06, 69.70, 388.80, -211.00)),
    ("3-ethyl-3-methylpentane",   "CCC(C)(CC)CC",     (0.306899, 9.081, 101.48, 69.30, 391.41, -214.80)),
    ("2,2,3-trimethylpentane",    "CC(C)(C)C(C)CC",   (0.300816, 8.826, 101.31, 67.30, 382.99, -219.90)),
    ("2,2,4-trimethylpentane",    "CC(C)(C)CC(C)C",   (0.305370, 8.402, 104.09, 64.87, 372.39, -224.00)),
    ("2,3,3-trimethylpentane",    "CC(C)C(C)(C)CC",   (0.293177, 8.897, 102.06, 68.10, 387.91, -216.30)),
    ("2,3,4-trimethylpentane",    "CC(C)C(C)C(C)C",   (0.317422, 9.014, 102.39, 68.37, 386.62, -217.30)),
    ("2,2,3,3-tetramethylbutane", "CC(C)(C)C(C)(C)C", (0.255294, 8.410,  93.06, 66.20, 379.62, -225.60)),
]

PROPERTY_NAMES = ["AcenFac", "DHvap", "Entropy", "Hvap", "BoilingP", "Enthalpy"]


def smiles_to_graph(smiles):
    """Hydrogen-suppressed skeleton of a branched alkane (C, ( and ) only)."""
    G = nx.Graph()
    stack, current, node_id = [], None, 0
    for ch in smiles:
        if ch == "C":
            if current is not None:
                G.add_edge(current, node_id)
            else:
                G.add_node(node_id)
            current = node_id
            node_id += 1
        elif ch == "(":
            stack.append(current)
        elif ch == ")":
            current = stack.pop()
        else:
            raise ValueError(f"unsupported SMILES character: {ch!r}")
    return G


def check_structures(graphs):
    for (name, _, _), G in zip(OCTANES, graphs):
        assert G.number_of_nodes() == 8, f"{name}: not 8 carbons"
        assert nx.is_tree(G), f"{name}: not a tree"
        assert max(dict(G.degree()).values()) <= 4, f"{name}: degree > 4"
    for i, j in itertools.combinations(range(len(graphs)), 2):
        assert not nx.is_isomorphic(graphs[i], graphs[j]), \
            f"{OCTANES[i][0]} and {OCTANES[j][0]} are isomorphic"
    print(f"Structure check: {len(graphs)} isomers, all valid and distinct.  OK")


# ---------------------------------------------------------------------
# 2. Indices -- same set as 06_correlation_trees.py
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
    ("CDZ",        CDZ),
    ("R",          R),
    ("GA",         GA),
    ("ABC",        ABC),
    ("SO2",        SO2),
    ("M2",         M2),
    ("SDD",        SDD),
    ("SO",         SO),
    ("R_1/4",      R_alpha(0.25)),
    ("R_1/2",      R_alpha(0.5)),
    ("chi_-1",     chi(-1)),
    ("chi_1/4",    chi(0.25)),
]

# ---------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------

graphs = [smiles_to_graph(s) for _, s, _ in OCTANES]
props = [p for _, _, p in OCTANES]
check_structures(graphs)

index_vals = {name: [f(G) for G in graphs] for name, f in INDICES}

print("\nTable: The absolute values of Pearson correlation coefficients (|R|)")
print("between the indices and the indicated chemical properties of octane isomers.\n")
names = [n for n, _ in INDICES]
w, w0 = 9, 9
print(f"{'':<{w0}}" + "".join(f"{n:>{w}}" for n in names))
for k, prop in enumerate(PROPERTY_NAMES):
    y = [p[k] for p in props]
    row = f"{prop:<{w0}}"
    for name in names:
        row += f"{abs(pearsonr(index_vals[name], y)[0]):>{w}.5f}"
    print(row)
