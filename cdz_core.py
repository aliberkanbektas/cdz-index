"""
Core definitions shared by the scripts in this repository.

The degree-atomic number commensurability index of a molecular graph G is

    CDZ(G) = sum over edges uv of  gcd(a_u, a_v) / sqrt(a_u * a_v),
    a_u    = d_u * Z_u,

where d_u is the degree of u in the hydrogen-suppressed graph and Z_u the
atomic number of the atom at u.  CDZ_hom is the same quantity with every
Z_u set to 6, which isolates the contribution of the atomic numbers.

parse_smiles() builds the hydrogen-suppressed graph of a SMILES string
without RDKit.  It was validated against an RDKit-based computation of CDZ
on 10 833 molecules (maximum absolute difference 0); see
scripts/12_partial_r2.py, which repeats that check on every run.
"""

import re
from math import gcd, sqrt

import networkx as nx


def w(a, b):
    """Contribution of one edge whose endpoint weights are a and b."""
    return gcd(a, b) / sqrt(a * b)


def cdz(G, use_Z=True):
    """CDZ of a graph whose nodes carry an atomic number in the attribute 'Z'.

    Nodes without a 'Z' attribute are treated as carbon, so a plain
    networkx graph gives the homogeneous (all-carbon) value.
    """
    d = dict(G.degree())
    a = {v: d[v] * (G.nodes[v].get("Z", 6) if use_Z else 6) for v in G}
    return sum(w(a[u], a[v]) for u, v in G.edges())


def cdz_hom(G):
    """CDZ with every atomic number set to 6."""
    return cdz(G, use_Z=False)


# ---------------------------------------------------------------------------
# SMILES -> hydrogen-suppressed graph
# ---------------------------------------------------------------------------

Z = {
    'H': 1, 'Li': 3, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11,
    'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'K': 19,
    'Ca': 20, 'Fe': 26, 'Zn': 30, 'As': 33, 'Se': 34, 'Br': 35, 'I': 53,
}


def parse_smiles(s):
    """
    Minimal SMILES parser: builds the heavy-atom skeleton (a networkx Graph
    with each node's element symbol stored as 'Z'). Handles multi-atom
    brackets, two-letter symbols (Cl, Br, Na, ...), aromatic lowercase atoms
    (treated as their element -- aromaticity does not change atomic number),
    ring closures (single-digit and %nn), branches, and bond symbols.
    Does not model charge, isotopes or stereochemistry beyond skipping them.
    """
    G = nx.Graph()
    stack, prev, node_id, rings = [], None, 0, {}
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '(':
            stack.append(prev); i += 1; continue
        if ch == ')':
            prev = stack.pop(); i += 1; continue
        if ch in '-=#$:/\\':
            i += 1; continue
        if ch == '.':
            prev = None; i += 1; continue
        if ch == '[':
            j = s.index(']', i)
            tok = s[i + 1:j]
            m = re.match(r'\d*([A-Za-z][a-z]?)', tok)
            sym = m.group(1)
            sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
            i = j + 1
        elif s[i:i + 2] in ('Cl', 'Br', 'Na', 'Si', 'Se', 'Fe', 'Zn', 'Al',
                             'Ca', 'Mg', 'Li', 'As', 'nH'):
            two = s[i:i + 2]
            sym = 'N' if two == 'nH' else two
            i += 2
        elif ch.isalpha():
            sym = ch.upper()
            i += 1
        elif ch.isdigit():
            key = ch
            if key in rings:
                G.add_edge(rings.pop(key), prev)
            else:
                rings[key] = prev
            i += 1
            continue
        elif ch == '%':
            key = s[i + 1:i + 3]
            if key in rings:
                G.add_edge(rings.pop(key), prev)
            else:
                rings[key] = prev
            i += 3
            continue
        else:
            i += 1
            continue
        if sym not in Z:
            raise ValueError(f"unknown element symbol: {sym!r}")
        G.add_node(node_id, Z=Z[sym])
        if prev is not None:
            G.add_edge(prev, node_id)
        prev = node_id
        node_id += 1
    return G


# ---------------------------------------------------------------------------
# Molecular formula from SMILES (used to identify true constitutional isomers)
# ---------------------------------------------------------------------------

_ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
_AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}
_VALENCES = {"B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5),
             "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,)}
_BOND = {"-": 1, "=": 2, "#": 3, "$": 4, ":": 1, "/": 1, "\\": 1}
MASS = {"H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007,
        "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
        "Si": 28.085, "P": 30.974, "S": 32.065, "Cl": 35.453, "K": 39.098,
        "Ca": 40.078, "Fe": 55.845, "Zn": 65.38, "As": 74.922, "Se": 78.971,
        "Br": 79.904, "Te": 127.60, "I": 126.904}


def molecular_formula(s):
    """Element counts, hydrogens included, following the OpenSMILES rules.

    Implicit hydrogens of an organic-subset atom outside brackets are the
    smallest normal valence not below the sum of its bond orders, minus that
    sum; an aromatic atom (lower case) receives max(0, lowest valence - sum
    of bond orders - 1), aromatic bonds counting as 1.  Bracket atoms carry
    exactly the hydrogens written inside the bracket.  Returns a dict.
    """
    atoms = []            # [element, aromatic, explicit H or None]
    order = []            # sum of bond orders per atom
    stack, prev, rings, pending = [], None, {}, None
    i = 0

    def bond(a, b, sym):
        o = _BOND[sym] if sym else 1
        order[a] += o
        order[b] += o

    while i < len(s):
        ch = s[i]
        if ch == "(":
            stack.append(prev); i += 1; continue
        if ch == ")":
            prev = stack.pop(); i += 1; continue
        if ch == ".":
            prev = None; i += 1; continue
        if ch in _BOND:
            pending = ch; i += 1; continue
        if ch.isdigit() or ch == "%":
            key = s[i + 1:i + 3] if ch == "%" else ch
            i += 3 if ch == "%" else 1
            if key in rings:
                j, sym = rings.pop(key)
                bond(prev, j, pending or sym)
            else:
                rings[key] = (prev, pending)
            pending = None
            continue
        if ch == "[":
            j = s.index("]", i)
            tok = s[i + 1:j]
            m = re.match(r"(\d*)([A-Z][a-z]?|[a-z][a-z]?)(.*)", tok)
            sym, rest = m.group(2), m.group(3)
            arom = sym[0].islower()
            el = sym.capitalize() if len(sym) > 1 or not arom else sym.upper()
            if arom and len(sym) == 1:
                el = _AROMATIC[sym]
            rest = re.sub(r"@+(TH|AL|SP|TB|OH)?\d*", "", rest)
            hm = re.match(r"H(\d*)", rest)
            nh = (int(hm.group(1)) if hm.group(1) else 1) if hm else 0
            atoms.append([el, arom, nh])
            i = j + 1
        else:
            two = s[i:i + 2]
            if two in ("Cl", "Br"):
                el, arom = two, False; i += 2
            elif ch in _AROMATIC:
                el, arom = _AROMATIC[ch], True; i += 1
            else:
                el, arom = ch, False; i += 1
            atoms.append([el, arom, None])
        order.append(0)
        k = len(atoms) - 1
        if prev is not None:
            bond(prev, k, pending)
        pending = None
        prev = k

    counts = {}
    for (el, arom, nh), v in zip(atoms, order):
        counts[el] = counts.get(el, 0) + 1
        if nh is None:
            vals = _VALENCES.get(el, (0,))
            if arom:
                nh = max(0, vals[0] - v - 1)
            else:
                target = next((x for x in vals if x >= v), None)
                nh = 0 if target is None else target - v
        counts["H"] = counts.get("H", 0) + nh
    return counts


def molecular_weight(s):
    return sum(MASS[el] * c for el, c in molecular_formula(s).items())
