"""
Separating structures versus separating them in the right direction
(Section 5.7, the isosteric ranking table).

Two molecules are *isosteric* here when their hydrogen-suppressed graphs are
isomorphic as unlabelled graphs.  Every degree-based index gives both members
of such a pair the same value; CDZ need not.  For every isosteric pair that
CDZ separates and whose measured property differs, the pair counts as
correctly ordered when the CDZ difference and the property difference have
the same sign.  A two-sided binomial test against 50 % is reported.

Three nested families of pairs are tested:
  all        every isosteric pair;
  formula    pairs with the same molecular formula (hydrogens included),
             i.e. true constitutional isomers -- the family reported in the
             paper as "true positional isomers";
  heavy      pairs with the same multiset of heavy atoms, hydrogen count not
             compared (a looser family, reported for transparency).

Molecular formulas come from cdz_core.molecular_formula, which agrees with
RDKit's molecular weight on all 1128 ESOL molecules (checked below).
"""

import csv
import itertools
import os
import sys
import warnings
from collections import defaultdict

import networkx as nx
from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdz_core import cdz, molecular_formula, molecular_weight, parse_smiles  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
DATA = os.path.join(ROOT, "data")

DATASETS = [
    ("ESOL", "esol_delaney.csv", "smiles", "measured log solubility in mols per litre"),
    ("Lipophilicity", "lipophilicity.csv", "smiles", "exp"),
]
# (dataset, family) -> (pairs, % correct, p) as reported in the paper
EXPECTED = {
    ("ESOL", "all"): (2124, 42.8, 3e-11), ("Lipophilicity", "all"): (1073, 55.7, 2e-4),
    ("ESOL", "formula"): (50, 54.0, 0.67), ("Lipophilicity", "formula"): (82, 52.4, 0.74),
    ("ESOL", "heavy"): (81, 56.8, 0.27), ("Lipophilicity", "heavy"): (89, 53.9, 0.53),
}


def unlabelled_hash(G):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    return nx.weisfeiler_lehman_graph_hash(H, iterations=5)


def load(fname, scol, pcol):
    out = []
    with open(os.path.join(DATA, fname)) as fh:
        for r in csv.DictReader(fh):
            smi = r[scol].strip()
            try:
                G = parse_smiles(smi)
                if G.number_of_nodes() < 2:
                    continue
                y = float(r[pcol])
            except Exception:
                continue
            out.append({"G": G, "y": y, "cdz": cdz(G), "hash": unlabelled_hash(G),
                        "heavy": tuple(sorted(G.nodes[v]["Z"] for v in G)),
                        "formula": tuple(sorted(molecular_formula(smi).items()))})
    return out


def ranking(entries, family):
    groups = defaultdict(list)
    for e in entries:
        groups[e["hash"]].append(e)
    pairs = []
    for members in groups.values():
        for a, b in itertools.combinations(members, 2):
            if not nx.is_isomorphic(nx.Graph(a["G"].edges()), nx.Graph(b["G"].edges())):
                continue
            if family == "heavy" and a["heavy"] != b["heavy"]:
                continue
            if family == "formula" and a["formula"] != b["formula"]:
                continue
            if abs(a["cdz"] - b["cdz"]) < 1e-9 or abs(a["y"] - b["y"]) < 1e-9:
                continue
            pairs.append((a, b))
    correct = sum(1 for a, b in pairs if (a["cdz"] - b["cdz"]) * (a["y"] - b["y"]) > 0)
    n = len(pairs)
    return correct, n, binomtest(correct, n, 0.5).pvalue


def main():
    ok = True
    # molecular-formula check against the molecular weights shipped with ESOL
    with open(os.path.join(DATA, "esol_delaney.csv")) as fh:
        rows = list(csv.DictReader(fh))
    worst = max(abs(molecular_weight(r["smiles"].strip()) - float(r["Molecular Weight"])) for r in rows)
    v = worst < 0.05
    ok &= v
    print(f"[{'OK' if v else 'FAIL'}] molecular formula: max |MW - ESOL molecular weight| = "
          f"{worst:.3f} over {len(rows)} molecules\n")

    print(f"{'dataset':15s}{'pairs':10s}{'n':>6s}{'correct':>10s}{'p':>11s}")
    print("-" * 52)
    for label, fname, scol, pcol in DATASETS:
        entries = load(fname, scol, pcol)
        for family in ("all", "formula", "heavy"):
            c, n, p = ranking(entries, family)
            en, epct, ep = EXPECTED[(label, family)]
            p_ok = abs(p - ep) <= 0.006 if ep >= 0.01 else abs(p - ep) <= 0.2 * ep
            row_ok = n == en and round(100 * c / n, 1) == epct and p_ok
            ok &= row_ok
            print(f"{label:15s}{family:10s}{n:6d}{100 * c / n:9.1f}%{p:11.2g}"
                  f"  {'OK' if row_ok else f'MISMATCH (paper n={en}, {epct}%, p={ep})'}")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
