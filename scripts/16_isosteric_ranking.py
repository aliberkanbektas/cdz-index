"""
Separating structures versus separating them in the right direction
(Section 5.7, the isosteric ranking table).

Two molecules are *isosteric* here when their hydrogen-suppressed graphs are
isomorphic as unlabelled graphs.  Every degree-based index gives both members
of such a pair the same value; CDZ need not.  For every isosteric pair that
CDZ separates and whose measured property differs, the pair counts as
correctly ordered when the CDZ difference and the property difference have
the same sign.

Pairs are not independent: a molecule with k isosteric partners enters k
pairs, so a binomial test against 50 % overstates the significance.  The
p value used in the paper is therefore a permutation test that keeps this
dependence: within each group of mutually isosteric molecules (refined by the
family below) the measured values are permuted at random, which leaves the
group structure and every CDZ value unchanged and destroys only the link
between CDZ order and property, and the fraction of correctly ordered pairs is
recomputed.  The p value is two-sided, from 20 000 permutations with a fixed
seed.  The binomial p value is printed alongside for comparison.

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
import numpy as np
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
N_PERM = 20000
SEED = 0
# (dataset, family) -> (pairs, % correct, binomial p, permutation p) as in the paper
EXPECTED = {
    ("ESOL", "all"): (2124, 42.8, 3e-11, 7.5e-4), ("Lipophilicity", "all"): (1073, 55.7, 2e-4, 0.0035),
    ("ESOL", "formula"): (50, 54.0, 0.67, 0.69), ("Lipophilicity", "formula"): (82, 52.4, 0.74, 0.68),
    ("ESOL", "heavy"): (81, 56.8, 0.27, 0.34), ("Lipophilicity", "heavy"): (89, 53.9, 0.53, 0.52),
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


def groups_of(entries, family):
    """Groups of mutually isosteric molecules, refined by the family key."""
    buckets = defaultdict(list)
    for e in entries:
        buckets[e["hash"]].append(e)
    out = []
    for members in buckets.values():
        iso = []
        for e in members:
            for c in iso:
                if nx.is_isomorphic(nx.Graph(c[0]["G"].edges()), nx.Graph(e["G"].edges())):
                    c.append(e)
                    break
            else:
                iso.append([e])
        for c in iso:
            sub = defaultdict(list)
            for e in c:
                sub[e[family] if family in ("formula", "heavy") else 0].append(e)
            out += [g for g in sub.values() if len(g) > 1]
    return out


def permutation_test(entries, family, rng):
    """Two-sided within-group permutation p value; also pairs, molecules, groups."""
    obs_c = obs_n = 0
    data, mols = [], set()
    for g in groups_of(entries, family):
        cz = np.array([e["cdz"] for e in g])
        y = np.array([e["y"] for e in g])
        I, J = np.triu_indices(len(g), 1)
        keep = np.abs(cz[I] - cz[J]) >= 1e-9
        I, J = I[keep], J[keep]
        if len(I) == 0:
            continue
        sc = np.sign(cz[I] - cz[J])
        dy = y[I] - y[J]
        ok = np.abs(dy) >= 1e-9
        if not ok.any():
            continue
        obs_c += int(((sc * dy > 0) & ok).sum())
        obs_n += int(ok.sum())
        mols.update((id(g), int(k)) for k in np.concatenate([I[ok], J[ok]]))
        data.append((y, I, J, sc))
    corr = np.zeros(N_PERM)
    tot = np.zeros(N_PERM)
    for y, I, J, sc in data:
        yp = y[np.argsort(rng.random((N_PERM, len(y))), axis=1)]
        dy = yp[:, I] - yp[:, J]
        ok = np.abs(dy) >= 1e-9
        corr += ((sc * dy > 0) & ok).sum(1)
        tot += ok.sum(1)
    f = corr / tot
    fo = obs_c / obs_n
    p = (np.sum(np.abs(f - f.mean()) >= abs(fo - f.mean()) - 1e-12) + 1) / (N_PERM + 1)
    n_mol = len(mols)
    return obs_n, n_mol, len(data), p


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

    rng = np.random.default_rng(SEED)
    print(f"{'dataset':15s}{'pairs':10s}{'n':>6s}{'mols':>6s}{'groups':>7s}{'correct':>9s}"
          f"{'p (perm)':>10s}{'p (binom)':>11s}")
    print("-" * 74)

    def close(p, ep):
        return abs(p - ep) <= 0.006 if ep >= 0.01 else abs(p - ep) <= 0.2 * ep

    for label, fname, scol, pcol in DATASETS:
        entries = load(fname, scol, pcol)
        for family in ("all", "formula", "heavy"):
            c, n, p = ranking(entries, family)
            n2, n_mol, n_grp, pp = permutation_test(entries, family, rng)
            en, epct, ep, epp = EXPECTED[(label, family)]
            pp_ok = abs(pp - epp) <= 0.006 if epp >= 0.01 else abs(pp - epp) <= 0.1 * epp
            row_ok = n == en == n2 and round(100 * c / n, 1) == epct and close(p, ep) and pp_ok
            ok &= row_ok
            print(f"{label:15s}{family:10s}{n:6d}{n_mol:6d}{n_grp:7d}{100 * c / n:8.1f}%{pp:10.2g}{p:11.2g}"
                  f"  {'OK' if row_ok else f'MISMATCH (paper n={en}, {epct}%, p={epp})'}")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
