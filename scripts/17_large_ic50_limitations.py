"""
Limitations on a carbon-rich set (Section 6, items 2 and 7).

On the large ChEMBL IC50 set (10 833 molecules) this script reproduces
  * leave-one-out Q^2 for molar refractivity (RDKit MolMR) from CDZ, the
    Randic index, the ABC index and the heavy-atom count;
  * the two nested model comparisons CDZ -> CDZ + Randic and
    Randic -> Randic + CDZ;
  * the comparison after controlling for size: the partial R^2 of MR on
    each index given the heavy-atom count, and the number of heavy-atom-count
    strata (with at least 100 molecules) in which the Randic index explains
    more of the variance in MR than CDZ;
  * leave-one-out Q^2 for pIC50 = 9 - log10(IC50 / nM) from each descriptor.

For a linear model the leave-one-out residual is e_i / (1 - h_ii), so Q^2 is
computed exactly from the hat matrix without refitting.
"""

import csv
import math
import os
import sys

import numpy as np
from scipy.stats import f as fdist

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdz_core import cdz, parse_smiles  # noqa: E402

EXPECTED_Q2_MR = {"CDZ": 0.792, "Randic": 0.969, "ABC": 0.961, "HAC": 0.967}


def randic(G):
    d = dict(G.degree())
    return sum(1 / math.sqrt(d[u] * d[v]) for u, v in G.edges())


def abc(G):
    d = dict(G.degree())
    return sum(math.sqrt((d[u] + d[v] - 2) / (d[u] * d[v])) for u, v in G.edges())


def fit(X, y):
    """Return (R^2, LOO Q^2) of the least-squares fit y ~ X + const."""
    X1 = np.column_stack([X, np.ones(len(y))])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta
    h = np.einsum("ij,jk,ik->i", X1, np.linalg.pinv(X1.T @ X1), X1)
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - (res ** 2).sum() / sst, 1 - ((res / (1 - h)) ** 2).sum() / sst


def main():
    rows = []
    with open(os.path.join(ROOT, "data", "chembl_large_ic50.csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                G = parse_smiles(r["Smiles"])
                vals = [float(r[k]) for k in ("MolWt", "MolLogP", "TPSA", "MolMR", "classic_CDZ")]
            except Exception:
                continue
            ic50 = float(r["Standard Value"]) if r["Standard Value"] else float("nan")
            rows.append((cdz(G), randic(G), abc(G), G.number_of_nodes(), vals[3], ic50))
    A = np.array(rows, dtype=float)
    n = len(A)
    ok = n == 10833
    print(f"[{'OK' if ok else 'FAIL'}] molecules analysed: {n} (paper: 10833)\n")

    mr = A[:, 4]
    print("Molar refractivity, single-descriptor models")
    for i, name in enumerate(("CDZ", "Randic", "ABC", "HAC")):
        r2, q2 = fit(A[:, [i]], mr)
        row_ok = round(q2, 3) == EXPECTED_Q2_MR[name]
        ok &= row_ok
        print(f"  {name:7s} R2 = {r2:.4f}  Q2_LOO = {q2:.4f}  {'OK' if row_ok else 'MISMATCH'}")

    r_c, _ = fit(A[:, [0]], mr)
    r_r, _ = fit(A[:, [1]], mr)
    r_cr, _ = fit(A[:, [0, 1]], mr)
    F1 = (r_cr - r_c) / ((1 - r_cr) / (n - 3))
    F2 = (r_cr - r_r) / ((1 - r_cr) / (n - 3))
    p2 = fdist.sf(F2, 1, n - 3)
    c_ok = (round(r_c, 3), round(r_cr, 3)) == (0.792, 0.969) and round(r_r, 4) == round(r_cr, 4) \
        and round(p2, 3) == 0.026
    ok &= c_ok
    print(f"\n  CDZ -> CDZ + Randic:    R2 {r_c:.4f} -> {r_cr:.4f}  (F = {F1:.0f})")
    print(f"  Randic -> Randic + CDZ: R2 {r_r:.4f} -> {r_cr:.4f}  (F = {F2:.2f}, p = {p2:.3f})"
          f"  {'OK' if c_ok else 'MISMATCH'}")

    def partial_r2(x, y, z):
        xr = x - np.polyval(np.polyfit(z, x, 1), z)
        yr = y - np.polyval(np.polyfit(z, y, 1), z)
        return np.corrcoef(xr, yr)[0, 1] ** 2

    pc, pr = partial_r2(A[:, 0], mr, A[:, 3]), partial_r2(A[:, 1], mr, A[:, 3])
    print(f"\n  Controlling for heavy-atom count: partial R2 of MR on CDZ = {pc:.3f}, on Randic = {pr:.3f}")
    print("  Within strata of equal heavy-atom count (strata with at least 100 molecules):")
    better = total = 0
    for hac in sorted(set(A[:, 3].astype(int))):
        s = A[A[:, 3] == hac]
        if len(s) < 100:
            continue
        rc = np.corrcoef(s[:, 0], s[:, 4])[0, 1] ** 2
        rr = np.corrcoef(s[:, 1], s[:, 4])[0, 1] ** 2
        total += 1
        better += rr > rc
    s_ok = (round(pc, 3), round(pr, 3)) == (0.008, 0.054) and (better, total) == (19, 28)
    ok &= s_ok
    print(f"  Randic explains more MR variance than CDZ in {better} of {total} strata"
          f"  {'OK' if s_ok else 'MISMATCH'}")

    good = ~np.isnan(A[:, 5]) & (A[:, 5] > 0)
    pic50 = 9 - np.log10(A[good, 5])
    print(f"\npIC50 (n = {good.sum()}), single-descriptor models")
    q2s = []
    for i, name in enumerate(("CDZ", "Randic", "ABC", "HAC")):
        _, q2 = fit(A[good][:, [i]], pic50)
        q2s.append(q2)
        print(f"  {name:7s} Q2_LOO = {q2:.4f}")
    p_ok = max(q2s) <= 0.073
    ok &= p_ok
    print(f"  [{'OK' if p_ok else 'FAIL'}] largest Q2 = {max(q2s):.4f} <= 0.073")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
