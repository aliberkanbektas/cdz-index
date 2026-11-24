"""
The 45-compound aliphatic set: alcohols, ethers and amines with normal
boiling points (Section 5.6, the single-descriptor and partial-F tables).

Reproduces
  * R^2, leave-one-out Q^2 and RMSE of single-descriptor linear models for
    CDZ, the Randic index, the ABC index and the heavy-atom count;
  * external validation over 20 random 70/30 splits (seeds 0..19): mean
    R^2_ext and the Tropsha regression-through-origin slopes k and k';
  * Y-randomisation over 200 permutations (seed 42);
  * the partial F-tests of Table "partialF";
  * the isosteric triples discussed in the text.

RMSE is sqrt(SS_res / n).  Data: data/aliphatic45.csv; every boiling point
is checked against the NIST Chemistry WebBook by 18_aliphatic45_vs_nist.py.

Notes on two corrections.  An earlier encoding of this set represented
2-methyl-1-butanol and 3-methyl-2-butanol by the graph of 3-methyl-1-butanol,
and gave methyl isopropyl ether a boiling point of 32.5 C, outside the WebBook
value of 303.9 K (30.75 C); the data file now has 30.8 C.  Set
USE_EARLIER_ENCODING = True to reproduce the numbers of the earlier version
exactly.
"""

import csv
import math
import os
import random
import sys

import numpy as np
from scipy.stats import f as fdist

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdz_core import cdz, parse_smiles  # noqa: E402

USE_EARLIER_ENCODING = False

EXPECTED = {   # corrected structures and boiling points
    "CDZ": (0.703, 0.678, 26.38), "Randic": (0.589, 0.546, 31.04),
    "ABC": (0.515, 0.450, 33.73), "HAC": (0.584, 0.537, 31.24),
    "ext": (0.673, 1.018, 0.910), "yrand": (0.022, 0.253),
    "F": {"Randic -> Randic + CDZ": (16.42, 2.1e-4, 0.580, 0.691),
          "ABC -> ABC + CDZ": (26.79, 6.0e-6, 0.504, 0.690),
          "CDZ -> CDZ + Randic": (0.19, 0.67, 0.696, 0.691)},
}
if USE_EARLIER_ENCODING:
    EXPECTED = {
        "CDZ": (0.714, 0.689, 25.87), "Randic": (0.591, 0.548, 30.95),
        "ABC": (0.516, 0.452, 33.65), "HAC": (0.584, 0.538, 31.19),
        "ext": (0.678, 1.020, 0.911), "yrand": (0.022, 0.254),
        "F": {"Randic -> Randic + CDZ": (18.75, 9.0e-5, 0.581, 0.704),
              "ABC -> ABC + CDZ": (29.08, 3.0e-6, 0.505, 0.701),
              "CDZ -> CDZ + Randic": (0.46, 0.50, 0.707, 0.704)},
    }


def randic(G):
    d = dict(G.degree())
    return sum(1 / math.sqrt(d[u] * d[v]) for u, v in G.edges())


def abc(G):
    d = dict(G.degree())
    return sum(math.sqrt((d[u] + d[v] - 2) / (d[u] * d[v])) for u, v in G.edges())


def linreg(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((p - mx) ** 2 for p in x)
    a = sum((p - mx) * (q - my) for p, q in zip(x, y)) / sxx
    b = my - a * mx
    ss_res = sum((q - (a * p + b)) ** 2 for p, q in zip(x, y))
    ss_tot = sum((q - my) ** 2 for q in y)
    return a, b, 1 - ss_res / ss_tot, math.sqrt(ss_res / n)


def q2_loo(x, y):
    n = len(x)
    my = sum(y) / n
    press = 0.0
    for i in range(n):
        a, b, _, _ = linreg(x[:i] + x[i + 1:], y[:i] + y[i + 1:])
        press += (y[i] - (a * x[i] + b)) ** 2
    return 1 - press / sum((q - my) ** 2 for q in y)


def external(x, y, seeds=range(20)):
    out = []
    for seed in seeds:
        idx = list(range(len(x)))
        random.Random(seed).shuffle(idx)
        ntr = int(round(0.7 * len(idx)))
        tr, te = idx[:ntr], idx[ntr:]
        a, b, _, _ = linreg([x[i] for i in tr], [y[i] for i in tr])
        yp = [a * x[i] + b for i in te]
        yo = [y[i] for i in te]
        mtr = sum(y[i] for i in tr) / len(tr)
        press = sum((o - p) ** 2 for o, p in zip(yo, yp))
        r2ext = 1 - press / sum((o - mtr) ** 2 for o in yo)
        k = sum(o * p for o, p in zip(yo, yp)) / sum(p * p for p in yp)
        kp = sum(o * p for o, p in zip(yo, yp)) / sum(o * o for o in yo)
        out.append((r2ext, k, kp))
    return [sum(c) / len(out) for c in zip(*out)]


def y_randomisation(x, y, trials=200, seed=42):
    rnd = random.Random(seed)
    ys = list(y)
    tot = best = 0.0
    for _ in range(trials):
        rnd.shuffle(ys)
        r2 = linreg(x, ys)[2]
        tot += r2
        best = max(best, r2)
    return tot / trials, best


def r2_multi(X, y):
    X1 = np.column_stack([X, np.ones(len(y))])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    with open(os.path.join(ROOT, "data", "aliphatic45.csv")) as fh:
        rows = list(csv.DictReader(fh))
    names = [r["name"] for r in rows]
    bp = [float(r["bp_C"]) for r in rows]
    graphs = [parse_smiles(r["smiles"]) for r in rows]
    if USE_EARLIER_ENCODING:
        for nm in ("2-methyl-1-butanol", "3-methyl-2-butanol"):
            graphs[names.index(nm)] = parse_smiles("CC(C)CCO")
        bp[names.index("methyl isopropyl ether")] = 32.5
    X = {"CDZ": [cdz(G) for G in graphs], "Randic": [randic(G) for G in graphs],
         "ABC": [abc(G) for G in graphs], "HAC": [float(G.number_of_nodes()) for G in graphs]}
    ok = True

    print(f"Single-descriptor models for the normal boiling point (n = {len(bp)})\n")
    print(f"{'descriptor':12s}{'R2':>8s}{'Q2_LOO':>9s}{'RMSE':>8s}")
    for k in ("CDZ", "Randic", "ABC", "HAC"):
        _, _, r2, rmse = linreg(X[k], bp)
        q2 = q2_loo(X[k], bp)
        row_ok = (round(r2, 3), round(q2, 3), round(rmse, 2)) == EXPECTED[k]
        ok &= row_ok
        print(f"{k:12s}{r2:8.3f}{q2:9.3f}{rmse:8.2f}  {'OK' if row_ok else 'MISMATCH'}")

    r2ext, k, kp = external(X["CDZ"], bp)
    e_ok = (round(r2ext, 3), round(k, 3), round(kp, 3)) == EXPECTED["ext"]
    ok &= e_ok
    print(f"\nExternal validation (20 splits): R2_ext = {r2ext:.3f}, k = {k:.3f}, k' = {kp:.3f}"
          f"  {'OK' if e_ok else 'MISMATCH'}")
    m, M = y_randomisation(X["CDZ"], bp)
    y_ok = (round(m, 3), round(M, 3)) == EXPECTED["yrand"]
    ok &= y_ok
    print(f"Y-randomisation (200): mean R2 = {m:.3f}, max R2 = {M:.3f}  {'OK' if y_ok else 'MISMATCH'}")

    print("\nPartial F-tests")
    y = np.array(bp)
    c, R, A = (np.array(X[k]) for k in ("CDZ", "Randic", "ABC"))
    n = len(y)
    adj = lambda r2, p: 1 - (1 - r2) * (n - 1) / (n - p - 1)   # noqa: E731
    for label, base, extra in (("Randic -> Randic + CDZ", R, c), ("ABC -> ABC + CDZ", A, c),
                               ("CDZ -> CDZ + Randic", c, R)):
        rb = r2_multi(base.reshape(-1, 1), y)
        rf = r2_multi(np.column_stack([base, extra]), y)
        F = (rf - rb) / ((1 - rf) / (n - 3))
        p = fdist.sf(F, 1, n - 3)
        eF, ep, ea, eb = EXPECTED["F"][label]
        row_ok = (round(F, 2) == eF and abs(p - ep) <= 0.05 * ep + 0.006
                  and (round(adj(rb, 1), 3), round(adj(rf, 2), 3)) == (ea, eb))
        ok &= row_ok
        print(f"  {label:24s} F(1,{n - 3}) = {F:6.2f}  p = {p:.1e}  adj R2 {adj(rb, 1):.3f} -> "
              f"{adj(rf, 2):.3f}  {'OK' if row_ok else 'MISMATCH'}")

    print("\nIsosteric triples (identical hydrogen-suppressed graphs)")
    for trio in (("1-propanol", "propylamine", "methyl ethyl ether"),
                 ("1-butanol", "butylamine", "diethyl ether"),
                 ("2-propanol", "isopropylamine")):
        idx = [names.index(t) for t in trio]
        cz = [X["CDZ"][i] for i in idx]
        b = [bp[i] for i in idx]
        same = sorted(range(len(idx)), key=lambda i: -cz[i]) == sorted(range(len(idx)), key=lambda i: -b[i])
        print("  " + " / ".join(f"{t} {z:.3f} ({v:g} C)" for t, z, v in zip(trio, cz, b))
              + f"   R = {X['Randic'][idx[0]]:.3f}, ABC = {X['ABC'][idx[0]]:.3f} for all;"
              f" CDZ order {'matches' if same else 'does not match'} boiling points")
        ok &= same

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
