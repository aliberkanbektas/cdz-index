"""
Variance in measured and computed properties explained by CDZ beyond size
(Section 5.6, the partial-R^2 table).

For each dataset and property Y, with a size covariate S (heavy-atom count
or molecular weight), the partial R^2 of a descriptor X is the squared
correlation between the residuals of X ~ S and of Y ~ S: the fraction of the
size-adjusted variance in Y that X explains.  It is reported for CDZ with the
true atomic numbers and for the control CDZ_hom (every Z set to 6), so that
the difference isolates the atomic-number weighting.  The partial F-test of
adding CDZ to a size-only model is also reported.

Datasets (see data/README.md for provenance):
  HER2        ChEMBL, EC50 set, RDKit descriptors      size: heavy-atom count
  large IC50  ChEMBL, IC50 set, RDKit descriptors      size: molecular weight
  E. coli     ChEMBL, IC50 set, RDKit descriptors      size: heavy-atom count
  ESOL        Delaney (2004), measured solubility       size: molecular weight
  Lipo        MoleculeNet, measured LogD                size: heavy-atom count
For E. coli, the analysis uses the 1082 molecules for which every descriptor
(including the molar volume) is present.

The script also re-validates the SMILES parser of cdz_core.py: on the large
IC50 set its CDZ must agree with the RDKit-based column 'classic_CDZ', and on
the E. coli set with the column 'CDZ'.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdz_core import cdz, parse_smiles  # noqa: E402

DATA = os.path.join(ROOT, "data")

# label, file, smiles column, property column, size column, expected (Z, Z=6), expected n
ROWS = [
    ("HER2",       "chembl_her2_ec50.csv",  "Smiles", "RDKit_LogP", "HAC",  (0.141, 0.025), 2589),
    ("large IC50", "chembl_large_ic50.csv", "Smiles", "MolLogP",    "MolWt", (0.097, 0.000), 10833),
    ("E. coli",    "chembl_ecoli_ic50.csv", "SMILES", "MolLogP",    "HAC",  (0.287, 0.123), 1082),
    ("HER2",       "chembl_her2_ec50.csv",  "Smiles", "RDKit_TPSA", "HAC",  (0.194, 0.095), 2589),
    ("E. coli",    "chembl_ecoli_ic50.csv", "SMILES", "TPSA",       "HAC",  (0.355, 0.162), 1082),
    ("ESOL",       "esol_delaney.csv",      "smiles", "measured log solubility in mols per litre",
                                                                  "Molecular Weight", (0.137, 0.040), 1128),
    ("Lipo",       "lipophilicity.csv",     "smiles", "exp",        "HAC",  (0.019, 0.012), 4200),
    ("large IC50", "chembl_large_ic50.csv", "Smiles", "MolMR",      "MolWt", (0.271, 0.319), 10833),
    ("E. coli",    "chembl_ecoli_ic50.csv", "SMILES", "MolarVolume", "HAC", (0.217, 0.292), 1082),
]
REQUIRED = {  # columns that must be present for a molecule to enter the analysis
    "chembl_large_ic50.csv": ["MolWt", "MolLogP", "TPSA", "MolMR", "classic_CDZ"],
    "chembl_ecoli_ic50.csv": ["MolLogP", "TPSA", "MolarVolume", "HeavyAtomCount", "MolWt"],
}


def partial_r2(x, y, z):
    xr = x - np.polyval(np.polyfit(z, x, 1), z)
    yr = y - np.polyval(np.polyfit(z, y, 1), z)
    return np.corrcoef(xr, yr)[0, 1] ** 2


def r2_multi(X, y):
    X1 = np.column_stack([X, np.ones(len(y))])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(fname, smiles_col):
    df = pd.read_csv(os.path.join(DATA, fname))
    keep, c_true, c_hom, hac = [], [], [], []
    for i, s in enumerate(df[smiles_col]):
        try:
            G = parse_smiles(str(s).strip())
        except Exception:
            continue
        keep.append(i)
        c_true.append(cdz(G))
        c_hom.append(cdz(G, use_Z=False))
        hac.append(G.number_of_nodes())
    df = df.iloc[keep].copy()
    df["CDZ_"], df["CDZhom_"], df["HAC"] = c_true, c_hom, hac
    for col in REQUIRED.get(fname, []):
        df = df[pd.to_numeric(df[col], errors="coerce").notna()]
    return df


def main():
    ok = True
    cache = {}
    print("Parser validation")
    ic = load("chembl_large_ic50.csv", "Smiles")
    d1 = np.abs(ic["CDZ_"].values - ic["classic_CDZ"].astype(float).values).max()
    ec = load("chembl_ecoli_ic50.csv", "SMILES")
    d2 = np.abs(ec["CDZ_"].values - ec["CDZ"].astype(float).values).max()
    v = d1 < 1e-9 and d2 < 1e-9
    ok &= v
    print(f"  [{'OK' if v else 'FAIL'}] max |CDZ - RDKit-based CDZ|: large IC50 {d1:.1e} "
          f"({len(ic)} molecules), E. coli {d2:.1e} ({len(ec)} molecules)\n")
    cache["chembl_large_ic50.csv"], cache["chembl_ecoli_ic50.csv"] = ic, ec

    print(f"{'dataset':11s}{'property':14s}{'n':>7s}{'size':>8s}{'pR2 (Z)':>10s}{'pR2 (Z=6)':>11s}"
          f"{'F (CDZ)':>10s}{'p':>10s}")
    print("-" * 81)
    for label, fname, scol, ycol, scov, (ez, eh), en in ROWS:
        df = cache.get(fname)
        if df is None:
            df = cache[fname] = load(fname, scol)
        y = df[ycol].astype(float).values
        z = df[scov].astype(float).values
        x, xh = df["CDZ_"].values, df["CDZhom_"].values
        pz, ph = partial_r2(x, y, z), partial_r2(xh, y, z)
        n = len(y)
        rb = r2_multi(z.reshape(-1, 1), y)
        rf = r2_multi(np.column_stack([z, x]), y)
        F = (rf - rb) / ((1 - rf) / (n - 3))
        p = fdist.sf(F, 1, n - 3)
        row_ok = (round(pz, 3), round(ph, 3), n) == (ez, eh, en)
        ok &= row_ok
        short = {"measured log solubility in mols per litre": "log S", "RDKit_LogP": "LogP",
                 "RDKit_TPSA": "TPSA", "MolLogP": "LogP", "exp": "LogD", "MolMR": "MR"}.get(ycol, ycol)
        size = "MW" if scov in ("MolWt", "Molecular Weight") else "HAC"
        print(f"{label:11s}{short:14s}{n:7d}{size:>8s}{pz:10.3f}{ph:11.3f}{F:10.1f}{p:10.1e}"
              f"  {'OK' if row_ok else f'MISMATCH (paper {ez}, {eh}, n={en})'}")
    print("\nThe first seven rows are polarity-related; in each the true atomic numbers explain")
    print("more size-adjusted variance than the homogeneous control. The last two rows (molar")
    print("refractivity, molar volume) reverse that order.")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
