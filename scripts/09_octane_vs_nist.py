"""
Verification of the octane boiling-point and enthalpy-of-formation columns
against the NIST Chemistry WebBook.
==========================================================================

The six octane property columns used in the paper have two different
provenances:

  AcenFac, DHvap, Entropy, Hvap   validated by reproducing, for SO/SDD/Tan, the
                                  correlations published in Oz & Gutman,
                                  Hacet. J. Math. Stat. 54(6) (2025), Table 1,
                                  to three decimal places.

  BoilingP, Enthalpy              a separate compilation; verified here, value
                                  by value, against the recommended values of
                                  the NIST Chemistry WebBook (retrieved
                                  2026-10-24).

The NIST values are stored alongside this script in data/octane_nist_boiling_points.tsv and
data/octane_nist_enthalpies.tsv, each row carrying the species name, CAS number, the value used in
the paper, and the WebBook's recommended value with its stated uncertainty.
Every WebBook enthalpy traces to the combustion calorimetry of Prosen and
Rossini, J. Res. NBS 34 (1945) 263-267.

This script checks three things:

  1. every paper value lies inside the WebBook's stated uncertainty;
  2. how much the correlations of Table 3 move if the WebBook's own
     (more coarsely rounded) values are substituted for the compilation;
  3. whether any ordering or slope sign in the paper depends on that choice.

Run:  python3 verify_octane_vs_nist.py
Requires: networkx, scipy; reads 07_octane_properties.py for the
property table and the index definitions, so the two never drift apart.
"""

import os
import statistics
from math import sqrt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "07_octane_properties.py")
DATA = os.path.join(HERE, "..", "data")

# Reuse the paper's own property table and index definitions: everything up to
# the reporting section of that module is definitions only.
_src = open(TABLE).read()
_defs = _src.split("# ---------------------------------------------------------------------\n# 3.")[0]
exec(_defs.replace('print(f"Structure check', 'pass  #'))        # noqa: S102

BP_COL, HF_COL = 4, 5          # positions in the property tuple


def read_reference(filename):
    """{species name: (paper value, NIST value, NIST uncertainty)}."""
    out = {}
    with open(os.path.join(DATA, filename)) as fh:
        for line in fh.read().strip().split("\n")[1:]:
            parts = line.split("\t")
            out[parts[0]] = (float(parts[2]), float(parts[3]), float(parts[4]))
    return out


def check_within_uncertainty(ref, label, unit):
    print(f"\n{label}")
    print(f"  {'isomer':28s}{'paper':>10}{'NIST':>10}{'±':>7}{'diff':>9}  ok")
    worst, failures = 0.0, []
    for name, (paper, nist, unc) in ref.items():
        diff = paper - nist
        ok = abs(diff) <= unc + 1e-9
        worst = max(worst, abs(diff))
        if not ok:
            failures.append(name)
        print(f"  {name:28s}{paper:10.2f}{nist:10.1f}{unc:7.1f}{diff:+9.2f}"
              f"  {'yes' if ok else 'NO'}")
    print(f"  -> {len(ref)} isomers, {len(failures)} outside uncertainty, "
          f"max |diff| = {worst:.2f} {unit}")
    return failures


def correlations(graphs, y):
    return {name: abs(pearsonr([f(G) for G in graphs], y)[0])
            for name, f in INDICES}


def fit_cdz(cdz_values, y):
    """Least-squares line y = slope*CDZ + intercept, with the standard error."""
    mx, my = statistics.fmean(cdz_values), statistics.fmean(y)
    slope = (sum((a - mx) * (b - my) for a, b in zip(cdz_values, y))
             / sum((a - mx) ** 2 for a in cdz_values))
    intercept = my - slope * mx
    resid = [b - (slope * a + intercept) for a, b in zip(cdz_values, y)]
    see = sqrt(sum(r * r for r in resid) / (len(y) - 2))
    return slope, intercept, see


def main():
    bp_ref, hf_ref = read_reference("octane_nist_boiling_points.tsv"), read_reference("octane_nist_enthalpies.tsv")
    names = [n for n, _, _ in OCTANES]
    assert set(names) == set(bp_ref) == set(hf_ref), \
        "species names in the reference files do not match the property table"

    print("=" * 72)
    print("1. Each paper value against the NIST WebBook's stated uncertainty")
    print("=" * 72)
    bad = check_within_uncertainty(bp_ref, "Normal boiling point (K)", "K")
    bad += check_within_uncertainty(
        hf_ref, "Enthalpy of formation, gas phase (kJ/mol)", "kJ/mol")

    graphs = [smiles_to_graph(s) for _, s, _ in OCTANES]
    paper_bp = [p[BP_COL] for _, _, p in OCTANES]
    paper_hf = [p[HF_COL] for _, _, p in OCTANES]
    nist_bp = [bp_ref[n][1] for n in names]
    nist_hf = [hf_ref[n][1] for n in names]

    print("\n" + "=" * 72)
    print("2. Correlations of Table 3 under each choice of source")
    print("=" * 72)
    print(f"{'Index':12s}{'BP paper':>10}{'BP NIST':>10}{'delta':>9}"
          f"   {'Hf paper':>10}{'Hf NIST':>10}{'delta':>9}")
    bp_a, bp_b = correlations(graphs, paper_bp), correlations(graphs, nist_bp)
    hf_a, hf_b = correlations(graphs, paper_hf), correlations(graphs, nist_hf)
    worst = 0.0
    for name, _ in INDICES:
        d1, d2 = bp_b[name] - bp_a[name], hf_b[name] - hf_a[name]
        worst = max(worst, abs(d1), abs(d2))
        print(f"{name:12s}{bp_a[name]:10.5f}{bp_b[name]:10.5f}{d1:+9.5f}"
              f"   {hf_a[name]:10.5f}{hf_b[name]:10.5f}{d2:+9.5f}")
    print(f"\n  -> largest change in any |R|: {worst:.5f}")

    print("\n" + "=" * 72)
    print("3. Do any conclusions depend on the choice?")
    print("=" * 72)
    for label, a, b in (("boiling point", bp_a, bp_b),
                        ("enthalpy of formation", hf_a, hf_b)):
        wa = sorted(a, key=a.get)[:4]
        wb = sorted(b, key=b.get)[:4]
        print(f"  {label}: four weakest indices")
        print(f"    paper source : {wa}")
        print(f"    NIST source  : {wb}")
        print(f"    CDZ is weakest overall: paper={wa[0]=='CDZ'}, "
              f"NIST={wb[0]=='CDZ'}")

    cdz_values = [CDZ(G) for G in graphs]
    print("\n  single-descriptor CDZ fits:")
    for label, y in (("BoilingP paper", paper_bp), ("BoilingP NIST", nist_bp),
                     ("Enthalpy paper", paper_hf), ("Enthalpy NIST", nist_hf)):
        slope, intercept, see = fit_cdz(cdz_values, y)
        print(f"    {label:16s} slope {slope:+9.5f}  intercept {intercept:+11.5f}"
              f"  SEE {see:8.5f}")

    print("\n" + "=" * 72)
    if bad:
        print(f"RESULT: {len(bad)} value(s) outside NIST uncertainty: {bad}")
    else:
        print("RESULT: all 36 values lie within the NIST WebBook's stated "
              "uncertainty.")
    print("=" * 72)


if __name__ == "__main__":
    main()
