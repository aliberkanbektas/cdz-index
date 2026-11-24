"""
Verification of the 45 boiling points of data/aliphatic45.csv against the
NIST Chemistry WebBook (retrieved 24 November 2026).

data/aliphatic45_nist_boiling_points.tsv holds, for each compound, its CAS
number, the WebBook value in K and the page it was read from.  For 32
compounds this is the WebBook's recommended value ("AVG", an average of the
compiled measurements) with its stated uncertainty.  For the other 13 the
WebBook page carrying that average could not be retrieved automatically; for
these the file gives the median of the individual WebBook measurements as
compiled by Cheméo (values more than 10 K from the median dropped) and their
range, and the Cheméo page is the stated URL.

A boiling point passes when it lies within the stated uncertainty of the
recommended value or, for the 13 compounds without one, within the range of
the WebBook measurements, widened by 0.05 K for rounding to 0.1 C.  The one
compound for which the WebBook gives a single value without an uncertainty
(methyl isopropyl ether, Majer and Svoboda 1985) passes when it lies within
0.1 C of it.
"""

import csv
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")


def main():
    with open(os.path.join(DATA, "aliphatic45.csv")) as fh:
        used = {r["name"]: float(r["bp_C"]) for r in csv.DictReader(fh)}
    with open(os.path.join(DATA, "aliphatic45_nist_boiling_points.tsv")) as fh:
        nist = {r["name"]: r for r in csv.DictReader(fh, delimiter="\t")}
    ok = set(used) == set(nist) and len(used) == 45
    print(f"[{'OK' if ok else 'FAIL'}] {len(used)} compounds, every one with a WebBook entry\n")

    print(f"{'compound':26s}{'used (C)':>9s}{'NIST (C)':>10s}{'diff':>7s}  tolerance")
    worst = 0.0
    for name, bp in used.items():
        r = nist[name]
        t = float(r["nist_Tboil_K"]) - 273.15
        diff = bp - t
        unc = r["nist_unc_K"].strip()
        m = re.match(r"range ([\d.]+)-([\d.]+)", unc)
        if m:
            lo, hi = float(m.group(1)) - 273.15 - 0.05, float(m.group(2)) - 273.15 + 0.05
            good = lo <= bp <= hi
            tol = f"WebBook range {lo + 0.05:.2f} to {hi - 0.05:.2f} C"
        elif unc:
            good = abs(diff) <= float(unc) + 0.05
            tol = f"+/- {float(unc):.1f} K"
        else:
            good = abs(diff) <= 0.1
            tol = "single value, 0.1 C"
        ok &= good
        worst = max(worst, abs(diff))
        print(f"{name:26s}{bp:9.1f}{t:10.2f}{diff:+7.2f}  {tol}  {'OK' if good else 'OUTSIDE'}")

    print(f"\nlargest |difference| = {worst:.2f} C")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
