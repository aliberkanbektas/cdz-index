"""
Run every script in scripts/ and check its output against the paper.

Scripts 01-05 and 14-17 check their own results and exit with a non-zero
status on any mismatch.  For the remaining scripts, which print tables, the
expected values below are matched against their output.  Each script's full
output is written to results/<script>.txt.

Usage:  python3 run_all.py            (about a minute in total)
"""

import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")
RESULTS = os.path.join(ROOT, "results")

# script -> regular expressions that must all occur in its output
EXPECT = {
    "02_subdivision.py": [r"ALL CHECKS PASSED"],
    "06_correlation_trees.py": [r"R\(G\)\s+\|R\| = 0\.76237", r"SO2\(G\)\s+\|R\| = 0\.86848",
                                r"SDD\(G\)\s+\|R\| = 0\.70431"],
    "07_octane_properties.py": [r"AcenFac\s+0\.63924\s+0\.90413", r"BoilingP\s+0\.67044\s+0\.82128",
                                r"Enthalpy\s+0\.50666\s+0\.84694"],
    "08_octane_regression.py": [r"AcenFac\s+= 0\.03271 \* CDZ\(G\) \+ 0\.18761,\s+SEE = 0\.02811",
                                r"BoilingP\s+= 5\.92305 \* CDZ\(G\) \+ 360\.03560,\s+SEE = 4\.68163"],
    "09_octane_vs_nist.py": [r"RESULT: all 36 values lie within"],
    "10_degeneracy.py": [r"Octanes\s+18\s+88\.9%", r"Decanes\s+75\s+65\.3%",
                         r"13 vertices\s+1301\s+0\.58493\s+0\.56188"],
    "11_mutual_correlation.py": [r"21/21 entries match", r"CDZ\s+1\.00000 0\.58966 0\.58144"],
    "12_ss_abr.py": [r"17/17 checks passed", r"n=6 SS\s+: CDZ=0\.21191",
                     r"n=13 Ratio: CDZ=0\.35994"],
    "13_labelled_trees.py": [r"4\s+75\s+90\.7%\s+2\.7%", r"7\s+9894\s+65\.3%\s+0\.1%"],
}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    failures = []
    for name in sorted(os.listdir(SCRIPTS)):
        if not name.endswith(".py"):
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, os.path.join(SCRIPTS, name)],
                              capture_output=True, text=True, cwd=SCRIPTS)
        out = proc.stdout + proc.stderr
        with open(os.path.join(RESULTS, name.replace(".py", ".txt")), "w") as fh:
            fh.write(out)
        missing = [p for p in EXPECT.get(name, []) if not re.search(p, out)]
        ok = proc.returncode == 0 and not missing
        if not ok:
            failures.append(name)
        print(f"  {'OK  ' if ok else 'FAIL'} {name:34s} {time.time() - t0:6.1f} s"
              + (f"   missing: {missing}" if missing else "")
              + (f"   exit code {proc.returncode}" if proc.returncode else ""))
    print("\n" + ("All results reproduced." if not failures else f"{len(failures)} script(s) failed: {failures}"))
    return not failures


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
