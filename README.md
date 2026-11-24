# CDZ index — code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24505775.svg)](https://doi.org/10.5281/zenodo.24505775)

Code and data accompanying

> A. B. Bektaş, İ. N. Cangül, Y. Kara Şen, *The Degree–Atomic Number Commensurability Index: Theory, Extremal
> Results and Chemical Applicability* (submitted).

The index is

    CDZ(G) = Σ_{uv ∈ E(G)} gcd(a_u, a_v) / sqrt(a_u · a_v),   a_u = d_u · Z_u,

where `d_u` is the heavy-atom degree and `Z_u` the atomic number of vertex `u`.
`CDZ_hom` is the ablation control with every `Z_u = 6`.

## Quick start

```bash
pip install -r requirements.txt
python3 run_all.py
```

`run_all.py` runs every script in `scripts/`, writes each output to
`results/<script>.txt`, and checks it against the values printed in the paper.
It ends with `All results reproduced.` when every check passes (about a minute
in total; `12_ss_abr.py` takes the longest).
Any single script can also be run on its own, e.g. `python3 scripts/03_local_moves.py`.
The `results/` folder in the repository holds the outputs of a reference run.

Tested with Python 3.13, networkx 3.7, numpy 2.5, scipy 1.18. RDKit is **not**
required: molecular graphs are read from SMILES by the parser in `cdz_core.py`.

## Where each result comes from

| Paper                                                    | Script                              |
|----------------------------------------------------------|-------------------------------------|
| Table 1, Remark 3.7, Conjecture 3.16 (exhaustive checks) | `01_minimisers.py`                  |
| §3.5.1–3.5.2: two-parameter family, Lemma 3.17, Prop. 3.18, Remark 3.19 | `02_subdivision.py`  |
| Prop. 3.20(b), Appendix A (10 818 trees, 8903 with a (3,3) edge, 373 uncovered (4,4) trees, margin 0.03626) | `03_local_moves.py` |
| Remark 3.22 (the tree H₁₂)                               | `04_h12_obstruction.py`             |
| Conjecture 3.25 and the unicyclic minimisers, 4 ≤ n ≤ 12 | `05_unicyclic.py`                   |
| Table 2 (correlation on trees)                           | `06_correlation_trees.py`           |
| Table 3 (octane properties)                              | `07_octane_properties.py`           |
| §5.2 regression equations                                | `08_octane_regression.py`           |
| Octane data checked against NIST (all 36 values)         | `09_octane_vs_nist.py`              |
| Tables 4–5 (degeneracy)                                  | `10_degeneracy.py`                  |
| Table 6 (mutual correlations)                            | `11_mutual_correlation.py`          |
| Tables 7–9 (SS, Abr, SS/Abr)                             | `12_ss_abr.py`                      |
| Table 10 (C/N/O-labelled trees)                          | `13_labelled_trees.py`              |
| Table 11 (partial R² beyond size)                        | `14_partial_r2.py`                  |
| Tables 12–13 and the isosteric triples (45 aliphatics)   | `15_aliphatic45.py`                 |
| Table 14 (ordering of isosteric pairs)                   | `16_isosteric_ranking.py`           |
| §6, items 2 and 7 (large IC50 set)                       | `17_large_ic50_limitations.py`      |

Theorem, table and section numbers refer to the submitted version.

## Layout

```
cdz_core.py      CDZ, CDZ_hom, SMILES parser, molecular formula / weight
run_all.py       runs and checks everything
scripts/         one script per result (see table above)
data/            datasets in the form used; provenance in data/README.md
results/         reference outputs
```

`cdz_core.parse_smiles` agrees with an RDKit-based computation of CDZ on all
10 833 molecules of the large IC50 set (maximum difference 6e-14), and
`cdz_core.molecular_weight` agrees with RDKit's `MolWt` on every ESOL, HER2 and
E. coli molecule.

## Licence

Code: MIT (see `LICENSE`). Data files keep the licences of their sources; see
`data/README.md`.

## Citation

See `CITATION.cff`. Please cite the paper and, for the code, the Zenodo DOI
[10.5281/zenodo.24505775](https://doi.org/10.5281/zenodo.24505775).
