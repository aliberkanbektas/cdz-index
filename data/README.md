# Data

| File | Rows | Used by | Source and licence |
|------|------|---------|--------------------|
| `esol_delaney.csv` | 1128 | 14, 16 | J. S. Delaney, *J. Chem. Inf. Comput. Sci.* 44 (2004) 1000–1005, as distributed in MoleculeNet (Wu et al., *Chem. Sci.* 9 (2018) 513). Measured log solubility (mol/L). |
| `lipophilicity.csv` | 4200 | 14, 16 | MoleculeNet Lipophilicity set (Wu et al. 2018), derived from ChEMBL; measured octanol/water LogD at pH 7.4. ChEMBL data: CC BY-SA 3.0. |
| `chembl_her2_ec50.csv` | 2589 | 14 | ChEMBL (Zdrazil et al., *Nucleic Acids Res.* 2024), HER2 ligands with reported EC50, single-protein target, standardised. RDKit descriptors (MolWt, LogP, TPSA, …) computed by the author. CC BY-SA 3.0. |
| `chembl_large_ic50.csv` | 10 851 (10 833 parsed) | 14, 17 | ChEMBL, ligands with reported IC50, single-protein target, standardised; RDKit descriptors as above. Rows whose SMILES the parser rejects are skipped. CC BY-SA 3.0. |
| `chembl_ecoli_ic50.csv` | 1092 (1082 parsed) | 14 | ChEMBL, *E. coli* targets with reported IC50; RDKit descriptors plus molar volume. CC BY-SA 3.0. |
| `octane_nist_boiling_points.tsv`, `octane_nist_enthalpies.tsv` | 18 + 18 | 07–09 | Octane-isomer boiling points and standard enthalpies of formation as used in the paper, side by side with the NIST Chemistry WebBook values (retrieved 24 October 2026) and their stated uncertainties. |
| `aliphatic45.csv` | 45 | 15 | 45 aliphatic alcohols, ethers and amines with normal boiling points (°C), compiled by the author. |

The graph-theoretic results (scripts 01–06, 10–13) need no data: the trees are
enumerated with `networkx.nonisomorphic_trees`.

Licence of the ChEMBL-derived files: Creative Commons Attribution-ShareAlike 3.0
Unported (https://creativecommons.org/licenses/by-sa/3.0/). Redistributing
these derived files requires attribution to ChEMBL and the same licence; this
applies to the files in this folder only, not to the code.
