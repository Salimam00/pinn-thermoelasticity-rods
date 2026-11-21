# Variational Deep Ritz thermoelastic rod benchmark

This repository contains the reproducibility materials for the manuscript **“Variational Deep Ritz Modeling of a Variable-Area Thermoelastic Rod with Finite-Element Verification.”**

## Authoritative version

The current verified release is **v3**. Use only these root-level files:

- `run_all_v3.py` — single executable script that regenerates all reported results;
- `summary_v3.json` — verified seed-42 summary metrics for Cases A and B;
- `reproducibility_package_v3.zip` — complete archive with the original directory structure, CSV data, figures, and instructions;
- `README_reproducibility_v3.md` — concise file-to-manuscript map.

The archive DOI is [10.5281/zenodo.24321123](https://doi.org/10.5281/zenodo.24321123).

## Reproduction

The reported calculations use Python 3.13, PyTorch 2.15, NumPy 2.5, pandas 3.0, CPU execution, and `float64` arithmetic.

```bash
python run_all_v3.py
```

The script generates seed-42 solutions for Cases A and B, ten-seed repeatability results, the quadrature-sensitivity study, independent FEM convergence, pointwise fields, Adam/L-BFGS energy histories, and all manuscript figures and tables.

## Verified headline results

- Case A relative L2 displacement error: `0.012501%`;
- Case A mechanically normalized displacement error: `0.084140%`;
- Case A relative L2 stress error: `0.633215%`;
- Case A corrected energy-seminorm error: `0.575485%`;
- Case B relative L2 displacement error: `0.151822%`;
- Case B relative L2 stress error: `0.089600%`;
- Case B reactions: `-59.793 kN` and `-59.871 kN` versus the analytical `-60.000 kN`.

## Legacy materials

The directory `deep_ritz_final/` and older figures are retained only for provenance. They correspond to superseded v1/v2 development runs and must not be used to verify the current manuscript. If an older file conflicts with the v3 archive, the v3 archive and `summary_v3.json` are authoritative.

## License and citation

The reproducibility release is archived on Zenodo under the Creative Commons Attribution 4.0 International license. Cite the Zenodo record and the associated manuscript.
