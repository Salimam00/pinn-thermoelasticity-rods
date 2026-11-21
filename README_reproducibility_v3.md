# Reproducibility package (manuscript v3)

`python run_all.py` reproduces every number, table and figure of the manuscript
(Case A and Case B seed 42, ten-seed repeatability for both cases, quadrature study,
FEM h-convergence, energy history). Results are written to `results_final/`.
`python run_all.py results_final --figures-only` regenerates the figures from the saved CSV files.

Environment used for the reported numbers: Python 3.13, PyTorch 2.15 (CPU, float64), NumPy 2.5, pandas 3.0, 2 CPU threads.
Total run time about 10 min on CPU. Case A reproduces the earlier Colab metrics (metrics_v2.json) to four significant digits.

Files that the manuscript relies on:
- results_final/summary.json — seed-42 metrics (Tables 1, 3, 5; Sections 4.1–4.9)
- results_final/case_A_repeatability_summary.csv — Table 2; case_B_repeatability_summary.csv — Section 4.9
- results_final/case_A_quadrature_sensitivity.csv — Table 4
- results_final/case_A_fem_convergence.csv — Table 6 and Section 4.10
- results_final/Fig1..Fig5 — Figures 1–3, 5 (Case B) and 4 (energy history)
