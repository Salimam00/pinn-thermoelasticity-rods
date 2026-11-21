# Deep Ritz Thermoelasticity Analysis

This folder contains the computational materials accompanying the manuscript:

**Variational Deep Ritz Modeling of a Variable-Area Thermoelastic Rod with Finite-Element Verification**

The study develops a Deep Ritz physics-informed neural network approach for the thermoelastic analysis of rods with variable cross-sectional area and temperature-dependent thermal loading.

## Contents

- `JCAMECH_Deep_Ritz_Thermoelastic_Rod_v2.ipynb` — verified Google Colab notebook used to regenerate the manuscript figures and Tables 1, 2, 4, and 5.
- `PINN_Deep_Ritz_Thermoelasticity_Analysis.ipynb` — earlier archived computational notebook retained for provenance.
- `run_benchmark.py` — command-line version of the authoritative calculation.
- `verify_results.py` — independent post-processing check with explicit numerical acceptance criteria.
- `run_repeatability.py` — ten-seed repeatability experiment used for Table 2.
- `verify_fem.py` — independent linear-bar FEM cross-check and mesh-convergence utility.
- `run_fixed_fixed_case.py` — restrained thermal-expansion case with both rod ends fixed.
- `results_fixed_fixed/` — fields, metrics, and figure for the fixed-fixed case.
- `fem_convergence.csv` — FEM h-convergence data for 50 to 2000 elements.
- `fem_verification.csv` — numerical error metrics from the FEM cross-check.
- `repeatability_results.csv` — individual results for seeds 0–9.
- `repeatability_summary.csv` — mean, sample standard deviation, minimum, and maximum for Table 2.
- `requirements.txt` — tested package versions for a CPU reproduction environment.
- `prediction_data.csv` — computed prediction data used for the analysis and visualization of the numerical results.
- `Figures/` — figures generated for the manuscript.
- `tables/` — numerical tables reported in the manuscript.

## Computational approach

The model is formulated using the total potential energy functional and incorporates the essential boundary condition through a hard constraint at the clamped end.

The computational model uses:

- Rod length: `L = 1.0`
- Reference cross-sectional area: `A0 = 1e-3`
- Variable-section parameter: `β = 3.0`
- Young's modulus: `E = 200e9 Pa`
- Thermal expansion coefficient: `α = 1.2e-5`
- Maximum temperature change: `ΔTmax = 100`
- Applied load: `P = 1e4 N`
- Random seed: `42`
- Neural network architecture: 4 hidden layers with 50 neurons per layer and `tanh` activation
- Quadrature points: `200`
- Adam optimization steps: `5000`
- L-BFGS optimization steps: `500`
- Validation points: `400`

## Reproducibility

The archived v2 results and manuscript numbers originate from the saved reference run with random seed 42. On the 400-point validation grid, that reference run gives MAE `3.1748775883e-08 m`, RMSE `3.9454811096e-08 m`, relative L2 error `0.0125044962%`, and maximum absolute displacement error `8.9782907290e-08 m`.

[Open the verified v2 notebook in Google Colab](https://colab.research.google.com/github/Salimam00/pinn-thermoelasticity-rods/blob/main/deep_ritz_final/JCAMECH_Deep_Ritz_Thermoelastic_Rod_v2.ipynb)

To reproduce the benchmark in a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_benchmark.py
python verify_results.py results
```

The verification passes when the reproduced result satisfies all of the following conservative criteria: MAE `<= 3.5e-08 m`, RMSE `<= 4.2e-08 m`, relative L2 error `<= 0.0135%`, maximum absolute displacement error `<= 9.5e-08 m`, maximum-stress relative error `<= 2.1%`, and end-force relative error `<= 2.1%`.

The same seed is used in every run. Nevertheless, neural-network optimization and floating-point reductions can differ slightly across PyTorch builds, BLAS libraries, processors, and CPU/GPU backends. Therefore, last-digit identity is not claimed across all platforms; reproducibility is assessed by the stated numerical criteria and by preservation of the physical conclusions.

The main notebook contains the same computational procedure, physical parameters, training schedule, validation calculations, and figure generation as the command-line script. The command-line script is the preferred route for an automated clean-environment reproduction.

The tested independent CPU reproduction used Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.5.5, pandas 3.1.1, and Matplotlib 3.11.2. The fresh result met all criteria; two consecutive executions in that environment produced identical CSV files.

The ten-seed experiment is reproduced with:

```bash
python run_repeatability.py --seeds 0 1 2 3 4 5 6 7 8 9
```

In the tested environment, relative L2 error ranged from `0.002112%` to `0.029975%`. This demonstrates high absolute accuracy for every tested seed, but also non-negligible initialization sensitivity. The archive therefore reports the full seed-level results rather than claiming last-digit or seed-independent repeatability.

The independent FEM check is reproduced with:

```bash
python verify_fem.py --elements 2000
```

With midpoint integration for the variable area and thermal strain, its relative L2 displacement error against the closed-form solution is `5.73e-07%`, well below the `0.001%` criterion stated in the manuscript.

The revised manuscript also reports FEM h-convergence. Even 50 linear elements (51 nodal degrees of freedom) give a relative L2 displacement error of `0.000908%`, compared with `0.012504%` for the 7801-parameter neural model. This benchmark is therefore not presented as an efficiency advantage over FEM.

## Fixed-fixed thermoelastic verification

The fixed-free benchmark is statically determinate: its axial stress is `P/A(x)` and is independent of the prescribed temperature field. To test genuinely thermally induced stress, the revised study adds a rod with both ends fixed and no applied end force. The hard trial function is proportional to `(x/L)(1-x/L)`, so both displacement conditions are exact.

Run the additional case with:

```bash
python run_fixed_fixed_case.py
```

For the manuscript parameters, the analytical reaction is `-60.0 kN` and the analytical stress varies from `-60 MPa` to `-240 MPa`. The saved seed-42 run gives left and right reactions of `-59.85 kN` and `-59.90 kN`, a relative L2 displacement error of `0.1121%`, and a relative L2 stress error of `0.0676%`.

Both authoritative scripts use PyTorch `float64`, dimensionless coordinates, displacement scaling, and positive energy scaling. These transformations improve conditioning without changing the physical minimizer.

## Relation to the manuscript

This repository contains the computational materials corresponding to the final methodology and results presented in the manuscript. The implementation in the main notebook uses hard enforcement of the essential boundary condition and should be considered the authoritative computational implementation for the study.

