# Deep Ritz Thermoelasticity Analysis

This folder contains the computational materials accompanying the manuscript:

**Deep Ritz Physics-Informed Neural Networks for Thermoelastic Analysis of Variable Cross-Section Rods: A Parametric Surrogate Approach**

The study develops a Deep Ritz physics-informed neural network approach for the thermoelastic analysis of rods with variable cross-sectional area and temperature-dependent thermal loading.

## Contents

- `PINN_Deep_Ritz_Thermoelasticity_Analysis.ipynb` — main computational notebook implementing the Deep Ritz thermoelastic model.
- `prediction_data.csv` — computed prediction data used for the analysis and visualization of the numerical results.
- `Figures/` — figures generated for the manuscript.

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

The main notebook contains the implementation of the computational procedure, model parameters, training procedure, validation calculations, and generation of the prediction data and figures reported in the manuscript.

The files in this folder are provided to facilitate reproducibility and allow readers to inspect and reproduce the computational results.

## Relation to the manuscript

This repository contains the computational materials corresponding to the final methodology and results presented in the manuscript. The implementation in the main notebook uses hard enforcement of the essential boundary condition and should be considered the authoritative computational implementation for the study.

