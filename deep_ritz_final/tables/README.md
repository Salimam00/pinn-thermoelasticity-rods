# Manuscript Tables

This directory contains all the quantitative data and comparison tables generated for the research manuscript on the Deep Ritz method applied to thermoelastic rods. All files are stored in clean, version-controlled CSV format.

## Table of Contents

| File Name | Title / Description |
| :--- | :--- |
| `table1_error_metrics.csv` | Overall displacement error metrics on the 400-point validation grid |
| `table2_repeatability.csv` | Repeatability across independent training runs (10 random seeds) |
| `table3_max_stress.csv` | Maximum axial stress: analytical versus Deep Ritz |
| `table4_boundary_conditions.csv` | Essential and natural boundary conditions: target versus recovered values |
| `table5_quadrature_sensitivity.csv` | Sensitivity of the Deep Ritz solution to the number of Gauss--Legendre quadrature points |
| `table6_comparison.csv` | Comparison of the proposed Deep Ritz formulation with FEM and strong-form PINNs |

## Usage Example (Python)

To load any table and automatically generate its LaTeX representation for the manuscript, you can use a short Python script with `pandas`:

```python
import pandas as pd

# Load the error metrics table
df = pd.read_csv("table1_error_metrics.csv")

# Generate LaTeX table code
latex_output = df.to_latex(
    index=False,
    escape=False,
    column_format="p{6cm}p{4cm}",
    caption="Overall displacement error metrics on the 400-point validation grid",
    label="tab:error_metrics"
)

print(latex_output)
