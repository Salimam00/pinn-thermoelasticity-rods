# Physics-Informed Neural Network (Deep Ritz Method) for Thermoelasticity Analysis

## Overview

This repository contains the implementation and numerical results of a Physics-Informed Neural Network (PINN) based on the Deep Ritz method for the thermoelastic analysis of rods with variable cross-section.

The proposed approach combines the principles of continuum mechanics, thermoelasticity, variational formulation, and deep neural networks to obtain numerical solutions without relying on conventional mesh-based discretization.

The model considers a thermoelastic rod subjected to mechanical and thermal loading, distributed forces, and appropriate boundary conditions.

---

## Problem Formulation

The considered problem describes the thermoelastic response of a rod with a variable cross-sectional area:

\[
A = A(x)
\]

where \(x\) is the longitudinal coordinate and \(A(x)\) represents the variable cross-sectional area.

The thermoelastic strain is represented as:

\[
\varepsilon(x)
=
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\]

where:

- \(u(x)\) is the axial displacement;
- \(\alpha(x)\) is the coefficient of thermal expansion;
- \(\Delta T(x)\) is the temperature change.

The corresponding stress is calculated using the thermoelastic constitutive relation:

\[
\sigma(x)
=
E(x)
\left[
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
\]

where \(E(x)\) is Young's modulus.

The axial force is given by:

\[
P(x)=\sigma(x)A(x)
\]

---

## Physical Model

The considered thermoelastic rod is subjected to distributed mechanical and thermal loading.

### Figure 1. Rod schematic and boundary conditions

![Rod schematic](rod_schematic_Fig1.png.png)

The schematic illustrates the variable cross-section rod, distributed loading, thermal loading, and the corresponding mechanical boundary conditions.

---

## Thermoelastic Mathematical Model

The mathematical formulation is based on the thermoelastic constitutive law and the Bernoulli-Euler hypothesis.

### Figure 2. Thermoelastic model

![Thermoelastic model](thermoelastic_model_Fig2.png)

The model describes the relationship between displacement, strain, stress, temperature variation, material properties, and cross-sectional area.

---

## Physics-Informed Neural Network

The displacement field is approximated using a neural network:

\[
u(x) \approx u_{\mathrm{pred}}(x)
\]

The neural network takes the spatial coordinate \(x\) as input and predicts the axial displacement.

Automatic differentiation is used to calculate the derivative:

\[
\frac{du_{\mathrm{pred}}}{dx}
\]

The physical constraints of the thermoelastic problem are incorporated into the variational loss function.

### Figure 3. PINN architecture

![PINN architecture](pinn_architecture.png_Fig.3.png)

The neural network consists of multiple fully connected layers. The architecture uses nonlinear activation functions and automatic differentiation to construct the variational formulation of the thermoelastic problem.

---

## Deep Ritz Variational Formulation

The Deep Ritz method is used to minimize the total potential energy of the system.

The variational formulation incorporates:

- elastic strain energy;
- thermal energy contribution;
- distributed mechanical loading;
- boundary conditions;
- axial forces at the rod boundaries.

The total loss function is minimized during neural network optimization.

The trained neural network therefore provides an approximation of the displacement field while satisfying the governing physical principles.

---

## Numerical Results

The trained PINN model is evaluated by comparing its predictions with the corresponding reference solution.

### Figure 4. PINN numerical results

![PINN results](pinn_results_en.png)

The results demonstrate the ability of the proposed Physics-Informed Neural Network to reproduce the thermoelastic response of the variable-section rod.

---

## Source Code

The main implementation is provided in Python.

### Main Python file

[pinn_variable_section_rod.py](pinn_variable_section_rod.py)

The script contains the implementation of the PINN/Deep Ritz approach for the thermoelastic rod with variable cross-section.

---

## Repository Structure

```text
pinn-thermoelasticity-rods/
│
├── README.md
├── .gitignore
│
├── rod_schematic_Fig1.png.png
├── thermoelastic_model_Fig2.png
├── pinn_architecture.png_Fig.3.png
├── pinn_results_en.png
│
└── pinn_variable_section_rod.py
