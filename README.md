# Physics-Informed Neural Network (Deep Ritz Method) for Thermoelasticity Analysis

## Overview

This repository contains the implementation and numerical results of a Physics-Informed Neural Network (PINN) based on the Deep Ritz method for the thermoelastic analysis of rods with variable cross-section.

The proposed approach combines continuum mechanics, thermoelasticity, variational formulation, and deep neural networks to obtain numerical solutions without relying on conventional mesh-based discretization.

The considered model describes a thermoelastic rod subjected to mechanical and thermal loading.

---

## 1. Problem Formulation

The considered problem describes the thermoelastic response of a rod with a variable cross-sectional area.

The longitudinal coordinate of the rod is denoted by $x$, where:

```math
0 \leq x \leq L
