# Physics-Informed Neural Network (Deep Ritz Method) for Thermoelasticity Analysis

## Overview

This repository presents a Physics-Informed Neural Network (PINN) based on the Deep Ritz method for the thermoelastic analysis of rods with variable cross-section.

The proposed computational framework combines continuum mechanics, thermoelasticity, variational principles, and deep neural networks to obtain numerical solutions of thermoelastic problems.

The main objective is to develop a mesh-free computational approach in which the physical laws, material properties, thermal effects, mechanical loading, and boundary conditions are incorporated directly into the variational formulation.

The considered problem describes the thermoelastic response of a rod with a variable cross-sectional area subjected to distributed mechanical loading, thermal loading, and prescribed boundary conditions.

---

# 1. Physical Problem

The considered one-dimensional thermoelastic rod occupies the domain

```math
0 \leq x \leq L
