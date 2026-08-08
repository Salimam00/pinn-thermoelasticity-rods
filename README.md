# Physics-Informed Neural Network (Deep Ritz Method) for Thermoelasticity Analysis

## Overview

This repository presents a Physics-Informed Neural Network (PINN) based on the Deep Ritz method for the thermoelastic analysis of rods with variable cross-section.

The proposed computational framework combines continuum mechanics, thermoelasticity, variational principles, and deep neural networks to obtain numerical solutions of thermoelastic problems without conventional mesh-based discretization.

The main objective is to develop a mesh-free computational approach in which physical laws, material properties, thermal effects, mechanical loading, and boundary conditions are incorporated directly into the variational formulation.

The considered problem describes the thermoelastic response of a one-dimensional rod with variable cross-sectional area subjected to distributed mechanical loading, thermal loading, and prescribed boundary conditions.

---

## 1. Physical Problem

The considered one-dimensional thermoelastic rod occupies the domain

$$
0 \leq x \leq L
$$

where:

- $x$ is the longitudinal coordinate;
- $L$ is the total length of the rod;
- $A(x)$ is the variable cross-sectional area;
- $u(x)$ is the longitudinal displacement;
- $E(x)$ is Young's modulus;
- $\alpha(x)$ is the coefficient of thermal expansion;
- $\Delta T(x)$ is the temperature change;
- $q(x)$ is the distributed axial load;
- $P_L$ is the prescribed axial force at $x=L$.

The rod has a variable cross-section and may have spatially varying material and thermal properties.

---

## 2. Geometry and Boundary Conditions

The left end of the rod is assumed to be clamped.

The displacement boundary condition is

$$
u(0)=0
$$

At the right boundary, the axial force is prescribed:

$$
P(L)=P_L
$$

The axial force is related to the stress and cross-sectional area by

$$
P(x)=\sigma(x)A(x)
$$

Therefore, the mechanical boundary conditions are

$$
u(0)=0,
\qquad
\sigma(L)A(L)=P_L
$$

### Figure 1. Schematic representation of the thermoelastic rod

![Figure 1 - Thermoelastic rod](rod_schematic_Fig1.png.png)

The rod is subjected to distributed mechanical loading $q(x)$ and thermal loading $\Delta T(x)$.

The variable cross-sectional area is represented by

$$
A=A(x)
$$

---

## 3. Thermoelastic Formulation

The total strain is defined as

$$
\varepsilon(x)=\frac{du(x)}{dx}
$$

The total strain consists of elastic and thermal components:

$$
\varepsilon(x)
=
\varepsilon_{\mathrm{el}}(x)
+
\varepsilon_{\mathrm{th}}(x)
$$

The thermal strain is

$$
\varepsilon_{\mathrm{th}}(x)
=
\alpha(x)\Delta T(x)
$$

Therefore, the elastic strain is

$$
\varepsilon_{\mathrm{el}}(x)
=
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
$$

---

## 4. Thermoelastic Constitutive Relation

The corresponding stress is calculated using the thermoelastic constitutive relation

$$
\sigma(x)
=
E(x)
\left[
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
$$

where $E(x)$ is Young's modulus.

The axial force is then obtained from

$$
P(x)=\sigma(x)A(x)
$$

or

$$
P(x)
=
E(x)A(x)
\left[
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
$$

---

## 5. Equilibrium Equation

For a one-dimensional rod subjected to a distributed axial load $q(x)$, the equilibrium equation is

$$
\frac{dP(x)}{dx}+q(x)=0
$$

Using the thermoelastic constitutive relation, this can be written as

$$
\frac{d}{dx}
\left\{
E(x)A(x)
\left[
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
\right\}
+
q(x)
=
0
$$

The Deep Ritz formulation avoids solving this differential equation directly and instead minimizes an appropriate energy functional.

---

## 6. Deep Ritz Method

The Deep Ritz method formulates the boundary-value problem as a variational minimization problem.

The neural network represents the unknown displacement field

$$
u(x)\approx u_{\mathrm{pred}}(x)
$$

The neural network receives the spatial coordinate $x$ as input and produces the predicted displacement $u_{\mathrm{pred}}(x)$ as output.

Automatic differentiation is used to calculate the spatial derivative

$$
\frac{du_{\mathrm{pred}}(x)}{dx}
$$

The elastic strain predicted by the neural network is therefore

$$
\varepsilon_{\mathrm{el}}(x)
=
\frac{du_{\mathrm{pred}}(x)}{dx}
-
\alpha(x)\Delta T(x)
$$

The corresponding stress is

$$
\sigma(x)
=
E(x)
\left[
\frac{du_{\mathrm{pred}}(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
$$

---

## 7. PINN Architecture

The neural network consists of multiple fully connected layers.

The input of the network is the spatial coordinate

$$
x\in[0,L]
$$

The output is the predicted displacement

$$
u_{\mathrm{pred}}(x)
$$

The implemented architecture uses:

- one input variable $x$;
- fully connected hidden layers;
- nonlinear activation functions;
- automatic differentiation;
- one output variable representing displacement;
- variational loss minimization.

The neural network can be represented as

$$
u_{\mathrm{pred}}(x)
=
\mathcal{N}(x;\theta)
$$

where $\theta$ represents all trainable neural-network parameters.

### Figure 3. PINN architecture and variational loss formulation

![Figure 3 - PINN architecture](pinn_architecture.png_Fig.3.png)

The neural network is trained by minimizing the total variational loss.

---

## 8. Neural Network Model

The neural network contains several fully connected layers.

A typical configuration used in the computational experiment contains four hidden layers with 50 neurons per layer.

The activation function is the hyperbolic tangent:

$$
\tanh(x)
$$

The neural network therefore represents the displacement field as

$$
u_{\mathrm{pred}}(x)
=
\mathcal{N}(x;\theta)
$$

where $\theta$ denotes the weights and biases of the network.

The optimization process updates $\theta$ in order to minimize the total loss.

---

## 9. Automatic Differentiation

Automatic differentiation is used to calculate derivatives of the neural-network output with respect to the spatial coordinate.

The first derivative is

$$
\frac{du_{\mathrm{pred}}}{dx}
$$

This derivative is required for calculating the elastic strain.

The strain is calculated as

$$
\varepsilon_{\mathrm{el}}
=
\frac{du_{\mathrm{pred}}}{dx}
-
\alpha\Delta T
$$

and the stress is

$$
\sigma
=
E
\left(
\frac{du_{\mathrm{pred}}}{dx}
-
\alpha\Delta T
\right)
$$

Automatic differentiation eliminates the need for manually calculating derivatives of the neural-network output.

---

## 10. Variational Energy Functional

The total potential energy functional for the thermoelastic rod can be expressed as

$$
\Pi[u]
=
\frac{1}{2}
\int_0^L
E(x)A(x)
\left(
\frac{du}{dx}
-
\alpha(x)\Delta T(x)
\right)^2
dx
-
\int_0^L
q(x)u(x)
dx
-
P_Lu(L)
$$

The neural network approximates the displacement field and minimizes this functional.

Therefore,

$$
u_{\mathrm{pred}}
=
\arg\min_u \Pi[u]
$$

In the Deep Ritz framework, the continuous integral is evaluated numerically using collocation or quadrature points.

---

## 11. Total Loss Function

The total loss function consists of several components.

It can be represented as

$$
L_{\mathrm{total}}
=
L_{\mathrm{energy}}
+
L_{\mathrm{BC}}
$$

where:

- $L_{\mathrm{energy}}$ is the variational energy loss;
- $L_{\mathrm{BC}}$ represents the boundary-condition losses.

The energy contribution is

$$
L_{\mathrm{energy}}
=
\frac{1}{2}
\int_0^L
E(x)A(x)
\left(
\frac{du_{\mathrm{pred}}}{dx}
-
\alpha(x)\Delta T(x)
\right)^2
dx
-
\int_0^L
q(x)u_{\mathrm{pred}}(x)
dx
-
P_Lu_{\mathrm{pred}}(L)
$$

---

## 12. Boundary Condition Loss

The displacement boundary condition at the left end is

$$
u(0)=0
$$

The corresponding penalty term is

$$
L_{x=0}
=
\left(
u_{\mathrm{pred}}(0)-0
\right)^2
$$

The force boundary condition at the right end is

$$
P(L)=P_L
$$

The corresponding loss term is

$$
L_{x=L}
=
\left(
P_{\mathrm{pred}}(L)-P_L
\right)^2
$$

where

$$
P_{\mathrm{pred}}(L)
=
\sigma_{\mathrm{pred}}(L)A(L)
$$

Therefore, the complete loss can be written as

$$
L_{\mathrm{total}}
=
L_{\mathrm{energy}}
+
\lambda_0L_{x=0}
+
\lambda_LL_{x=L}
$$

where $\lambda_0$ and $\lambda_L$ are penalty coefficients.

---

## 13. Variable Cross-Section

The cross-sectional area of the rod is not necessarily constant.

It is represented by

$$
A=A(x)
$$

A variable area affects both the stiffness and the axial force:

$$
P(x)=\sigma(x)A(x)
$$

Therefore,

$$
P(x)
=
E(x)A(x)
\left[
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
$$

The proposed PINN formulation can therefore be used for rods with arbitrary spatially varying cross-sections.

---

## 14. Material Properties

The material properties may also depend on the spatial coordinate.

Young's modulus is represented as

$$
E=E(x)
$$

The coefficient of thermal expansion is

$$
\alpha=\alpha(x)
$$

The temperature distribution is

$$
T=T(x)
$$

and the temperature change is

$$
\Delta T(x)
=
T(x)-T_{\mathrm{ref}}
$$

where $T_{\mathrm{ref}}$ is the reference temperature.

---

## 15. Thermal Loading

Thermal deformation is introduced through the thermal strain

$$
\varepsilon_{\mathrm{th}}(x)
=
\alpha(x)\Delta T(x)
$$

The thermoelastic stress is therefore

$$
\sigma(x)
=
E(x)
\left[
\varepsilon(x)
-
\varepsilon_{\mathrm{th}}(x)
\right]
$$

or

$$
\sigma(x)
=
E(x)
\left[
\frac{du(x)}{dx}
-
\alpha(x)\Delta T(x)
\right]
$$

This allows the PINN model to account for spatially varying temperature fields.

---

## 16. Distributed Mechanical Load

The rod may be subjected to a distributed axial load

$$
q=q(x)
$$

The potential energy contribution of the distributed load is

$$
\Pi_q
=
-
\int_0^L
q(x)u(x)
dx
$$

The corresponding term is included in the variational loss.

---

## 17. External Axial Force

An axial force $P_L$ may be applied at the right boundary.

Its potential energy contribution is

$$
\Pi_P
=
-
P_Lu(L)
$$

Therefore, the complete potential energy contains the terms associated with:

1. elastic deformation;
2. thermal deformation;
3. distributed mechanical loading;
4. external axial force.

---

## 18. Computational Workflow

The computational procedure consists of the following steps:

1. Define the geometry of the rod.
2. Define the variable cross-sectional area $A(x)$.
3. Define Young's modulus $E(x)$.
4. Define the coefficient of thermal expansion $\alpha(x)$.
5. Define the temperature distribution $T(x)$.
6. Define the distributed load $q(x)$.
7. Define the boundary conditions.
8. Generate collocation points.
9. Construct the neural network.
10. Predict the displacement field.
11. Calculate derivatives using automatic differentiation.
12. Calculate elastic strain.
13. Calculate thermoelastic stress.
14. Calculate the axial force.
15. Evaluate the variational energy.
16. Evaluate the boundary-condition losses.
17. Construct the total loss.
18. Optimize the neural-network parameters.
19. Repeat the optimization until convergence.
20. Post-process the displacement and mechanical response.

---

## 19. Numerical Implementation

The computational implementation is provided in Python.

The main source code is:

`pinn_variable_section_rod.py`

The implementation uses a neural network to approximate the displacement field.

The computational model includes:

- variable cross-sectional geometry;
- thermoelastic constitutive equations;
- distributed mechanical loading;
- prescribed boundary conditions;
- automatic differentiation;
- variational loss calculation;
- neural-network optimization.

---

## 20. Python Implementation

The main program can be executed using Python.

Example execution:

```bash
python pinn_variable_section_rod.py
