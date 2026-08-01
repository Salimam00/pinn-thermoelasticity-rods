# Physics-Informed Neural Network (Deep Ritz Method) for Thermoelasticity Analysis

This repository provides the official implementation of the **Weak-Form Physics-Informed Neural Network (Deep Ritz Method)** with **Hard Boundary Constraints** for analyzing the thermomechanical behavior of variable cross-section rods with temperature-dependent material properties.

---

## 📌 Features
- **Variational Formulation (Deep Ritz):** Minimizes the total potential energy functional without higher-order automatic differentiation.
- **Hard Constraints:** Parametric ansatz guarantees exact satisfaction of kinematic ($u(0)=0$) and static boundary conditions.
- **Gauss-Legendre Quadrature:** Numerical integration over $K = 20$ subintervals with $N_g = 5$ Gauss points per element.
- **Thermomechanical Coupling:** Supports temperature-dependent Elastic Modulus $E(T)$ and Thermal Expansion Coefficient $\alpha(T)$.

---

## 📊 Numerical Results & Model Validation

The figure below demonstrates:
1. **Displacement Field $u(x)$:** Comparison between the Deep Ritz PINN prediction and the analytical solution.
2. **Energy Loss Convergence:** Two-stage optimization process using **Adam** (epochs 0–5000) followed by **L-BFGS** fine-tuning.

![PINN Validation and Loss Convergence](results/pinn_validation_and_convergence.png)

---

## 🛠 Project Structure
- `train_pinn.py` — Core training pipeline for the Deep Ritz model.
- `results/` — Directory containing generated plots and validation benchmarks.
- `.gitignore` — Specifies untracked files to ignore in Git.
- `README.md` — Project description and documentation.

---

## 🚀 Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

---

## 📜 Citation
If you find this code useful for your research, please cite our paper:
> *Physics-Informed Neural Networks for Thermomechanical Analysis of Variable Cross-Section Rods.*