"""
Single reproducibility script for the manuscript
"Variational Deep Ritz Modeling of a Variable-Area Thermoelastic Rod
 with Finite-Element Verification".

Every number reported in the manuscript is produced by this script in one run.
Usage:  python run_all.py            (writes everything to ./results_final)
"""
import json, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
OUT = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "results_final"))
OUT.mkdir(exist_ok=True)

# ---------------- physical parameters ----------------
L, A0, BETA = 1.0, 1.0e-3, 3.0
E, ALPHA, DT_MAX, P = 200.0e9, 1.2e-5, 100.0, 1.0e4
# ---------------- numerical settings ----------------
LAYERS, WIDTH = 4, 50
N_QUAD, N_VAL = 200, 400
ADAM_EPOCHS, ADAM_LR, LBFGS_MAX_ITER = 5000, 1.0e-3, 500
MAIN_SEED, REPEAT_SEEDS = 42, list(range(10))

REACTION_B = -ALPHA * DT_MAX * E * A0 / (2.0 * (1.0 + BETA / 3.0))
CASES = {
    "A": dict(u_scale=P * L / (E * A0) + ALPHA * DT_MAX * L, end_force=P),
    "B": dict(u_scale=ALPHA * DT_MAX * L, end_force=0.0),
}


def area(x):
    return A0 / (1.0 + BETA * (x / L) ** 2)


def dtemp(x):
    return DT_MAX * x / L


def u_exact(case, x):
    N = P if case == "A" else REACTION_B
    return N / (E * A0) * (x + BETA * x**3 / (3 * L**2)) + ALPHA * DT_MAX * x**2 / (2 * L)


def du_exact(case, x):
    N = P if case == "A" else REACTION_B
    return N / (E * area(x)) + ALPHA * dtemp(x)


def u_mech(case, x):  # mechanical (force-induced) part of the exact displacement
    N = P if case == "A" else REACTION_B
    return N / (E * A0) * (x + BETA * x**3 / (3 * L**2))


def sigma_exact(case, x):
    N = P if case == "A" else REACTION_B
    return N / area(x)


class Net(nn.Module):
    def __init__(self, case):
        super().__init__()
        layers = [nn.Linear(1, WIDTH), nn.Tanh()]
        for _ in range(LAYERS - 1):
            layers += [nn.Linear(WIDTH, WIDTH), nn.Tanh()]
        layers.append(nn.Linear(WIDTH, 1))
        self.net = nn.Sequential(*layers)
        self.case = case
        self.us = CASES[case]["u_scale"]

    def forward(self, x):
        xi = x / L
        g = xi if self.case == "A" else xi * (1.0 - xi)   # hard constraint factor
        return self.us * g * self.net(xi)


def quad(n):
    nodes, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor(0.5 * L * (nodes + 1)).view(-1, 1),
            torch.tensor(0.5 * L * w).view(-1, 1))


def energy(model, xq, wq):
    """Total potential energy, Eq. (9), divided by E*A0*u_ref^2/L."""
    x = xq.clone().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    dens = 0.5 * E * area(x) * du**2 - E * area(x) * ALPHA * dtemp(x) * du
    pi = torch.sum(wq * dens)
    F = CASES[model.case]["end_force"]
    if F != 0.0:
        pi = pi - F * model(torch.tensor([[L]])).squeeze()
    return pi / (E * A0 * model.us**2 / L)


def train(case, seed, n_quad=N_QUAD):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(case)
    xq, wq = quad(n_quad)
    hist = []
    t0 = time.perf_counter()
    adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    for _ in range(ADAM_EPOCHS):
        adam.zero_grad()
        loss = energy(model, xq, wq)
        loss.backward()
        adam.step()
        hist.append(("Adam", float(loss.detach())))
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_MAX_ITER,
                              tolerance_grad=1e-10, tolerance_change=1e-14,
                              history_size=100, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = energy(model, xq, wq)
        loss.backward()
        hist.append(("L-BFGS", float(loss)))
        return loss
    lbfgs.step(closure)
    return model, hist, time.perf_counter() - t0


def evaluate(case, model):
    x = np.linspace(0.0, L, N_VAL)
    xt = torch.tensor(x).view(-1, 1).requires_grad_(True)
    ut = model(xt)
    dut = torch.autograd.grad(ut, xt, torch.ones_like(ut))[0]
    u = ut.detach().numpy().ravel()
    du = dut.detach().numpy().ravel()
    s = E * (du - ALPHA * dtemp(x))
    ue, se = u_exact(case, x), sigma_exact(case, x)
    e = u - ue
    # energy-norm error on Gauss points (elastic-strain normalisation)
    xg, wg = quad(N_QUAD)
    xgt = xg.clone().requires_grad_(True)
    ug = model(xgt)
    dug = torch.autograd.grad(ug, xgt, torch.ones_like(ug))[0].detach().numpy().ravel()
    xg_np, wg_np = xg.numpy().ravel(), wg.numpy().ravel()
    de = dug - du_exact(case, xg_np)
    eps_el = du_exact(case, xg_np) - ALPHA * dtemp(xg_np)
    num = np.sum(wg_np * E * area(xg_np) * de**2)
    den_el = np.sum(wg_np * E * area(xg_np) * eps_el**2)
    den_tot = np.sum(wg_np * E * area(xg_np) * du_exact(case, xg_np)**2)
    um = u_mech(case, x)
    N0 = float(s[0] * area(0.0))
    NL = float(s[-1] * area(L))
    return dict(
        MAE_m=float(np.mean(np.abs(e))),
        RMSE_m=float(np.sqrt(np.mean(e**2))),
        rel_L2_u_pct=float(100 * np.linalg.norm(e) / np.linalg.norm(ue)),
        rel_L2_u_mech_pct=float(100 * np.linalg.norm(e) / np.linalg.norm(um)),
        max_abs_u_m=float(np.max(np.abs(e))),
        max_abs_u_over_peak_pct=float(100 * np.max(np.abs(e)) / np.max(np.abs(ue))),
        argmax_abs_u_x=float(x[np.argmax(np.abs(e))]),
        rel_L2_sigma_pct=float(100 * np.linalg.norm(s - se) / np.linalg.norm(se)),
        max_sigma_err_over_peak_pct=float(100 * np.max(np.abs(s - se)) / np.max(np.abs(se))),
        peak_sigma_exact_MPa=float(se[np.argmax(np.abs(se))] / 1e6),
        peak_sigma_dr_MPa=float(s[np.argmax(np.abs(s))] / 1e6),
        energy_norm_err_pct=float(100 * np.sqrt(num / den_el)),
        energy_norm_err_totalstrain_pct=float(100 * np.sqrt(num / den_tot)),
        force_x0_N=N0, force_xL_N=NL,
        u0_m=float(u[0]), uL_m=float(u[-1]),
    ), dict(x=x, u_exact=ue, u_dr=u, sigma_exact=se, sigma_dr=s)


def fem(n_el, case="A", n_gauss=4):
    """Linear bar elements; element stiffness and thermal load integrated with n_gauss-point Gauss rule."""
    xn = np.linspace(0, L, n_el + 1)
    K = np.zeros((n_el + 1, n_el + 1))
    f = np.zeros(n_el + 1)
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    for e in range(n_el):
        a, b = xn[e], xn[e + 1]
        h = b - a
        xg = 0.5 * (a + b) + 0.5 * h * gp
        wg = 0.5 * h * gw
        EA_int = np.sum(wg * E * area(xg))
        th_int = np.sum(wg * E * area(xg) * ALPHA * dtemp(xg))
        K[e:e + 2, e:e + 2] += EA_int / h**2 * np.array([[1, -1], [-1, 1]])
        f[e:e + 2] += th_int / h * np.array([-1, 1])
    f[-1] += P
    u = np.zeros(n_el + 1)
    u[1:] = np.linalg.solve(K[1:, 1:], f[1:])
    xv = np.linspace(0, L, N_VAL)
    uv = np.interp(xv, xn, u)
    ue = u_exact(case, xv)
    ev = uv - ue
    en = u - u_exact(case, xn)
    # element stresses (constant per element) at the 400 validation points
    idx = np.clip(np.searchsorted(xn, xv, side="right") - 1, 0, n_el - 1)
    s_el = E * ((u[idx + 1] - u[idx]) / (xn[idx + 1] - xn[idx]) - ALPHA * dtemp(xv))
    se = sigma_exact(case, xv)
    # standard recovery: nodal averaging of element axial forces, divided by local area
    xm = 0.5 * (xn[1:] + xn[:-1])
    Ne = E * area(xm) * (np.diff(u) / np.diff(xn) - ALPHA * dtemp(xm))
    Nn = np.r_[Ne[0], 0.5 * (Ne[1:] + Ne[:-1]), Ne[-1]]
    s_rec = np.interp(xv, xn, Nn) / area(xv)
    return dict(elements=n_el, dof=n_el,  # u(0)=0 removed
                nodal_max_abs_err_m=float(np.max(np.abs(en))),
                rel_L2_u_val_pct=float(100 * np.linalg.norm(ev) / np.linalg.norm(ue)),
                max_abs_u_val_m=float(np.max(np.abs(ev))),
                rel_L2_sigma_val_pct=float(100 * np.linalg.norm(s_el - se) / np.linalg.norm(se)),
                rel_L2_sigma_recovered_pct=float(100 * np.linalg.norm(s_rec - se) / np.linalg.norm(se)))


def make_figures(OUT):
    """All manuscript figures, regenerated from the saved CSV files."""
    fa = pd.read_csv(OUT / "case_A_fields_seed42.csv")
    fb = pd.read_csv(OUT / "case_B_fields_seed42.csv")
    x = fa["x"]
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    ax.plot(x, 1e3 * fa["u_exact"], lw=2.5, label="Analytical")
    ax.plot(x, 1e3 * fa["u_dr"], "--", lw=2, label="Deep Ritz")
    ax.set(xlabel="x (m)", ylabel="Displacement u (mm)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "Fig1_caseA_displacement.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    ax.plot(x, 1e9 * np.abs(fa["u_dr"] - fa["u_exact"]), lw=1.8)
    ax.set(xlabel="x (m)", ylabel="Absolute displacement error (nm)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "Fig2_caseA_abs_error.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    ax.plot(x, fa["sigma_exact"] / 1e6, lw=2.5, label="Analytical")
    ax.plot(x, fa["sigma_dr"] / 1e6, "--", lw=2, label="Deep Ritz")
    ax.set(xlabel="x (m)", ylabel="Axial stress σ (MPa)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "Fig3_caseA_stress.png", dpi=300); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].plot(fb["x"], 1e6 * fb["u_exact"], lw=2.5, label="Analytical")
    axs[0].plot(fb["x"], 1e6 * fb["u_dr"], "--", lw=2, label="Deep Ritz")
    axs[0].set(xlabel="x (m)", ylabel="Displacement u (µm)")
    axs[1].plot(fb["x"], fb["sigma_exact"] / 1e6, lw=2.5, label="Analytical")
    axs[1].plot(fb["x"], fb["sigma_dr"] / 1e6, "--", lw=2, label="Deep Ritz")
    axs[1].set(xlabel="x (m)", ylabel="Axial stress σ (MPa)")
    for a in axs:
        a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "Fig4_caseB_displacement_stress.png", dpi=300); plt.close(fig)
    # energy gap to the exact minimum (same 200-point quadrature)
    xq, wq = quad(N_QUAD)
    xg, wg = xq.numpy().ravel(), wq.numpy().ravel()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, c in zip(axs, "AB"):
        du = du_exact(c, xg)
        us, F = CASES[c]["u_scale"], CASES[c]["end_force"]
        pi = np.sum(wg * (0.5 * E * area(xg) * du**2 - E * area(xg) * ALPHA * dtemp(xg) * du)) - F * u_exact(c, L)
        pi /= E * A0 * us**2 / L
        h = pd.read_csv(OUT / f"case_{c}_loss_history.csv")
        gap = h["dimensionless_energy"].to_numpy() - pi
        na = int((h["stage"] == "Adam").sum())
        ax.semilogy(np.arange(len(gap)), np.maximum(gap, 1e-16), lw=1)
        ax.axvline(na, color="k", ls=":", lw=1)
        ax.text(na + 40, gap.max() / 3, "L-BFGS", va="top")
        ax.set(xlabel="Adam epoch / L-BFGS function evaluation",
               ylabel=r"$\tilde\Pi[u_\theta]-\tilde\Pi[u_{exact}]$", title=f"Case {c}")
        ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "Fig5_energy_history.png", dpi=300); plt.close(fig)


def main():
    env = dict(python=sys.version.split()[0], platform=platform.platform(),
               torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
               threads=torch.get_num_threads())
    print(env)
    summary = {"environment": env}

    # ---- main runs (seed 42) ----
    for case in ["A", "B"]:
        model, hist, t = train(case, MAIN_SEED)
        m, f = evaluate(case, model)
        m["training_time_s"] = t
        m["lbfgs_closure_calls"] = sum(1 for h in hist if h[0] == "L-BFGS")
        summary[f"case_{case}_seed42"] = m
        pd.DataFrame(f).to_csv(OUT / f"case_{case}_fields_seed42.csv", index=False)
        pd.DataFrame(hist, columns=["stage", "dimensionless_energy"]).to_csv(
            OUT / f"case_{case}_loss_history.csv", index_label="iteration")
        print(case, json.dumps(m, indent=1))


    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(OUT)

    # ---- repeatability (seeds 0-9), both cases ----
    for case in ["A", "B"]:
        rows = []
        for s in REPEAT_SEEDS:
            model, _, t = train(case, s)
            m, _ = evaluate(case, model)
            m.update(seed=s, training_time_s=t)
            rows.append(m)
            print(case, s, round(m["rel_L2_u_pct"], 5), round(m["rel_L2_sigma_pct"], 4), flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"case_{case}_repeatability_seeds0-9.csv", index=False)
        cols = ["MAE_m", "RMSE_m", "rel_L2_u_pct", "rel_L2_u_mech_pct", "rel_L2_sigma_pct",
                "max_sigma_err_over_peak_pct", "energy_norm_err_pct"]
        df[cols].agg(["mean", "std", "min", "max"]).to_csv(OUT / f"case_{case}_repeatability_summary.csv")

    # ---- quadrature sensitivity (Case A, seed 42) ----
    rows = []
    for nq in [50, 100, 200, 400]:
        model, _, t = train("A", MAIN_SEED, n_quad=nq)
        m, _ = evaluate("A", model)
        rows.append(dict(N_quad=nq, rel_L2_u_pct=m["rel_L2_u_pct"], max_abs_u_m=m["max_abs_u_m"],
                         rel_L2_sigma_pct=m["rel_L2_sigma_pct"], training_time_s=t))
        print("quad", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "case_A_quadrature_sensitivity.csv", index=False)

    # ---- FEM h-convergence (Case A) ----
    ft = pd.DataFrame([fem(n) for n in [50, 100, 200, 400, 2000]])
    ft.to_csv(OUT / "case_A_fem_convergence.csv", index=False)
    print(ft)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        make_figures(OUT)
    else:
        main()
