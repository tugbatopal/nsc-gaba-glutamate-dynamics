#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Information S1
Analytical equilibrium, stability, verification, and figure-generation code for:
"Glutamate-Driven Neurogenesis Under Michaelis-Menten GABA Inhibitory Feedback:
A Dynamical Systems Analysis of Neural Stem Cell Lineage Progression"

Author: Tugba Topal
"""

import os
import sys
import csv
import time
import argparse
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov, expm
from scipy.optimize import brentq, fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

# Default illustrative parameter set (see Table 1 of the manuscript for units)
PARAMS = dict(
    a1=0.10, a2=0.80, a3=0.10,      # proliferation rates
    b=1.00,                          # inhibition half-saturation constant
    c1=0.12, c2=0.90, c3=0.90,      # compartment loss/differentiation rates
    s1=0.10, s2=0.30,                # TAC->PN and PN->MN transition rates
    d=0.50,                          # MN clearance rate
    k=0.01, delta=1.00,              # GABA synthesis and degradation rates
    g=1.00                           # glutamate coupling gain
)

IC0 = np.array([1.0, 2.0, 5.0, 3.0, 0.05])  # initial state (X, Y, Z, U, A)
NAMES = ["X", "Y", "Z", "U", "A"]
PNAMES = ["a1", "a2", "a3", "b", "c1", "c2", "c3", "s1", "s2", "d", "k", "delta", "g"]
PLABELS = [
    r"$a_1$", r"$a_2$", r"$a_3$", r"$b$", r"$c_1$", r"$c_2$", r"$c_3$",
    r"$s_1$", r"$s_2$", r"$d$", r"$k$", r"$\delta$", r"$g$"
]

OUT_FIG = "./figures"
OUT_DAT = "./output"


# ----------------------------------------------------------------------------
# Model equations and Jacobian
# ----------------------------------------------------------------------------
def inhibition(A, Z, p):
    """Saturating GABA inhibition term h(A, Z) = b / (b + A*Z)."""
    return p["b"] / (p["b"] + A * Z)


def rhs(t, y, G, p):
    """Right-hand side of the five-compartment ODE system."""
    X, Y, Z, U, A = y
    h = inhibition(A, Z, p)
    e = p["g"] * G
    return [
        (p["a1"] * h + e - p["c1"]) * X,
        p["c1"] * X + (p["a2"] * h + e - p["c2"]) * Y,
        p["s1"] * Y + (p["a3"] * h + e - p["c3"]) * Z,
        p["s2"] * Z - p["d"] * U,
        p["k"] * Z - p["delta"] * A,
    ]


def rhs_vec(Y, G, p):
    """Vectorized right-hand side for an ensemble Y of shape (M, 5)."""
    X, Yc, Z, U, A = Y.T
    h = p["b"] / (p["b"] + A * Z)
    e = p["g"] * G
    return np.stack([
        (p["a1"] * h + e - p["c1"]) * X,
        p["c1"] * X + (p["a2"] * h + e - p["c2"]) * Yc,
        p["s1"] * Yc + (p["a3"] * h + e - p["c3"]) * Z,
        p["s2"] * Z - p["d"] * U,
        p["k"] * Z - p["delta"] * A,
    ], axis=1)


def jac(y, G, p):
    """Analytical 5x5 Jacobian matrix evaluated at state y."""
    X, Y, Z, U, A = y
    e = p["g"] * G
    den = p["b"] + A * Z
    h = p["b"] / den
    hZ = -p["b"] * A / (den ** 2)
    hA = -p["b"] * Z / (den ** 2)

    J = np.zeros((5, 5))
    J[0, 0] = p["a1"] * h + e - p["c1"]
    J[0, 2] = p["a1"] * X * hZ
    J[0, 4] = p["a1"] * X * hA

    J[1, 0] = p["c1"]
    J[1, 1] = p["a2"] * h + e - p["c2"]
    J[1, 2] = p["a2"] * Y * hZ
    J[1, 4] = p["a2"] * Y * hA

    J[2, 1] = p["s1"]
    J[2, 2] = p["a3"] * h + e - p["c3"] + p["a3"] * Z * hZ
    J[2, 4] = p["a3"] * Z * hA

    J[3, 2] = p["s2"]
    J[3, 3] = -p["d"]

    J[4, 2] = p["k"]
    J[4, 4] = -p["delta"]
    return J


def numjac(f, y, eps=1e-6):
    """Central finite-difference Jacobian used for independent verification."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    J = np.zeros((n, n))
    for j in range(n):
        step = eps * max(1.0, abs(y[j]))
        dy = np.zeros(n)
        dy[j] = step
        J[:, j] = (np.asarray(f(y + dy)) - np.asarray(f(y - dy))) / (2.0 * step)
    return J


# ----------------------------------------------------------------------------
# Closed-form equilibria and stability classification
# ----------------------------------------------------------------------------
def thresholds(p=PARAMS):
    """Lower (G_L) and upper (G_U) glutamate boundaries of E_XYZ."""
    return (p["c1"] - p["a1"]) / p["g"], p["c1"] / p["g"]


def h_star(G, p=PARAMS):
    """Equilibrium inhibition level h* = (c1 - g*G) / a1 for X* > 0."""
    return (p["c1"] - p["g"] * G) / p["a1"]


def Z_from_h(h, p=PARAMS):
    """Steady-state PN count Z* corresponding to inhibition level h."""
    return np.sqrt(p["delta"] * p["b"] / p["k"] * (1.0 / h - 1.0))


def _fill(X, Y, Z, p):
    return np.array([X, Y, Z, p["s2"] * Z / p["d"], p["k"] * Z / p["delta"]])


def eq_trivial():
    """Extinction equilibrium E0 = (0, 0, 0, 0, 0)."""
    return np.zeros(5)


def eq_pn_only(G, p=PARAMS):
    """PN-only equilibrium E_Z = (0, 0, Z*, U*, A*)."""
    h = (p["c3"] - p["g"] * G) / p["a3"]
    if not 0.0 < h < 1.0:
        return None
    return _fill(0.0, 0.0, Z_from_h(h, p), p)


def eq_tac_pn(G, p=PARAMS):
    """TAC-PN equilibrium E_YZ = (0, Y*, Z*, U*, A*)."""
    h = (p["c2"] - p["g"] * G) / p["a2"]
    if not 0.0 < h < 1.0:
        return None
    al3 = p["c3"] - p["a3"] * h - p["g"] * G
    if al3 <= 0.0:
        return None
    Z = Z_from_h(h, p)
    return _fill(0.0, al3 * Z / p["s1"], Z, p)


def eq_coexist(G, p=PARAMS):
    """Stem-cell-supported coexistence equilibrium E_XYZ."""
    h = h_star(G, p)
    if not 0.0 < h < 1.0:
        return None
    e = p["g"] * G
    al2 = p["c2"] - p["a2"] * h - e
    al3 = p["c3"] - p["a3"] * h - e
    if al2 <= 0.0 or al3 <= 0.0:
        return None
    Z = Z_from_h(h, p)
    Y = al3 * Z / p["s1"]
    X = al2 * Y / p["c1"]
    return _fill(X, Y, Z, p)


def max_re_eig(y, G, p=PARAMS):
    """Largest real part among eigenvalues of the Jacobian at y."""
    return float(np.max(np.linalg.eigvals(jac(y, G, p)).real))


def all_equilibria(G, p=PARAMS):
    """Return (label, state, max_re_lambda, is_stable) for all existing equilibria."""
    out = []
    candidates = [
        ("E0", eq_trivial()),
        ("E_Z", eq_pn_only(G, p)),
        ("E_YZ", eq_tac_pn(G, p)),
        ("E_XYZ", eq_coexist(G, p)),
    ]
    for name, y in candidates:
        if y is None:
            continue
        lam = max_re_eig(y, G, p)
        out.append((name, y, lam, lam < -1e-12))
    return out


def attractor_label(G, p=PARAMS):
    """Classify the attracting state from positive initial conditions."""
    stable = [n for n, _, _, is_stab in all_equilibria(G, p) if is_stab]
    if len(stable) == 0:
        return "runaway"
    if len(stable) > 1:
        return "bistable"
    return {
        "E0": "extinction",
        "E_Z": "PN-only",
        "E_YZ": "TAC-PN",
        "E_XYZ": "coexistence",
    }[stable[0]]


def elasticity_Z(G, p=PARAMS):
    """Analytical elasticities d ln(Z*) / d ln(p) on the coexistence branch."""
    h = h_star(G, p)
    den = 2.0 * p["a1"] * h * (1.0 - h)
    S = {n: 0.0 for n in PNAMES}
    S["a1"] = 1.0 / (2.0 * (1.0 - h))
    S["c1"] = -p["c1"] / den
    S["g"] = p["g"] * G / den
    S["b"] = 0.5
    S["delta"] = 0.5
    S["k"] = -0.5
    return S


def elasticity_numeric(G, p=PARAMS, rel=1e-6):
    """Central-difference elasticities across all five state variables (5 x 13)."""
    y0 = eq_coexist(G, p)
    E = np.zeros((5, len(PNAMES)))
    for j, n in enumerate(PNAMES):
        pu = dict(p)
        pd = dict(p)
        pu[n] = p[n] * (1.0 + rel)
        pd[n] = p[n] * (1.0 - rel)
        yu, yd = eq_coexist(G, pu), eq_coexist(G, pd)
        E[:, j] = (np.log(yu) - np.log(yd)) / (np.log(1.0 + rel) - np.log(1.0 - rel))
    return E, y0


# ----------------------------------------------------------------------------
# Numerical integration, return times, and linear-noise statistics
# ----------------------------------------------------------------------------
def simulate(G, p=PARAMS, ic=IC0, T=600.0, n=1500, rtol=1e-9, atol=1e-12, stop_at=None):
    """Integrate the ODE system using the implicit Radau method."""
    ev = None
    if stop_at is not None:
        ev = lambda t, y, *a: y[0] + y[1] + y[2] - stop_at
        ev.terminal = True
    return solve_ivp(
        rhs, (0.0, T), ic, args=(G, p), method="Radau",
        rtol=rtol, atol=atol, t_eval=np.linspace(0.0, T, n), events=ev
    )


def leading_eigs(y, G, p=PARAMS):
    """Eigenvalues of the Jacobian sorted by descending real part."""
    ev = np.linalg.eigvals(jac(y, G, p))
    return ev[np.argsort(-ev.real)]


def return_time_sim(G, p=PARAMS, amp=1e-4, seed=0):
    """Estimate asymptotic relaxation time from a small perturbation of E_XYZ."""
    y0 = eq_coexist(G, p)
    ev = leading_eigs(y0, G, p)
    lam1 = ev[0].real
    lam2 = ev[1].real
    ta = 8.0 / abs(min(lam2, -1e-3)) if lam2 < -1e-3 else 80.0
    T = ta + 7.0 / abs(lam1)

    rng = np.random.default_rng(seed)
    ic = y0 * (1.0 + amp * rng.choice([-1.0, 1.0], 5))
    t_eval = np.linspace(0.0, T, 6000)
    sol = solve_ivp(
        rhs, (0.0, T), ic, args=(G, p), method="Radau",
        rtol=1e-12, atol=1e-14, t_eval=t_eval
    )
    dev = np.linalg.norm((sol.y.T - y0) / y0, axis=1)
    mask = (sol.t > ta) & (dev > 1e-11)
    slope = np.polyfit(sol.t[mask], np.log(dev[mask]), 1)[0]
    return -1.0 / slope, -1.0 / lam1


def lna_stats(G, p=PARAMS, eps=0.01, lag=10.0):
    """Stationary CV^2 and lag autocorrelation of Z under the linear-noise approximation."""
    y0 = eq_coexist(G, p)
    J = jac(y0, G, p)
    D = np.diag((eps * y0) ** 2)
    S = solve_continuous_lyapunov(J, -D)
    cv2 = S[2, 2] / (y0[2] ** 2)
    ac = (expm(J * lag) @ S)[2, 2] / S[2, 2]
    return cv2, ac


def em_ensemble(G, p=PARAMS, eps=0.01, lag=10.0, M=800, dt=0.05, seed=1):
    """Euler-Maruyama ensemble simulation of the stochastic nonlinear model."""
    y0 = eq_coexist(G, p)
    lam1 = abs(leading_eigs(y0, G, p)[0].real)
    T = 6.0 / lam1 + 100.0
    rng = np.random.default_rng(seed)
    sig = eps * y0
    Y = np.tile(y0, (M, 1))
    sq = np.sqrt(dt)

    for _ in range(int(T / dt)):
        Y = Y + rhs_vec(Y, G, p) * dt + sig * sq * rng.standard_normal((M, 5))
        np.maximum(Y, 0.0, out=Y)
    Z0 = Y[:, 2].copy()

    for _ in range(int(lag / dt)):
        Y = Y + rhs_vec(Y, G, p) * dt + sig * sq * rng.standard_normal((M, 5))
        np.maximum(Y, 0.0, out=Y)
    Z1 = Y[:, 2]

    cv2 = np.var(Z0) / (y0[2] ** 2)
    ac = np.corrcoef(Z0, Z1)[0, 1]
    return cv2, ac


# ----------------------------------------------------------------------------
# Model extensions for structural robustness (Section 3.8)
# ----------------------------------------------------------------------------
def rhs_capacity(t, y, G, p, K):
    """Finite niche capacity extension with logistic crowding factor phi = 1 - (X+Y+Z)/K."""
    X, Y, Z, U, A = y
    h = inhibition(A, Z, p)
    e = p["g"] * G
    phi = 1.0 - (X + Y + Z) / K
    return [
        (p["a1"] * h + e) * phi * X - p["c1"] * X,
        p["c1"] * X + (p["a2"] * h + e) * phi * Y - p["c2"] * Y,
        p["s1"] * Y + (p["a3"] * h + e) * phi * Z - p["c3"] * Z,
        p["s2"] * Z - p["d"] * U,
        p["k"] * Z - p["delta"] * A,
    ]


def eq_capacity(G, p, K):
    """Coexistence equilibrium under finite niche capacity K."""
    kk = p["k"] / (p["delta"] * p["b"])
    e = p["g"] * G

    def parts(Z):
        h = 1.0 / (1.0 + kk * Z * Z)
        phi = p["c1"] / (p["a1"] * h + e)
        al2 = p["c2"] - (p["a2"] * h + e) * phi
        al3 = p["c3"] - (p["a3"] * h + e) * phi
        Y = Z * al3 / p["s1"]
        X = Y * al2 / p["c1"]
        return phi, al2, al3, X, Y

    def residual(Z):
        phi, _, _, X, Y = parts(Z)
        return K * (1.0 - phi) - (X + Y + Z)

    if residual(1e-9) <= 0.0:
        return None
    Zs = brentq(residual, 1e-9, 1e8)
    phi, al2, al3, X, Y = parts(Zs)
    if al2 <= 0.0 or al3 <= 0.0:
        return None
    return _fill(X, Y, Zs, p)


def rhs_gabaonly(t, y, G, p):
    """Variant with receptor saturation on GABA alone: h(A) = b / (b + A)."""
    X, Y, Z, U, A = y
    h = p["b"] / (p["b"] + A)
    e = p["g"] * G
    return [
        (p["a1"] * h + e - p["c1"]) * X,
        p["c1"] * X + (p["a2"] * h + e - p["c2"]) * Y,
        p["s1"] * Y + (p["a3"] * h + e - p["c3"]) * Z,
        p["s2"] * Z - p["d"] * U,
        p["k"] * Z - p["delta"] * A,
    ]


def eq_gabaonly(G, p=PARAMS):
    """Closed-form coexistence equilibrium for GABA-only inhibition."""
    h = h_star(G, p)
    if not 0.0 < h < 1.0:
        return None
    Z = p["delta"] * p["b"] / p["k"] * (1.0 / h - 1.0)
    e = p["g"] * G
    al2 = p["c2"] - p["a2"] * h - e
    al3 = p["c3"] - p["a3"] * h - e
    if al2 <= 0.0 or al3 <= 0.0:
        return None
    Y = al3 * Z / p["s1"]
    return _fill(al2 * Y / p["c1"], Y, Z, p)


def rhs_source(t, y, G, p, S):
    """Variant with constant exogenous GABA source S."""
    X, Y, Z, U, A = y
    h = inhibition(A, Z, p)
    e = p["g"] * G
    return [
        (p["a1"] * h + e - p["c1"]) * X,
        p["c1"] * X + (p["a2"] * h + e - p["c2"]) * Y,
        p["s1"] * Y + (p["a3"] * h + e - p["c3"]) * Z,
        p["s2"] * Z - p["d"] * U,
        S + p["k"] * Z - p["delta"] * A,
    ]


def eq_source(G, p, S):
    """Closed-form coexistence equilibrium with constant exogenous GABA source S."""
    h = h_star(G, p)
    if not 0.0 < h < 1.0:
        return None
    q = p["delta"] * p["b"] * (1.0 / h - 1.0)
    Z = (-S + np.sqrt(S * S + 4.0 * p["k"] * q)) / (2.0 * p["k"])
    e = p["g"] * G
    al2 = p["c2"] - p["a2"] * h - e
    al3 = p["c3"] - p["a3"] * h - e
    Y = al3 * Z / p["s1"]
    return np.array([
        al2 * Y / p["c1"], Y, Z, p["s2"] * Z / p["d"],
        (S + p["k"] * Z) / p["delta"],
    ])


def rhs_gains(t, y, G, p, g1, g2, g3):
    """Variant with compartment-specific glutamate gains (g1, g2, g3)."""
    X, Y, Z, U, A = y
    h = inhibition(A, Z, p)
    return [
        (p["a1"] * h + g1 * G - p["c1"]) * X,
        p["c1"] * X + (p["a2"] * h + g2 * G - p["c2"]) * Y,
        p["s1"] * Y + (p["a3"] * h + g3 * G - p["c3"]) * Z,
        p["s2"] * Z - p["d"] * U,
        p["k"] * Z - p["delta"] * A,
    ]


def sim_label(G, p=PARAMS, T=6000.0):
    """Classify long-time outcome from direct ODE integration."""
    ev = lambda t, y, *a: y[0] + y[1] + y[2] - 1e8
    ev.terminal = True
    sol = solve_ivp(
        rhs, (0.0, T), IC0, args=(G, p), method="Radau",
        rtol=1e-8, atol=1e-11, events=ev, t_eval=np.array([T / 2.0, T])
    )
    if sol.status == 1 or sol.t_events[0].size > 0:
        return "runaway"
    y_half, y_end = sol.y[:, 0], sol.y[:, -1]
    tot_half, tot_end = y_half[:3].sum(), y_end[:3].sum()
    if tot_end > 1e5 or (tot_end > 1.5 * tot_half and tot_end > 1e3):
        return "runaway"
    X, Y, Z = y_end[:3]
    if max(X, Y, Z) < 1e-3:
        return "extinction"
    if X > 1e-2 and Y > 1e-2 and Z > 1e-2:
        return "coexistence"
    if X < 1e-3 and Y < 1e-3 and Z > 1e-2:
        return "PN-only"
    if X < 1e-3 and Y > 1e-2 and Z > 1e-2:
        return "TAC-PN"
    return "other"


# ----------------------------------------------------------------------------
# Numerical verification suite
# ----------------------------------------------------------------------------
class VerificationLog:
    def __init__(self):
        self.lines = []
        self.ok = True

    def log(self, text=""):
        print(text)
        self.lines.append(text)

    def check(self, label, passed, detail=""):
        self.ok &= bool(passed)
        tag = "OK" if passed else "FAIL"
        self.log(f"  [{tag}] {label}  {detail}")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


def verify(fast=False):
    log = VerificationLog()
    p = PARAMS
    GL, GU = thresholds(p)
    rng = np.random.default_rng(12345)

    log.log("Numerical Verification Report")
    log.log("-" * 72)
    log.log(f"Thresholds: G_L = (c1 - a1)/g = {GL:.6f},  G_U = c1/g = {GU:.6f}")

    log.log("\n1. Closed-form coexistence equilibrium residuals")
    for G in (0.03, 0.05, 0.08, 0.10, 0.118):
        y = eq_coexist(G, p)
        res = np.max(np.abs(rhs(0.0, y, G, p)))
        log.check(
            f"G={G}: Z*={y[2]:.4f}, X*={y[0]:.2f}, Y*={y[1]:.2f}",
            res < 1e-10, f"max|f|={res:.1e}"
        )
    z_formula = lambda G: np.sqrt(p["delta"] * p["b"] / p["k"] * (G - GL) / (GU - G))
    log.check(
        "Z*^2 = (delta*b/k)*(G - G_L)/(G_U - G)",
        abs(z_formula(0.08) - eq_coexist(0.08, p)[2]) < 1e-9
    )

    log.log("\n2. Analytical Jacobian vs central finite differences")
    for G in (0.05, 0.118):
        y = eq_coexist(G, p)
        diff = np.max(np.abs(jac(y, G, p) - numjac(lambda v: rhs(0.0, v, G, p), y)))
        log.check(f"equilibrium G={G}", diff < 1e-6, f"max diff={diff:.1e}")
    y_test = np.array([3.0, 2.0, 1.5, 0.7, 0.2])
    diff = np.max(np.abs(jac(y_test, 0.07, p) - numjac(lambda v: rhs(0.0, v, 0.07, p), y_test)))
    log.check("arbitrary off-equilibrium state", diff < 1e-6, f"max diff={diff:.1e}")

    log.log("\n3. Asymptotic scaling of leading eigenvalue near thresholds")
    for dist in (1e-3, 1e-4):
        r_lo = leading_eigs(eq_coexist(GL + dist, p), GL + dist, p)[0].real / (-2.0 * p["g"] * dist)
        r_up = leading_eigs(eq_coexist(GU - dist, p), GU - dist, p)[0].real / (-2.0 * p["g"] * dist)
        log.check(
            f"dist={dist:.0e}",
            abs(r_lo - 1.0) < 0.2 and abs(r_up - 1.0) < 0.05,
            f"ratio lower={r_lo:.4f}, upper={r_up:.4f}"
        )
    all_neg = all(max_re_eig(eq_coexist(G, p), G, p) < 0.0 for G in np.linspace(GL + 1e-4, GU - 1e-4, 400))
    log.check("Re(lambda_1) < 0 across entire coexistence interval (400 points)", all_neg)

    log.log("\n4. Random-start Newton census of equilibria")
    def census(G, pp, n_starts):
        found = []
        for _ in range(n_starts):
            x0 = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), 5)) * rng.choice([0, 1], 5, p=[0.3, 0.7])
            sol, info, ier, _ = fsolve(lambda v: rhs(0.0, np.abs(v), G, pp), x0, full_output=True)
            if ier == 1 and np.max(np.abs(info["fvec"])) < 1e-9:
                s = np.abs(sol)
                if s[:3].max() > 1e-3 and not any(np.allclose(s, q, rtol=1e-4, atol=1e-6) for q in found):
                    found.append(s)
        return found

    n_cen = 400 if fast else 1500
    for G, pp, tag in (
        (0.05, p, "default"),
        (0.115, p, "default"),
        (0.13, p, "default"),
        (0.04, {**p, "c3": 0.08}, "c3=0.08"),
        (0.07, {**p, "c3": 0.08}, "c3=0.08"),
    ):
        known = [y for _, y, _, _ in all_equilibria(G, pp) if y[:3].max() > 1e-3]
        found = census(G, pp, n_cen)
        matched = all(any(np.allclose(f, k, rtol=1e-3, atol=1e-5) for k in known) for f in found)
        log.check(
            f"G={G} ({tag}): {len(found)} nonzero equilibria found, {len(known)} analytical",
            matched
        )

    log.log("\n5. Uniqueness of stable attractor (absence of bistability)")
    n_stab = [sum(s for *_, s in all_equilibria(G, p)) for G in np.linspace(0.0, 0.3, 601)]
    log.check("1D sweep over G in [0, 0.3]", max(n_stab) <= 1, f"max stable={max(n_stab)}")
    mx = 0
    for c3 in np.linspace(0.02, 1.2, 40 if fast else 80):
        for G in np.linspace(0.0, 0.2, 40 if fast else 80):
            mx = max(mx, sum(s for *_, s in all_equilibria(G, {**p, "c3": c3})))
    log.check("2D sweep over (G, c3) plane", mx <= 1, f"max stable={mx}")

    log.log("\n6. Global convergence to E_XYZ from random positive initial states")
    n_ic = 4 if fast else 10
    for G in (0.03, 0.06, 0.09, 0.11):
        ys = eq_coexist(G, p)
        lam = abs(leading_eigs(ys, G, p)[0].real)
        Tn = max(3000.0, 25.0 / lam)
        converged = 0
        for _ in range(n_ic):
            ic = np.exp(rng.uniform(np.log(1e-2), np.log(1e3), 5))
            sol = solve_ivp(rhs, (0.0, Tn), ic, args=(G, p), method="Radau", rtol=1e-9, atol=1e-12, t_eval=[Tn])
            converged += int(np.allclose(sol.y[:, -1], ys, rtol=2e-2))
        log.check(f"G={G}", converged == n_ic, f"{converged}/{n_ic} converged")

    log.log("\n7. Positivity and exponential growth rate above G_U")
    sol = simulate(0.05, T=300.0, n=300)
    log.check("solution positivity", sol.y.min() > -1e-9)
    for G in (0.15, 0.30):
        sol = solve_ivp(
            rhs, (0.0, 300.0), IC0, args=(G, p), method="Radau",
            rtol=1e-9, atol=1e-12, t_eval=np.linspace(0.0, 300.0, 301)
        )
        rate = np.polyfit(sol.t[200:], np.log(sol.y[0, 200:]), 1)[0]
        log.check(
            f"G={G}: growth rate={rate:.4f} vs g*G - c1={p['g']*G - p['c1']:.4f}",
            abs(rate - (p["g"] * G - p["c1"])) < 0.02 * max(1.0, G) and sol.status == 0
        )

    log.log("\n8. Reduced PN loss rate (c3 = 0.08): PN-only equilibrium E_Z")
    p8 = {**p, "c3": 0.08}
    log.check(
        "coexistence equilibrium absent for G in {0.01, 0.04, 0.07}",
        all(eq_coexist(G, p8) is None for G in (0.01, 0.04, 0.07))
    )
    for G in (0.01, 0.04, 0.052, 0.07):
        y = eq_pn_only(G, p8)
        lam = max_re_eig(y, G, p8)
        sol = solve_ivp(rhs, (0.0, 4000.0), IC0, args=(G, p8), method="Radau", rtol=1e-10, atol=1e-13, t_eval=[4000.0])
        log.check(
            f"G={G}: Z*={y[2]:.3f}, Re(lambda_1)={lam:.4f}, end=({sol.y[0,-1]:.1e},{sol.y[1,-1]:.1e},{sol.y[2,-1]:.3f})",
            lam < 0.0 and sol.y[0, -1] < 1e-6 and abs(sol.y[2, -1] - y[2]) < 1e-3
        )
    log.check(
        "upper existence boundary of E_Z at c3/g = 0.08",
        eq_pn_only(0.0799, p8) is not None and eq_pn_only(0.0801, p8) is None
    )

    log.log("\n9. Stability of extinction state E0 for G < G_L")
    for G in (0.0, 0.01):
        max_lam = max(max_re_eig(eq_trivial(), G, {**p, "c3": c3}) for c3 in np.linspace(0.2, 1.2, 21))
        log.check(f"G={G} across c3 in [0.2, 1.2]", max_lam < 0.0, f"max Re(lambda)={max_lam:.4f}")

    log.log("\n10. Analytical elasticities vs central finite differences")
    Sa = elasticity_Z(0.08, p)
    En, _ = elasticity_numeric(0.08, p)
    err = max(abs(Sa[n] - En[2, j]) for j, n in enumerate(PNAMES))
    log.check("elasticities of Z* across 13 parameters", err < 1e-5, f"max diff={err:.1e}")

    log.log("\n11. Return times from small perturbations vs 1/|Re(lambda_1)|")
    for G in (0.021, 0.05, 0.08, 0.118):
        ts, te = return_time_sim(G, p)
        log.check(f"G={G}: tau_eig={te:.1f}, tau_sim={ts:.1f}", abs(ts / te - 1.0) < 0.03)

    log.log("\n12. Structural robustness extensions")
    K = 5000.0
    for G in (0.05, 0.118, 0.20, 0.40):
        y = eq_capacity(G, p, K)
        f = lambda v, G=G: rhs_capacity(0.0, v, G, p, K)
        lam = np.max(np.linalg.eigvals(numjac(f, y)).real)
        sol = solve_ivp(rhs_capacity, (0.0, 20000.0), IC0, args=(G, p, K), method="Radau", rtol=1e-10, atol=1e-13, t_eval=[20000.0])
        log.check(
            f"finite capacity K={K:g}, G={G}: Z*={y[2]:.2f}, total={y[:3].sum():.0f}<K, Re(lambda_1)={lam:.3f}",
            lam < 0.0 and y[:3].sum() < K and np.allclose(sol.y[:, -1], y, rtol=2e-2)
        )

    for G in (0.03, 0.08, 0.115):
        y = eq_gabaonly(G, p)
        f = lambda v, G=G: rhs_gabaonly(0.0, v, G, p)
        lam = np.max(np.linalg.eigvals(numjac(f, y)).real)
        sol = solve_ivp(rhs_gabaonly, (0.0, 40000.0), IC0, args=(G, p), method="Radau", rtol=1e-10, atol=1e-13, t_eval=[40000.0])
        log.check(
            f"GABA-only inhibition, G={G}: Z*={y[2]:.2f}, Re(lambda_1)={lam:.4f}",
            lam < 0.0 and np.allclose(sol.y[:, -1], y, rtol=2e-2) and np.max(np.abs(f(y))) < 1e-9
        )

    S0 = 0.05
    for G in (0.05, 0.10):
        y = eq_source(G, p, S0)
        res = np.max(np.abs(rhs_source(0.0, y, G, p, S0)))
        log.check(f"exogenous GABA source S={S0}, G={G}", res < 1e-9, f"res={res:.1e}")

    g1, g2, g3 = 1.5, 0.6, 0.4
    GLg, GUg = (p["c1"] - p["a1"]) / g1, p["c1"] / g1
    end_vals = []
    for G in (GLg - 0.004, 0.5 * (GLg + GUg), GUg + 0.004):
        sol = solve_ivp(rhs_gains, (0.0, 30000.0), IC0, args=(G, p, g1, g2, g3), method="Radau", rtol=1e-9, atol=1e-12, t_eval=[30000.0])
        end_vals.append(sol.y[0, -1])
    log.check(
        f"compartment-specific gains (G_L={GLg:.4f}, G_U={GUg:.4f})",
        end_vals[0] < 1e-3 and 1.0 < end_vals[1] < 1e5 and end_vals[2] > 1e6
    )

    log.log("\nSummary: " + ("All verification checks passed." if log.ok else "One or more checks failed."))
    os.makedirs(OUT_DAT, exist_ok=True)
    log.save(os.path.join(OUT_DAT, "verification_report.txt"))
    return log.ok


# ----------------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------------
def make_tables():
    os.makedirs(OUT_DAT, exist_ok=True)
    p = PARAMS
    rows = []
    for G in (0.03, 0.05, 0.08, 0.10, 0.118):
        y = eq_coexist(G, p)
        ev = leading_eigs(y, G, p)
        rows.append([G] + list(np.round(y, 4)) + [round(ev[0].real, 5), round(1.0 / abs(ev[0].real), 1)])

    with open(os.path.join(OUT_DAT, "Table3_equilibria.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["GLU", "X*", "Y*", "Z*", "U*", "A*", "Re_lambda1", "tau_return"])
        writer.writerows(rows)

    E, _ = elasticity_numeric(0.08, p)
    with open(os.path.join(OUT_DAT, "Table4_elasticities.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["output"] + PNAMES)
        for i, nm in enumerate(["X*", "Y*", "Z*", "U*", "A*"]):
            writer.writerow([nm] + [round(v, 4) for v in E[i]])
    return rows


# ----------------------------------------------------------------------------
# Publication figures
# ----------------------------------------------------------------------------
CB = dict(
    blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
    purple="#CC79A7", sky="#56B4E9", black="#000000", grey="#7F7F7F"
)


def set_style():
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "mathtext.fontset": "stixsans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, letter, dx=-0.13, dy=1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT_FIG, exist_ok=True)
    fig.savefig(f"{OUT_FIG}/{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT_FIG}/{name}.tif", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{OUT_FIG}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {name}")


GL_, GU_ = thresholds(PARAMS)


def fig1():
    Gs, cols = (0.03, 0.06, 0.09), (CB["blue"], CB["green"], CB["verm"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    ax = axes.ravel()
    ttl = ["NSC  X", "TAC  Y", "PN  Z", "MN  U", "GABA  A"]
    for G, c in zip(Gs, cols):
        sol = simulate(G, T=1000.0, n=1500)
        ys = eq_coexist(G)
        for i in range(5):
            ax[i].plot(sol.t, sol.y[i], color=c, label=f"GLU = {G:g}")
            ax[i].axhline(ys[i], color=c, ls=":", lw=0.9)
    for i in range(5):
        ax[i].set_title(ttl[i], loc="left", fontsize=8.5)
        ax[i].set_xlabel("Time (a.u.)")
        ax[i].set_ylabel("Cells" if i < 4 else "Concentration (a.u.)")
        ax[i].set_ylim(bottom=0)
        panel(ax[i], "ABCDEF"[i])
    ax[0].legend(loc="upper left", title="dotted: analytic $x^*$", title_fontsize=6.5)

    G = 0.118
    sol = simulate(G, T=3000.0, n=3000)
    ys = eq_coexist(G)
    lam1 = leading_eigs(ys, G)[0].real
    for i, (nm, c) in enumerate(zip(["X", "Y", "Z"], (CB["blue"], CB["orange"], CB["verm"]))):
        ax[5].plot(sol.t, sol.y[i] / ys[i], color=c, label=f"{nm}(t) / {nm}$^*$")
    ax[5].axhline(1.0, color="k", ls=":", lw=0.9)
    ax[5].axvline(5.0 / abs(lam1), color=CB["grey"], ls="--", lw=0.9)
    ax[5].text(5.0 / abs(lam1) - 40.0, 0.45, r"$5/|\lambda_1|$", color=CB["grey"], fontsize=7, ha="right")
    ax[5].set_title("Near-critical, GLU = 0.118", loc="left", fontsize=8.5)
    ax[5].set_xlabel("Time (a.u.)")
    ax[5].set_ylabel("Fraction of equilibrium")
    ax[5].set_ylim(0, 1.15)
    ax[5].legend(loc="lower right")
    panel(ax[5], "F")
    fig.tight_layout()
    save(fig, "Fig1")


def fig2():
    Gg = np.linspace(GL_ + 1e-4, GU_ - 1e-4, 700)
    Y = np.array([eq_coexist(G) for G in Gg])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    cols = [CB["blue"], CB["orange"], CB["verm"], CB["purple"], CB["green"]]
    nm = ["NSC  $X^*$", "TAC  $Y^*$", "PN  $Z^*$", "MN  $U^*$", "GABA  $A^*$"]
    for i in range(5):
        a.plot(Gg, Y[:, i], color=cols[i], label=nm[i])
    for G in (0.03, 0.06, 0.09, 0.115):
        ys = eq_coexist(G)
        lam = abs(leading_eigs(ys, G)[0].real)
        s = simulate(G, T=max(3000.0, 20.0 / lam), n=2)
        for i in range(5):
            a.plot(G, s.y[i, -1], "o", mfc="none", mec="k", ms=4.5, mew=0.7, zorder=5)
    a.plot([], [], "o", mfc="none", mec="k", ms=4.5, label="simulation")
    a.set_yscale("log")
    a.set_ylim(1e-2, 1e4)
    a.axvline(GL_, color=CB["grey"], ls="--", lw=0.8)
    a.axvline(GU_, color=CB["grey"], ls="--", lw=0.8)
    a.set_xlabel("GLU (a.u.)")
    a.set_ylabel("Equilibrium value (cells or conc.)")
    a.legend(ncol=2, loc="upper left", fontsize=6.5)
    panel(a, "A")

    lam = np.array([leading_eigs(y, G)[0].real for y, G in zip(Y, Gg)])
    b.plot(Gg, lam, "k-", label=r"Re $\lambda_1$ (exact)")
    dd = np.linspace(0.0, 0.03, 50)
    b.plot(GL_ + dd, -2.0 * PARAMS["g"] * dd, color=CB["blue"], ls="--", label=r"$-2g\,(G-G_L)$")
    b.plot(GU_ - dd, -2.0 * PARAMS["g"] * dd, color=CB["verm"], ls="--", label=r"$-2g\,(G_U-G)$")
    b.axhline(0.0, color="k", lw=0.5)
    b.set_ylim(-0.065, 0.004)
    b.set_xlabel("GLU (a.u.)")
    b.set_ylabel(r"Leading eigenvalue Re $\lambda_1$")
    b.text(GL_, 0.0012, r"$G_L$", ha="center", fontsize=7)
    b.text(GU_, 0.0012, r"$G_U$", ha="center", fontsize=7)
    b.legend(loc="lower center")
    panel(b, "B")
    fig.tight_layout()
    save(fig, "Fig2")


def fig3():
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    Gg = np.linspace(GL_ + 1e-6, GU_ - 1e-7, 2000)
    Z = np.array([eq_coexist(G)[2] for G in Gg])
    ax.axvspan(GU_, 0.2, color=CB["verm"], alpha=0.10, lw=0)
    ax.plot([0.0, GL_], [0.0, 0.0], color="k", lw=1.6)
    ax.plot([GL_, 0.2], [0.0, 0.0], color="k", lw=1.0, ls=(0, (3, 2)))
    ax.plot(Gg, Z, color=CB["green"], lw=1.8)
    ax.axvline(GU_, color=CB["verm"], ls="--", lw=0.9)
    ax.plot([GL_], [0.0], "o", mfc="white", mec="k", ms=5, zorder=5)
    ax.annotate("transcritical\n(extinction state\nloses stability)", xy=(GL_, 0.0), xytext=(0.045, 40.0),
                fontsize=6.5, ha="center", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate(r"$Z^*\to\infty$ as G$\to G_U$" + "\n(no fold, no\nsecond equilibrium)",
                xy=(GU_ - 0.0008, 90.0), xytext=(0.062, 84.0), fontsize=6.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(0.16, 62.0, "no equilibrium:\nexponential\ngrowth", color=CB["verm"], fontsize=7, ha="center")
    ax.text(0.006, 4.0, "extinction\nstable", fontsize=6.5, ha="left", va="bottom")
    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(-3.0, 100.0)
    ax.set_xlabel("GLU (a.u.)")
    ax.set_ylabel(r"PN equilibrium $Z^*$ (cells)")
    ax.legend(
        handles=[
            Line2D([0], [0], color="k", lw=1.6, label="extinction E$_0$ (stable)"),
            Line2D([0], [0], color="k", lw=1.0, ls=(0, (3, 2)), label="extinction E$_0$ (unstable)"),
            Line2D([0], [0], color=CB["green"], lw=1.8, label="coexistence E$_{XYZ}$ (stable)"),
        ],
        loc="lower right", fontsize=6.3, bbox_to_anchor=(1.0, 0.07)
    )
    ax.set_xticks([0.0, GL_, 0.06, 0.1, GU_, 0.16, 0.2])
    ax.set_xticklabels(["0", r"$G_L$", "0.06", "0.10", r"$G_U$", "0.16", "0.20"])
    fig.tight_layout()
    save(fig, "Fig3")


def fig4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    d = np.geomspace(1e-4, 0.05, 80)
    lo = np.array([-leading_eigs(eq_coexist(GL_ + x), GL_ + x)[0].real for x in d])
    up = np.array([-leading_eigs(eq_coexist(GU_ - x), GU_ - x)[0].real for x in d])
    a.loglog(d, lo, color=CB["blue"], label=r"lower: $|\lambda_1|$ vs $G-G_L$")
    a.loglog(d, up, color=CB["verm"], label=r"upper: $|\lambda_1|$ vs $G_U-G$")
    a.loglog(d, 2.0 * PARAMS["g"] * d, "k--", lw=1.0, label=r"$2g\,d$")
    a.set_xlabel(r"Distance to threshold $d$ (a.u.)")
    a.set_ylabel(r"$|{\rm Re}\,\lambda_1|$ (a.u.$^{-1}$)")
    a.legend(loc="upper left")
    panel(a, "A")

    for G, m, c in ((0.08, "o", CB["blue"]), (0.118, "^", CB["verm"])):
        ev = np.linalg.eigvals(jac(eq_coexist(G), G, PARAMS))
        b.plot(ev.real, ev.imag, m, color=c, ms=5, label=f"GLU = {G}")
    b.axhline(0.0, color="k", lw=0.5)
    b.axvline(0.0, color="k", lw=0.5)
    b.annotate("slow real mode\n(controls the approach)", xy=(-0.0038, 0.0), xytext=(-0.62, 0.30),
               fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    b.annotate("strongly damped\ncomplex pairs", xy=(-0.51, 0.52), xytext=(-1.2, 0.62),
               fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    b.set_xlim(-1.4, 0.1)
    b.set_ylim(-0.75, 0.75)
    b.set_xlabel(r"Re $\lambda$")
    b.set_ylabel(r"Im $\lambda$")
    b.legend(loc="lower left")
    panel(b, "B")
    fig.tight_layout()
    save(fig, "Fig4")


def fig5():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    cols = (CB["blue"], CB["green"], CB["orange"], CB["verm"])
    ic2 = np.array([800.0, 300.0, 1.0, 1.0, 0.01])
    for G, c in zip((0.03, 0.06, 0.09, 0.115), cols):
        ys = eq_coexist(G)
        T = max(2000.0, 12.0 / abs(leading_eigs(ys, G)[0].real))
        for ic, ls in ((IC0, "-"), (ic2, "--")):
            s = simulate(G, ic=ic, T=T, n=3000)
            a.plot(s.y[0], s.y[2], color=c, ls=ls, lw=1.1)
        a.plot(ys[0], ys[2], "o", color=c, mec="k", mew=0.6, ms=5.5, zorder=6, label=f"GLU = {G:g}")
    a.set_xlabel("NSC  $X$ (cells)")
    a.set_ylabel("PN  $Z$ (cells)")
    a.legend(
        loc="upper left",
        title="solid / dashed: two initial\nconditions; circles: $E_{XYZ}$",
        title_fontsize=6.3
    )
    panel(a, "A")

    for G, c in ((0.08, CB["blue"]), (0.118, CB["verm"])):
        ys = eq_coexist(G)
        lam1 = leading_eigs(ys, G)[0].real
        T = 14.0 / abs(lam1)
        s = simulate(G, T=T, n=4000)
        dev = np.linalg.norm((s.y.T - ys) / ys, axis=1)
        b.semilogy(s.t, dev, color=c, label=f"GLU = {G}")
        i0 = np.searchsorted(s.t, 0.25 * T)
        b.semilogy(s.t[i0:], dev[i0] * np.exp(lam1 * (s.t[i0:] - s.t[i0])), color=c, ls=":", lw=1.0)
    b.set_ylim(1e-8, 10.0)
    b.set_xlabel("Time (a.u.)")
    b.set_ylabel(r"$\|(x-x^*)/x^*\|$")
    b.text(0.97, 0.95, r"dotted: $e^{\lambda_1 t}$", transform=b.transAxes, ha="right", va="top", fontsize=7)
    b.legend(loc="lower left")
    panel(b, "B")
    fig.tight_layout()
    save(fig, "Fig5")


def fig6(fast=False):
    nd = 7 if fast else 12
    dd = np.geomspace(2e-4, 0.0599, nd)
    Gs = np.unique(np.round(np.concatenate([GL_ + dd, GU_ - dd[::-1]]), 8))
    tsim = np.array([return_time_sim(G)[0] for G in Gs])
    Gf = np.linspace(GL_ + 1e-4, GU_ - 1e-4, 500)
    teig = np.array([1.0 / abs(leading_eigs(eq_coexist(G), G)[0].real) for G in Gf])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    a, b, c = axes
    a.semilogy(Gf, teig, "k-", label=r"$1/|{\rm Re}\,\lambda_1|$")
    a.semilogy(Gs, tsim, "o", mfc="none", mec=CB["verm"], ms=4, label="simulated (small perturbation)")
    a.set_ylim(10.0, 2e4)
    a.set_xlabel("GLU (a.u.)")
    a.set_ylabel("Return time (a.u.)")
    a.legend(loc="upper center", fontsize=6.3)
    panel(a, "A", dx=-0.2)

    Gn = np.linspace(GL_ + 3e-4, GU_ - 3e-4, 300)
    st = np.array([lna_stats(G) for G in Gn])
    b.semilogy(Gn, st[:, 0], "k-", label="linear-noise theory")
    c.plot(Gn, st[:, 1], "k-", label="linear-noise theory")
    gem = [0.05, 0.09, 0.11, 0.116] if not fast else [0.05, 0.11]
    for G in gem:
        cv, ac = em_ensemble(G, M=600 if not fast else 300)
        b.semilogy(G, cv, "o", mfc="none", mec=CB["verm"], ms=5)
        c.plot(G, ac, "o", mfc="none", mec=CB["verm"], ms=5)
    b.plot([], [], "o", mfc="none", mec=CB["verm"], label="stochastic simulation")
    b.set_xlabel("GLU (a.u.)")
    b.set_ylabel(r"CV$^2$ of $Z$")
    b.legend(loc="upper center", fontsize=6.3)
    panel(b, "B", dx=-0.2)

    c.set_xlabel("GLU (a.u.)")
    c.set_ylabel(r"Autocorrelation of $Z$, lag 10")
    panel(c, "C", dx=-0.2)

    for ax_ in axes:
        ax_.axvline(GL_, color=CB["grey"], ls="--", lw=0.7)
        ax_.axvline(GU_, color=CB["grey"], ls="--", lw=0.7)
    fig.tight_layout()
    save(fig, "Fig6")


def fig7():
    G0 = 0.08
    p = PARAMS
    SZ = elasticity_Z(G0, p)
    En, _ = elasticity_numeric(G0, p)
    SU = {n: En[3, j] for j, n in enumerate(PNAMES)}
    order = sorted(range(len(PNAMES)), key=lambda j: (-abs(SZ[PNAMES[j]]), -abs(SU[PNAMES[j]])))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    yy = np.arange(len(order))[::-1]
    hgt = 0.38
    a.barh(yy + hgt / 2.0, [SZ[PNAMES[j]] for j in order], height=hgt, color=CB["verm"], label=r"PN, $Z^*$")
    a.barh(yy - hgt / 2.0, [SU[PNAMES[j]] for j in order], height=hgt, color=CB["sky"], label=r"MN, $U^*$")
    a.set_yticks(yy)
    a.set_yticklabels([PLABELS[j] for j in order])
    a.axvline(0.0, color="k", lw=0.7)
    for yv, j in zip(yy, order):
        for off, v in ((hgt / 2.0, SZ[PNAMES[j]]), (-hgt / 2.0, SU[PNAMES[j]])):
            if abs(v) > 1e-9:
                a.text(v + (0.06 if v > 0 else -0.06), yv + off, f"{v:+.2f}", va="center",
                       ha="left" if v > 0 else "right", fontsize=6)
    a.set_xlabel(r"Elasticity $\partial\ln x^*/\partial\ln p$ at GLU = 0.08")
    a.set_xlim(-3.2, 2.4)
    a.legend(loc="lower right")
    panel(a, "A", dx=-0.16)

    Gg = np.linspace(GL_ + 2e-4, GU_ - 2e-4, 600)
    for n, c, lab in (("c1", CB["blue"], r"$c_1$"), ("g", CB["verm"], r"$g$ (and GLU)"),
                      ("a1", CB["green"], r"$a_1$")):
        b.plot(Gg, [elasticity_Z(G, p)[n] for G in Gg], color=c, label=lab)
    b.axhline(0.0, color="k", lw=0.5)
    b.axvline(G0, color=CB["grey"], ls=":", lw=0.9)
    b.set_ylim(-12.0, 12.0)
    b.set_xlabel("GLU (a.u.)")
    b.set_ylabel(r"Elasticity of $Z^*$")
    b.legend(loc="upper center", ncol=3)
    panel(b, "B")
    fig.tight_layout()
    save(fig, "Fig7")


def fig8():
    G = 0.04
    p8 = {**PARAMS, "c3": 0.08}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    ax = axes.ravel()
    s0 = simulate(G, T=500.0, n=1500)
    s1 = simulate(G, p=p8, T=500.0, n=1500)
    ttl = ["NSC  X", "TAC  Y", "PN  Z", "GABA  A"]
    idx = [0, 1, 2, 4]
    for k, i in enumerate(idx):
        ax[k].plot(s0.t, s0.y[i], color=CB["blue"], label=r"default $c_3=0.90$")
        ax[k].plot(s1.t, np.maximum(s1.y[i], 1e-12), color=CB["verm"], ls="--", label=r"reduced $c_3=0.08$")
        ax[k].set_title(ttl[k], loc="left", fontsize=8.5)
        ax[k].set_xlabel("Time (a.u.)")
        ax[k].set_ylabel("Cells" if i < 4 else "Concentration (a.u.)")
        panel(ax[k], "ABCD"[k])
    for k in (0, 1):
        ax[k].set_yscale("log")
        ax[k].set_ylim(1e-9, 1e3)
    ax[0].legend(loc="lower left")
    y0 = eq_coexist(G)
    y8 = eq_pn_only(G, p8)
    ax[0].text(0.97, 0.52, f"GLU = {G:g}\ndefault $X^*$ = {y0[0]:.0f}; reduced $c_3$: $X\\to 0$",
               transform=ax[0].transAxes, fontsize=6.5, ha="right")
    ax[2].text(0.5, 0.45, f"$Z^*$ = {y0[2]:.2f} (default)\n$Z^*$ = {y8[2]:.2f} (X = Y = 0)",
               transform=ax[2].transAxes, fontsize=6.5)
    ax[2].set_ylim(bottom=0)
    ax[3].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "Fig8")


MAPCOL = {
    "extinction": "#BDBDBD",
    "PN-only": "#9ECAE1",
    "coexistence": "#74C476",
    "runaway": "#FC9272",
    "TAC-PN": "#FDD0A2",
    "bistable": "#000000",
}


def fig9(fast=False):
    Gg = np.linspace(0.0, 0.2, 200 if fast else 360)
    cc = np.geomspace(0.02, 1.2, 160 if fast else 300)
    keys = ["extinction", "PN-only", "TAC-PN", "coexistence", "runaway", "bistable"]
    M = np.zeros((len(cc), len(Gg)), dtype=int)
    for i, c3 in enumerate(cc):
        pp = {**PARAMS, "c3": c3}
        for j, G in enumerate(Gg):
            M[i, j] = keys.index(attractor_label(G, pp))

    cm = ListedColormap([MAPCOL[k] for k in keys])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.pcolormesh(
        Gg, cc, M, cmap=cm,
        norm=BoundaryNorm(np.arange(-0.5, len(keys) + 0.5), len(keys)),
        shading="nearest", rasterized=True
    )
    ax.set_yscale("log")
    ax.axhline(PARAMS["c3"], color="white", ls="--", lw=1.2)
    ax.axhline(0.08, color="k", ls=":", lw=1.2)
    ax.axhline(PARAMS["c1"], color="k", ls="-", lw=0.5)
    ax.text(0.197, PARAMS["c3"] * 1.06, r"default $c_3=0.90$", color="white", ha="right", fontsize=7, va="bottom")
    ax.text(0.197, 0.08 * 0.93, r"$c_3=0.08$ (reduced)", color="k", ha="right", fontsize=7, va="top")
    ax.text(0.197, PARAMS["c1"] * 1.06, r"$c_3=c_1=0.12$", color="k", ha="right", fontsize=7, va="bottom")
    ax.text(0.0105, 0.5, "extinction", rotation=90, ha="center", va="center", fontsize=7)
    ax.text(0.07, 0.5, "coexistence", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="#74C476", ec="none", alpha=0.85))
    ax.text(0.16, 0.55, "no stable equilibrium\n(runaway growth)", ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FC9272", ec="none", alpha=0.85))
    ax.text(0.03, 0.045, "PN-only", ha="center", va="center", fontsize=7)

    ng, nc = (9, 8) if fast else (17, 15)
    agree = tot = skipped = 0
    for c3 in np.geomspace(0.03, 1.15, nc):
        for G in np.linspace(0.005, 0.195, ng):
            pp = {**PARAMS, "c3": c3}
            lab = attractor_label(G, pp)
            slow = [abs(l) for _, _, l, is_stab in all_equilibria(G, pp) if is_stab]
            if lab != "runaway" and slow and min(slow) < 4e-3:
                skipped += 1
                continue
            sl = sim_label(G, pp, T=6000.0)
            tot += 1
            agree += int(sl == lab)
            ax.plot(G, c3, "o", ms=3.0, mfc=MAPCOL.get(sl, "w"), mec="k", mew=0.5, zorder=6)

    ax.set_xlabel("GLU (a.u.)")
    ax.set_ylabel(r"PN loss rate $c_3$ (a.u.$^{-1}$)")
    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(0.02, 1.2)
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
    ax.legend(
        handles=[
            Patch(fc=MAPCOL["extinction"], ec="k", label="extinction (E$_0$ stable)"),
            Patch(fc=MAPCOL["PN-only"], ec="k", label="PN-only, X = Y = 0 (E$_Z$ stable)"),
            Patch(fc=MAPCOL["coexistence"], ec="k", label="coexistence (E$_{XYZ}$ stable)"),
            Patch(fc=MAPCOL["runaway"], ec="k", label="no stable equilibrium"),
            Line2D([0], [0], marker="o", color="none", mfc="w", mec="k", ms=3.5,
                   label="outcome of direct simulation"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=6.5, frameon=False
    )
    fig.tight_layout()
    save(fig, "Fig9")
    print(f"Map vs simulation: {agree}/{tot} agree ({skipped} slow-boundary points skipped)")
    return agree, tot, skipped


def fig10():
    p = PARAMS
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    Gg = np.linspace(0.021, 0.30, 600)
    Gb = np.linspace(0.021, GU_ - 1e-4, 600)
    a.plot(Gb, [eq_coexist(G)[2] for G in Gb], "k-", lw=1.6, label=r"$K=\infty$ (baseline)")
    for K, c in ((2e4, CB["blue"]), (5e3, CB["green"]), (2e3, CB["verm"])):
        zz = [eq_capacity(G, p, K)[2] for G in Gg]
        a.plot(Gg, zz, color=c, label=f"$K$ = {K:g}")
    a.axvline(GU_, color=CB["grey"], ls="--", lw=0.8)
    a.set_ylim(0.0, 120.0)
    a.set_xlim(0.0, 0.3)
    a.set_xlabel("GLU (a.u.)")
    a.set_ylabel(r"PN equilibrium $Z^*$ (cells)")
    a.text(GU_ + 0.004, 112.0, r"$G_U$", fontsize=7)
    a.legend(loc="center right")
    panel(a, "A")

    G, K = 0.15, 5000.0
    s0 = solve_ivp(rhs, (0.0, 400.0), IC0, args=(G, p), method="Radau", rtol=1e-9, atol=1e-12,
                   t_eval=np.linspace(0.0, 400.0, 800))
    s1 = solve_ivp(rhs_capacity, (0.0, 400.0), IC0, args=(G, p, K), method="Radau", rtol=1e-9, atol=1e-12,
                   t_eval=np.linspace(0.0, 400.0, 800))
    b.semilogy(s0.t, s0.y[:3].sum(0), "k-", label=r"$K=\infty$")
    b.semilogy(s1.t, s1.y[:3].sum(0), color=CB["green"], label=f"$K$ = {K:g}")
    b.axhline(K, color=CB["grey"], ls=":", lw=0.9)
    b.text(5.0, K * 1.25, r"capacity $K$", fontsize=7, color=CB["grey"])
    b.set_xlabel("Time (a.u.)")
    b.set_ylabel(r"Total cells $X+Y+Z$")
    b.set_title("GLU = 0.15 (above $G_U$)", loc="left", fontsize=8)
    b.legend(loc="lower right")
    panel(b, "B")
    fig.tight_layout()
    save(fig, "Fig10")


def figS1():
    p = PARAMS
    Gg = np.linspace(GL_ + 2e-4, GU_ - 2e-4, 500)
    Zp = np.array([eq_coexist(G)[2] for G in Gg])
    Zg = np.array([eq_gabaonly(G)[2] for G in Gg])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a.semilogy(Gg, Zp, color=CB["blue"], label=r"product form  $b/(b+A\,Z)$")
    a.semilogy(Gg, Zg, color=CB["verm"], label=r"GABA only  $b/(b+A)$")
    a.set_xlabel("GLU (a.u.)")
    a.set_ylabel(r"PN equilibrium $Z^*$ (cells)")
    a.legend(loc="upper left", fontsize=6.5)
    panel(a, "A")

    lam_p, lam_g, Gl = [], [], []
    for G in Gg[::5]:
        yp, yg = eq_coexist(G), eq_gabaonly(G)
        lam_p.append(max_re_eig(yp, G))
        lam_g.append(np.max(np.linalg.eigvals(numjac(lambda v, G=G: rhs_gabaonly(0.0, v, G, p), yg)).real))
        Gl.append(G)
    b.plot(Gl, lam_p, color=CB["blue"], label="product form")
    b.plot(Gl, lam_g, color=CB["verm"], label="GABA only")
    b.axhline(0.0, color="k", lw=0.5)
    b.set_xlabel("GLU (a.u.)")
    b.set_ylabel(r"Leading eigenvalue Re $\lambda_1$")
    b.legend(loc="upper center", fontsize=6.5)
    panel(b, "B")
    for ax_ in (a, b):
        ax_.axvline(GL_, color=CB["grey"], ls="--", lw=0.8)
        ax_.axvline(GU_, color=CB["grey"], ls="--", lw=0.8)
    fig.tight_layout()
    save(fig, "FigS1")


def main():
    parser = argparse.ArgumentParser(description="Simulation and analysis code for NSC GABA-glutamate lineage model.")
    parser.add_argument("--verify", action="store_true", help="Run analytical and numerical verification suite only.")
    parser.add_argument("--figs", nargs="*", default=None, help="Specific figure IDs to generate (e.g., 1 3 9 S1).")
    parser.add_argument("--fast", action="store_true", help="Use coarser grids for rapid execution.")
    args = parser.parse_args()

    set_style()
    t0 = time.time()
    if args.verify:
        sys.exit(0 if verify(args.fast) else 1)

    make_tables()
    figures = {
        "1": fig1,
        "2": fig2,
        "3": fig3,
        "4": fig4,
        "5": fig5,
        "6": lambda: fig6(args.fast),
        "7": fig7,
        "8": fig8,
        "9": lambda: fig9(args.fast),
        "10": fig10,
        "S1": figS1,
    }
    ids = args.figs if args.figs else list(figures.keys())
    if args.figs is None:
        verify(args.fast)
    for fig_id in ids:
        print(f"Generating Figure {fig_id}...")
        figures[str(fig_id)]()
    print(f"Completed in {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
