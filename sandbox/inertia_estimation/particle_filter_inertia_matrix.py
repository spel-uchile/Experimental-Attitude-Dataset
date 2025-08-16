"""
Created by Elias Obreque
Date: 05/08/2025
email: els.obrq@gmail.com
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

###############################################################################
# 1. Physical & Simulation Parameters                                          #
###############################################################################

@dataclass
class SimConfig:
    """Constants mirroring Table 1 of Helmuth&DeMars (2025)."""

    I_true: np.ndarray = np.array(
        [[67946.0,  -83.0, 11129.0],
         [ -83.0, 90061.0,   103.0],
         [11129.0,  103.0, 45821.0]]
    )  # kg·m²
    # Constant reaction‑wheel torque applied in body frame (≤ τ_max = 0.072 N·m)
    tau_body: np.ndarray = np.array([0.06, -0.04, 0.02])  # N·m
    # Gyro noise 0.005 deg/s 1‑σ  → rad²/s²
    R_gyro: np.ndarray = (0.005 * np.pi / 180.0) ** 2 * np.eye(3)
    # Very small process noise on inertia parameters (numerical stability)
    Q_j: np.ndarray = 1e-11 * np.eye(6)
    # Simulation horizon & step
    t_final: float = 600.0  # s
    dt: float = 0.1  # s
    # Particle‑filter hyper‑parameters
    N_particles: int = 3000
    ess_trigger: float = 0.5
    n_lambda: int = 25

###############################################################################
# 2. Inertia Parametrisation (log‑Cholesky)                                    #
###############################################################################

def u_to_L(u: np.ndarray) -> np.ndarray:
    """ℝ⁶ → lower‑triangular Cholesky factor L."""
    L = np.zeros((3, 3))
    L[0, 0] = math.exp(u[0])
    L[1, 0] = u[3]
    L[1, 1] = math.exp(u[1])
    L[2, 0] = u[4]
    L[2, 1] = u[5]
    L[2, 2] = math.exp(u[2])
    return L

def u_to_I(u: np.ndarray) -> np.ndarray:
    """Convert log‑Cholesky vector to inertia tensor (SPD)."""
    L = u_to_L(u)
    return L @ L.T

def I_to_u(I: np.ndarray) -> np.ndarray:
    """SPD tensor → log‑Cholesky vector."""
    L = np.linalg.cholesky(I)
    return np.array([
        math.log(L[0, 0]), math.log(L[1, 1]), math.log(L[2, 2]),
        L[1, 0], L[2, 0], L[2, 1]
    ])

###############################################################################
# 3. Rigid‑Body Dynamics (RK4 integrator)                                      #
###############################################################################

def euler_rhs(omega: np.ndarray, I: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Euler equation: ω̇ = I⁻¹ ( τ − ω × (I ω) )."""
    return np.linalg.solve(I, tau - np.cross(omega, I @ omega))

def rk4_step(omega: np.ndarray, I: np.ndarray, tau: np.ndarray, dt: float) -> np.ndarray:
    """Classic RK4 step for ω."""
    k1 = euler_rhs(omega, I, tau)
    k2 = euler_rhs(omega + 0.5 * dt * k1, I, tau)
    k3 = euler_rhs(omega + 0.5 * dt * k2, I, tau)
    k4 = euler_rhs(omega + dt * k3, I, tau)
    return omega + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def propagate_state(x: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Propagate a particle for one dt, adding small RW to inertia params."""
    omega, u = x[:3], x[3:]
    I = u_to_I(u)
    omega_new = rk4_step(omega, I, cfg.tau_body, cfg.dt)
    u_new = u + np.random.multivariate_normal(np.zeros(6), cfg.Q_j)
    return np.hstack((omega_new, u_new))

###############################################################################
# 4. Constrained Exact Gaussian Particle Flow                                  #
###############################################################################

def cegpf_update(particles: np.ndarray, weights: np.ndarray, y: np.ndarray, cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Linear‑measurement CEGPF update (Daum–Huang exact flow)."""
    H = np.hstack((np.eye(3), np.zeros((3, 6))))
    R = cfg.R_gyro
    n_lambda = cfg.n_lambda
    dλ = 1.0 / n_lambda

    m = np.average(particles, axis=0, weights=weights)
    P = np.cov(particles.T, aweights=weights, ddof=0)
    HPHT = H @ P @ H.T

    for _ in range(n_lambda):
        S = dλ * R + HPHT
        K = P @ H.T @ np.linalg.inv(S)
        innov = y - (H @ particles.T).T  # (N×3)
        particles += (K @ innov.T).T * dλ

    return particles, weights  # weights unchanged (unit Jacobian)

###############################################################################
# 5. Monte‑Carlo Simulation loop                                               #
###############################################################################

def run_trial(cfg: SimConfig, quick: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run one complete PF trial; return time vector and inertia‑error history."""
    N = 100 if quick else cfg.N_particles
    steps = int(cfg.t_final / cfg.dt)

    # ---------------- Initial particle cloud --------------------------------
    particles = np.zeros((N, 9))
    particles[:, :3] = np.random.randn(N, 3) * 1e-2  # ω0

    jitter = (np.random.rand(N, 1) - 0.5) * 0.4  # ±20 % isotropic scaling
    for i in tqdm(range(N), desc="Iteration...", total=N):
        I_init = cfg.I_true * (1.0 + jitter[i, 0])
        particles[i, 3:] = I_to_u(I_init) + np.random.multivariate_normal(np.zeros(6), 1e-4 * np.eye(6))

    weights = np.full(N, 1.0 / N)

    # ---------------- Storage ----------------------------------------------
    I_err_hist = np.zeros((steps, 3))
    t_vec = np.arange(steps) * cfg.dt

    # ---------------- True state -------------------------------------------
    omega_true = np.array([0.05, -0.02, 0.04])

    # Pre‑invert R once for likelihood
    R_inv = np.linalg.inv(cfg.R_gyro)

    for k in tqdm(range(steps), "Pre steps...", total=steps):
        # Truth propagation & noisy measurement
        omega_true = rk4_step(omega_true, cfg.I_true, cfg.tau_body, cfg.dt)
        y_meas = omega_true + np.random.multivariate_normal(np.zeros(3), cfg.R_gyro)

        # Predict step for every particle
        for i in range(N):
            particles[i] = propagate_state(particles[i], cfg)

        # Deterministic flow measurement update
        particles, weights = cegpf_update(particles, weights, y_meas, cfg)

        # Likelihood weighting
        innov = y_meas - particles[:, :3]
        llh = np.exp(-0.5 * np.sum(innov @ R_inv * innov, axis=1))
        weights *= llh
        weights /= np.sum(weights)

        # Resampling (systematic via numpy choice)
        ess = 1.0 / np.sum(weights ** 2)
        if ess < cfg.ess_trigger * N:
            idx = np.random.choice(N, N, p=weights)
            particles = particles[idx]
            weights.fill(1.0 / N)

        # Log principal‑moment errors
        u_est = np.average(particles[:, 3:], axis=0, weights=weights)
        I_est = u_to_I(u_est)
        I_err_hist[k] = np.diag(I_est) - np.diag(cfg.I_true)

    return t_vec, I_err_hist

###############################################################################
# 6. Plotting / Data saving                                                    #
###############################################################################

def save_or_plot(t: np.ndarray, I_err: np.ndarray, out_dir: Path, plot: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if plot and HAS_MPL:
        labels = [r"$I_{xx}-I_{xx}^{true}$", r"$I_{yy}-I_{yy}^{true}$", r"$I_{zz}-I_{zz}^{true}$"]
        for i in range(3):
            plt.figure()
            plt.plot(t, I_err[:, i])
            plt.xlabel("Time [s]")
            plt.ylabel(labels[i] + " [kg·m²]")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"principal_moment_error_{i+1}.png", dpi=300)
            plt.close()
    else:
        np.savez(out_dir / "principal_moment_errors.npz", time=t, I_err=I_err)
        if plot and not HAS_MPL:
            print("matplotlib not installed – saved data to .npz. Run `pip install matplotlib` to enable plotting.")

###############################################################################
# 7. Smoke test                                                               #
###############################################################################

def smoke_test():
    cfg = SimConfig(t_final=20.0, N_particles=400)
    t, I_err = run_trial(cfg, quick=True)
    assert not np.isnan(I_err).any(), "NaN detected in error history"
    final_err = np.abs(I_err[-1]) / np.diag(cfg.I_true)
    assert np.all(final_err < 0.05), "Filter failed to converge below 5 %"
    print("Smoke test passed ✅")

###############################################################################
# 8. Command‑line interface                                                    #
###############################################################################

def main(argv: list[str]):
    parser = argparse.ArgumentParser(description="CEGPF inertia estimation (Helmuth & DeMars 2025)")
    parser.add_argument("--quick", action="store_true", help="reduced particle count & horizon for fast debug")
    parser.add_argument("--no-plot", action="store_true", help="skip plotting even if matplotlib is available")
    parser.add_argument("--test", action="store_true", help="run smoke test and exit")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="output directory")
    args = parser.parse_args(argv)

    if args.test:
        smoke_test()
        return

    cfg = SimConfig()
    t, I_err = run_trial(cfg, quick=args.quick)
    save_or_plot(t, I_err, args.out, plot=not args.no_plot)
    print(f"Done. Results stored in {args.out.resolve()}")

if __name__ == "__main__":
    # main(sys.argv[1:])
    main(["--quick"])

