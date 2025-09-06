# phase1_adiabatic_alpha.py
"""
Phase 1 — Adiabatic Compression with δ-framework probes (no kernel changes)

What this script does
---------------------
- Initializes a 3D hard-sphere gas (GasSim).
- Warms up in a fixed box, then turns on a piston (x-max wall) moving inward at constant speed.
- At each sampling window Δt, collects macroscopic observables and computes three δ-quantities:
  1) Internal energy U(t)  (thermal, subtracting bulk drift)
  2) δa = -P * ΔV + T_avg * ΔS_IG   where ΔS_IG uses ideal-gas entropy difference
  3) α = ΔU - δa                    (vertical residual / dissipation measure)
- Writes a CSV with time series: t, V, Lx, P, U, T, Work, ΔS_IG, δa, α, α/T, PV^γ, T*V^{γ-1}
- Optionally plots quick-look figures.

Sign conventions & units
------------------------
- k_B = 1 in code units.
- Pressure proxy uses piston-impulse sum over the window:  P ≈ (Σ|Δp|)/(A*Δt).
- Work returned by sim.get_work_done() is the cumulative *mechanical* work ON the gas.
- Volume V(t) is tracked from the commanded piston trajectory (no kernel query needed).

Why these formulas
------------------
- δa implements the reversible part  (-P dV + T dS); α = dU - δa is the "vertical" remainder.
  Along quasi-static adiabats α ≈ 0; for finite-rate compression α > 0 typically grows with rate.
  (See your Thermo correspondence draft: contactomorphism and δ-projection)  [citation below]

Run
---
  python phase1_adiabatic_alpha.py

Requires
--------
  - maturin develop -m pyproject.toml   (to build/install gassim)
  - numpy
  - matplotlib (optional for plotting)

References
----------
- AEG ↔ Thermo contactomorphism; δ a, α definitions; Maxwell/Legendre dictionary.
- These steps implement the "Phase 1" consistency test discussed in our plan.

"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from gassim import GasSim


# -------------------------------
# Configuration
# -------------------------------

@dataclass
class Config:
    # Gas & box
    num_particles: int = 9128
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)  # (Lx0, Ly, Lz)
    radius: float = 0.2
    mass: float = 1.0
    dim: int = 3
    seed: int = 123

    # Piston (x-max wall => wall_id=1). Use negative velocity to compress (inward).
    piston_wall_id: int = 1
    piston_velocity: float = -0.05  # inward (negative along +x outward normal)

    # Time control
    t_warmup: float = 0.5   # warm up with static walls
    t_final: float = 200.0
    dt: float = 0.2

    # Geometry / safety
    min_Lx: float = 3.0     # stop early if commanded Lx would drop below this (avoid geometric degeneracy)

    # Phys constants (code units)
    kB: float = 1.0
    gamma: float = 5.0 / 3.0  # monatomic ideal gas in 3D

    # Numerics
    subtract_bulk_drift: bool = True   # use thermal energy (remove bulk COM drift) for U and T

    # Output files
    csv_path: str = "phase1_adiabatic_alpha.csv"
    fig_path: str = "phase1_adiabatic_alpha.png"


# -------------------------------
# Helpers
# -------------------------------

def piston_Lx_at_time(t: float, cfg: Config) -> float:
    """Commanded piston position -> Lx(t). Warmup: static; after warmup: Lx = Lx0 + u_piston*(t - t_warmup)."""
    Lx0, _, _ = cfg.box_size
    if t <= cfg.t_warmup:
        return Lx0
    return Lx0 + cfg.piston_velocity * (t - cfg.t_warmup)


def internal_energy_and_temperature(
    vel: np.ndarray, mass: float, kB: float = 1.0, subtract_bulk: bool = True
) -> Tuple[float, float]:
    """
    Return thermal internal energy U and thermal T (equipartition),
    optionally subtracting bulk drift (recommended for non-equilibrium flows).
    """
    assert vel.ndim == 2 and vel.shape[1] in (2, 3), "velocities shape must be (N, D)"
    if subtract_bulk:
        vcm = np.mean(vel, axis=0, keepdims=True)
        v_rel = vel - vcm
    else:
        v_rel = vel
    Ek = 0.5 * mass * float(np.sum(v_rel * v_rel))  # total kinetic energy (thermal part if subtract_bulk=True)
    N, D = vel.shape
    T = (2.0 * Ek) / (D * N * kB)
    return Ek, T


def pressure_from_impulses(impulses: np.ndarray, area: float, window: float) -> float:
    """
    Estimate pressure from piston-wall impulse events in the last window Δt:
        P ≈ (Σ |Δp|) / (A * Δt)
    impulses array has shape (M, 2): [time, |impulse|]
    """
    if impulses is None or impulses.size == 0 or area <= 0.0 or window <= 0.0:
        return 0.0
    s = float(np.sum(impulses[:, 1]))
    return s / (area * window)


def safe_log_ratio(a2: float, a1: float, eps: float = 1e-16) -> float:
    """Return log(a2/a1) with clipping to avoid -inf/inf from numerical noise."""
    a2c = max(a2, eps)
    a1c = max(a1, eps)
    return float(np.log(a2c) - np.log(a1c))


# -------------------------------
# Main
# -------------------------------

def main(cfg: Config) -> None:
    # Construct simulator
    sim = GasSim(
        num_particles=cfg.num_particles,
        box_size=list(cfg.box_size),
        radius=cfg.radius,
        mass=cfg.mass,
        dim=cfg.dim,
        seed=cfg.seed,
    )

    Lx0, Ly, Lz = cfg.box_size
    piston_area = Ly * Lz

    print(f"[init] N={cfg.num_particles}, box={cfg.box_size}, piston_area={piston_area:.6g}")

    # Warmup
    sim.advance_to(cfg.t_warmup)
    print(f"[warmup] to t={cfg.t_warmup:g}, W={sim.get_work_done():.6g}")

    # Enable piston (x-max wall) with constant inward velocity
    sim.set_piston(cfg.piston_wall_id, cfg.piston_velocity)
    print(f"[piston] wall_id={cfg.piston_wall_id}, velocity={cfg.piston_velocity:+.6g}")

    # Time loop
    t = cfg.t_warmup

    # Storage
    times: List[float] = []
    vols: List[float] = []
    Lxs: List[float] = []
    presses: List[float] = []
    works: List[float] = []
    Us: List[float] = []
    Ts: List[float] = []
    dS_IGs: List[float] = []
    delta_as: List[float] = []
    alphas: List[float] = []
    alpha_over_T: List[float] = []
    PVg: List[float] = []
    TVgm1: List[float] = []
    num_events: List[int] = []

    # Previous step memory
    U_prev: float | None = None
    V_prev: float | None = None
    T_prev: float | None = None
    P_prev: float | None = None

    N = cfg.num_particles
    g = cfg.gamma

    # Main sampling loop
    while t < cfg.t_final - 1e-15:
        t_next = min(cfg.t_final, t + cfg.dt)

        # Safety: if commanded Lx would be too small, stop early
        Lx_next = piston_Lx_at_time(t_next, cfg)
        if Lx_next <= cfg.min_Lx:
            print(f"[stop] commanded Lx(t) would hit {Lx_next:.3f} < min_Lx={cfg.min_Lx:.3f} at t={t_next:.3f}; stopping.")
            break

        # Advance simulation to t_next
        sim.advance_to(t_next)

        # Measurements from simulator
        W = sim.get_work_done()  # cumulative work ON gas
        impulses = sim.get_pressure_history(window=(t_next - t))
        P = pressure_from_impulses(impulses, piston_area, (t_next - t))

        v = sim.get_velocities()
        U, T_th = internal_energy_and_temperature(
            v, mass=cfg.mass, kB=cfg.kB, subtract_bulk=cfg.subtract_bulk_drift
        )

        # Geometric volume from commanded piston trajectory (no kernel query required)
        V = Lx_next * Ly * Lz

        # Invariants for quick quasi-static check
        PV_gamma = P * (V ** g)
        TV_gm1 = T_th * (V ** (g - 1.0))

        # Stepwise δ-quantities
        if U_prev is not None:
            dU = U - U_prev
            dV = V - V_prev  # >0 expansion, <0 compression

            # Ideal-gas entropy difference as baseline gauge (k_B = 1)
            # ΔS_IG = N * [ (3/2) ln(T2/T1) + ln(V2/V1) ]
            dS_IG = N * (1.5 * safe_log_ratio(T_th, T_prev) + safe_log_ratio(V, V_prev))

            # Window-averaged T and P (simple choice)
            Tavg = 0.5 * (T_th + T_prev)
            Pavg = P  # P we computed from impulses over this very window

            # δa and α
            delta_a = -Pavg * dV + Tavg * dS_IG
            alpha = dU - delta_a

            # Save
            dS_IGs.append(dS_IG)
            delta_as.append(delta_a)
            alphas.append(alpha)
            alpha_over_T.append(alpha / max(Tavg, 1e-16))
        else:
            # First step placeholders
            dS_IGs.append(0.0)
            delta_as.append(0.0)
            alphas.append(0.0)
            alpha_over_T.append(0.0)

        # Loggers
        times.append(t_next)
        vols.append(V)
        Lxs.append(Lx_next)
        presses.append(P)
        works.append(W)
        Us.append(U)
        Ts.append(T_th)
        PVg.append(PV_gamma)
        TVgm1.append(TV_gm1)
        num_events.append(0 if impulses is None else impulses.shape[0])

        # Console line
        print(
            f"t={t_next:8.3f} | Lx={Lx_next:7.3f} V={V:8.3f} | "
            f"P={P:11.4e} T={T_th:9.5f} U={U:12.6e} | "
            f"dS_IG={dS_IGs[-1]:+10.6e} δa={delta_as[-1]:+10.6e} α={alphas[-1]:+10.6e} | "
            f"events={num_events[-1]}"
        )

        # Update prev
        U_prev, V_prev, T_prev, P_prev = U, V, T_th, P
        t = t_next

    # Write CSV
    with open(cfg.csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time", "Lx", "V", "pressure", "U", "T",
                "work_cum",
                "dS_IG", "delta_a", "alpha", "alpha_over_T",
                "PV_gamma", "T_V_gamma_minus1", "impulse_events",
            ]
        )
        for row in zip(
            times, Lxs, vols, presses, Us, Ts, works,
            dS_IGs, delta_as, alphas, alpha_over_T,
            PVg, TVgm1, num_events,
        ):
            w.writerow([f"{x:.12g}" for x in row])

    print(f"[done] wrote CSV: {cfg.csv_path} (rows={len(times)})")

    # Optional quick-look plots
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))

        ax[0].plot(times, Ts, label="T(th)", lw=1.5)
        ax[0].set_ylabel("T")
        ax[0].legend()

        ax[1].plot(times, presses, label="P (impulse proxy)", lw=1.2)
        ax[1].set_ylabel("P")
        ax[1].legend()

        ax[2].plot(times, vols, label="V", lw=1.2)
        ax[2].set_ylabel("V")
        ax[2].legend()

        ax[3].plot(times, alphas, label="alpha", lw=1.2)
        ax[3].plot(times, alpha_over_T, label="alpha/T", lw=1.0, ls="--")
        ax[3].set_ylabel("α, α/T")
        ax[3].set_xlabel("time")
        ax[3].legend()

        fig.tight_layout()
        plt.savefig(cfg.fig_path, dpi=150)
        print(f"[plot] wrote figure: {cfg.fig_path}")
        # plt.show()
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    main(Config())
