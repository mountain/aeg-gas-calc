# phase1_adiabatic_sweep.py
"""
Phase 1 — Adiabatic compression sweep with δ-framework probes (no kernel changes)

What this program does
----------------------
For each (piston_velocity, dt, N) combo it:
  • Runs a 3D hard-sphere gas with a moving piston (x-max wall).
  • Collects time series and computes the 6 classes of summary metrics we agreed:
      1) avg_sigma = mean(-alpha/T)                      # 熵产率的平均
      2) cum_sigma = sum(-alpha/T)                       # 累计熵增
      3) Baseline deviations vs reversible adiabat:
           chi_T_end = T_end / T_ad(V_end) - 1
           chi_P_end = P_end / P_ad(V_end) - 1
      4) RMS drift of adiabatic invariants (IG pressure baseline to reduce noise):
           rms_PV_gamma, rms_TV_gm1
      5) Work–energy residual (adiabatic boundary):      res_WE = (U_end-U0) - (W_end-W0)
      6) EOS consistency (impulse vs ideal-gas):
           eos_rel_diff_rms, eos_rel_diff_end
     (Also outputs alpha_upper = dU + P*dV as an immediate upper bound when ΔS≥0.)
  • Writes per-run CSV (time series) and appends a summary row to `phase1_sweep_summary.csv`.
  • Optionally saves small figures per run.

Engineering improvements implemented
------------------------------------
  (i) Thermal U,T computed after removing bulk drift (center-of-mass velocity).
 (ii) Pressure dual-track: impulse-based P for mechanics + smooth IG baseline P_IG = N k_B T / V.
      Additionally, an *adaptive time window* keeps the number of piston-impulse events near a target,
      stabilizing the impulse pressure estimator’s variance.

"Must-pass" checks (printed at the end)
---------------------------------------
  PASS #1 Quasi-static limit: for the same (N, dt), avg_sigma should rise as |piston_velocity| increases.
  PASS #2 Work–energy: |res_WE| is small (relative to |ΔU|+|W|) for every run.
  PASS #3 Baseline sign: chi_T_end, chi_P_end are ≥ 0 on average and correlate with avg_sigma.

Run
---
  python phase1_adiabatic_sweep.py               # default 3×3×1 sweep (9 runs)
  python phase1_adiabatic_sweep.py --full        # 3×3×3 sweep (27 runs, heavier)
  python phase1_adiabatic_sweep.py --vels -0.005 -0.05 -0.2 --dts 0.1 0.2 0.5 --Ns 9128
  python phase1_adiabatic_sweep.py --no-fig

Requirements
------------
  - maturin develop -m pyproject.toml   (to build/install gassim)
  - numpy, matplotlib

Theory anchors
--------------
- Contact correspondence: Φ^*(dU+p dV - T dS) = da - μ du - λ a dv
- δ-projection and vertical residual: δa = -p dV + T dS,  α = dU - δa
- Non-commutativity / curvature: δ^2 a = μ λ du∧dv; circulation–area
(See your drafts, Thermodynamic correspondence §3 and AEG basics §6–§7.)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

from gassim import GasSim


# -------------------------------
# Configuration dataclasses
# -------------------------------

@dataclass
class BaseConfig:
    # Gas & box
    num_particles: int = 9128
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)  # (Lx0, Ly, Lz)
    radius: float = 0.2
    mass: float = 1.0
    dim: int = 3
    seed: int = 123

    # Piston (x-max wall => wall_id=1). Negative => compress inward.
    piston_wall_id: int = 1
    piston_velocity: float = -0.05

    # Time control
    t_warmup: float = 0.5
    t_final: float = 200.0
    dt_init: float = 0.2

    # Geometry / safety
    min_Lx: float = 3.0

    # Physics
    kB: float = 1.0
    gamma: float = 5.0 / 3.0  # monatomic ideal gas, D=3

    # Numerics / engineering
    subtract_bulk_drift: bool = True
    adapt_window_by_events: bool = True
    target_events: int = 200      # aim impulse events per window
    dt_min: float = 0.02
    dt_max: float = 0.5

    # Output
    out_dir: str = "phase1_out"
    save_fig: bool = True
    show_fig: bool = False


@dataclass
class SweepConfig:
    velocities: List[float]
    dts: List[float]
    Ns: List[int]
    full: bool = False


# -------------------------------
# Utilities
# -------------------------------

def piston_Lx_at_time(t: float, cfg: BaseConfig) -> float:
    """Commanded Lx(t): static before warmup; then linear with piston_velocity."""
    Lx0, _, _ = cfg.box_size
    if t <= cfg.t_warmup:
        return Lx0
    return Lx0 + cfg.piston_velocity * (t - cfg.t_warmup)


def internal_energy_and_temperature(
    vel: np.ndarray, mass: float, kB: float = 1.0, subtract_bulk: bool = True
) -> Tuple[float, float]:
    """
    Thermal internal energy U and thermal temperature T from velocities,
    optionally removing bulk (COM) drift.
    """
    assert vel.ndim == 2 and vel.shape[1] in (2, 3), "velocities shape must be (N, D)"
    if subtract_bulk:
        vcm = np.mean(vel, axis=0, keepdims=True)
        v_rel = vel - vcm
    else:
        v_rel = vel
    Ek = 0.5 * mass * float(np.sum(v_rel * v_rel))  # total kinetic (thermal) energy
    N, D = vel.shape
    T = (2.0 * Ek) / (D * N * kB)
    return Ek, T


def pressure_from_impulses(impulses: np.ndarray, area: float, window: float) -> float:
    """
    Impulse pressure estimator over time window Δt:
        P ≈ (Σ |Δp|) / (A * Δt)
    impulses: shape (M, 2): [time, |impulse|]
    """
    if impulses is None or impulses.size == 0 or area <= 0.0 or window <= 0.0:
        return 0.0
    s = float(np.sum(impulses[:, 1]))
    return s / (area * window)


def safe_log_ratio(a2: float, a1: float, eps: float = 1e-16) -> float:
    a2c = max(a2, eps)
    a1c = max(a1, eps)
    return float(np.log(a2c) - np.log(a1c))


def adiabatic_T_of_V(T0: float, V0: float, V: float, gamma: float) -> float:
    return T0 * (V0 / V) ** (gamma - 1.0)


def adiabatic_P_of_V(P0: float, V0: float, V: float, gamma: float) -> float:
    return P0 * (V0 / V) ** gamma


def ensure_dir(d: str) -> None:
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


# -------------------------------
# Single experiment
# -------------------------------

def run_one_experiment(
    base_cfg: BaseConfig,
    piston_velocity: float,
    dt_init: float,
    N: int,
    tag: str,
) -> Dict[str, float]:
    """
    Run one (v_pis, dt, N) experiment and return a dict of summary metrics.
    Also writes per-run CSV and optional figure.
    """

    cfg = BaseConfig(**asdict(base_cfg))
    cfg.num_particles = N
    cfg.piston_velocity = piston_velocity
    cfg.dt_init = dt_init

    ensure_dir(cfg.out_dir)
    fig_dir = os.path.join(cfg.out_dir, "figs")
    ensure_dir(fig_dir)
    ts_dir = os.path.join(cfg.out_dir, "timeseries")
    ensure_dir(ts_dir)

    # Build simulator
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

    # Warmup
    sim.advance_to(cfg.t_warmup)
    W0 = sim.get_work_done()

    # First thermal read at warmup
    v_now = sim.get_velocities()
    U0, T0 = internal_energy_and_temperature(
        v_now, mass=cfg.mass, kB=cfg.kB, subtract_bulk=cfg.subtract_bulk_drift
    )
    V0 = piston_Lx_at_time(cfg.t_warmup, cfg) * Ly * Lz
    P0_IG = (cfg.num_particles * cfg.kB * T0) / V0

    # Enable piston
    sim.set_piston(cfg.piston_wall_id, cfg.piston_velocity)

    # Time loop
    t = cfg.t_warmup
    dt = cfg.dt_init

    # Storage
    times: List[float] = []
    vols: List[float] = []
    Lxs: List[float] = []
    P_imp: List[float] = []
    P_ig: List[float] = []
    works: List[float] = []
    Us: List[float] = []
    Ts: List[float] = []
    PVg: List[float] = []
    TVgm1: List[float] = []
    num_events: List[int] = []
    alpha_upper_list: List[float] = []
    alpha_ig_list: List[float] = []
    sigma_upper_list: List[float] = []   # -alpha_upper/T
    sigma_ig_list: List[float] = []      # -alpha_ig/T

    # Previous state
    U_prev: float | None = U0
    V_prev: float | None = V0
    T_prev: float | None = T0

    # Sampling loop
    while t < cfg.t_final - 1e-15:
        t_next = min(cfg.t_final, t + dt)

        # Safety: stop if Lx too small
        Lx_next = piston_Lx_at_time(t_next, cfg)
        if Lx_next <= cfg.min_Lx:
            break

        # Advance to t_next
        sim.advance_to(t_next)

        # Measurements
        impulses = sim.get_pressure_history(window=(t_next - t))
        events = 0 if impulses is None else impulses.shape[0]
        P = pressure_from_impulses(impulses, piston_area, (t_next - t))

        v = sim.get_velocities()
        U, T_th = internal_energy_and_temperature(
            v, mass=cfg.mass, kB=cfg.kB, subtract_bulk=cfg.subtract_bulk_drift
        )
        W = sim.get_work_done()
        V = Lx_next * Ly * Lz

        # Dual pressure track: impulse + IG baseline
        PIG = (cfg.num_particles * cfg.kB * T_th) / V

        # Invariants (use IG pressure to reduce noise)
        PV_gamma = PIG * (V ** cfg.gamma)
        TV_gm1 = T_th * (V ** (cfg.gamma - 1.0))

        # δ-quantities per window
        if U_prev is not None:
            dU = U - U_prev
            dV = V - V_prev
            # IG entropy difference for calibration-only use
            dS_IG = cfg.num_particles * (1.5 * safe_log_ratio(T_th, T_prev) + safe_log_ratio(V, V_prev))
            Tavg = 0.5 * (T_th + T_prev)
            Pavg = P  # impulse window

            alpha_upper = dU + Pavg * dV                   # upper bound (ΔS ≥ 0 ⇒ α ≤ α_upper)
            delta_a_ig = -Pavg * dV + Tavg * dS_IG
            alpha_ig = dU - delta_a_ig

            alpha_upper_list.append(alpha_upper)
            alpha_ig_list.append(alpha_ig)
            sigma_upper_list.append(-alpha_upper / max(Tavg, 1e-16))
            sigma_ig_list.append(-alpha_ig / max(Tavg, 1e-16))
        else:
            alpha_upper_list.append(0.0)
            alpha_ig_list.append(0.0)
            sigma_upper_list.append(0.0)
            sigma_ig_list.append(0.0)

        # Log step
        times.append(t_next)
        vols.append(V)
        Lxs.append(Lx_next)
        P_imp.append(P)
        P_ig.append(PIG)
        works.append(W)
        Us.append(U)
        Ts.append(T_th)
        PVg.append(PV_gamma)
        TVgm1.append(TV_gm1)
        num_events.append(events)

        # Adaptive window: aim for constant event count
        if cfg.adapt_window_by_events:
            # crude proportional controller
            target = cfg.target_events
            if events > 0:
                ratio = target / max(events, 1)
                # limit changes to avoid jitter
                ratio = max(0.5, min(2.0, ratio))
                dt = max(cfg.dt_min, min(cfg.dt_max, dt * ratio))
            else:
                # if no events, increase window gently
                dt = min(cfg.dt_max, dt * 1.5)

        # Update prev
        U_prev, V_prev, T_prev = U, V, T_th
        t = t_next

    # Final values
    if len(times) == 0:
        raise RuntimeError("No data collected; consider extending t_final or adjusting dt.")

    t_end = times[-1]
    U_end = Us[-1]
    T_end = Ts[-1]
    V_end = vols[-1]
    W_end = works[-1]
    P_end = P_imp[-1]
    PIG_end = P_ig[-1]

    # Baseline (reversible adiabat) from warmup state (use IG P0 to reduce noise)
    T_ad_end = adiabatic_T_of_V(T0, V0, V_end, cfg.gamma)
    P_ad_end = adiabatic_P_of_V(P0_IG, V0, V_end, cfg.gamma)
    chi_T_end = T_end / max(T_ad_end, 1e-30) - 1.0
    chi_P_end = P_end / max(P_ad_end, 1e-30) - 1.0

    # RMS drifts of invariants (normalized to warmup)
    PVg0 = P0_IG * (V0 ** cfg.gamma)
    TVgm10 = T0 * (V0 ** (cfg.gamma - 1.0))
    pv_rat = np.array(PVg) / max(PVg0, 1e-30)
    tv_rat = np.array(TVgm1) / max(TVgm10, 1e-30)
    rms_pv = float(np.sqrt(np.mean((pv_rat - 1.0) ** 2)))
    rms_tv = float(np.sqrt(np.mean((tv_rat - 1.0) ** 2)))

    # EOS consistency (impulse vs IG)
    P_imp_arr = np.array(P_imp)
    P_ig_arr = np.array(P_ig)
    eos_rel_diff = np.abs(P_imp_arr - P_ig_arr) / np.maximum(P_ig_arr, 1e-30)
    eos_rel_diff_rms = float(np.sqrt(np.mean(eos_rel_diff ** 2)))
    eos_rel_diff_end = float(eos_rel_diff[-1])

    # Work–energy residual
    dU_total = U_end - U0
    dW_total = W_end - W0
    res_WE = dU_total - dW_total

    # Entropy production summaries
    # drop the first element (it is zero-by-construction)
    s_up = np.array(sigma_upper_list[1:], dtype=float)
    s_ig = np.array(sigma_ig_list[1:], dtype=float)
    avg_sigma_upper = float(np.mean(s_up)) if s_up.size else 0.0
    cum_sigma_upper = float(np.sum(s_up)) if s_up.size else 0.0
    avg_sigma_ig = float(np.mean(s_ig)) if s_ig.size else 0.0
    cum_sigma_ig = float(np.sum(s_ig)) if s_ig.size else 0.0

    # Must-pass checks (per-run)
    tol_rel_WE = 0.05  # 5% tolerance
    denom = max(1e-12, abs(dU_total) + abs(dW_total))
    pass_WE = (abs(res_WE) / denom) < tol_rel_WE

    # Save per-run time series
    ts_path = os.path.join(ts_dir, f"{tag}.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time", "Lx", "V", "P_impulse", "P_IG", "U", "T", "W_cum",
                "PV_gamma(IG)", "T*V^(gamma-1)", "events",
                "alpha_upper", "alpha_IG", "sigma_upper", "sigma_IG",
            ]
        )
        for row in zip(
            times, Lxs, vols, P_imp, P_ig, Us, Ts, works, PVg, TVgm1, num_events,
            alpha_upper_list, alpha_ig_list, sigma_upper_list, sigma_ig_list
        ):
            w.writerow([f"{x:.12g}" for x in row])

    # Optional figure
    if base_cfg.save_fig:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
            ax[0].plot(times, Ts, label="T(th)")
            ax[0].plot(times, [adiabatic_T_of_V(T0, V0, v, cfg.gamma) for v in vols], label="T_ad(V)", ls="--")
            ax[0].set_ylabel("T"); ax[0].legend()

            ax[1].plot(times, P_imp, label="P (impulse)")
            ax[1].plot(times, P_ig, label="P_IG = NkBT/V", ls="--")
            ax[1].plot(times, [adiabatic_P_of_V(P0_IG, V0, v, cfg.gamma) for v in vols], label="P_ad(V)", ls=":")
            ax[1].set_ylabel("P"); ax[1].legend()

            ax[2].plot(times, vols, label="V")
            ax[2].set_ylabel("V"); ax[2].legend()

            ax[3].plot(times[1:], sigma_ig_list[1:], label="-alpha_IG/T")
            ax[3].plot(times[1:], sigma_upper_list[1:], label="-alpha_upper/T", ls="--")
            ax[3].set_ylabel("σ"); ax[3].set_xlabel("time"); ax[3].legend()

            fig.tight_layout()
            fig_path = os.path.join(fig_dir, f"{tag}.png")
            plt.savefig(fig_path, dpi=140)
            if base_cfg.show_fig:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"[plot] skipped ({e})")

    # Pack summary
    summary = dict(
        tag=tag,
        N=cfg.num_particles,
        dt_init=cfg.dt_init,
        adapt=cfg.adapt_window_by_events,
        target_events=cfg.target_events,
        piston_velocity=cfg.piston_velocity,
        t_end=t_end,
        # initial state
        T0=T0, V0=V0, P0_IG=P0_IG, W0=W0, U0=U0,
        # final
        T_end=T_end, V_end=V_end, P_end=P_end, PIG_end=PIG_end, W_end=W_end, U_end=U_end,
        # summaries
        avg_sigma_IG=avg_sigma_ig, cum_sigma_IG=cum_sigma_ig,
        avg_sigma_upper=avg_sigma_upper, cum_sigma_upper=cum_sigma_upper,
        chi_T_end=chi_T_end, chi_P_end=chi_P_end,
        rms_PV_gamma=rms_pv, rms_TV_gm1=rms_tv,
        res_WE=res_WE, pass_WE=int(pass_WE),
        eos_rel_diff_rms=eos_rel_diff_rms, eos_rel_diff_end=eos_rel_diff_end,
        ts_path=os.path.relpath(ts_path),
    )
    return summary


# -------------------------------
# Sweep and checks
# -------------------------------

def sweep_and_summarize(base_cfg: BaseConfig, sweep: SweepConfig) -> None:
    ensure_dir(base_cfg.out_dir)
    summary_path = os.path.join(base_cfg.out_dir, "phase1_sweep_summary.csv")
    summary_rows: List[Dict[str, float]] = []

    # Choose default sweep sizes
    velocities = sweep.velocities
    dts = sweep.dts
    Ns = sweep.Ns

    print(f"[sweep] velocities={velocities}, dts={dts}, Ns={Ns}")

    # Run experiments
    for N in Ns:
        for dt in dts:
            for vel in velocities:
                tag = f"N{N}_dt{dt:.3f}_v{vel:+.4f}"
                print(f"[run] {tag}")
                row = run_one_experiment(base_cfg, vel, dt, N, tag)
                summary_rows.append(row)

    # Write summary CSV
    header = [
        "tag", "N", "dt_init", "adapt", "target_events", "piston_velocity",
        "t_end",
        "T0", "V0", "P0_IG", "W0", "U0",
        "T_end", "V_end", "P_end", "PIG_end", "W_end", "U_end",
        "avg_sigma_IG", "cum_sigma_IG",
        "avg_sigma_upper", "cum_sigma_upper",
        "chi_T_end", "chi_P_end",
        "rms_PV_gamma", "rms_TV_gm1",
        "res_WE", "pass_WE",
        "eos_rel_diff_rms", "eos_rel_diff_end",
        "ts_path",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in summary_rows:
            w.writerow([row.get(k, "") for k in header])

    print(f"[done] wrote sweep summary: {summary_path}")

    # ---- Must-pass checks across runs ----

    # #1 Quasi-static trend: within each (N, dt), avg_sigma_IG should increase with |vel|
    # Evaluate simple monotonicity: slow < mid < fast
    def groups(rows: List[Dict[str, float]]):
        key2rows: Dict[Tuple[int, float], List[Dict[str, float]]] = {}
        for r in rows:
            key = (int(r["N"]), float(r["dt_init"]))
            key2rows.setdefault(key, []).append(r)
        return key2rows

    group_ok = []
    for (N, dt) in sorted(groups(summary_rows).keys()):
        rows = groups(summary_rows)[(N, dt)]
        rows_sorted = sorted(rows, key=lambda r: abs(float(r["piston_velocity"])))
        if len(rows_sorted) >= 3:
            s0 = float(rows_sorted[0]["avg_sigma_IG"])
            s1 = float(rows_sorted[1]["avg_sigma_IG"])
            s2 = float(rows_sorted[2]["avg_sigma_IG"])
            ok = (s0 <= s1 + 1e-12) and (s1 <= s2 + 1e-12)
            group_ok.append(ok)
            print(f"[QS] (N={N}, dt={dt:.3f}) avg_sigma_IG: {s0:.3e} ≤ {s1:.3e} ≤ {s2:.3e}  -> {'PASS' if ok else 'FAIL'}")
        else:
            print(f"[QS] (N={N}, dt={dt:.3f}) needs ≥3 velocities to check monotonicity.")

    pass_quasi_static = all(group_ok) if group_ok else False

    # #2 Work–energy per-run: already flagged
    pass_WE_all = all(int(r["pass_WE"]) == 1 for r in summary_rows)

    # #3 Baseline sign & correlation (soft): mean chi_T_end, chi_P_end >= 0; correlation with avg_sigma_IG >= 0
    chiT = np.array([float(r["chi_T_end"]) for r in summary_rows], dtype=float)
    chiP = np.array([float(r["chi_P_end"]) for r in summary_rows], dtype=float)
    sigA = np.array([float(r["avg_sigma_IG"]) for r in summary_rows], dtype=float)

    mean_chiT = float(np.mean(chiT))
    mean_chiP = float(np.mean(chiP))
    # Pearson correlations
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2:
            return 0.0
        a = a - np.mean(a); b = b - np.mean(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    corr_T = corr(chiT, sigA)
    corr_P = corr(chiP, sigA)

    pass_baseline = (mean_chiT >= -1e-3 and mean_chiP >= -1e-3 and corr_T >= 0.2 and corr_P >= 0.2)

    print("\n=== MUST-PASS SUMMARY ===")
    print(f"PASS #1 Quasi-static limit (avg_sigma ↑ with |v_pis|) : {'PASS' if pass_quasi_static else 'FAIL'}")
    print(f"PASS #2 Work–energy (per-run residual small)        : {'PASS' if pass_WE_all else 'FAIL'}")
    print(f"PASS #3 Baseline sign & correlation                 : {'PASS' if pass_baseline else 'FAIL'}")
    print(f"  mean chi_T_end={mean_chiT:.3e}, chi_P_end={mean_chiP:.3e}, corr(chi_T,avg_sigma)={corr_T:.2f}, corr(chi_P,avg_sigma)={corr_P:.2f}")
    print("===============================================")


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 adiabatic sweep with δ-framework probes")
    p.add_argument("--vels", type=float, nargs="+", default=[-0.005, -0.05, -0.2],
                   help="piston velocities (negative=inward)")
    p.add_argument("--dts", type=float, nargs="+", default=[0.1, 0.2, 0.5], help="initial Δt list")
    p.add_argument("--Ns", type=int, nargs="+", default=[9128], help="particle counts")
    p.add_argument("--full", action="store_true", help="3×3×3 sweep (overrides Ns with 3 values)")
    p.add_argument("--no-fig", action="store_true", help="disable per-run figure output")
    p.add_argument("--no-adapt", action="store_true", help="disable adaptive window by events")
    p.add_argument("--out", type=str, default="phase1_out", help="output directory")
    p.add_argument("--tfinal", type=float, default=200.0, help="final time")
    p.add_argument("--warmup", type=float, default=0.5, help="warmup time")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    return p.parse_args()


def main():
    args = parse_args()

    Ns = args.Ns
    if args.full:
        # a heavier default set for universality/size check
        Ns = [4096, 9128, 20000]

    base = BaseConfig(
        num_particles=Ns[0],
        box_size=(20.0, 20.0, 20.0),
        radius=0.2,
        mass=1.0,
        dim=3,
        seed=args.seed,
        piston_wall_id=1,
        piston_velocity=-0.05,
        t_warmup=args.warmup,
        t_final=args.tfinal,
        dt_init=0.2,
        min_Lx=3.0,
        kB=1.0,
        gamma=5.0 / 3.0,
        subtract_bulk_drift=True,
        adapt_window_by_events=not args.no_adapt,
        target_events=200,
        dt_min=0.02,
        dt_max=0.5,
        out_dir=args.out,
        save_fig=not args.no_fig,
        show_fig=False,
    )
    sweep = SweepConfig(
        velocities=args.vels,
        dts=args.dts,
        Ns=Ns,
        full=args.full,
    )
    sweep_and_summarize(base, sweep)


if __name__ == "__main__":
    main()
