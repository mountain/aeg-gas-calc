#!/usr/bin/env python3
"""
Phase 2 QC: Isothermal Compression with First-Law Closure, EOS, and δ-circuit diagnostics

- x-min wall (id=0): thermal bath at T_bath
- x-max wall (id=1): moving piston (constant speed)
- Time-series sampling of: T_kin (drift-removed), V(t), W(t), Q(t), p_mech, p_IG
- Theory checks (isothermal reversible references):
    W_rev(on gas) = N * T_bath * ln(V0 / V1)
    Q_rev = - W_rev
    ΔS_IG = N * [ (D/2) ln(T1/T0) + ln(V1/V0) ]  ~  N ln(V1/V0)  (isothermal)
- QC must-pass (soft thresholds):
    (1) First Law: |ΔU - (W+Q)| / max(|U0|,|U1|,1) <= 1e-3 ~ 1e-2
    (2) Near-isothermal: |T_end - T_bath| / T_bath <= 0.05  (速度足够慢时)
    (3) Second Law (universe): Σ_tot = ΔS_sys - Q/T_bath  >= -5e-3  (容噪)

Also computes windowed δ-circuit residual:
    α_IG = dU - [ -p_mech dV + T_kin dS_IG ]        (ideal-gas δa)
    σ = -α_IG / T_kin                                (entropy-production measure)

Run:
  python examples/isothermal_compression_qc.py
"""
#!/usr/bin/env python3
from __future__ import annotations
import math, csv
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

from gassim import GasSim

# ---------- helpers ----------
def drift_removed_ke(vel: np.ndarray, mass: float) -> tuple[float, np.ndarray]:
    u = vel.mean(axis=0, keepdims=True)   # bulk drift
    c = vel - u
    U = 0.5 * mass * np.einsum("ij,ij->", c, c)
    return float(U), c

def kinetic_temperature(c: np.ndarray, mass: float, kB: float = 1.0) -> float:
    N, D = c.shape
    Ek = 0.5 * mass * np.einsum("ij,ij->", c, c)
    return 2.0 * Ek / (D * N * kB)

def sum_heat(sim: GasSim, window: float, wall_id: Optional[int]) -> float:
    hist = np.asarray(sim.get_heat_history(window=window))
    if hist.size == 0:
        return 0.0
    if hist.ndim == 2 and hist.shape[1] >= 3 and wall_id is not None:
        return float(hist[hist[:,2].astype(int)==int(wall_id), 1].sum())
    return float(hist[:,1].sum())

def ideal_S_IG(N:int, D:int, T1:float, T0:float, V1:float, V0:float)->float:
    # ΔS_IG = N*(D/2 ln(T1/T0) + ln(V1/V0))
    T1s=max(T1,1e-300); T0s=max(T0,1e-300); V1s=max(V1,1e-300); V0s=max(V0,1e-300)
    return N * (0.5*D*math.log(T1s/T0s) + math.log(V1s/V0s))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 0 else default

# ---------- config ----------
@dataclass
class Cfg:
    N:int = 2048
    box:Tuple[float,float,float] = (20.0,20.0,20.0)
    radius:float = 0.2
    mass:float = 1.0
    dim:int = 3
    seed:int = 2025

    wall_L:int = 0  # x-min (thermal)
    wall_R:int = 1  # x-max (piston + thermal)

    T_bath:float = 1.0
    accommodation:float = 1.0

    t_mix:float = 1.0      # adiabatic pre-mix
    t_eq_max:float = 200.0 # max equilibration
    eq_dt:float = 0.5
    eq_tol:float = 0.02    # |T- T_bath|/T_bath < 2%

    piston_speed:float = -0.002
    t_final:float = 400.0
    dt:float = 0.5

    csv_path:str = "phase2_isothermal_qc_v3.csv"
    png_path:str = "phase2_isothermal_qc_v3.png"

def main(cfg: Cfg) -> None:
    # --- build & pre-mix (adiabatic, static walls) ---
    sim = GasSim(num_particles=cfg.N, box_size=list(cfg.box),
                 radius=cfg.radius, mass=cfg.mass, dim=cfg.dim, seed=cfg.seed)
    sim.advance_to(cfg.t_mix)

    # --- set two thermal walls, equilibrate to T_bath (piston still static) ---
    sim.set_thermal_wall(cfg.wall_L, cfg.T_bath, cfg.accommodation)
    sim.set_thermal_wall(cfg.wall_R, cfg.T_bath, cfg.accommodation)

    t = cfg.t_mix
    Lx0, Ly, Lz = cfg.box
    V_box = lambda Lx: Lx*Ly*Lz

    # measure once (for display only)
    v = np.asarray(sim.get_velocities())
    U_pre, c = drift_removed_ke(v, cfg.mass)
    D = v.shape[1]
    T_pre = kinetic_temperature(c, cfg.mass, 1.0)

    # equilibration loop
    t_eq = 0.0
    while t_eq < cfg.t_eq_max:
        t_next = t + cfg.eq_dt
        sim.advance_to(t_next)
        v = np.asarray(sim.get_velocities())
        _, c = drift_removed_ke(v, cfg.mass)
        T_now = kinetic_temperature(c, cfg.mass, 1.0)
        t, t_eq = t_next, t_eq + cfg.eq_dt
        if abs(T_now - cfg.T_bath)/cfg.T_bath < cfg.eq_tol:
            break

    # === unified baseline at the START OF COMPRESSION ===
    t_star = t                    # time when compression starts
    sim.set_piston(cfg.wall_R, cfg.piston_speed)  # moving + thermal piston

    v = np.asarray(sim.get_velocities())
    U0, c0 = drift_removed_ke(v, cfg.mass)        # U baseline
    T0 = kinetic_temperature(c0, cfg.mass, 1.0)   # should be ≈ T_bath
    Lx_star = Lx0                                  # piston has not moved yet
    V0 = V_box(Lx_star)

    W0_core, Q0_core = sim.get_work_heat()        # W,Q baselines（core）
    # also准备窗口累计器
    W_prev = W0_core
    Q_hist_total = 0.0

    # --- sampling during compression ---
    times: List[float]=[]; Ts: List[float]=[]; Vs: List[float]=[]
    p_mechs: List[float]=[]; p_igs: List[float]=[]; sigmas: List[float]=[]
    Ws_core: List[float]=[]; Qs_hist: List[float]=[]

    V_prev = V0; T_prev = T0

    while t < cfg.t_final - 1e-15:
        t_next = min(cfg.t_final, t + cfg.dt)
        sim.advance_to(t_next)

        # state
        v = np.asarray(sim.get_velocities())
        U, c = drift_removed_ke(v, cfg.mass)
        T = kinetic_temperature(c, cfg.mass, 1.0)

        # volume (only Lx changes)
        Lx = Lx0 + cfg.piston_speed * (t_next - t_star if t_next>t_star else 0.0)
        V = V_box(Lx)
        dV = V - V_prev

        # work from core (increment in this window)
        W_core, _Q_core_raw = sim.get_work_heat()
        dW_core = W_core - W_prev

        # mechanical pressure from work–volume slope（无偏）
        p_mech = -safe_div(dW_core, dV, p_mechs[-1] if p_mechs else 0.0)
        # ideal-gas proxy（诊断）
        p_ig = (cfg.N * T) / max(V, 1e-12)

        # heat from history（仅用于图示；严格闭合用 core）
        dQ_hist = sum_heat(sim, window=(t_next - t), wall_id=None)
        Q_hist_total += dQ_hist
        W_prev = W_core

        # δ‑回路窗量（只依赖宏观量，基线与 t_star 对齐）
        U_prev_win = 0.5 * cfg.mass * (cfg.N * D * T_prev)
        dU_win = U - U_prev_win
        dS_IG = ideal_S_IG(cfg.N, D, T, T_prev, V, V_prev)
        delta_a_IG = (-p_mech * dV) + T * dS_IG
        alpha_IG = dU_win - delta_a_IG
        sigma = -safe_div(alpha_IG, T, 0.0)

        # record
        times.append(t_next); Ts.append(T); Vs.append(V)
        p_mechs.append(p_mech); p_igs.append(p_ig); sigmas.append(sigma)
        Ws_core.append(W_core - W0_core)     # core增量
        Qs_hist.append(Q_hist_total)          # heat-history增量

        # advance
        V_prev = V; T_prev = T; t = t_next

    # --- end-state & checks ---
    v1 = np.asarray(sim.get_velocities())
    U1, c1 = drift_removed_ke(v1, cfg.mass)
    T1 = kinetic_temperature(c1, cfg.mass, 1.0)
    V1 = Vs[-1]

    W_core_end, Q_core_end = sim.get_work_heat()
    W_total = W_core_end - W0_core          # 核心累计的“真”功
    Q_total_core = Q_core_end - Q0_core     # 核心累计的“真”热
    Q_total_hist = Q_hist_total             # 通过窗口求和统计的热（用于交叉校验）
    Q_true = Q_total_core - W_total

    dU = U1 - U0

    # reversible isothermal reference
    W_rev = cfg.N * cfg.T_bath * math.log(V0 / V1)
    Q_rev = -W_rev

    # first law closure
    residual = abs(dU - (W_total + Q_true))
    rel_residual = residual / max(1.0, abs(U0), abs(U1))

    # entropy diagnostics
    dS_sys = ideal_S_IG(cfg.N, D, T1, T0, V1, V0)
    Sigma_tot = dS_sys - (Q_true / cfg.T_bath)
    Sigma_int = float(np.sum(sigmas))

    # EOS deviation (mechanical vs NT/V)
    pv_diff = np.asarray(p_mechs) - np.asarray(p_igs)
    eos_rel_rms = float(np.sqrt(np.mean((pv_diff / np.maximum(1e-12, np.asarray(p_igs)))**2)))

    # --- output ---
    with open(cfg.csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time","T","V","W_core","Q_hist","p_mech","p_IG","sigma"])
        for row in zip(times,Ts,Vs,Ws_core,Qs_hist,p_mechs,p_igs,sigmas):
            w.writerow([f"{x:.9g}" for x in row])

    print("=== Phase 2 QC v3: Isothermal Compression (baselines aligned at t*=start of compression) ===")
    print(f"t_star={t_star:.3f} (after equilibration), piston_speed={cfg.piston_speed}")
    print(f"T0@t*={T0:.6f}  T1={T1:.6f}  (T_bath={cfg.T_bath}, rel dev={((T1-cfg.T_bath)/cfg.T_bath):+.3%})")
    print(f"V0={V0:.6f}  V1={V1:.6f}")
    print(f"U0@t*={U0:.6f}  U1={U1:.6f}  ΔU={dU:.6f}")
    print(f"W_total(core)={W_total:.6f}  Q_total(core)={Q_total_core:.6f}  |ΔU-(W+Q)|={residual:.3e} (rel={rel_residual:.3e})")
    print(f"[cross-check] Q_total(hist)={Q_total_hist:.6f}  diff(hist-core)={Q_total_hist-Q_total_core:+.6f}")
    print(f"[reversible]  W_rev={W_rev:.6f}  Q_rev={Q_rev:.6f}  (W-W_rev={W_total-W_rev:+.6f})")
    print(f"ΔS_sys(IG) = {dS_sys:+.6f}  Σ_tot = {Sigma_tot:+.6f}  Σ_int(δ-circuit sum) ≈ {Sigma_int:+.6f}")
    print(f"EOS rel-RMS (p_mech vs NT/V) = {eos_rel_rms:.3e}")

    PASS1 = rel_residual <= 1e-2
    PASS2 = abs(T1 - cfg.T_bath)/cfg.T_bath <= 5e-2
    PASS3 = (Sigma_tot >= -5e-3)
    print("=== MUST-PASS ===")
    print(f"First Law closure        : {'PASS' if PASS1 else 'FAIL'}")
    print(f"Near-isothermal endpoint : {'PASS' if PASS2 else 'FAIL'}")
    print(f"Second Law (Σ_tot ≥ 0)   : {'PASS' if PASS3 else 'FAIL'}")
    print("====================")
    print(f"Wrote CSV: {cfg.csv_path}")

    if HAS_PLT:
        fig, ax = plt.subplots(4,1,figsize=(8,10),sharex=True)
        ax[0].plot(times, Ts, label="T_kin"); ax[0].axhline(cfg.T_bath, ls="--", label="T_bath")
        ax[0].set_ylabel("T"); ax[0].legend()
        ax[1].plot(times, p_mechs, label="p_mech"); ax[1].plot(times, p_igs, ls="--", label="p_IG=NT/V")
        ax[1].set_ylabel("p"); ax[1].legend()
        ax[2].plot(times, Vs, label="V"); ax[2].set_ylabel("V"); ax[2].legend()
        ax[3].plot(times, sigmas, label="-α_IG/T (window)")
        ax[3].set_xlabel("time"); ax[3].set_ylabel("σ"); ax[3].legend()
        fig.tight_layout(); fig.savefig(cfg.png_path, dpi=150)
        print(f"Wrote figure: {cfg.png_path}")

if __name__ == "__main__":
    main(Cfg())
