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

def drift_removed_ke(vel: np.ndarray, mass: float) -> tuple[float, np.ndarray]:
    u = vel.mean(axis=0, keepdims=True)
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
    return N * (0.5*D*math.log(max(T1,1e-300)/max(T0,1e-300)) + math.log(max(V1,1e-300)/max(V0,1e-300)))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 0 else default

@dataclass
class Cfg:
    N:int = 2048
    box:Tuple[float,float,float] = (20.0,20.0,20.0)
    radius:float = 0.2
    mass:float = 1.0
    dim:int = 3
    seed:int = 2025

    # thermal & piston
    wall_L:int = 0    # x-min
    wall_R:int = 1    # x-max (piston face)
    T_bath:float = 1.0
    accommodation:float = 1.0

    # timing
    t_mix:float = 1.0     # pre-mix (adiabatic)
    t_eq_max:float = 200.0 # <-- 新增：最长等温化时间
    eq_dt:float = 0.5
    eq_tol:float = 0.02    # |T - T_bath|/T_bath < 2% 视为等温化完成

    piston_speed:float = -0.002   # <-- 更慢的活塞速度
    t_final:float = 400.0         # <-- 拉长总时长以保持同终点压缩
    dt:float = 0.5                # 采样节距

    csv_path:str = "phase2_isothermal_qc_v2.csv"
    png_path:str = "phase2_isothermal_qc_v2.png"

def main(cfg: Cfg) -> None:
    sim = GasSim(num_particles=cfg.N, box_size=list(cfg.box),
                 radius=cfg.radius, mass=cfg.mass, dim=cfg.dim, seed=cfg.seed)

    # A) 预混合（全绝热，静壁）
    sim.advance_to(cfg.t_mix)

    # B) 预等温化阶段：两侧都设为热浴，活塞仍静止
    sim.set_thermal_wall(cfg.wall_L, cfg.T_bath, cfg.accommodation)
    sim.set_thermal_wall(cfg.wall_R, cfg.T_bath, cfg.accommodation)  # 双热浴以加速等温化

    t = cfg.t_mix
    v = np.asarray(sim.get_velocities())
    U0, c = drift_removed_ke(v, cfg.mass)
    D = v.shape[1]
    T0 = kinetic_temperature(c, cfg.mass, 1.0)

    Lx0, Ly, Lz = cfg.box
    V0 = Lx0*Ly*Lz

    t_eq = 0.0
    while t_eq < cfg.t_eq_max:
        t += cfg.eq_dt; t_eq += cfg.eq_dt
        sim.advance_to(t)
        v = np.asarray(sim.get_velocities())
        _, c = drift_removed_ke(v, cfg.mass)
        T = kinetic_temperature(c, cfg.mass, 1.0)
        if abs(T - cfg.T_bath)/cfg.T_bath < cfg.eq_tol:
            break  # 等温化达标

    # C) 开始压缩：保持右壁为“移动且热”的活塞
    sim.set_piston(cfg.wall_R, cfg.piston_speed)

    # 采样容器
    times: List[float] = []; Ts: List[float]=[]; Vs: List[float]=[]
    Ws: List[float]=[]; Qs: List[float]=[]; p_mechs: List[float]=[]; p_igs: List[float]=[]
    sigmas: List[float] = []; dS_sys_hist: List[float]=[]

    # 基线
    W_prev = 0.0; Q_prev = 0.0
    V_prev = V0
    T_prev = T0

    while t < cfg.t_final - 1e-15:
        t_next = min(cfg.t_final, t + cfg.dt)
        sim.advance_to(t_next)

        # 状态
        v = np.asarray(sim.get_velocities())
        U, c = drift_removed_ke(v, cfg.mass)
        T = kinetic_temperature(c, cfg.mass, 1.0)

        # 功与热：热量取“全部热边界”的和（两侧皆热）
        W_total, _Q_core = sim.get_work_heat()
        dW = W_total - W_prev
        dQ = sum_heat(sim, window=(t_next - t), wall_id=None)  # sum all thermal walls
        Q_total = Q_prev + dQ

        # 体积与压强
        Lx = Lx0 + cfg.piston_speed * max(0.0, t_next - (cfg.t_mix + t_eq))
        V = Lx * Ly * Lz
        dV = V - V_prev
        p_mech = -safe_div(dW, dV, default=(p_mechs[-1] if p_mechs else 0.0))
        p_ig = (cfg.N * T) / max(V, 1e-12)

        # δ‑回路（窗量）
        # 用 T_prev,V_prev -> T,V 的窗增量
        U_prev_win = 0.5 * cfg.mass * (cfg.N * D * T_prev)
        dU = U - U_prev_win
        dS_IG = ideal_S_IG(cfg.N, D, T, T_prev, V, V_prev)
        delta_a_IG = (-p_mech * dV) + T * dS_IG
        alpha_IG = dU - delta_a_IG
        sigma = -safe_div(alpha_IG, T, 0.0)

        # 记录
        times.append(t_next); Ts.append(T); Vs.append(V)
        Ws.append(W_total); Qs.append(Q_total)
        p_mechs.append(p_mech); p_igs.append(p_ig)
        sigmas.append(sigma); dS_sys_hist.append(dS_IG)

        # 更新窗基线
        W_prev = W_total; Q_prev = Q_total
        V_prev = V; T_prev = T; t = t_next

    # 末态与核对
    v1 = np.asarray(sim.get_velocities())
    U1, c1 = drift_removed_ke(v1, cfg.mass)
    T1 = kinetic_temperature(c1, cfg.mass, 1.0)
    V1 = Vs[-1]
    W_total, Q_core_raw = sim.get_work_heat()
    Q_total = Qs[-1]
    dU = U1 - U0

    W_rev = cfg.N * cfg.T_bath * math.log(V0 / V1)
    Q_rev = -W_rev

    residual = abs(dU - (W_total + Q_total))
    rel_residual = residual / max(1.0, abs(U0), abs(U1))

    dS_sys = ideal_S_IG(cfg.N, D, T1, T0, V1, V0)
    Sigma_tot = dS_sys - (Q_total / cfg.T_bath)
    Sigma_int = float(np.sum(sigmas))

    pv_diff = np.asarray(p_mechs) - np.asarray(p_igs)
    eos_rel_rms = float(np.sqrt(np.mean((pv_diff / np.maximum(1e-12, np.asarray(p_igs)))**2)))

    with open(cfg.csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time","T","V","W","Q","p_mech","p_IG","sigma","dS_IG_win"])
        for row in zip(times,Ts,Vs,Ws,Qs,p_mechs,p_igs,sigmas,dS_sys_hist):
            w.writerow([f"{x:.9g}" for x in row])

    print("=== Phase 2 QC v2: Isothermal Compression ===")
    print(f"T0={T0:.6f}  T1={T1:.6f}  (to T_bath={cfg.T_bath}, rel dev={((T1-cfg.T_bath)/cfg.T_bath):+.3%})")
    print(f"V0={V0:.6f}  V1={V1:.6f}")
    print(f"U0={U0:.6f}  U1={U1:.6f}  ΔU={dU:.6f}")
    print(f"W_total={W_total:.6f}  Q_total={Q_total:.6f}")
    print(f"|ΔU-(W+Q)|={residual:.3e} (rel={rel_residual:.3e})")
    print(f"[reversible]  W_rev={W_rev:.6f}  Q_rev={Q_rev:.6f}")
    print(f"Excess work  W-W_rev = {W_total-W_rev:+.6f}")
    print(f"Excess heat  Q-Q_rev = {Q_total-Q_rev:+.6f}")
    print(f"ΔS_sys(IG) = {dS_sys:+.6f}  Σ_tot = {Sigma_tot:+.6f}")
    print(f"Σ_int (δ-circuit sum) ≈ {Sigma_int:+.6f}")
    print(f"EOS rel-RMS = {eos_rel_rms:.3e}")
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
