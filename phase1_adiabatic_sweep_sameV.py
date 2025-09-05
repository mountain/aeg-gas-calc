# phase1_adiabatic_sweep_sameV.py
"""
Phase 1 — Adiabatic compression sweep (N‑扩展 + 同终点压缩) with δ-framework probes

改动要点
--------
1) N‑扩展：保持初始数密度/体积分数不变。对每个 N，按基准盒子体积 V_ref 与 N_ref
   计算缩放因子 s=(N/N_ref)^(1/3)，把 (Lx0,Ly0,Lz0) 同比缩放为 s*(Lx_ref,Ly_ref,Lz_ref)。

2) 同终点压缩：所有 run 压到相同体积比 V_end/V0 = rV（默认 0.5）。程序精确对齐最后一步，
   令各 run 在一致的 V_end 停止，从而终点 T、P、U 可直接横向比较。

保留并扩展的指标
----------------
- 平均/累计熵产（IG 量尺与“上界版”alpha_upper 两轨）
- 绝热基线偏离 chi_T_end, chi_P_end
- 绝热不变量 RMS 漂移: rms_PV_gamma, rms_TV_gm1
- 功–能量一致性残差 res_WE
- 状态方程一致性（冲量压强 vs IG 压强）
- 新增：熵产归一化（单位粒子、单位 ln 压缩量）:
    sigma_IG_norm = cum_sigma_IG / (N * ln(V0/V_end))
    sigma_upper_norm = cum_sigma_upper / (N * ln(V0/V_end))

“必过线”
--------
#1 准静态极限：同 (N, dt) 下，avg_sigma_IG 随 |v_pis| 单调上升
#2 功–能量：每组 |ΔU-W|/(|ΔU|+|W|) 小于阈值
#3 基线偏离与耗散同向：mean(chi_T_end), mean(chi_P_end) ≥ 0 且与 avg_sigma_IG 的相关系数为正
#0 终点一致性（新增）：每组 |V_end/(rV*V0)-1| 小于阈值（默认 5e-4）

运行
----
  python phase1_adiabatic_sweep_sameV.py
  python phase1_adiabatic_sweep_sameV.py --full                # 3×3×3（27组）
  python phase1_adiabatic_sweep_sameV.py --Vratio 0.4          # 压到 40% 体积
  python phase1_adiabatic_sweep_sameV.py --Ns 4096 9128 20000  # 自定义 N
  python phase1_adiabatic_sweep_sameV.py --no-fig --no-adapt

依赖
----
  - maturin develop -m pyproject.toml （安装 gassim）
  - numpy, matplotlib
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
# 配置
# -------------------------------

@dataclass
class BaseConfig:
    # 基准盒子与 N（用于密度保持的缩放参考）
    num_particles: int = 9128
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)  # reference (Lx0, Ly0, Lz0)
    radius: float = 0.2
    mass: float = 1.0
    dim: int = 3
    seed: int = 123

    # 活塞（x-max 壁为 1；负号=向内压缩）
    piston_wall_id: int = 1
    piston_velocity: float = -0.05

    # 采样与时间
    t_warmup: float = 0.5
    t_final_soft: float = 1e9  # “软”上限，不实际触发；我们按体积终点停
    dt_init: float = 0.2

    # 几何与安全
    min_Lx: float = 3.0
    gamma: float = 5.0 / 3.0
    kB: float = 1.0

    # 数值工程
    subtract_bulk_drift: bool = True
    adapt_window_by_events: bool = True
    target_events: int = 200
    dt_min: float = 0.02
    dt_max: float = 0.5

    # 输出
    out_dir: str = "phase1_out_sameV"
    save_fig: bool = True
    show_fig: bool = False

    # N‑扩展：保持初始数密度
    keep_initial_density: bool = True  # True: 盒子随 N 缩放，保持 ρ0 常数

    # 同终点压缩：统一体积比
    Vratio_end: float = 0.5            # V_end / V0
    end_tolerance: float = 5e-4        # V 终点相对误差阈值


@dataclass
class SweepConfig:
    velocities: List[float]
    dts: List[float]
    Ns: List[int]
    full: bool = False


# -------------------------------
# 工具函数
# -------------------------------

def scaled_box_for_N(N: int, base_cfg: BaseConfig) -> Tuple[float, float, float]:
    """
    若 keep_initial_density=True，则按 (N/N_ref)^(1/3) 同比缩放盒子；
    否则直接使用基准盒子。
    """
    if not base_cfg.keep_initial_density:
        return base_cfg.box_size
    N_ref = base_cfg.num_particles
    Lx_ref, Ly_ref, Lz_ref = base_cfg.box_size
    s = (N / N_ref) ** (1.0 / 3.0)
    return (Lx_ref * s, Ly_ref * s, Lz_ref * s)


def piston_Lx_at_time(t: float, warmup: float, Lx0: float, vpis: float) -> float:
    if t <= warmup:
        return Lx0
    return Lx0 + vpis * (t - warmup)


def internal_energy_and_temperature(
    vel: np.ndarray, mass: float, kB: float, subtract_bulk: bool
) -> Tuple[float, float]:
    assert vel.ndim == 2 and vel.shape[1] in (2, 3)
    v_rel = vel - np.mean(vel, axis=0, keepdims=True) if subtract_bulk else vel
    Ek = 0.5 * mass * float(np.sum(v_rel * v_rel))
    N, D = vel.shape
    T = (2.0 * Ek) / (D * N * kB)
    return Ek, T


def pressure_from_impulses(impulses: np.ndarray, area: float, window: float) -> float:
    if impulses is None or impulses.size == 0 or area <= 0.0 or window <= 0.0:
        return 0.0
    return float(np.sum(impulses[:, 1])) / (area * window)


def safe_log_ratio(a2: float, a1: float, eps: float = 1e-16) -> float:
    return float(np.log(max(a2, eps)) - np.log(max(a1, eps)))


def adiabatic_T_of_V(T0: float, V0: float, V: float, gamma: float) -> float:
    return T0 * (V0 / V) ** (gamma - 1.0)


def adiabatic_P_of_V(P0: float, V0: float, V: float, gamma: float) -> float:
    return P0 * (V0 / V) ** gamma


def ensure_dir(d: str) -> None:
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


# -------------------------------
# 单次实验
# -------------------------------

def run_one_experiment(
    base_cfg: BaseConfig,
    piston_velocity: float,
    dt_init: float,
    N: int,
    tag: str,
) -> Dict[str, float]:
    """
    执行一组 (N, dt, v_pis)。盒子按需要缩放以保持初始数密度，所有 run 压到相同 V_end/V0。
    返回一行 summary，并落盘时序 CSV 与小图。
    """
    cfg = BaseConfig(**asdict(base_cfg))
    cfg.num_particles = N
    cfg.piston_velocity = piston_velocity
    cfg.dt_init = dt_init

    # 输出目录
    ensure_dir(cfg.out_dir)
    ts_dir = os.path.join(cfg.out_dir, "timeseries")
    fig_dir = os.path.join(cfg.out_dir, "figs")
    ensure_dir(ts_dir); ensure_dir(fig_dir)

    # 盒子缩放以保持 ρ0
    Lx0, Ly0, Lz0 = scaled_box_for_N(N, cfg)
    box_now = (Lx0, Ly0, Lz0)

    # 构造模拟器
    sim = GasSim(
        num_particles=N,
        box_size=list(box_now),
        radius=cfg.radius,
        mass=cfg.mass,
        dim=cfg.dim,
        seed=cfg.seed,
    )
    piston_area = Ly0 * Lz0

    # 预热
    sim.advance_to(cfg.t_warmup)
    W0 = sim.get_work_done()
    v_now = sim.get_velocities()
    U0, T0 = internal_energy_and_temperature(
        v_now, cfg.mass, cfg.kB, cfg.subtract_bulk_drift
    )
    V0 = Lx0 * Ly0 * Lz0
    P0_IG = (N * cfg.kB * T0) / V0

    # 统一体积终点
    V_end_target = V0 * cfg.Vratio_end
    Lx_end_target = V_end_target / (Ly0 * Lz0)
    if Lx_end_target <= cfg.min_Lx:
        raise RuntimeError(
            f"目标 Lx_end={Lx_end_target:.3f} 小于 min_Lx={cfg.min_Lx:.3f}；"
            f"请提高 Vratio 或放宽 min_Lx。"
        )
    # 理想到达时间（线性活塞）
    if piston_velocity >= 0:
        raise RuntimeError("活塞速度应为负（向内压缩）。")
    t_hit = cfg.t_warmup + (Lx_end_target - Lx0) / piston_velocity  # v<0, 分子<0 ⇒ 正

    # 打开活塞
    sim.set_piston(cfg.piston_wall_id, piston_velocity)

    # 时序存储
    times: List[float] = []
    Lxs: List[float] = []
    Vs:  List[float] = []
    P_imp: List[float] = []
    P_ig: List[float] = []
    Us: List[float] = []
    Ts: List[float] = []
    Ws: List[float] = []
    PVg: List[float] = []
    TVgm1: List[float] = []
    events_list: List[int] = []
    alpha_upper_list: List[float] = []
    alpha_ig_list: List[float] = []
    sigma_upper_list: List[float] = []
    sigma_ig_list: List[float] = []

    # 迭代
    t = cfg.t_warmup
    dt = cfg.dt_init
    U_prev, V_prev, T_prev = U0, V0, T0

    while True:
        # 截断到终点
        t_next = min(t + dt, t_hit)
        # 对齐几何：若 t_next==t_hit，对应 Lx=Lx_end_target
        Lx_next = piston_Lx_at_time(t_next, cfg.t_warmup, Lx0, piston_velocity)
        if Lx_next <= cfg.min_Lx:
            break  # 容错：不应触发

        # 推进一步
        sim.advance_to(t_next)

        impulses = sim.get_pressure_history(window=(t_next - t))
        events = 0 if impulses is None else impulses.shape[0]
        P = pressure_from_impulses(impulses, piston_area, (t_next - t))

        v_now = sim.get_velocities()
        U, T_th = internal_energy_and_temperature(
            v_now, cfg.mass, cfg.kB, cfg.subtract_bulk_drift
        )
        W = sim.get_work_done()
        V = Lx_next * Ly0 * Lz0
        PIG = (N * cfg.kB * T_th) / V

        PV_gamma = PIG * (V ** cfg.gamma)
        TV_gm1 = T_th * (V ** (cfg.gamma - 1.0))

        # 窗口 δ 量
        dU = U - U_prev
        dV = V - V_prev
        dS_IG = N * (1.5 * safe_log_ratio(T_th, T_prev) + safe_log_ratio(V, V_prev))
        Tavg = 0.5 * (T_th + T_prev)
        Pavg = P

        alpha_upper = dU + Pavg * dV
        delta_a_ig = -Pavg * dV + Tavg * dS_IG
        alpha_ig = dU - delta_a_ig

        alpha_upper_list.append(alpha_upper)
        alpha_ig_list.append(alpha_ig)
        sigma_upper_list.append(-alpha_upper / max(Tavg, 1e-16))
        sigma_ig_list.append(-alpha_ig / max(Tavg, 1e-16))

        # 记录
        times.append(t_next);  Lxs.append(Lx_next);  Vs.append(V)
        P_imp.append(P);  P_ig.append(PIG)
        Us.append(U);  Ts.append(T_th);  Ws.append(W)
        PVg.append(PV_gamma);  TVgm1.append(TV_gm1)
        events_list.append(events)

        # 自适应窗口：以事件数为目标稳定方差
        if cfg.adapt_window_by_events and t_next < t_hit:
            target = cfg.target_events
            if events > 0:
                ratio = target / max(events, 1)
                ratio = max(0.5, min(2.0, ratio))
                dt = max(cfg.dt_min, min(cfg.dt_max, dt * ratio))
            else:
                dt = min(cfg.dt_max, dt * 1.5)

        # 更新
        U_prev, V_prev, T_prev = U, V, T_th
        t = t_next

        # 到达终点则退出
        if abs(t - t_hit) < 1e-12:
            break

    # 终点与检查
    t_end = times[-1]
    U_end, T_end, V_end, W_end = Us[-1], Ts[-1], Vs[-1], Ws[-1]
    P_end, PIG_end = P_imp[-1], P_ig[-1]

    # 终点一致性
    end_ok = abs(V_end / V_end_target - 1.0) <= cfg.end_tolerance

    # 绝热基线（以 IG 压强作降噪基线）
    T_ad_end = adiabatic_T_of_V(T0, V0, V_end, cfg.gamma)
    P_ad_end = adiabatic_P_of_V(P0_IG, V0, V_end, cfg.gamma)
    chi_T_end = T_end / max(T_ad_end, 1e-30) - 1.0
    chi_P_end = P_end / max(P_ad_end, 1e-30) - 1.0

    PVg0 = P0_IG * (V0 ** cfg.gamma)
    TVgm10 = T0 * (V0 ** (cfg.gamma - 1.0))
    pv_rat = np.array(PVg) / max(PVg0, 1e-30)
    tv_rat = np.array(TVgm1) / max(TVgm10, 1e-30)
    rms_pv = float(np.sqrt(np.mean((pv_rat - 1.0) ** 2)))
    rms_tv = float(np.sqrt(np.mean((tv_rat - 1.0) ** 2)))

    # 功–能量一致性
    dU_total = U_end - U0
    dW_total = W_end - W0
    res_WE = dU_total - dW_total
    tol_rel_WE = 0.05
    denom_WE = max(1e-12, abs(dU_total) + abs(dW_total))
    pass_WE = (abs(res_WE) / denom_WE) < tol_rel_WE

    # 熵产汇总（去掉首窗）
    s_up = np.array(sigma_upper_list, dtype=float)
    s_ig = np.array(sigma_ig_list, dtype=float)
    avg_sigma_upper = float(np.mean(s_up)) if s_up.size else 0.0
    cum_sigma_upper = float(np.sum(s_up)) if s_up.size else 0.0
    avg_sigma_ig = float(np.mean(s_ig)) if s_ig.size else 0.0
    cum_sigma_ig = float(np.sum(s_ig)) if s_ig.size else 0.0

    # 归一化（单位粒子、单位 ln 压缩量）
    dlnV = math.log(V0 / V_end)
    sigma_IG_norm = (cum_sigma_ig / (N * dlnV)) if dlnV > 0 else 0.0
    sigma_upper_norm = (cum_sigma_upper / (N * dlnV)) if dlnV > 0 else 0.0

    # EOS 一致性
    P_imp_arr = np.array(P_imp);  P_ig_arr = np.array(P_ig)
    eos_rel_diff = np.abs(P_imp_arr - P_ig_arr) / np.maximum(P_ig_arr, 1e-30)
    eos_rel_diff_rms = float(np.sqrt(np.mean(eos_rel_diff ** 2)))
    eos_rel_diff_end = float(eos_rel_diff[-1])

    # 时序 CSV
    ts_path = os.path.join(ts_dir, f"{tag}.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time","Lx","V","P_impulse","P_IG","U","T","W_cum",
            "PV_gamma(IG)","T*V^(gamma-1)","events",
            "alpha_upper","alpha_IG","sigma_upper","sigma_IG"
        ])
        for row in zip(times, Lxs, Vs, P_imp, P_ig, Us, Ts, Ws, PVg, TVgm1, events_list,
                       alpha_upper_list, alpha_ig_list, sigma_upper_list, sigma_ig_list):
            w.writerow([f"{x:.12g}" for x in row])

    # 小图
    if cfg.save_fig:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
            ax[0].plot(times, Ts, label="T(th)")
            ax[0].plot(times, [adiabatic_T_of_V(T0, V0, v, cfg.gamma) for v in Vs], ls="--", label="T_ad(V)")
            ax[0].set_ylabel("T"); ax[0].legend()

            ax[1].plot(times, P_imp, label="P (impulse)")
            ax[1].plot(times, P_ig, ls="--", label="P_IG=NkBT/V")
            ax[1].plot(times, [adiabatic_P_of_V(P0_IG, V0, v, cfg.gamma) for v in Vs], ls=":", label="P_ad(V)")
            ax[1].set_ylabel("P"); ax[1].legend()

            ax[2].plot(times, Vs, label="V"); ax[2].axhline(V_end_target, ls=":", label="V_end target")
            ax[2].set_ylabel("V"); ax[2].legend()

            ax[3].plot(times, sigma_ig_list, label="-alpha_IG/T")
            ax[3].plot(times, sigma_upper_list, ls="--", label="-alpha_upper/T")
            ax[3].set_ylabel("σ"); ax[3].set_xlabel("time"); ax[3].legend()

            fig.tight_layout()
            fig_path = os.path.join(fig_dir, f"{tag}.png")
            fig.savefig(fig_path, dpi=140)
            if cfg.show_fig:
                import matplotlib.pyplot as plt2  # noqa
                plt.show()
        except Exception as e:
            print(f"[plot] skipped: {e}")

    # 汇总
    phi0 = N * (4.0/3.0) * math.pi * (cfg.radius ** 3) / V0
    phi_end = N * (4.0/3.0) * math.pi * (cfg.radius ** 3) / V_end

    summary = dict(
        tag=tag,
        N=N,
        Lx0=Lx0, Ly0=Ly0, Lz0=Lz0,
        V0=V0, V_end_target=V_end_target, V_end=V_end,
        Vratio_end=cfg.Vratio_end, end_ok=int(1 if end_ok else 0),
        piston_velocity=piston_velocity, dt_init=dt_init,
        t_end=t_end, t_hit=t_hit,
        T0=T0, P0_IG=P0_IG, U0=U0, W0=W0,
        T_end=T_end, P_end=P_end, PIG_end=PIG_end, U_end=U_end, W_end=W_end,
        chi_T_end=chi_T_end, chi_P_end=chi_P_end,
        rms_PV_gamma=rms_pv, rms_TV_gm1=rms_tv,
        res_WE=res_WE, pass_WE=int(pass_WE),
        avg_sigma_IG=avg_sigma_ig, cum_sigma_IG=cum_sigma_ig,
        avg_sigma_upper=avg_sigma_upper, cum_sigma_upper=cum_sigma_upper,
        sigma_IG_norm=sigma_IG_norm, sigma_upper_norm=sigma_upper_norm,
        eos_rel_diff_rms=eos_rel_diff_rms, eos_rel_diff_end=eos_rel_diff_end,
        phi0=phi0, phi_end=phi_end,
        ts_path=os.path.relpath(ts_path),
    )
    return summary


# -------------------------------
# 扫描与总表 + 必过线
# -------------------------------

def sweep_and_summarize(base_cfg: BaseConfig, sweep: SweepConfig) -> None:
    ensure_dir(base_cfg.out_dir)
    summary_path = os.path.join(base_cfg.out_dir, "phase1_sweep_summary.csv")
    rows: List[Dict[str, float]] = []

    print(f"[sweep] v_list={sweep.velocities}, dt_list={sweep.dts}, N_list={sweep.Ns}, "
          f"Vratio={base_cfg.Vratio_end}, density_mode={'keep' if base_cfg.keep_initial_density else 'fixed_box'}")

    for N in sweep.Ns:
        for dt in sweep.dts:
            for vel in sweep.velocities:
                tag = f"N{N}_dt{dt:.3f}_v{vel:+.4f}"
                print(f"[run] {tag}")
                rows.append(run_one_experiment(base_cfg, vel, dt, N, tag))

    # 写总表
    header = [
        "tag","N","Lx0","Ly0","Lz0","V0","V_end_target","V_end","Vratio_end","end_ok",
        "piston_velocity","dt_init","t_hit","t_end",
        "T0","P0_IG","U0","W0","T_end","P_end","PIG_end","U_end","W_end",
        "chi_T_end","chi_P_end","rms_PV_gamma","rms_TV_gm1",
        "res_WE","pass_WE",
        "avg_sigma_IG","cum_sigma_IG","avg_sigma_upper","cum_sigma_upper",
        "sigma_IG_norm","sigma_upper_norm",
        "eos_rel_diff_rms","eos_rel_diff_end",
        "phi0","phi_end",
        "ts_path"
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])

    print(f"[done] wrote summary: {summary_path}")

    # ===== “必过线”判据 =====

    # #0 终点一致性
    pass_end = all(int(r["end_ok"]) == 1 for r in rows)

    # #1 准静态极限：固定 (N,dt) 下，avg_sigma_IG 随 |v| 单调上升
    def groups(rows: List[Dict[str, float]]):
        g: Dict[Tuple[int, float], List[Dict[str, float]]] = {}
        for r in rows:
            g.setdefault((int(r["N"]), float(r["dt_init"])), []).append(r)
        return g

    gmap = groups(rows)
    qs_flags = []
    for (N, dt), rs in sorted(gmap.items()):
        seq = sorted(rs, key=lambda r: abs(float(r["piston_velocity"])))
        if len(seq) >= 3:
            s0, s1, s2 = (float(seq[i]["avg_sigma_IG"]) for i in range(3))
            ok = (s0 <= s1 + 1e-12) and (s1 <= s2 + 1e-12)
            qs_flags.append(ok)
            print(f"[QS] (N={N}, dt={dt:.3f}) avg_sigma_IG: {s0:.3e} ≤ {s1:.3e} ≤ {s2:.3e} -> {'PASS' if ok else 'FAIL'}")
        else:
            print(f"[QS] (N={N}, dt={dt:.3f}) need ≥3 velocities.")
    pass_qs = all(qs_flags) if qs_flags else False

    # #2 功–能量一致
    pass_WE_all = all(int(r["pass_WE"]) == 1 for r in rows)

    # #3 基线偏离与耗散同向（软判据）
    chiT = np.array([float(r["chi_T_end"]) for r in rows], dtype=float)
    chiP = np.array([float(r["chi_P_end"]) for r in rows], dtype=float)
    sA   = np.array([float(r["avg_sigma_IG"]) for r in rows], dtype=float)
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2: return 0.0
        a = a - np.mean(a); b = b - np.mean(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
    mean_chiT, mean_chiP = float(np.mean(chiT)), float(np.mean(chiP))
    corr_T, corr_P = corr(chiT, sA), corr(chiP, sA)
    pass_baseline = (mean_chiT >= -1e-3 and mean_chiP >= -1e-3 and corr_T >= 0.2 and corr_P >= 0.2)

    print("\n=== MUST-PASS SUMMARY ===")
    print(f"PASS #0 Same end volume (all runs)             : {'PASS' if pass_end else 'FAIL'}")
    print(f"PASS #1 Quasi-static limit (avg_sigma ↑ |v|)  : {'PASS' if pass_qs else 'FAIL'}")
    print(f"PASS #2 Work–energy consistency               : {'PASS' if pass_WE_all else 'FAIL'}")
    print(f"PASS #3 Baseline sign & correlation           : {'PASS' if pass_baseline else 'FAIL'}")
    print(f"  mean chi_T_end={mean_chiT:.3e}, chi_P_end={mean_chiP:.3e}, "
          f"corr(chi_T,avg_sigma)={corr_T:.2f}, corr(chi_P,avg_sigma)={corr_P:.2f}")
    print("=============================================")


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 sweep (N‑扩展 + 同终点压缩)")
    p.add_argument("--vels", type=float, nargs="+", default=[-0.005, -0.05, -0.2],
                   help="piston velocities (negative=inward)")
    p.add_argument("--dts", type=float, nargs="+", default=[0.1, 0.2, 0.5], help="initial Δt list")
    p.add_argument("--Ns", type=int, nargs="+", default=[9128], help="particle counts")
    p.add_argument("--full", action="store_true", help="3×3×3 sweep with Ns=[4096,9128,20000]")
    p.add_argument("--no-fig", action="store_true", help="disable per-run figure")
    p.add_argument("--no-adapt", action="store_true", help="disable adaptive impulse-window")
    p.add_argument("--out", type=str, default="phase1_out_sameV", help="output directory")
    p.add_argument("--warmup", type=float, default=0.5, help="warmup time")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--keep-density", action="store_true", default=True,
                   help="keep initial number density across N (default True)")
    p.add_argument("--no-keep-density", dest="keep_density", action="store_false")
    p.add_argument("--Vratio", type=float, default=0.5, help="V_end / V0 (0<ratio<1)")
    p.add_argument("--end-tol", type=float, default=5e-4, help="end volume relative tolerance")
    return p.parse_args()


def main():
    args = parse_args()

    Ns = args.Ns
    if args.full:
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
        t_final_soft=1e9,
        dt_init=0.2,
        min_Lx=3.0,
        gamma=5.0/3.0,
        kB=1.0,
        subtract_bulk_drift=True,
        adapt_window_by_events=not args.no_adapt,
        target_events=200,
        dt_min=0.02,
        dt_max=0.5,
        out_dir=args.out,
        save_fig=not args.no_fig,
        show_fig=False,
        keep_initial_density=True if args.keep_density else False,
        Vratio_end=args.Vratio,
        end_tolerance=args.end_tol,
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
