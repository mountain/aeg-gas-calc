# -*- coding: utf-8 -*-
"""
熵产生的 δ–环流—面积验证（两热浴稳态导热）
------------------------------------------------
- 设计：两端恒温热壁（TL > TR），盒内硬球气体进入非平衡稳态（恒定热流）。
- 目标：用 δ–环流 S_delta = β_L dQ_L + β_R dQ_R 计算“每窗熵增”，
        并验证逐窗恒等式：
          S_delta - [(β_R-β_L)/2]*(dQ_R - dQ_L) = β_avg*(dQ_L + dQ_R)
        从而得到 S_delta_corr ≡ S_classic_sym。
- 物理记号：流入“气体”为正号；k_B = 1；β = 1/T。
- 依赖：gassim >= 0.0.6；numpy。
"""

from gassim import GasSim
import numpy as np

# ---------- 0) 基本设置 ----------
np.set_printoptions(suppress=True, linewidth=120)  # 让数组打印更易读

# 热浴温度与逆温（物理输入量）
TL, TR = 2.0, 1.0
beta_L, beta_R = 1.0/TL, 1.0/TR
beta_avg = 0.5*(beta_L + beta_R)   # 暂存项的“平均逆温”权重

# 几何尺寸与时间窗口（数值输入量）
Lx, Ly, Lz = 40.0, 20.0, 20.0
A = Ly * Lz            # 左/右壁面积，用于把“热量”换算成“热流×时间”
dt = 100.0             # 采样窗口长度
windows = 50           # 窗口数
warmup_t = 2000.0      # 预热到稳态的时间

# ---------- 1) 构造系统 ----------
# 说明：硬球 EDMD；dim=3；半径/质量在内部单位下设置。
sim = GasSim(num_particles=4000, box_size=[Lx, Ly, Lz],
             radius=0.25, mass=1.0, dim=3, seed=2025)

# 两端热壁：左壁 id=0；右壁 id=1（沿用你的 API：参数名 t）
sim.set_thermal_wall(wall_id=0, t=TL)   # 左：热浴（高温）
sim.set_thermal_wall(wall_id=1, t=TR)   # 右：冷浴（低温）
# 其余四面默认弹性反射（绝热），不参与能量交换

# ---------- 2) 预热到稳态 ----------
# 目的：让体系进入非平衡稳态（NES S），此后热流/能量统计更平稳
sim.advance_to(warmup_t)

# ---------- 3) 滑窗采样 ----------
# 将所有待比较的“每窗积分量”存为列表
S_delta_list       = []    # δ–环流：S_delta = β_L*dQ_L + β_R*dQ_R
S_classic_R_list   = []    # 经典（单用右壁）：(β_R - β_L)*dQ_R
S_classic_sym_list = []    # 经典（双壁对称）：(β_R - β_L)*(dQ_R - dQ_L)/2
S_delta_corr_list  = []    # 校正后 δ–环流：S_delta - β_avg*(dQ_L + dQ_R)
dE_list            = []    # 暂存能量：dE = dQ_L + dQ_R
ratios_raw         = []    # δ / 经典(右壁)：平均应接近 1（长时无偏）
ratios_corr        = []    # δ(校正) / 经典(对称)：逐窗应等于 1

# 累积热量的初值（注意 get_heat_by_wall() 返回“从 t=0 起累计到当前”的热量）
Q_by_wall = sim.get_heat_by_wall()
Q_L_prev, Q_R_prev = Q_by_wall[0], Q_by_wall[1]

t_now = warmup_t
print("Sampling windows:", end=" ", flush=True)

for k in range(windows):
    # 前进到下一个窗口末端时刻
    t_now += dt
    sim.advance_to(t_now)
    print(".", end="", flush=True)

    # 读取两壁的“到当前时刻为止”的累计热
    Q_by_wall = sim.get_heat_by_wall()
    Q_L, Q_R = Q_by_wall[0], Q_by_wall[1]

    # 形成“本窗口内”的热量交换：流入“气体”为正
    dQL = Q_L - Q_L_prev
    dQR = Q_R - Q_R_prev
    Q_L_prev, Q_R_prev = Q_L, Q_R

    # 3.1 δ–环流（边界线积分：∮ β dQ）
    S_delta = beta_L * dQL + beta_R * dQR

    # 3.2 经典表达（两种写法）：
    #     - 仅用右壁（与很多教材里的边界化公式一致）
    S_classic_R = (beta_R - beta_L) * dQR
    #     - 双壁对称（热流的对称估计，降低短窗噪声）
    S_classic_sym = (beta_R - beta_L) * 0.5 * (dQR - dQL)

    # 3.3 暂存项与校正：dE 表示窗口内系统能量的净变化
    dE = dQL + dQR
    S_delta_corr = S_delta - beta_avg * dE  # 理论：应与 S_classic_sym 逐窗相等

    # 3.4 记录
    S_delta_list.append(S_delta)
    S_classic_R_list.append(S_classic_R)
    S_classic_sym_list.append(S_classic_sym)
    S_delta_corr_list.append(S_delta_corr)
    dE_list.append(dE)

    # 用于比值与收敛性检查（避免除零）
    if abs(S_classic_R)  > 1e-12: ratios_raw.append(S_delta      / S_classic_R)
    if abs(S_classic_sym)> 1e-12: ratios_corr.append(S_delta_corr/ S_classic_sym)

print(" done.")

# ---------- 4) 转为数组 ----------
S_delta_arr        = np.array(S_delta_list)
S_classic_R_arr    = np.array(S_classic_R_list)
S_classic_sym_arr  = np.array(S_classic_sym_list)
S_delta_corr_arr   = np.array(S_delta_corr_list)
dE_arr             = np.array(dE_list)

# ---------- 5) 统计与误差学 ----------
def stats(name, x):
    print(f"{name:16s} min={x.min(): .6f}  max={x.max(): .6f}  mean={x.mean(): .6f}  std={x.std(): .6f}")

print("\n=== Window statistics (per-window integrals; sign: heat into gas > 0) ===")
stats("S_delta",        S_delta_arr)
stats("classic_right",  S_classic_R_arr)
stats("classic_sym",    S_classic_sym_arr)
stats("S_delta_corr",   S_delta_corr_arr)
stats("dE_sys",         dE_arr)

rmse_raw  = np.sqrt(np.mean((S_delta_arr      - S_classic_R_arr  )**2))   # δ vs 单壁经典
rmse_sym  = np.sqrt(np.mean((S_delta_arr      - S_classic_sym_arr)**2))   # δ vs 对称经典（未校正）
rmse_corr = np.sqrt(np.mean((S_delta_corr_arr - S_classic_sym_arr)**2))   # δ(校正) vs 对称经典

print("\n=== Consistency checks ===")
print("ratio (delta/classic_right)        :", np.mean(ratios_raw)  if ratios_raw  else np.nan)
print("ratio (delta_corr/classic_sym)     :", np.mean(ratios_corr) if ratios_corr else np.nan)
print("RMSE(delta vs classic_right)       :", rmse_raw)
print("RMSE(delta vs classic_sym)         :", rmse_sym)
print("RMSE(delta_corr vs classic_sym)    :", rmse_corr)

# ---------- 6) 恒等式的数值检验：残差与暂存项的线性关系 ----------
# 理论恒等：S_delta - S_classic_sym = beta_avg * dE_sys  （逐窗恒等）
residual = S_delta_arr - S_classic_sym_arr
X = beta_avg * dE_arr

if np.allclose(X.var(), 0.0):
    print("\n[Warn] dE variance ~ 0，暂存项几乎为零，线性拟合跳过。")
else:
    a, b = np.polyfit(X, residual, 1)  # 拟合 residual = a*X + b
    yhat = a*X + b
    ss_res = np.sum((residual - yhat)**2)
    ss_tot = np.sum((residual - residual.mean())**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0

    print("\n=== Storage-term identity check ===")
    print("Expected slope a ≈ 1, intercept b ≈ 0")
    print(f"Fit: residual ≈ a*(beta_avg*dE) + b  ->  a={a:.6f}, b={b:.6e},  R^2={R2:.6f}")
    print(f"Max |residual - beta_avg*dE|       : {np.max(np.abs(residual - X)):.6e}")

# ---------- 7) 便捷的宏观量（熵产生率与平均热流密度） ----------
entropy_rate = - S_delta_corr_arr.mean() / dt                    # 正值：单位时间全系统熵产生率
Jq_mean      =   S_classic_sym_arr.mean() / ((beta_R-beta_L)*A*dt)  # 平均热流密度（方向：左→右为负）
print("\n=== Macroscopic observables ===")
print("Mean entropy production rate  :", entropy_rate)
print("Mean heat flux density J_q    :", Jq_mean)
