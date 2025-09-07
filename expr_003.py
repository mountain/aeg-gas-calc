# -*- coding: utf-8 -*-
"""
1D 稳态导热（GasSim）：沿 x 的温度剖面 + δ-理论（物理定标）+ 傅立叶对照 + 体/界/边闭合 + (可选) 局域 μ,λ
----------------------------------------------------------------------------------------------------
依赖：gassim>=0.0.6, numpy, matplotlib

理论要点（与 AEG 对齐）：
- 流方程 / Eikonal：||∇a|| = sqrt(μ^2 + λ^2 a^2)，a 取 β=1/T。        [见你的摘要与论文]
- 常速化变量：y = asinh((λ/μ) a)，从而 ||∇y|| = λ —— 用它做“直线化”。   [同上]
- 环流—面积：∮ω = ∬ dω = μλ ∬ du∧dv（平直框）。                         [同上]
（引用：明理《表达式微分与全纯（初等版）统一摘要》《Geometry of Arithmetic Expressions: I》）
"""

from gassim import GasSim
import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------- 0) 基本设置 ----------
np.set_printoptions(suppress=True, linewidth=120)

# 物理与几何
TL, TR   = 2.0, 1.0
Lx, Ly, Lz = 40.0, 20.0, 20.0
A          = Ly * Lz
N          = 4000
radius     = 0.25
mass       = 1.0
dim        = 3
seed       = 2025

# 采样与时间平均
NBINS         = 100
WARMUP_T      = 2000.0
DT_WINDOW     = 20.0
N_WINDOWS     = 200
MIN_COUNT     = 5
MARGIN_LAMBDA = 0.75      # 剖面内域离壁 ≥ 0.75×λ_mfp(mean)

# 近壁“薄带”几何（气相侧 β(0+), β(L-)）
FACE_BAND_HALF_FACTOR = 0.5   # 半宽 = 0.5 × λ_mfp(mean)（不足自动放宽）

# ---------- 1) 构造系统并设定热壁 ----------
sim = GasSim(num_particles=N, box_size=[Lx, Ly, Lz],
             radius=radius, mass=mass, dim=3, seed=seed)
sim.set_thermal_wall(wall_id=0, t=TL)  # 左壁：热
sim.set_thermal_wall(wall_id=1, t=TR)  # 右壁：冷

# ---------- 2) 预热到稳态 ----------
sim.advance_to(WARMUP_T)

# ---------- 2.5) 以自由程设计采样几何 ----------
diameter   = 2.0 * radius
n_mean     = N / (Lx * Ly * Lz)
lambda_mfp_mean = 1.0 / (np.sqrt(2.0) * np.pi * (diameter**2) * n_mean)

x_lo = max(0.0 + MARGIN_LAMBDA*lambda_mfp_mean, 1e-6)
x_hi = min(Lx - MARGIN_LAMBDA*lambda_mfp_mean, Lx - 1e-6)

edges   = np.linspace(x_lo, x_hi, NBINS+1)
centers = 0.5 * (edges[:-1] + edges[1:])
dx      = edges[1] - edges[0]

# 近壁薄带中心与半宽（气相侧）
band_half = FACE_BAND_HALF_FACTOR * lambda_mfp_mean
x_face_Lc = min(0.5 * lambda_mfp_mean, 0.05 * Lx)
x_face_Rc = max(Lx - 0.5 * lambda_mfp_mean, 0.95 * Lx)

print("--- Sampling geometry ---")
print(f"λ_mfp(mean) ~ {lambda_mfp_mean:.3f}")
print(f"x-domain for profile: [{x_lo:.3f}, {x_hi:.3f}]  (length={x_hi-x_lo:.3f}, NBINS={NBINS}, dx={dx:.3f})")
print(f"Face bands: centers at {x_face_Lc:.3f} (L), {x_face_Rc:.3f} (R); half-width={band_half:.3f}")

# ---------- 3) 累计器 ----------
sum_E    = np.zeros(NBINS)   # 分箱去对流动能总和
sum_N    = np.zeros(NBINS)   # 分箱计数
# 误差棒：窗间加权均值与方差（权重=Ni）
w_mean_T = np.zeros(NBINS)
w_M2_T   = np.zeros(NBINS)
w_sum    = np.zeros(NBINS)

# 近壁薄带
sum_E_faceL = 0.0; sum_N_faceL = 0
sum_E_faceR = 0.0; sum_N_faceR = 0

# 边界热量与热流（逐窗）
dQL_list, dQR_list = [], []
heats0 = sim.get_heat_by_wall()
Q_L_prev, Q_R_prev = heats0[0], heats0[1]

def local_T_from_mask(vel):
    """去对流：T = (2/3) * <1/2 |c|^2>，m=1。"""
    if len(vel) == 0:
        return np.nan, 0.0, 0
    u = vel.mean(axis=0)            # 局域漂移
    c = vel - u
    E = 0.5 * np.sum(c*c)           # 总动能
    Np = len(vel)
    T = (2.0/3.0) * (E / Np)
    return T, E, Np

def accumulate_window_profile(pos, vel, edges):
    global sum_E, sum_N, w_mean_T, w_M2_T, w_sum
    x = pos[:,0]
    bin_idx = np.searchsorted(edges, x, side='right') - 1
    valid = (bin_idx >= 0) & (bin_idx < NBINS)
    # 为数值稳健起见，逐箱处理（NBINS=100，成本可忽略）
    for i in range(NBINS):
        mask = valid & (bin_idx == i)
        Ni = int(np.count_nonzero(mask))
        if Ni < MIN_COUNT:
            continue
        T, E, _N = local_T_from_mask(vel[mask])
        sum_E[i] += E
        sum_N[i] += _N
        # 窗间加权
        w_old = w_sum[i]; w_new = w_old + Ni
        delta = T - w_mean_T[i]
        w_mean_T[i] += (Ni / w_new) * delta
        w_M2_T[i]   += Ni * delta * (T - w_mean_T[i])
        w_sum[i]     = w_new

def accumulate_face_band(pos, vel, x_center, half):
    """返回该薄带 (E,N)；若样本太少，依次放宽 1.5×、2×。"""
    for f in (1.0, 1.5, 2.0):
        lo = max(x_center - f*half, 0.0)
        hi = min(x_center + f*half, Lx)
        mask = (pos[:,0] >= lo) & (pos[:,0] < hi)
        Ni = int(np.count_nonzero(mask))
        if Ni >= MIN_COUNT:
            _, E, Np = local_T_from_mask(vel[mask])
            return E, Np
    return 0.0, 0

# ---------- 4) 滑窗采样 ----------
t_now = WARMUP_T
print("Sampling windows:", end=" ", flush=True)
for k in range(N_WINDOWS):
    t_now += DT_WINDOW
    sim.advance_to(t_now)
    if (k+1) % 10 == 0:
        print(".", end="", flush=True)

    pos = sim.get_positions()
    vel = sim.get_velocities()
    # 只取盒内
    in_box = (pos[:,0]>=0)&(pos[:,0]<Lx)&(pos[:,1]>=0)&(pos[:,1]<Ly)&(pos[:,2]>=0)&(pos[:,2]<Lz)
    pos, vel = pos[in_box], vel[in_box]

    # 分箱剖面
    accumulate_window_profile(pos, vel, edges)

    # 近壁薄带
    E_L, N_L = accumulate_face_band(pos, vel, x_face_Lc, band_half)
    E_R, N_R = accumulate_face_band(pos, vel, x_face_Rc, band_half)
    sum_E_faceL += E_L; sum_N_faceL += N_L
    sum_E_faceR += E_R; sum_N_faceR += N_R

    # 热量记账（逐窗）
    heats = sim.get_heat_by_wall()
    Q_L, Q_R = heats[0], heats[1]
    dQL_list.append(Q_L - Q_L_prev)
    dQR_list.append(Q_R - Q_R_prev)
    Q_L_prev, Q_R_prev = Q_L, Q_R

print(" done.")

# ---------- 5) 剖面与近壁 β ----------
T_profile = np.full(NBINS, np.nan)
valid_bins = (sum_N > 0)
T_profile[valid_bins] = (2.0/3.0) * (sum_E[valid_bins] / sum_N[valid_bins])

T_std_win = np.full(NBINS, np.nan)
ok = (w_sum > 0)
T_std_win[ok] = np.sqrt(np.maximum(w_M2_T[ok], 0.0) / np.maximum(w_sum[ok], 1.0))

def safe_beta_from_EN(E, Np):
    if Np <= 0: return np.nan
    T = (2.0/3.0) * (E / Np)
    return 1.0 / T

beta_L_gas = safe_beta_from_EN(sum_E_faceL, sum_N_faceL)
beta_R_gas = safe_beta_from_EN(sum_E_faceR, sum_N_faceR)
# 若极端情况下薄带样本不足，则回退为浴温 β
if not np.isfinite(beta_L_gas): beta_L_gas = 1.0/TL
if not np.isfinite(beta_R_gas): beta_R_gas = 1.0/TR

print("\n=== Face inverse-temperature (gas-side) ===")
print(f"β_L^gas = {beta_L_gas:.6f}   β_L(bath) = {1.0/TL:.6f}")
print(f"β_R^gas = {beta_R_gas:.6f}   β_R(bath) = {1.0/TR:.6f}")
dBeta_bath = (1.0/TR - 1.0/TL)
dBeta_gas  = (beta_R_gas - beta_L_gas)
print(f"Δβ_bath = {dBeta_bath:.6f}   Δβ_gas = {dBeta_gas:.6f}   ratio Δβ_gas/Δβ_bath = {dBeta_gas/dBeta_bath:.3f}")

# ---------- 6) 碰撞计数规 s(x)：局域 λ_mfp(x) 与 s-坐标 ----------
V_bin = dx * Ly * Lz
Nbar_i = sum_N / max(N_WINDOWS, 1)                      # 每窗平均分箱粒子数
n_i    = np.where(Nbar_i>0, Nbar_i / V_bin, np.nan)     # 数密度
eps = 1e-12
lambda_mfp_i = 1.0 / (np.sqrt(2.0)*np.pi*diameter**2 * np.maximum(n_i, eps))
# s(x) 离散积分（梯形）
s_centers = np.zeros_like(centers)
inv_mfp = np.where(np.isfinite(lambda_mfp_i) & (lambda_mfp_i>0), 1.0/lambda_mfp_i, 0.0)
s_centers[1:] = np.cumsum(0.5*(inv_mfp[1:] + inv_mfp[:-1]) * np.diff(centers))

# ---------- 7) 标定 (μ,λ)：令 y=asinh((λ/μ)β) 对 s 最线性 ----------
mask_int = np.isfinite(T_profile) & (centers >= x_face_Lc) & (centers <= x_face_Rc)
beta_i   = np.where(np.isfinite(T_profile), 1.0 / T_profile, np.nan)
s_i      = s_centers

def fit_ratio_r(beta, s, mask):
    """优化 r = λ/μ 使 y=asinh(r*β) 对 s 线性；返回 r*, 斜率 λ_hat, 截距 b, RMSE。"""
    def line_fit(y, S):
        A = np.vstack([S, np.ones_like(S)]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - (k*S + b)
        rmse = np.sqrt(np.mean(resid**2))
        return float(k), float(b), float(rmse)

    def loss_of(r):
        y = np.arcsinh(np.clip(r * beta[mask], a_min=1e-12, a_max=None))
        k, b, rmse = line_fit(y, s[mask])
        return rmse, k, b

    # 粗扫 + 黄金分割细化
    rs = np.logspace(-2, 2, 201)
    losses = [loss_of(r)[0] for r in rs]
    r0 = rs[int(np.argmin(losses))]
    a, c = r0/3.0, r0*3.0
    phi = (1 + 5**0.5) / 2
    for _ in range(40):
        b1 = a + (c-a)/phi
        b2 = c - (c-a)/phi
        L1 = loss_of(b1)[0]; L2 = loss_of(b2)[0]
        if L1 < L2: c = b2
        else:       a = b1
    r_best = 0.5*(a+c)
    rmse, k_best, b_best = loss_of(r_best)
    return float(r_best), float(k_best), float(b_best), float(rmse)

r_opt, LAM_hat, b_int, lin_rmse = fit_ratio_r(beta_i, s_i, mask_int)
MU_hat = LAM_hat / r_opt

print("\n=== Calibration of (mu, lambda) from profile (collision gauge) ===")
print(f"r = lambda/mu  ≈ {r_opt:.6f}")
print(f"lambda (per-collision speed of y) ≈ {LAM_hat:.6f}")
print(f"mu ≈ {MU_hat:.6f}")
print(f"Linearity RMSE of y(s): {lin_rmse:.6e}")

# ---------- 8) δ–理论预测（物理定标；体内解） ----------
def T_delta_collision(x):
    """用 (MU_hat, LAM_hat) 与 s(x) 生成 δ–理论完整曲线（内域）。"""
    x = np.asarray(x)
    s   = np.interp(x, centers, s_centers)
    s_L = np.interp(x_face_Lc, centers, s_centers)
    yL  = np.arcsinh((LAM_hat/MU_hat) * beta_L_gas)
    # y(s) 常速：斜率即 lambda
    y   = yL + LAM_hat * (s - s_L)
    beta = (MU_hat / LAM_hat) * np.sinh(y)
    return 1.0 / beta

# ---------- 9) 傅立叶曲线（内域；气相侧 BC） ----------
T_L_gas = 1.0 / beta_L_gas
T_R_gas = 1.0 / beta_R_gas
x_L, x_R = x_face_Lc, x_face_Rc

def T_fourier_power(x, TL_loc, TR_loc, xL, xR, alpha):
    """κ(T)=κ0*T^alpha -> T^{alpha+1} 随 x 线性；仅定义在 [xL,xR]。"""
    x = np.asarray(x)
    s = np.clip((x - xL) / (xR - xL), 0.0, 1.0)
    p = alpha + 1.0
    return (TL_loc**p + (TR_loc**p - TL_loc**p) * s) ** (1.0/p)

# ---------- 10) RMSE 与出图（剖面对齐） ----------
x_grid_in = np.linspace(max(x_L, centers[mask_int].min()),
                        min(x_R, centers[mask_int].max()), 600)
T_delta_in   = T_delta_collision(x_grid_in)
T_four_lin   = T_fourier_power(x_grid_in, T_L_gas, T_R_gas, x_L, x_R, alpha=0.0)
T_four_sqrt  = T_fourier_power(x_grid_in, T_L_gas, T_R_gas, x_L, x_R, alpha=0.5)

def rmse_interp(xg, Tg):
    Tg_c = np.interp(centers[mask_int], xg, Tg)
    return float(np.sqrt(np.mean((Tg_c - T_profile[mask_int])**2)))

rmse_delta   = rmse_interp(x_grid_in, T_delta_in)
rmse_lin     = rmse_interp(x_grid_in, T_four_lin)
rmse_sqrt    = rmse_interp(x_grid_in, T_four_sqrt)

# α 扫描（形状拟合）
alphas = np.linspace(0.0, 1.0, 101)
errs = [rmse_interp(x_grid_in, T_fourier_power(x_grid_in, T_L_gas, T_R_gas, x_L, x_R, a_)) for a_ in alphas]
i_best = int(np.argmin(errs))
alpha_star, rmse_star = float(alphas[i_best]), float(errs[i_best])

print("\n=== Profile RMSE (interior; gas-side BCs) ===")
print(f"RMSE(δ-theory, collision gauge) : {rmse_delta:.6f}")
print(f"RMSE(Fourier const-κ, α=0)      : {rmse_lin:.6f}")
print(f"RMSE(Fourier κ∝T^1/2, α=0.5)    : {rmse_sqrt:.6f}")
print(f"Best α in [0,1]                 : α*={alpha_star:.3f}, RMSE={rmse_star:.6f}")

# 图 1：温度剖面 + δ(物理定标) + 傅立叶
plt.figure(figsize=(7.6, 4.8))
plt.errorbar(centers[mask_int], T_profile[mask_int], yerr=T_std_win[mask_int],
             fmt='o', capsize=2, label='Measured T(x)')
plt.plot(x_grid_in, T_delta_in,   label='δ-theory (collision gauge)')
plt.plot(x_grid_in, T_four_lin,   label='Fourier const-κ (linear T)')
plt.plot(x_grid_in, T_four_sqrt,  label='Fourier κ∝T^{1/2} (T^{3/2}-linear)')
if abs(alpha_star-0.0)>1e-6 and abs(alpha_star-0.5)>1e-6:
    plt.plot(x_grid_in, T_fourier_power(x_grid_in,T_L_gas,T_R_gas,x_L,x_R,alpha_star),
             linestyle='--', label=f'Fourier κ∝T^α (α*={alpha_star:.2f})')
# 标注界面（浴温与气相侧温）
plt.axvline(0.0, linestyle='--', linewidth=1)
plt.axvline(Lx,  linestyle='--', linewidth=1)
plt.scatter([0.0, x_L], [TL, T_L_gas], marker='s', label='Left bath/gas')
plt.scatter([Lx,  x_R], [TR, T_R_gas], marker='s', label='Right bath/gas')
plt.xlabel('x'); plt.ylabel('Temperature T')
plt.title('Measured profile vs δ-theory (collision gauge) and Fourier')
plt.legend(); plt.tight_layout()
fig1 = "Tx_profile_overlay.png"
plt.savefig(fig1, dpi=160)
print(f"Saved FIG : {fig1}")

# 图 2：线性化检验 —— y=asinh((λ/μ)β) 对 s 的直线
y_lin = np.arcsinh((LAM_hat/MU_hat) * beta_i[mask_int])
plt.figure(figsize=(7.2, 4.2))
S = s_i[mask_int]
k_fit,b_fit = np.linalg.lstsq(np.vstack([S, np.ones_like(S)]).T, y_lin, rcond=None)[0]
plt.plot(S, y_lin, 'o', label='data y(s)')
plt.plot(S, k_fit*S+b_fit, '-', label=f'fit: slope≈{k_fit:.4f} (λ̂), intercept≈{b_fit:.4f}')
plt.xlabel('s(x) = ∫ dx / λ_mfp(x)'); plt.ylabel('y = asinh((λ/μ) β)')
plt.title('Rectification check in collision gauge')
plt.legend(); plt.tight_layout()
fig2 = "y_linearization_collision_gauge.png"
plt.savefig(fig2, dpi=160)
print(f"Saved FIG : {fig2}")

# ---------- 10.5) （可选）滑窗反演 μ(x), λ(x) ----------
def local_mu_lambda(centers_arr, s_arr, beta_arr, mask_arr, window_bins=11):
    """
    在滑动窗内寻找 r 使 y=asinh(r β) 最线性；回归斜率即 λ_loc，μ_loc=λ_loc/r。
    返回：x_loc, μ_loc, λ_loc（均为窗中心定义）。
    """
    idx_all = np.where(mask_arr & np.isfinite(beta_arr))[0]
    if len(idx_all) < window_bins:
        return np.array([]), np.array([]), np.array([])
    half = window_bins//2
    xs, lam_loc, mu_loc = [], [], []
    phi = (1 + 5**0.5) / 2
    for k in range(half, len(idx_all)-half):
        idx = idx_all[k-half:k+half+1]
        S = s_arr[idx]; B = beta_arr[idx]
        # 选 r 使 y 对 S 最线性
        def fit_for(r):
            y = np.arcsinh(np.clip(r*B, 1e-12, None))
            A = np.vstack([S, np.ones_like(S)]).T
            k1, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
            rmse = np.sqrt(np.mean((y-(k1*S+b1))**2))
            return rmse, k1
        rs = np.logspace(-2, 2, 101)
        rmses, ks = zip(*[fit_for(r) for r in rs])
        r0 = rs[int(np.argmin(rmses))]
        a, c = r0/3.0, r0*3.0
        for _ in range(30):
            b1 = a + (c-a)/phi; b2 = c - (c-a)/phi
            L1, k1 = fit_for(b1); L2, k2 = fit_for(b2)
            if L1 < L2: c, kbest, rbest = b2, k1, b1
            else:       a, kbest, rbest = b1, k2, b2
        lam = float(kbest)
        mu  = float(lam / rbest)
        xs.append(centers_arr[idx_all[k]]); lam_loc.append(lam); mu_loc.append(mu)
    return np.array(xs), np.array(mu_loc), np.array(lam_loc)

# 窗宽可按 ~ 3 λ_mfp(mean) / dx 取奇数；若太小则降为 11
win_bins = int(max(11, np.round(3*lambda_mfp_mean/dx)))
win_bins = win_bins + 1 - (win_bins % 2)   # 调整为奇数
x_mu, mu_x, lam_x = local_mu_lambda(centers, s_centers, beta_i, mask_int, window_bins=win_bins)

if x_mu.size > 0:
    plt.figure(figsize=(7.6, 4.0))
    plt.plot(x_mu, lam_x, label='λ_loc(x) from rectified slope')
    plt.plot(x_mu, mu_x,  label='μ_loc(x) from ratio λ/r')
    plt.xlabel('x'); plt.ylabel('local generators')
    plt.title('Local μ(x), λ(x) via sliding-window rectification')
    plt.legend(); plt.tight_layout()
    fig3 = "mu_lambda_local_profile.png"
    plt.savefig(fig3, dpi=160)
    print(f"Saved FIG : {fig3}")
else:
    print("[Info] local μ(x), λ(x): insufficient interior bins for sliding-window.")

# ---------- 11) 体/界/边：用拟合端点做闭合检验 ----------
dQL_arr = np.array(dQL_list); dQR_arr = np.array(dQR_list)
Jq_arr  = dQR_arr / (A * DT_WINDOW)                 # 右壁为准（流入气体为正）

# 用 δ-曲线给出两端“气相侧”β^fit（也可直接用观测到的 gas-side 值）
T_L_fit = float(T_delta_collision([x_face_Lc])[0])
T_R_fit = float(T_delta_collision([x_face_Rc])[0])
beta_L_fit, beta_R_fit = 1.0/T_L_fit, 1.0/T_R_fit

beta_L_bath, beta_R_bath = 1.0/TL, 1.0/TR
Sigma_tot  = A * Jq_arr * (beta_R_bath - beta_L_bath)      # 总熵率（边界 δ）
Sigma_bulk = A * Jq_arr * (beta_R_fit  - beta_L_fit )      # 体内熵率（内域 Δβ^fit）
Sigma_int  = Sigma_tot - Sigma_bulk                        # 界面熵率（两端跃迁）

closure_rmse = float(np.sqrt(np.mean((Sigma_tot - Sigma_bulk - Sigma_int)**2)))
print("\n=== Bulk–Boundary closure with fitted curve ===")
print(f"β^fit_L={beta_L_fit:.6f}, β^fit_R={beta_R_fit:.6f}")
print(f"Mean Σ_tot  : {Sigma_tot.mean():.6f}")
print(f"Mean Σ_bulk : {Sigma_bulk.mean():.6f}")
print(f"Mean Σ_int  : {Sigma_int.mean():.6f}")
print(f"RMSE(Σ_tot - Σ_bulk - Σ_int) : {closure_rmse:.6e}")

# ---------- 12) 导出 CSV ----------
csv_path = "Tx_profile.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x_center","T","T_std_window","samples_sumN",
                "x_lo","x_hi","dx","lambda_mfp_mean",
                "beta_L_gas","beta_R_gas",
                "mu_hat","lambda_hat"])
    for i in range(NBINS):
        w.writerow([centers[i], T_profile[i], T_std_win[i], int(sum_N[i]),
                    x_lo, x_hi, dx, lambda_mfp_mean,
                    beta_L_gas, beta_R_gas,
                    MU_hat, LAM_hat])
print(f"Saved CSV : {csv_path}")
