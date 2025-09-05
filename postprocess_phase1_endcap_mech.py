# postprocess_phase1_endcap_mech.py
import pandas as pd, numpy as np, math, os, sys
gamma = 5/3; rho = 0.03

sum_path = "phase1_out_sameV/phase1_sweep_summary.csv"
df = pd.read_csv(sum_path)
rows = []
for _,r in df.iterrows():
    ts = pd.read_csv(r["ts_path"])
    V, W, t, P = ts["V"].values, ts["W_cum"].values, ts["time"].values, ts["P_impulse"].values
    V0, Vend = r["V0"], r["V_end"]
    Vcut = Vend + rho*(V0-Vend)
    i0 = np.argmax(V <= Vcut); i1 = len(V)-1
    Pmech = -(W[i1]-W[i0])/(V[i1]-V[i0])
    Pad   = r["P0_IG"] * (r["V0"]/Vend)**gamma
    chiP_mech = Pmech/Pad - 1.0
    rows.append(dict(tag=r["tag"], chi_P_end_mech=chiP_mech))
out = pd.DataFrame(rows)
out.to_csv("phase1_out_sameV/post_mech_pressure.csv", index=False)
print(out)
