"""Direct vs indirect method on level track (Example 1 without the downhill section, T = 600 s).

The indirect solution is continuous in x; NLP_R3 is solved on meshes of 30, 15 and 7.5 m.

Time bookkeeping: NLP_R3 charges t_k - t_{k-1} = ds_k / v_k (speed at the END of the segment). On the last
segment v_N = eps = 0.1 m/s, so the last metre costs 10 s, while braking through it takes about 1.6 s. The
discrete trip therefore has ~9 s less physical driving time than T. For a like-for-like comparison, the
physical trip time of each NLP solution is recomputed with the trapezoidal rule, T_phys = sum 2 ds/(v_{k-1}+v_k),
and the indirect method is solved for that T_phys.

    python scripts/compare_indirect.py --solver scip     # three NLPs, ~40 min with SCIP
    python scripts/compare_indirect.py --reuse           # only the indirect part, NLPs from results/
"""
import argparse

import numpy as np
import pandas as pd

from _path import ROOT
from train_eco.export import load_meta, load_profile, save_solution
from train_eco.indirect import LevelTrack
from train_eco.model import Train, level_route
from train_eco.nlpr3 import solve_nlpr3
from train_eco.plotting import plot_comparison, plot_energy_time, plot_phase_portrait, plot_tangent

ap = argparse.ArgumentParser()
ap.add_argument("--solver", default="gurobi", choices=["gurobi", "scip"])
ap.add_argument("--T", type=float, default=600.0)
ap.add_argument("--time-limit", type=float, default=3600)
ap.add_argument("--reuse", action="store_true", help="read the NLP solutions from results/ instead of solving")
a = ap.parse_args()
res = ROOT / "results"

# --- indirect method (continuous) -------------------------------------------------------------
lt = LevelTrack(Train(), 15020.0)
ind = lt.solve(a.T)
print(f"indirect: V = {ind.V * 3.6:.3f} km/h, brake speed U = {ind.U * 3.6:.3f} km/h, "
      f"switches at {np.round(ind.x[1:4], 1)} m, energy {ind.energy_kwh:.4f} kWh")

# --- direct method on refined meshes ---------------------------------------------------------
rows, profiles = [], []
for refine in (1, 2, 4):
    route = level_route(a.T, refine)
    stem = res / f"level_T{a.T:.0f}_N{route.N}"
    if a.reuse:
        s, v, F = load_profile(stem)
        meta = load_meta(stem)
        E, gap = meta["energy_kWh"], meta["gap"]
    else:
        sol = solve_nlpr3(route, solver=a.solver, time_limit=a.time_limit)
        save_solution(sol, stem)
        s, v, E, gap = route.s, sol.v, sol.energy_kwh, sol.gap
        print(f"  N = {route.N}: {sol.status}, {sol.runtime:.0f} s, gap {100 * sol.gap:.3f} %")
    T_phys = float(np.sum(2 * route.ds / (v[1:] + v[:-1])))
    E_ind = lt.solve(T_phys).energy_kwh
    rows.append(dict(N=route.N, ds_m=30 / refine, nlp_kWh=E, nlp_gap_pct=100 * gap, T_phys_s=T_phys,
                     indirect_at_T_phys_kWh=E_ind, diff_pct=100 * (E / E_ind - 1)))
    profiles.append((f"NLP_R3, ds = {30 / refine:g} m", s, v))

tab = pd.DataFrame(rows)
print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
tab.to_csv(res / "direct_vs_indirect.csv", index=False)

T_ref = rows[-1]["T_phys_s"]
plot_comparison(profiles, lt.profile(lt.solve(T_ref)), ROOT / "figures" / "direct_vs_indirect.png",
                f"Level track, 15.02 km, T = {a.T:.0f} s booked (indirect solved for T_phys = {T_ref:.1f} s)")
plot_phase_portrait(*ind.eta_profile, ind.U, ind.V, ROOT / "figures" / "phase_portrait.png")
plot_tangent(lt, ind.V, ROOT / "figures" / "brake_speed_tangent.png")

# --- energy-time trade-off (indirect method: milliseconds per journey) ----------------------
Ts = np.linspace(560, 1200, 60)
E = [lt.solve(T).energy_kwh for T in Ts]
pd.DataFrame({"T_s": Ts, "energy_kWh": E}).to_csv(res / "level_energy_vs_time.csv", index=False)
plot_energy_time(Ts, E, ROOT / "figures" / "energy_vs_time.png",
                 marks=[(a.T, ind.energy_kwh, f"T = {a.T:.0f} s")])
