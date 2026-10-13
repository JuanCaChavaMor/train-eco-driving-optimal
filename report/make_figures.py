"""Re-create every figure of the report (PNG + PDF in figures/) from the stored results; no solver needed.

    python report/make_figures.py
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_eco.model import example1                                  # noqa: E402
from train_eco.plotting import plot_mccormick_latency, plot_solution  # noqa: E402

FIG = ROOT / "figures"

# Example 1 from the stored global optimum
n = pd.read_csv(ROOT / "results" / "example1_nodes.csv")
g = pd.read_csv(ROOT / "results" / "example1_segments.csv")
sol = SimpleNamespace(route=example1(), v=n.v_mps.to_numpy(), t=n.t_s.to_numpy(), F=g.F_N.to_numpy())
E = float((g.F_N.clip(lower=0) * g.ds_m).sum()) / 3.6e6
plot_solution(sol, FIG / "example1.png", f"Example 1, global optimum: {E:.2f} kWh in {sol.t[-1]:.0f} s")

# latency relaxation vs. McCormick envelope (Example 1 bounds: 0.1 m/s <= v <= 140 km/h)
plot_mccormick_latency(0.1, 140 / 3.6, FIG / "mccormick_latency.png")

# direct vs. indirect, phase portrait, tangent construction, energy-time trade-off
subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_indirect.py"), "--reuse"], check=True)
