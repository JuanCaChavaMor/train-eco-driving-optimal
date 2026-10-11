"""Example 1 of Xiao et al. (2025): global optimum of NLP_R3 (15.02 km, N = 520, T = 540 s).

    python scripts/solve_example1.py                  # Gurobi (needs a full license)
    python scripts/solve_example1.py --solver scip    # open source, ~8 min
"""
import argparse

from _path import ROOT
from train_eco.checks import check
from train_eco.export import save_solution
from train_eco.model import example1
from train_eco.nlpr3 import solve_nlpr3
from train_eco.plotting import plot_solution

ap = argparse.ArgumentParser()
ap.add_argument("--solver", default="gurobi", choices=["gurobi", "scip"])
ap.add_argument("--T", type=float, default=540.0)
ap.add_argument("--time-limit", type=float, default=3600)
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()

sol = solve_nlpr3(example1(a.T), solver=a.solver, time_limit=a.time_limit, verbose=a.verbose)
print(f"{a.solver}: {sol.status}, {sol.runtime:.1f} s, gap {100 * sol.gap:.4f} %")
check(sol)
save_solution(sol, ROOT / "results" / "example1")
plot_solution(sol, ROOT / "figures" / "example1.png",
              f"Example 1, global optimum: {sol.energy_kwh:.2f} kWh in {sol.t[-1]:.0f} s")
