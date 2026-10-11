"""Why the latency constraint z >= 1/v must not be replaced by a McCormick envelope.

The envelope of y = v z on the box [eps, vmax] x [1/vmax, 1/eps] is an OUTER approximation, so
"y >= 1" no longer implies z >= 1/v. The solver then books less time than the train needs: the trip
seems to take T seconds, but driving the computed speed profile takes longer, or the profile only
fits T because time is booked where the speed is low.

    python scripts/pitfall_mccormick.py --solver scip
"""
import argparse

import pandas as pd

from _path import ROOT
from train_eco.checks import check
from train_eco.model import example1
from train_eco.nlpr3 import solve_nlpr3

ap = argparse.ArgumentParser()
ap.add_argument("--solver", default="gurobi", choices=["gurobi", "scip"])
a = ap.parse_args()

rows = []
for lat in ("exact", "mccormick"):
    sol = solve_nlpr3(example1(), solver=a.solver, latency=lat, time_limit=3600, gap=1e-3)
    print(f"\nlatency = {lat}: {sol.status}, {sol.runtime:.0f} s")
    rows.append(dict(latency=lat, **check(sol)))
pd.DataFrame(rows).to_csv(ROOT / "results" / "pitfall_mccormick.csv", index=False)
