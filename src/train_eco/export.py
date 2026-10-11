"""Save and load discrete solutions as CSV (nodes and segments) plus solver metadata (JSON)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Train


def save_solution(sol, stem):
    """Write <stem>_nodes.csv (s, v, t) and <stem>_segments.csv (s_end, ds, alpha, F, z, P)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    r = sol.route
    pd.DataFrame({"s_m": r.s, "v_mps": sol.v, "t_s": sol.t}).to_csv(f"{stem}_nodes.csv", index=False)
    pd.DataFrame({"s_end_m": r.s[1:], "ds_m": r.ds, "alpha": r.alpha, "vmax_mps": r.vmax,
                  "F_N": sol.F, "z_spm": sol.z, "P_W": sol.F * sol.v[1:]}).to_csv(f"{stem}_segments.csv", index=False)
    meta = dict(route=r.name, N=r.N, T=r.T, solver=sol.solver, status=sol.status, gap=float(sol.gap),
                runtime_s=round(sol.runtime, 1), energy_kWh=sol.energy_kwh)
    Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))


def load_meta(stem):
    return json.loads(Path(f"{stem}_meta.json").read_text())


def load_profile(stem):
    """Return (s, v, F_on_segments) from saved CSVs."""
    n = pd.read_csv(f"{stem}_nodes.csv")
    sg = pd.read_csv(f"{stem}_segments.csv")
    return n.s_m.to_numpy(), n.v_mps.to_numpy(), sg.F_N.to_numpy()


def classify_modes(v_end, F, train: Train = Train(), tol=1e-3):
    """Label each segment as MaxPower / Hold / Coast / MaxBrake from (v, F)."""
    Fp, Fb = train.max_traction(v_end), train.max_braking(v_end)
    lab = np.where(F >= Fp * (1 - tol), "MaxPower",
          np.where(F <= Fb * (1 - tol), "MaxBrake",
          np.where(np.abs(F) <= tol * train.Fmax, "Coast", np.where(F > 0, "HoldP", "Brake (partial)"))))
    return lab
