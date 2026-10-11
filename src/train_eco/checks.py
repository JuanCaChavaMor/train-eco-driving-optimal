"""Physical consistency checks of a discrete trajectory (independent of the solver)."""
import numpy as np

from .model import G


def check(sol, verbose=True):
    """Residuals of the discrete model, evaluated from (v, F) only.

    The trip time is recomputed as sum ds_k / v_k, i.e. without the latency variables z: if the
    relaxation z >= 1/v were not tight, this time would differ from the booked time t_N.
    """
    r, tr = sol.route, sol.train
    v, F = sol.v, sol.F
    vk = v[1:]
    acc = (F - tr.resistance(vk) - tr.m * G * np.sin(r.alpha)) / tr.m
    dyn = (vk ** 2 - v[:-1] ** 2) / 2 - r.ds * acc            # (18b) with E = v^2/2
    T_true = float(np.sum(r.ds / vk))
    power = F * vk
    out = {
        "energy_kWh": sol.energy_kwh,
        "booked_time_s": float(sol.t[-1]),
        "true_time_s": T_true,
        "max_latency_gap": float(np.max(sol.z - 1 / vk)),     # z - 1/v  (>= 0, should be ~0)
        "min_v_times_z": float(np.min(vk * sol.z)),
        "max_dyn_residual": float(np.max(np.abs(dyn))),       # J/kg
        "max_kin_residual": float(np.max(np.abs(sol.E - v ** 2 / 2))),
        "max_power_kW": float(power.max() / 1e3),
        "min_power_kW": float(power.min() / 1e3),
        "max_overspeed": float(np.max(vk - r.vmax)),
    }
    if verbose:
        for k, x in out.items():
            print(f"  {k:18s} {x: .6g}")
    return out
