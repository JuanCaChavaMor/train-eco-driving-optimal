"""Quick regression tests (a few seconds):  python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_eco.checks import check                      # noqa: E402
from train_eco.indirect import LevelTrack               # noqa: E402
from train_eco.model import Route, Train                # noqa: E402
from train_eco.nlpr3 import solve_nlpr3                 # noqa: E402

SMALL = Route(np.r_[np.ones(5), np.full(20, 50.0), np.ones(5)],
              np.r_[np.zeros(15), np.full(5, -0.04), np.zeros(10)], np.full(30, 140 / 3.6), 120.0)


def _solve(solver):
    try:
        return solve_nlpr3(SMALL, solver=solver)
    except ImportError:
        pytest.skip(f"{solver} not installed")


@pytest.mark.parametrize("solver", ["scip", "gurobi"])
def test_nlpr3_is_physical(solver):
    """Global optimum on a small route: z = 1/v is tight and the dynamics hold."""
    c = check(_solve(solver), verbose=False)
    assert abs(c["energy_kWh"] - 7.0558) < 1e-3
    assert abs(c["true_time_s"] - SMALL.T) < 1e-4
    assert c["max_latency_gap"] < 1e-6 and c["max_dyn_residual"] < 1e-6


def test_brake_speed_is_tangent_intercept():
    """U = psi(V)/phi'(V) equals the root of eta(U) = -1 on the coast curve."""
    lt = LevelTrack(Train(), 15020.0)
    V = 30.0
    U = brentq(lambda u: lt.EV(u, V) - lt.EV(V, V) - lt.r(u), 0.1, V - 1e-9)
    assert abs(U - lt.brake_speed(V)) < 1e-8


def test_indirect_closes_distance_and_time():
    lt = LevelTrack(Train(), 15020.0)
    sol = lt.solve(600.0)
    assert abs(sol.T - 600.0) < 1e-6 and sol.x[-1] == 15020.0
    assert np.all(np.diff(sol.x) >= 0)
