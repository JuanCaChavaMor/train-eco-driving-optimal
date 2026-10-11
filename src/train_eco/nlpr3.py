"""Direct method: the NLP_R3 model of Xiao, Ye & Chung (2025), solved to global optimality.

Discretisation (implicit Euler in the position s, all quantities of segment k at its end node s_k):

    min   sum_k F+_k ds_k
    s.t.  (E_k - E_{k-1}) / ds_k = (F_k - 2 c2 E_k - c1 v_k - c0 - m g sin a_k) / m     (18b)
          (t_k - t_{k-1}) / ds_k = z_k                                               (18c)
          z_k >= 1 / v_k            convex: written as  1 <= v_k z_k  (rotated cone)  (18d)
          E_k  = v_k^2 / 2          bilinear equality                                 (18e)
          Fmin <= F_k <= Fmax,  Pmin <= F_k v_k <= Pmax  (bilinear)                   (18f-g)
          eps <= v_k <= vmax_k,  F+_k >= F_k,  F+_k >= eta_reg F_k                    (18i-k)
          boundary values, t_N - t_0 <= T                                             (18l-n)
          valid inequalities:  E_k >= v_k^2 / 2  (19),  Pmin z_k <= F_k <= Pmax z_k  (20)

The only nonconvex constraints are the bilinear ones (18e) and (18g); Gurobi (NonConvex = 2) and
SCIP solve them exactly by spatial branch-and-bound. Proposition 1 of the paper guarantees that
(18d) is tight at the optimum, i.e. z_k = 1 / v_k: the time is booked correctly.

Note: (18d) must NOT be replaced by a McCormick envelope of y = v z with y >= 1. The envelope is
an outer approximation, so it admits z_k < 1 / v_k and the "trip time" t_N is then shorter than
the time the train actually needs.
"""
from dataclasses import dataclass
import time

import numpy as np

from .model import G, Route, Train


@dataclass
class Solution:
    route: Route
    train: Train
    v: np.ndarray       # nodes 0..N
    E: np.ndarray
    t: np.ndarray
    F: np.ndarray       # segments 1..N (array index k-1)
    z: np.ndarray
    energy: float       # objective sum F+ ds  [J]
    status: str
    gap: float
    runtime: float
    solver: str

    @property
    def energy_kwh(self):
        return self.energy / 3.6e6


# ---------------------------------------------------------------------------------------------
# Thin adapters so that the model is written once for Gurobi and SCIP
# ---------------------------------------------------------------------------------------------
class _Gurobi:
    name = "gurobi"

    def __init__(self, verbose):
        import gurobipy as gp
        self.gp = gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", int(verbose))
        env.start()
        self.m = gp.Model("nlpr3", env=env)
        self.m.Params.NonConvex = 2

    def var(self, lb, ub, name):
        return self.m.addVar(lb=lb, ub=ub, name=name)

    def con(self, c, name=""):
        self.m.addConstr(c, name=name)

    def qcon(self, c, name=""):
        self.m.addQConstr(c, name=name)

    def minimize(self, expr):
        self.m.setObjective(expr, self.gp.GRB.MINIMIZE)

    def solve(self, time_limit, gap, threads):
        self.m.Params.TimeLimit = time_limit
        self.m.Params.MIPGap = gap
        if threads:
            self.m.Params.Threads = threads
        self.m.optimize()
        st = {2: "optimal", 9: "time_limit"}.get(self.m.Status, str(self.m.Status))
        g = self.m.MIPGap if self.m.SolCount else np.inf
        return st, g, self.m.SolCount > 0

    def val(self, x):
        return x.X


class _Scip:
    name = "scip"

    def __init__(self, verbose):
        import pyscipopt
        self.m = pyscipopt.Model("nlpr3")
        self.quicksum = pyscipopt.quicksum
        if not verbose:
            self.m.hideOutput()

    def var(self, lb, ub, name):
        return self.m.addVar(lb=lb, ub=ub, name=name)

    def con(self, c, name=""):
        self.m.addCons(c, name=name)

    qcon = con

    def minimize(self, expr):
        self.m.setObjective(expr, "minimize")

    def solve(self, time_limit, gap, threads):
        self.m.setParam("limits/time", time_limit)
        self.m.setParam("limits/gap", gap)
        self.m.optimize()
        st = self.m.getStatus()
        ok = self.m.getNSols() > 0
        return st, (self.m.getGap() if ok else np.inf), ok

    def val(self, x):
        return self.m.getVal(x)


BACKENDS = {"gurobi": _Gurobi, "scip": _Scip}


def solve_nlpr3(route: Route, train: Train = Train(), solver: str = "gurobi", time_limit: float = 3600,
                gap: float = 1e-4, threads: int = 0, valid_inequalities: bool = True,
                latency: str = "exact", verbose: bool = False) -> Solution:
    """Build NLP_R3 for `route` and solve it with Gurobi or SCIP.

    latency = "exact"      z_k >= 1/v_k  as in the paper (default);
    latency = "mccormick"  z_k v_k >= 1 replaced by the McCormick envelope of y = v z with y >= 1.
                           This is NOT a valid model; it is kept only to reproduce the pitfall.
    """
    b = BACKENDS[solver](verbose)
    N, ds, eps = route.N, route.ds, route.eps
    # Work in kN, t and kW (well-scaled coefficients); E stays in J/kg = m^2/s^2.
    sc = 1e-3
    tr = Train(train.m * sc, train.Fmax * sc, train.Fmin * sc, train.Pmax * sc, train.Pmin * sc,
               train.c0 * sc, train.c1 * sc, train.c2 * sc, train.eta_reg)
    vU = float(route.vmax.max())
    inf = 1e20

    # nodes 0..N
    v = [b.var(eps, (route.vmax[k - 1] if k else vU), f"v{k}") for k in range(N + 1)]
    E = [b.var(eps ** 2 / 2, (route.vmax[k - 1] if k else vU) ** 2 / 2, f"E{k}") for k in range(N + 1)]
    t = [b.var(0.0, inf, f"t{k}") for k in range(N + 1)]
    # segments 1..N (list index k-1)
    F = [b.var(tr.Fmin, tr.Fmax, f"F{k}") for k in range(1, N + 1)]
    Fp = [b.var(tr.eta_reg * tr.Fmin, tr.Fmax, f"Fp{k}") for k in range(1, N + 1)]  # net force paid
    z = [b.var(1.0 / route.vmax[k - 1], 1.0 / eps, f"z{k}") for k in range(1, N + 1)]
    one = b.var(1.0, 1.0, "one")   # lets "1 <= v z" be stated as the rotated cone one^2 <= v z

    for k in range(1, N + 1):
        i = k - 1
        rhs = (F[i] - 2 * tr.c2 * E[k] - tr.c1 * v[k] - tr.c0 - tr.m * G * np.sin(route.alpha[i])) / tr.m
        b.con(E[k] - E[k - 1] == ds[i] * rhs, f"dynE{k}")                 # (18b)
        b.con(t[k] - t[k - 1] == ds[i] * z[i], f"dynT{k}")                # (18c)
        if latency == "exact":
            b.qcon(one * one <= v[k] * z[i], f"latency{k}")               # (18d) z >= 1/v
        else:                                                              # pitfall: outer envelope
            vL, vU_, zL, zU = eps, route.vmax[i], 1 / route.vmax[i], 1 / eps
            y = b.var(1.0, inf, f"y{k}")                                   # y = v z >= 1
            b.con(y >= vL * z[i] + zL * v[k] - vL * zL)
            b.con(y >= vU_ * z[i] + zU * v[k] - vU_ * zU)
            b.con(y <= vU_ * z[i] + zL * v[k] - vU_ * zL)
            b.con(y <= vL * z[i] + zU * v[k] - vL * zU)
        b.qcon(E[k] == 0.5 * v[k] * v[k], f"kin{k}")                       # (18e)
        b.qcon(F[i] * v[k] <= tr.Pmax, f"Pmax{k}")                         # (18g)
        b.qcon(F[i] * v[k] >= tr.Pmin, f"Pmin{k}")
        b.con(Fp[i] >= F[i], f"Fp{k}")                                     # (18j)
        b.con(Fp[i] >= tr.eta_reg * F[i], f"Fpreg{k}")                     # (18k)
        if valid_inequalities:
            b.qcon(v[k] * v[k] <= 2 * E[k], f"vi19_{k}")                    # (19)
            b.con(F[i] <= tr.Pmax * z[i], f"vi20a_{k}")                     # (20)
            b.con(F[i] >= tr.Pmin * z[i], f"vi20b_{k}")

    b.con(E[0] == route.V0 ** 2 / 2); b.con(E[N] == route.Vf ** 2 / 2)    # (18l)
    b.con(v[0] == route.V0); b.con(v[N] == route.Vf)                       # (18m)
    b.con(t[0] == 0.0); b.con(t[N] <= route.T)                             # (18n)

    obj = sum(Fp[i] * ds[i] for i in range(N))
    b.minimize(obj)

    t0 = time.perf_counter()
    status, g, ok = b.solve(time_limit, gap, threads)
    runtime = time.perf_counter() - t0
    if not ok:
        raise RuntimeError(f"{solver}: no feasible solution ({status})")

    val = lambda xs: np.array([b.val(x) for x in xs])
    Fv = val(F) * 1e3                                                      # back to N
    energy = float(np.sum(np.maximum(Fv, tr.eta_reg * Fv) * ds))
    return Solution(route, train, val(v), val(E), val(t), Fv, val(z), energy, status, g, runtime, b.name)
