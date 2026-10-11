"""Train and route data (Xiao, Ye & Chung 2025, Table 1 and Example 1)."""
from dataclasses import dataclass, field

import numpy as np

G = 9.81  # m/s^2


@dataclass(frozen=True)
class Train:
    """Point-mass train. Forces in N, power in W, speeds in m/s."""
    m: float = 400e3          # mass [kg]
    Fmax: float = 240e3       # maximum traction force
    Fmin: float = -240e3      # maximum braking force (negative)
    Pmax: float = 3.6e6       # maximum traction power
    Pmin: float = -3.6e6      # maximum braking power (negative)
    c0: float = 5840.0        # Davis resistance R(v) = c0 + c1 v + c2 v^2  [N]
    c1: float = 400.0
    c2: float = 15.0
    eta_reg: float = 0.0      # share of braking energy that is recovered

    def resistance(self, v):
        return self.c0 + self.c1 * v + self.c2 * v * v

    def max_traction(self, v):
        """Largest traction force at speed v: min(Fmax, Pmax / v)."""
        return np.minimum(self.Fmax, self.Pmax / np.maximum(v, 1e-9))

    def max_braking(self, v):
        """Largest braking force (negative) at speed v: max(Fmin, Pmin / v)."""
        return np.maximum(self.Fmin, self.Pmin / np.maximum(v, 1e-9))


@dataclass
class Route:
    """Discretised route: N segments, nodes s_0 < ... < s_N.

    Segment k (k = 1..N) is [s_{k-1}, s_k]; alpha[k-1] and vmax[k-1] are its
    gradient (rad, negative = downhill) and speed limit.
    """
    ds: np.ndarray
    alpha: np.ndarray
    vmax: np.ndarray
    T: float                      # planned trip time [s]
    V0: float = 0.1               # initial speed (= epsilon: start from standstill)
    Vf: float = 0.1               # final speed
    eps: float = 0.1              # lower speed bound, avoids 1/v singularity
    name: str = "route"
    s: np.ndarray = field(init=False)

    def __post_init__(self):
        self.ds = np.asarray(self.ds, float)
        self.alpha = np.asarray(self.alpha, float)
        self.vmax = np.asarray(self.vmax, float)
        assert len(self.ds) == len(self.alpha) == len(self.vmax)
        self.s = np.concatenate(([0.0], np.cumsum(self.ds)))

    @property
    def N(self):
        return len(self.ds)

    @property
    def length(self):
        return self.s[-1]


def example1(T: float = 540.0) -> Route:
    """Example 1 of Xiao et al. (2025): 15.02 km, N = 520, 140 km/h, downhill -0.04 on k = 311..483.

    The first and last 10 segments are 1 m long, the other 500 are 30 m.
    """
    ds = np.full(520, 30.0)
    ds[:10] = ds[-10:] = 1.0
    alpha = np.zeros(520)
    alpha[310:483] = -0.04            # paper indices k = 311..483 (1-based)
    vmax = np.full(520, 140 / 3.6)
    return Route(ds, alpha, vmax, T, name="example1")


def level_route(T: float = 600.0, refine: int = 1) -> Route:
    """Example 1 without the downhill section (level track), for the comparison with the indirect method.

    `refine` splits every 30 m segment into `refine` equal parts (N = 20 + 500 * refine).
    """
    ds = np.r_[np.ones(10), np.full(500 * refine, 30.0 / refine), np.ones(10)]
    return Route(ds, np.zeros(len(ds)), np.full(len(ds), 140 / 3.6), T, name=f"level_N{len(ds)}")


def coarsen(route: Route, factor: int) -> Route:
    """Merge the 30 m segments of a route in groups of `factor` (for small test instances)."""
    ds, al, vm = [], [], []
    k = 0
    while k < route.N:
        if route.ds[k] > 1.0 and k + factor <= route.N - 10:
            ds.append(route.ds[k:k + factor].sum()); al.append(route.alpha[k]); vm.append(route.vmax[k])
            k += factor
        else:
            ds.append(route.ds[k]); al.append(route.alpha[k]); vm.append(route.vmax[k])
            k += 1
    return Route(ds, al, vm, route.T, route.V0, route.Vf, route.eps, name=f"{route.name}_x{factor}")
