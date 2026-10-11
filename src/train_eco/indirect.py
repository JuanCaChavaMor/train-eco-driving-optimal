"""Indirect method: Pontryagin's principle for the train control problem on level track
(Albrecht, Howlett, Pudney, Vu & Zhou 2016, Part 1, Sections 3.1-3.6), without regenerative braking.

Position x is the independent variable; per unit mass the dynamics are  v dv/dx = u - r(v),  dt/dx = 1/v,
with traction/braking bounds U-(v) <= u <= U+(v). Pontryagin gives five modes; on level track the optimal
strategy is

    MaxPower  ->  HoldP at speed V  ->  Coast  ->  MaxBrake.

With the modified adjoint eta = mu2/v - 1, the phase switches are (Part 1, eq. 32-33, rho = 0):

    MaxPower -> HoldP   at  v = V        (eta = 0)
    Coast    -> Brake   at  v = U        (eta = -1),   U = psi(V) / phi'(V),

where phi(v) = v r(v) and psi(v) = v^2 r'(v). Geometrically, U is where the tangent of the convex curve
phi at v = V crosses the v-axis. Every phase is then an explicit integral in v; the length of the HoldP
phase closes the distance, and an outer root search on V meets the trip time T.

If the route is too short for a HoldP phase, MaxPower switches directly to Coast at a speed V' < V and
the brake speed follows from eta = -1 on the Coast curve that starts at (V', 0), see `_brake_speed_no_hold`.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .model import Train


@dataclass
class IndirectSolution:
    V: float            # optimal driving speed (hold speed)
    Vsw: float          # speed at the end of MaxPower (= V if there is a HoldP phase)
    U: float            # speed at which braking begins
    x: np.ndarray       # switching points [0, x_power_end, x_coast_start, x_brake_start, X]
    T: float
    energy: float       # J
    eta_profile: tuple  # (v, eta) along the journey, for the phase portrait

    @property
    def energy_kwh(self):
        return self.energy / 3.6e6


class LevelTrack:
    """Optimal strategy on level track with length X [m] and trip time T [s] (no speed limits, rho = 0)."""

    def __init__(self, train: Train, X: float):
        self.tr, self.X = train, X
        m = train.m
        self.r = lambda v: train.resistance(v) / m                    # resistance per unit mass
        self.dr = lambda v: (train.c1 + 2 * train.c2 * v) / m
        self.Up = lambda v: train.max_traction(v) / m                 # U+(v) > 0
        self.Um = lambda v: train.max_braking(v) / m                  # U-(v) < 0
        self.phi = lambda v: v * self.r(v)
        self.dphi = lambda v: self.r(v) + v * self.dr(v)
        self.psi = lambda v: v * v * self.dr(v)
        # E_V(v) = int (psi(v) - psi(V)) / v^2 dv = r1 v + r2 v^2 + psi(V)/v   (quadratic Davis resistance)
        self.EV = lambda v, V: (train.c1 * v + train.c2 * v * v) / m + self.psi(V) / v

    # ---- phases as integrals in v: (distance, time, traction work per unit mass) -------------
    def power_phase(self, v1):
        f = lambda v: self.Up(v) - self.r(v)
        kink = [self.tr.Pmax / self.tr.Fmax] if v1 > self.tr.Pmax / self.tr.Fmax else None  # U+ = Fmax -> Pmax/v
        dx = quad(lambda v: v / f(v), 0, v1, points=kink, limit=200)[0]
        dt = quad(lambda v: 1 / f(v), 0, v1, points=kink, limit=200)[0]
        work = quad(lambda v: self.Up(v) * v / f(v), 0, v1, points=kink, limit=200)[0]
        return dx, dt, work

    def coast_phase(self, v1, v2):          # from v1 down to v2
        dx = quad(lambda v: v / self.r(v), v2, v1)[0]
        dt = quad(lambda v: 1 / self.r(v), v2, v1)[0]
        return dx, dt

    def brake_phase(self, v1):              # from v1 down to 0
        g = lambda v: self.r(v) - self.Um(v)
        dx = quad(lambda v: v / g(v), 0, v1, limit=200)[0]
        dt = quad(lambda v: 1 / g(v), 0, v1, limit=200)[0]
        return dx, dt

    def brake_speed(self, V):
        """U = psi(V) / phi'(V): the tangent of phi at V meets the v-axis (eta = -1 at the brake switch)."""
        return self.psi(V) / self.dphi(V)

    def _brake_speed_no_hold(self, V, Vsw):
        """Brake speed when MaxPower ends at Vsw < V and Coast starts directly (Part 1, end of Sec. 3.6).

        Coast from (Vsw, eta = 0): eta(v) = (E_V(v) - E_V(Vsw)) / (-r(v)); braking starts at eta = -1.
        """
        return brentq(lambda U: self.EV(U, V) - self.EV(Vsw, V) - self.r(U), 1e-6, Vsw * (1 - 1e-12))

    # ---- journey for a nominated V --------------------------------------------------------------
    def journey(self, V):
        U = self.brake_speed(V)
        xp, tp, wp = self.power_phase(V)
        xc, tc = self.coast_phase(V, U)
        xb, tb = self.brake_phase(U)
        hold = self.X - xp - xc - xb
        if hold >= 0:
            Vsw = V
            T = tp + hold / V + tc + tb
            work = wp + self.r(V) * hold
            x = np.array([0, xp, xp + hold, xp + hold + xc, self.X])
            return dict(V=V, Vsw=V, U=U, T=T, work=work, x=x)

        # no HoldP phase: find the switch speed Vsw < V that closes the distance
        def dist(Vsw):
            U_ = self._brake_speed_no_hold(V, Vsw)
            return self.power_phase(Vsw)[0] + self.coast_phase(Vsw, U_)[0] + self.brake_phase(U_)[0] - self.X
        Vsw = brentq(dist, 1e-2 * V, V)
        U = self._brake_speed_no_hold(V, Vsw)
        xp, tp, wp = self.power_phase(Vsw)
        xc, tc = self.coast_phase(Vsw, U)
        xb, tb = self.brake_phase(U)
        x = np.array([0, xp, xp, xp + xc, self.X])
        return dict(V=V, Vsw=Vsw, U=U, T=tp + tc + tb, work=wp, x=x)

    def v_limit_power(self):
        """Limiting speed under Maximum Power on level track: U+(v) = r(v)."""
        return brentq(lambda v: self.Up(v) - self.r(v), 1.0, 300.0)

    def solve(self, T):
        """Outer loop: find the driving speed V whose journey takes exactly T."""
        V = brentq(lambda V: self.journey(V)["T"] - T, 0.5, 0.999 * self.v_limit_power(), xtol=1e-10)
        j = self.journey(V)
        return IndirectSolution(V, j["Vsw"], j["U"], j["x"], j["T"], self.tr.m * j["work"], self.eta(V, j))

    # ---- adjoint along the optimal journey (phase portrait (v, eta)) ----------------------------
    def eta(self, V, j, n=200):
        Vsw, U = j["Vsw"], j["U"]
        E0 = self.EV(Vsw, V)
        vp = np.linspace(0.5, Vsw, n)
        eta_p = (self.EV(vp, V) - E0) / (self.Up(vp) - self.r(vp))     # MaxPower: eta >= 0
        vc = np.linspace(Vsw, U, n)
        eta_c = (self.EV(vc, V) - E0) / (-self.r(vc))                   # Coast: -1 <= eta <= 0
        vb = np.linspace(U, 0.0, n)
        eta_b = -np.ones(n)          # MaxBrake with rho = 0: zeta = eta + 1 solves a homogeneous ODE, zeta(U) = 0
        return np.r_[vp, vc, vb], np.r_[eta_p, eta_c, eta_b]

    # ---- speed profile v(x) of the optimal journey ---------------------------------------------
    def profile(self, sol: IndirectSolution, n=150):
        """Speed profile (x, v) of an IndirectSolution, from the phase integrals (x as a function of v)."""
        x0, x1, x2, x3, X = sol.x
        vp = np.linspace(0, sol.Vsw, n)
        xp = np.array([self.power_phase(v)[0] for v in vp])
        vc = np.linspace(sol.Vsw, sol.U, n)
        xc = x2 + np.array([self.coast_phase(sol.Vsw, v)[0] for v in vc])
        vb = np.linspace(sol.U, 0, n)
        xb = x3 + np.array([self.brake_phase(sol.U)[0] - self.brake_phase(v)[0] for v in vb])
        x = np.r_[xp, xc, xb]
        v = np.r_[vp, vc, vb]
        return x, v
