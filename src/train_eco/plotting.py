"""Figures for the README (matplotlib, static)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .export import classify_modes
from .model import Train

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots, fixed order
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
MODE_COLOR = {"MaxPower": C[1], "HoldP": C[0], "Coast": C[2], "MaxBrake": C[4], "Brake (partial)": C[3]}

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
})


def _save(fig, path):
    """PNG for the README and a vector PDF (same name) for the report."""
    from pathlib import Path
    path = Path(path)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_mccormick_latency(v_lo, v_hi, path):
    """Feasible set of (v, z): exact latency z >= 1/v vs. the McCormick outer approximation of v z >= 1."""
    zL, zU = 1 / v_hi, 1 / v_lo
    v = np.linspace(v_lo, v_hi, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    # McCormick: y <= vU z + zL v - vU zL  and  y <= vL z + zU v - vL zU, with y >= 1
    z_mc = np.maximum((1 - zL * v + v_hi * zL) / v_hi, (1 - zU * v + v_lo * zU) / v_lo)
    z_mc = np.clip(z_mc, zL, zU)
    ax.fill_between(v, z_mc, 1 / v, where=1 / v > z_mc, color=C[1], alpha=0.25, lw=0,
                    label="admitted by McCormick, z < 1/v")
    ax.plot(v, 1 / v, color=C[0], label="exact: z = 1/v (feasible above)")
    ax.plot(v, z_mc, color=C[1], lw=1.5, ls="--", label="McCormick lower boundary")
    ax.set_yscale("log")
    ax.set_xlabel("speed v [m/s]"); ax.set_ylabel("latency z [s/m]")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def _shade_modes(ax, s, labels):
    """Background bands showing the driving mode of each segment."""
    start = 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[start]:
            ax.axvspan(s[start] / 1e3, s[k] / 1e3, color=MODE_COLOR[labels[start]], alpha=0.12, lw=0)
            start = k


def plot_solution(sol, path, title=""):
    """Speed and force profiles of a discrete solution, with the driving modes shaded."""
    r, tr = sol.route, Train()
    s = r.s
    lab = classify_modes(sol.v[1:], sol.F, tr)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, constrained_layout=True)
    for ax in (a1, a2):
        _shade_modes(ax, s, lab)
        down = np.where(r.alpha < 0)[0]
        if len(down):
            ax.axvspan(s[down[0]] / 1e3, s[down[-1] + 1] / 1e3, ymin=0, ymax=0.03, color=INK, lw=0)
    if len(down):
        a1.text((s[down[0]] + s[down[-1] + 1]) / 2e3, 8, f"downhill {100 * r.alpha[down[0]]:.0f} %",
                ha="center", color=INK, fontsize=8)
    a1.plot(s / 1e3, sol.v * 3.6, color=C[0], label="speed")
    a1.plot(s[1:] / 1e3, r.vmax * 3.6, color=MUTED, lw=1, ls="--", label="speed limit")
    a1.set_ylabel("speed [km/h]")
    a1.legend(loc="center", bbox_to_anchor=(0.4, 0.3), ncol=2)
    a2.step(s[1:] / 1e3, sol.F / 1e3, where="pre", color=C[0], label="applied force")
    vv = np.linspace(0.5, r.vmax.max(), 300)
    a2.plot(s[1:] / 1e3, tr.max_traction(sol.v[1:]) / 1e3, color=MUTED, lw=1, ls="--", label="force bounds at v")
    a2.plot(s[1:] / 1e3, tr.max_braking(sol.v[1:]) / 1e3, color=MUTED, lw=1, ls="--")
    a2.set_ylabel("force [kN]")
    a2.set_xlabel("position [km]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLOR[m], alpha=0.25) for m in MODE_COLOR if m in set(lab)]
    a2.legend(handles + a2.get_legend_handles_labels()[0],
              [m for m in MODE_COLOR if m in set(lab)] + a2.get_legend_handles_labels()[1],
              loc="lower left", ncol=3, fontsize=8)
    if title:
        a1.set_title(title, loc="left", color=INK)
    _save(fig, path)


def plot_comparison(profiles, ind_xv, path, title=""):
    """Speed profiles of several NLP solutions against the continuous indirect solution."""
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    x, v = ind_xv
    ax.plot(x / 1e3, v * 3.6, color=INK, lw=3, alpha=0.25, label="indirect (Pontryagin)")
    for (lbl, s, vs), c in zip(profiles, C):
        ax.plot(s / 1e3, vs * 3.6, color=c, lw=1.5, label=lbl)
    ax.set_xlabel("position [km]"); ax.set_ylabel("speed [km/h]")
    ax.legend(loc="lower center", ncol=2)
    if title:
        ax.set_title(title, loc="left", color=INK)
    _save(fig, path)


def plot_phase_portrait(v, eta, U, V, path):
    """Modified adjoint eta against speed along the optimal journey (cf. Albrecht et al., Part 1, Fig. 5)."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.axvline(0, color=MUTED, lw=1); ax.axvline(-1, color=MUTED, lw=1, ls="--")
    ax.plot(eta, v * 3.6, color=C[0])
    ax.annotate(f"HoldP at V = {V * 3.6:.1f} km/h", (0, V * 3.6), xytext=(0.12, V * 3.6 - 6), color=INK,
                fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate(f"brake at U = {U * 3.6:.1f} km/h", (-1, U * 3.6), xytext=(-0.9, U * 3.6 - 22), color=INK,
                fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.text(0.5, 5, "MaxPower\n(eta > 0)", ha="center", color=MUTED, fontsize=8)
    ax.text(-0.5, 5, "Coast\n(-1 < eta < 0)", ha="center", color=MUTED, fontsize=8)
    ax.set_xlim(-1.3, 1.2); ax.set_ylim(0, V * 3.6 * 1.12)
    ax.set_xlabel("modified adjoint eta = mu2/v - 1"); ax.set_ylabel("speed [km/h]")
    _save(fig, path)


def plot_energy_time(Ts, E, path, marks=()):
    """Energy-time trade-off of the optimal journeys."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4), constrained_layout=True)
    ax.plot(Ts, E, color=C[0])
    for T, e, lbl in marks:
        ax.plot([T], [e], "o", color=C[1], ms=7, mec="white", mew=2)
        ax.annotate(lbl, (T, e), xytext=(8, 4), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlabel("trip time T [s]"); ax.set_ylabel("traction energy [kWh]")
    _save(fig, path)


def plot_tangent(lt, V, path):
    """Geometric construction of the brake speed: tangent to phi(v) = v r(v) at V meets the v-axis at U."""
    U = lt.brake_speed(V)
    v = np.linspace(0, 1.25 * V, 300)
    fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    ax.plot(v * 3.6, lt.phi(v), color=C[0], label="phi(v) = v r(v)")
    ax.plot(v * 3.6, lt.phi(V) + lt.dphi(V) * (v - V), color=C[1], lw=1.5, label="tangent at V")
    ax.plot([V * 3.6, U * 3.6], [lt.phi(V), 0], "o", color=INK, ms=6, mec="white", mew=1.5)
    ax.annotate("V", (V * 3.6, lt.phi(V)), xytext=(-14, 6), textcoords="offset points", color=INK)
    ax.annotate("U = psi(V) / phi'(V)", (U * 3.6, 0), xytext=(6, -14), textcoords="offset points", color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-0.25 * lt.phi(V), None)
    ax.set_xlabel("speed [km/h]"); ax.set_ylabel("phi [m^2/s^3]")
    ax.legend(loc="upper left")
    _save(fig, path)
