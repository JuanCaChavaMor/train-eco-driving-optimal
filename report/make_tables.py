"""Write the LaTeX tables of the report from the CSV/JSON files in results/.

    python report/make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


# direct vs. indirect
d = pd.read_csv(RES / "direct_vs_indirect.csv")
rows = "\n".join(
    f"{int(r.N)} & {r.ds_m:g} & {r.nlp_kWh:.3f} & {r.nlp_gap_pct:.3f} & {r.T_phys_s:.1f} & "
    f"{r.indirect_at_T_phys_kWh:.3f} & {r.diff_pct:+.3f} \\\\" for r in d.itertuples())
write("direct_vs_indirect", r"""\begin{tabular}{rrrrrrr}
\toprule
$N$ & $\Delta s$ [m] & NLP$_{R3}$ [kWh] & gap [\%] & $T_\mathrm{phys}$ [s] & indirect [kWh] & diff. [\%] \\
\midrule
""" + rows + "\n\\bottomrule\n\\end{tabular}\n")

# McCormick pitfall
p = pd.read_csv(RES / "pitfall_mccormick.csv").set_index("latency")
lab = {"exact": r"exact, $v_k z_k \ge 1$", "mccormick": "McCormick envelope"}
rows = "\n".join(
    f"{lab[k]} & {r.energy_kWh:.1f} & {r.booked_time_s:.0f} & {r.true_time_s:.1f} & {r.min_v_times_z:.3f} \\\\"
    for k, r in p.iterrows())
write("pitfall", r"""\begin{tabular}{lrrrr}
\toprule
latency constraint & energy [kWh] & booked $t_N$ [s] & $\sum_k \Delta s_k / v_k$ [s] & $\min_k v_k z_k$ \\
\midrule
""" + rows + "\n\\bottomrule\n\\end{tabular}\n")

# Example 1 key figures (as macros)
m = json.loads((RES / "example1_meta.json").read_text())
write("example1_macros", "\n".join([
    rf"\newcommand{{\exEnergy}}{{{m['energy_kWh']:.2f}}}",
    rf"\newcommand{{\exEnergyMJ}}{{{m['energy_kWh'] * 3.6:.1f}}}",
    rf"\newcommand{{\exGap}}{{{100 * m['gap']:.3f}}}",
    rf"\newcommand{{\exRuntime}}{{{m['runtime_s']:.1f}}}",
]) + "\n")
