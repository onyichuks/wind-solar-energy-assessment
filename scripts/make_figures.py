"""Static figures for the README, drawn from the analysis outputs.

Usage:  python scripts/make_figures.py   (after run_wind.py and run_solar.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
S1, S2, S3, POS, NEG, TOTAL, INK2, LINE = "#2a78d6", "#eb6834", "#1baf7a", "#1baf7a", "#eb6834", "#475665", "#475665", "#d6dee5"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def main() -> None:
    FIG.mkdir(exist_ok=True)
    w = json.loads((ROOT / "outputs" / "wind_dashboard.json").read_text())
    s = json.loads((ROOT / "outputs" / "solar_dashboard.json").read_text())
    R = w["results"]

    # 1. monthly predicted against actual
    m = w["monthly"]
    x = np.arange(len(m["month"]))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x, np.array(m["actual"]) / 1000, color=S1, width=0.75, label="Actual (metered)")
    ax.plot(x, np.array(m["predicted"]) / 1000, color=S2, lw=2, marker="o", ms=3, label="Predicted (P50 for that month's wind)")
    ax.set_xticks(x[::6], [m["month"][i] for i in range(0, len(x), 6)], rotation=0)
    ax.set_ylabel("GWh per month")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title(f"Penmanshiel: actual {R['predicted_vs_actual']['gap_pct']:+.1f}% against the pre-construction estimate", loc="left", fontsize=11)
    fig.savefig(FIG / "fig01_predicted_vs_actual.png")
    plt.close(fig)

    # 2. gap waterfall
    g = w["gap"]
    short = {"Site flow": "Site flow", "Power curve": "Power\ncurve", "Wakes": "Wakes", "Availability": "Availability",
             "Electrical": "Electrical", "Curtailment": "Curtailment", "Environmental": "Residual"}
    labels = ["P50\n(estimate)"] + [next(v for k, v in short.items() if st.startswith(k)) for st in g["step"]] + ["Actual,\n2017-18"]
    p50 = R["energy_yield"]["p50_mwh"] / 1000
    actual = R["predicted_vs_actual"]["gap_target_mwh"] / 1000
    ch = np.array(g["change_mwh"]) / 1000
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(0, p50, color=TOTAL, width=0.6)
    run = p50
    for i, c in enumerate(ch, start=1):
        ax.bar(i, c, bottom=run, color=POS if c >= 0 else NEG, width=0.6)
        ax.text(i, run + max(c, 0) + 0.15, f"{c:+.2f}", ha="center", fontsize=9, color=INK2)
        run += c
    ax.bar(len(ch) + 1, actual, color=TOTAL, width=0.6)
    for xi, v in [(0, p50), (len(ch) + 1, actual)]:
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylim(min(p50, actual) * 0.92, max(p50, actual) * 1.03)
    ax.set_ylabel("GWh per year")
    ax.set_title("2017 and 2018: each assumption replaced by its measured value (long-term basis)", loc="left", fontsize=11)
    fig.savefig(FIG / "fig02_gap_waterfall.png")
    plt.close(fig)

    # 3. wake efficiency by sector, measured against models
    e = w["sector_eff"]
    lab = [f"{int(d)}°" for d in e["centre_deg"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = {k: v["label"] for k, v in R["energy_yield"]["wake_models"].items()}
    for k in ["bastankhah", "turbopark"]:
        ax.plot(lab, 100 * np.array(e[k]), color="#b8c3cc", lw=1.3)
    ax.plot(lab, 100 * np.array(e["niayifar"]), color=S2, lw=2, marker="o", ms=4, label=names["niayifar"])
    ax.plot(lab, 100 * np.array(e["jensen"]), color="#8796a3", lw=1.6, label=names["jensen"] + " (central estimate)")
    ax.plot(lab, 100 * np.array(e["measured"]), color=S1, lw=2, marker="o", ms=4, label="Measured, terrain-adjusted")
    ax.set_ylabel("Wake efficiency (%), 5 to 11 m/s")
    ax.set_xlabel("Wind direction sector")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Measured wake losses are well below every model's prediction", loc="left", fontsize=11)
    fig.savefig(FIG / "fig03_wake_efficiency.png")
    plt.close(fig)

    # 4. layout coloured by wake loss, and terrain
    L = w["layout"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    sc = a1.scatter(L["x_m"], L["y_m"], c=L["wake_loss_pct_jensen"], cmap="Blues", vmin=0, vmax=16, s=110, edgecolors="white", linewidths=1.5)
    for t, xx, yy in zip(L["turbine"], L["x_m"], L["y_m"]):
        a1.annotate(t, (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    a1.margins(y=0.12)  # headroom for the top label
    a1.set_aspect("equal")
    a1.set_xlabel("East (m)")
    a1.set_ylabel("North (m)")
    fig.colorbar(sc, ax=a1, label="Modelled wake loss (%)", shrink=0.8, pad=0.02)
    a1.set_title("Layout and modelled wake loss (Park, k = 0.075)", loc="left", fontsize=11)
    T = w["terrain"]
    a2.scatter(T["elevation_m"], T["speed_factor"], color=S1, s=50, edgecolors="white")
    # turbines that sit close together get their labels moved so none overlap
    place = {"T06": (-5, 3, "right"), "T12": (-5, 0, "right"), "T14": (4, -10, "left")}
    for t, xx, yy in zip(T["turbine"], T["elevation_m"], T["speed_factor"]):
        dx, dy, ha = place.get(t, (4, 3, "left"))
        a2.annotate(t, (xx, yy), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK2)
    a2.set_xlabel("Ground elevation (m)")
    a2.set_ylabel("Measured free-stream speed factor")
    a2.set_title("Terrain: higher positions see more wind", loc="left", fontsize=11)
    fig.savefig(FIG / "fig04_layout_terrain.png")
    plt.close(fig)

    # 5. solar monthly performance index
    sm = s["monthly"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    xs = np.arange(len(sm["month"]))
    ax.plot(xs, sm["performance_index"], color=S1, lw=2, marker="o", ms=4)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(xs[::4], [sm["month"][i] for i in range(0, len(xs), 4)])
    ax.set_ylabel("Actual over expected")
    st = s["results"]["totals"]
    ax.set_title(f"Solar PV performance index by month (availability {st['time_availability_pct']:.1f}%, outages {st['outage_loss_pct']:.2f}%, snow {st['snow_loss_pct']:.2f}%)", loc="left", fontsize=11)
    fig.savefig(FIG / "fig05_solar_performance.png")
    plt.close(fig)

    # 6. turbine health: model accuracy and the daily residual map
    hp = ROOT / "outputs" / "health_dashboard.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        acc = h["accuracy"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2.4]})
        y = np.arange(len(acc["turbine"]))
        a1.barh(y + 0.2, acc["mae_kw_power_curve"], height=0.38, color=S2, label="Power curve (method of bins)")
        a1.barh(y - 0.2, acc["mae_kw_model"], height=0.38, color=S1, label="Machine learning model")
        a1.set_yticks(y, acc["turbine"])
        a1.invert_yaxis()
        a1.set_xlabel("Mean absolute error, 2018 (kW)")
        a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=8)
        a1.set_title(f"Error {h['results']['accuracy']['mae_reduction_pct']:.0f}% lower than the power curve", loc="left", fontsize=11)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("res", [S2, "#ffffff", S1])
        names = sorted(h["turbines"])
        grid = np.array([[np.nan if v is None else v for v in h["turbines"][t]["residual"]] for t in names], dtype=float)
        im = a2.imshow(grid, aspect="auto", cmap=cmap, vmin=-10, vmax=10, interpolation="nearest")
        a2.set_yticks(range(len(names)), names)
        months = [i for i, d in enumerate(h["days"]) if d.endswith("-01") or i == 0]
        a2.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(months)], fontsize=8)
        a2.grid(False)
        cb = fig.colorbar(im, ax=a2, fraction=0.03, pad=0.01)
        cb.set_label("Actual against expected (%)")
        a2.set_title("2018, day by day: shortfalls (orange) that hit every turbine are external; single-turbine ones are not", loc="left", fontsize=10)
        fig.savefig(FIG / "fig06_turbine_health.png")
        plt.close(fig)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
