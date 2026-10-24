"""Figures for docs/ML_TECHNICAL_NOTE.md (figures/fig07 to fig10), drawn from outputs/ml_results.json.

Usage:  python scripts/make_ml_figures.py      (after run_ml_analysis.py)
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
BLUE, ORANGE, GREY, INK, INK2, LINE = "#2a78d6", "#eb6834", "#8a96a3", "#0f1a24", "#475665", "#d6dee5"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
REFS = {"Gradient boosting": BLUE, "Spline regression (linear)": ORANGE, "Power curve": GREY}


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=10, color=INK)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    R = json.loads((ROOT / "outputs" / "ml_results.json").read_text())

    # 7. Training curve, mean of 14 turbines, with each turbine's stopping points in a strip below
    tc = R["training_curve"]
    it = np.array(tc["iterations"])
    v = np.array(tc["validation_mean"])
    es = [t["early_stopping_iteration"] for t in tc["per_turbine"]]
    best = [t["best_iteration"] for t in tc["per_turbine"]]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
    ax.plot(it, tc["train_mean"], color=GREY, lw=1.6, label="Training months (January to September 2017)")
    ax.plot(it, v, color=BLUE, lw=2, label="Held-out quarter (October to December 2017)")
    b = int(it[np.argmin(v)])
    ax.plot([b], [v.min()], "o", ms=7, color=BLUE, mec="white", mew=2)
    ax.annotate(f"lowest at {b} iterations ({v.min():.1f} kW)", (b, v.min()), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(14, 34)
    ax.set_ylabel("Mean absolute error (kW)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    title(ax, "Training curve, mean of 14 turbines (first iterations, above 34 kW, not shown)")
    ax2.scatter(best, [1] * len(best), s=36, color=BLUE, alpha=0.6, edgecolors="white", linewidths=0.8)
    ax2.scatter(es, [0] * len(es), s=30, marker="D", color=ORANGE, alpha=0.6, edgecolors="white", linewidths=0.8)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["early stopping\n(random 10%)", "lowest held-out\nerror"], fontsize=8)
    ax2.set_ylim(-0.6, 1.6)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("Boosting iterations (each dot is one turbine)")
    fig.savefig(FIG / "fig07_ml_training_curve.png")
    plt.close(fig)

    # 8. Learning curve: how much training data
    lc = R["learning_curve"]
    order = [r for r in lc if "spread" not in r["training"]]
    spread = [r for r in lc if "spread" in r["training"]]
    months = {"last 1 month": 1, "last 2 months": 2, "last 3 months": 3, "last 6 months": 6, "last 9 months": 9, "all 12 months": 12}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, yl in ((axes[0], "mae_kw", "Mean absolute error, 2018 (kW)"),
                        (axes[1], "daily_spread_pct", "Spread of daily results, 2018 (percent)")):
        ax.plot([months[r["training"]] for r in order], [r[key] for r in order], "-o", color=BLUE, lw=2, ms=6, mec="white",
                mew=1.5, label="Most recent months of 2017")
        for r in spread:
            ax.plot([3], [r[key]], "D", color=ORANGE, ms=7, mec="white", mew=1.5, label="3 months spread through the year")
        ax.set_xlabel("Months of 2017 used for training")
        ax.set_ylabel(yl)
        ax.set_xticks([1, 2, 3, 6, 9, 12])
    pc = R["accuracy_2018"]["Power curve"]["mae_kw"]
    axes[0].axhline(pc, color=GREY, lw=1.4, ls=":", label=f"Power curve, 12 months ({pc:.1f} kW)")
    axes[0].set_ylim(24, 38)
    axes[1].set_ylim(2.0, 3.2)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    title(axes[0], "Learning curve")
    title(axes[1], "What it does to monitoring")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_ml_learning_curve.png")
    plt.close(fig)

    # 9. Errors by month and by wind direction
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, xl in ((axes[0], "month", "Month of 2018"), (axes[1], "direction_sector_deg", "Wind direction sector (degrees)")):
        rows = R["segments"][key]
        for lab, c in REFS.items():
            ax.plot([r[key] for r in rows], [r[f"mae_kw {lab}"] for r in rows], "-o", color=c, lw=1.8, ms=4, label=lab)
        ax.set_xlabel(xl)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(list("JFMAMJJASOND"))
    axes[1].set_xticks(range(0, 360, 60))
    axes[0].set_ylabel("Mean absolute error (kW)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    title(axes[0], "Error by month, 2018")
    title(axes[1], "By wind direction")
    fig.tight_layout()
    fig.savefig(FIG / "fig09_ml_errors.png")
    plt.close(fig)

    # 10. Fault injection
    ft = R["fault_test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, thr, lab in ((axes[0], "day_detection_at_1pct_pct", "day", "Daily check"),
                              (axes[1], "week_detection_at_1pct_pct", "week", "Weekly check")):
        for ref, c in REFS.items():
            f = ft[ref]["faults"]
            ax.plot([x["fault_pct"] for x in f], [x[key] for x in f], "-o", color=c, lw=1.8, ms=5, mec="white", mew=1,
                    label=f"{ref}: flag below {ft[ref][f'threshold_{thr}_at_1pct_pct']:.1f}%")
        ax.set_xlabel("Shortfall added (percent of energy)")
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        title(ax, lab)
    axes[0].set_ylabel("Affected days or weeks detected (percent)")
    fig.suptitle("Detection when each reference flags 1 percent of untouched days or weeks", x=0.01, ha="left",
                 fontsize=10, color=INK2, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_ml_fault_detection.png")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
