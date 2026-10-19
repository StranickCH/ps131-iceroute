# -*- coding: utf-8 -*-
"""Figures 5, 8 and 9 of article 3: simulation scheme, regret tails, two-panel robustness plot.
Run: python make_figures_shrink.py <runs folder> <output folder>"""
import sys, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100, "font.family": "DejaVu Sans"})
COL = {"amsr": "#7f7f7f", "camera": "#4C72B0", "fusion": "#C44E52"}
PLC = {"hard": "#bdbdbd", "bayes": "#3182bd", "cvar": "#31a354"}
SRC = {"amsr": "AMSR2", "fusion": "Fusion", "camera": "Camera"}


def run_dir(tag):
    return sorted(glob.glob(os.path.join(ROOT, f"routev3_{tag}_2*")))[-1]


# ------------------------------------------------------------------ 1. scheme
def fig_scheme():
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#555555", lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold")
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.2, linespacing=1.35)

    def arr(x0, y0, x1, y1, txt=None, rad=0.0, dy=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.0,
                                     color="#333333", connectionstyle=f"arc3,rad={rad}"))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", va="bottom", fontsize=6.8, style="italic")

    W, H = 29, 19
    # top row
    box(1, 27, W, H, "PS131 data", "out-of-fold predictions\nfor camera frames,\nbridge labels, AMSR2,\nDMI ice charts", "#f2f2f2")
    box(35.5, 27, W, H, "True field 100 × 100 km", "ice classes on a 1 km grid:\nAMSR2 + floe clusters\n(main setup)\nor DMI chart (reference)", "#fde0dd")
    box(70, 27, W, H, "Observation model P(o | k, v)", "real frame of the same class k\nand visibility v from other episodes;\nAMSR2: 6 km blocks; camera\nand fusion: 3 km blocks within\n1 / 3 / 5 km", "#deebf7")
    # bottom row
    box(70, 1, W, H, "Cost map", "hard: c(o)\nbayes: E[c | o, v] from M⁽ᵛ⁾ and π\nprobs / probs_t: network probabilities\ncvar: CVaR₀.₈ of the cell cost\n(M, π from other episodes only)", "#e5f5e0")
    box(35.5, 1, W, H, "MCP route", "8-neighbour, 1 km grid;\nreplanning every 3 km\nand when new cells appear\nin the field of view", "#fff2cc")
    box(1, 1, W, H, "Evaluation", "true cost of the sailed\nroute C(r) vs oracle C(r*):\nregret, tail metrics,\nnominal transit time", "#f2f2f2")
    arr(31, 36.5, 35, 36.5)
    arr(65.5, 36.5, 69.5, 36.5)
    arr(84.5, 26.6, 84.5, 20.6)
    ax.text(83.5, 23.6, "observations", fontsize=6.8, style="italic", ha="right", va="center")
    arr(69.5, 10.5, 65.5, 10.5)
    arr(35, 10.5, 31, 10.5)
    arr(50, 20.6, 74, 26.6, rad=-0.25)
    ax.text(48.5, 23.3, "ship motion\nreveals new cells", fontsize=6.8, style="italic", ha="right", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_scheme.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2. tails
def fig_tails():
    d = pd.read_csv(os.path.join(run_dir("main32"), "trials.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    for ax, src in zip(axes, ("amsr", "fusion", "camera")):
        for pl in ("hard", "bayes", "cvar"):
            r = np.sort(100 * d[(d.source == src) & (d.planner == pl)]["regret"].values)
            ex = 1 - np.arange(1, len(r) + 1) / len(r)
            ax.step(r, np.maximum(ex, 1e-4), where="post", color=PLC[pl], lw=1.6, label=pl)
            p95 = np.percentile(r, 95)
            ax.plot([p95], [0.05], "o", ms=4, color=PLC[pl], mec="k", mew=0.4)
        ax.axhline(0.05, color="#999999", lw=0.6, ls=":")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1.05); ax.set_xlim(0, 200)
        ax.set_title(SRC[src], fontsize=9)
        ax.set_xlabel("regret, %")
    axes[0].set_ylabel("fraction of routes with regret > x")
    axes[0].legend(fontsize=7, loc="lower left", title="planner", title_fontsize=7)
    axes[0].text(198, 0.058, "5 %", fontsize=6.5, ha="right", color="#777777")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_tails.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 3. robustness
VAR = [("Simulation assumptions", None),
       ("main32", "main run"), ("matrix", "sampling from confusion matrices"), ("floe3", "floe clusters σ = 3 km"),
       ("floe12", "floe clusters σ = 12 km"), ("thr25_75", "thresholds 0.25 / 0.75"), ("thr45_90", "thresholds 0.45 / 0.90"),
       ("cost148", "costs 1 / 4 / 8"), ("cost1212", "costs 1 / 2 / 12"), ("cost124", "costs 1 / 2 / 4 (time)"),
       ("range2", "camera range ×2"), ("block1", "error block 1 km"), ("block6", "error block 6 km"),
       ("donor32", "frames from a donor episode"),
       ("Condition shift and labels", None),
       ("timesplit32", "shift between cruise halves"), ("lab60", "label window 60 min (26 ep.)"),
       ("Reference: DMI ice charts", None),
       ("dmi70", "coverage ≥ 70%, no data impassable"), ("dmi90", "coverage ≥ 90%"),
       ("dmi70water", "no data = open water"), ("dmi70heavy", "no data = heavy ice"),
       ("dmi70amsr", "no data from AMSR2"), ("dmi50amsr", "coverage ≥ 50%, from AMSR2"),
       ("dmi70donor", "donor episode")]


def boot_ci(x, n=1000, seed=0):
    """95% CI by episode bootstrap, as boot_ci in route_ps131_v3.py."""
    rng = np.random.default_rng(seed)
    g = x.groupby("episode")["d"].mean().values
    return np.percentile([rng.choice(g, len(g)).mean() for _ in range(n)], [2.5, 97.5])


def parse(tag):
    """Regret differences (pp) and 95% CI from trials.csv, as in summary.txt of route_ps131_v3.py."""
    d = pd.read_csv(os.path.join(run_dir(tag), "trials.csv"))
    w = d.pivot_table(index=["episode", "vis", "trial"], columns=["source", "planner"], values="regret")
    out = {}
    for grp, vis in (("all", None), ("good", 2), ("poor", 0)):
        wv = w if vis is None else w[w.index.get_level_values("vis") == vis]
        for a, b in [((s_, "bayes"), (s_, "hard")) for s_ in ("amsr", "fusion", "camera")] + [(("fusion", "bayes"), ("amsr", "bayes"))]:
            if a in wv.columns and b in wv.columns:
                x = pd.DataFrame({"episode": wv.index.get_level_values("episode"), "d": (wv[a] - wv[b]).values}).dropna()
                lo, hi = boot_ci(x)
                out[(grp, *a, *b)] = (100 * x["d"].mean(), 100 * lo, 100 * hi)
    return out


def fig_robust():
    rows = []
    for tag, lab in VAR:
        if lab is None:
            rows.append((tag, None)); continue
        rows.append((lab, parse(tag)))
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 0.27 * n + 1.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(n)[::-1]
    for i, (lab, P) in enumerate(rows):
        yy = y[i]
        if P is None:
            for ax in axes:
                ax.axhspan(yy - 0.5, yy + 0.5, color="#f0f0f0", zorder=0)
            continue
        for k, src in enumerate(("amsr", "fusion", "camera")):
            v = P.get(("all", src, "bayes", src, "hard"))
            if v:
                axes[0].errorbar(v[0], yy + (k - 1) * 0.22, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=3.2,
                                 color=COL[src], lw=0.9, capsize=0)
        for k, (grp, mk) in enumerate((("all", "o"), ("good", "^"), ("poor", "v"))):
            v = P.get((grp, "fusion", "bayes", "amsr", "bayes"))
            if v:
                axes[1].errorbar(v[0], yy + (k - 1) * 0.22, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, ms=3.4,
                                 color=["#222222", "#d95f02", "#7570b3"][k], lw=0.9, capsize=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels([("" if P is None else lab) for lab, P in rows], fontsize=7.3)
    for i, (lab, P) in enumerate(rows):
        if P is None:
            axes[0].text(-0.02, y[i], lab, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                         fontsize=7.5, weight="bold")
    for ax in axes:
        ax.axvline(0, color="#444444", lw=0.7)
        ax.set_ylim(-0.6, n - 0.4)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel("bayes − hard, pp")
    axes[1].set_xlabel("fusion − AMSR2 (bayes), pp")
    axes[0].set_title("a) gain from error-aware planning", fontsize=9, loc="left")
    axes[1].set_title("b) difference between sensors", fontsize=9, loc="left")
    from matplotlib.lines import Line2D
    h0 = [Line2D([], [], marker="o", ls="", color=COL[s], label=SRC[s]) for s in ("amsr", "fusion", "camera")]
    axes[0].legend(handles=h0, fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
    h1 = [Line2D([], [], marker=m, ls="", color=c, label=l) for m, c, l in
          (("o", "#222222", "all episodes"), ("^", "#d95f02", "visibility > 10 km"), ("v", "#7570b3", "visibility < 1 km"))]
    axes[1].legend(handles=h1, fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_robustness.png"), bbox_inches="tight")
    plt.close(fig)


fig_scheme(); fig_tails(); fig_robust()
print("done:", sorted(os.listdir(OUT)))
