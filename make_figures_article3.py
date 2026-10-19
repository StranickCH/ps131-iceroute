"""
Figures and tables for article 3 from finished results (no training; ~5-10 min, CPU).

RUN:
  python make_figures_article3.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --lang en
Uses: fusion_oof.csv; runs\\cvrep_* (latest); runs\\routev3_<tag>_* (latest for main, matrix,
floe3, floe12, thr25_75, thr45_90); AMSR2 maps from the AMSR folder (as collocate_ps131.py); frames from the paths in oof.
OUTPUT: runs\\article3_figs_<lang>\\ :
  fig1_track_map.png      PS131 track and episodes by visibility over AMSR2 (mid-cruise)
  fig2_frames_grid.png    example frames: ice class x visibility
  fig3_confusion.png      confusion matrices AMSR2 / camera / fusion (episode weight = 1)
  fig4_reliability.png    reliability diagrams before/after temperature calibration
  fig5_regret.png         regret by source and planner (95% CI)
  fig6_forest.png         fusion - AMSR2 (bayes) by visibility in all robustness runs
  fig7_route_example.png  example routes in one ice situation
  table1_classification.csv, table2_calibration.csv, table3_routing.csv, table4_sensitivity.csv
Each figure is built independently: if data for one are missing, the others are still saved.
"""
import sys
import json
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T   # noqa: E402

L = {
    "en": dict(ice=["open/very open ≤3/10", "close 4–8/10", "very close ≥9/10"],
               vis=["<1 km", "1–10 km", ">10 km"], vis_t="visibility",
               src={"amsr": "AMSR2", "camera": "camera", "fusion": "fusion", "cam": "camera", "concat": "fusion",
                    "stack": "stacking", "gated": "gated"},
               pl={"hard": "hard labels", "bayes": "expected cost (confusion)", "probs": "network probabilities",
                   "probs_t": "calibrated probabilities", "cvar": "CVaR (α = 0.8)"},
               regret="regret vs oracle, %", true="true", pred="predicted", conf="model confidence",
               acc="accuracy", ideal="perfect calibration", raw="uncalibrated", cal="temperature-scaled",
               diff="fusion − AMSR2, p.p. (expected cost)", better="← fusion better", worse="fusion worse →",
               runs={"main": "main", "main32": "main", "timesplit32": "condition shift (frames)", "dmi70": "DMI chart truth", "matrix": "confusion sampling", "floe3": "smaller floes",
                     "floe12": "larger floes", "thr25_75": "thresholds 0.25/0.75", "thr45_90": "thresholds 0.45/0.90",
                     "timesplit": "condition shift (frames)", "timesplit_matrix": "condition shift (confusion)",
                     "cost148": "class cost 1/4/8", "cost1212": "class cost 1/2/12", "range2": "camera range ×2",
                     "block1": "error block 1 km", "block6": "error block 6 km"},
               all="all", track="PS131: episodes in time order", sic="AMSR2 sea-ice concentration", oracle="oracle",
               route_t="example routes (true field)", lon="longitude", lat="latitude"),
}
COL = {"amsr": "#7f7f7f", "camera": "#4C72B0", "cam": "#4C72B0", "fusion": "#C44E52", "concat": "#C44E52",
       "stack": "#DD8452", "gated": "#8172B2"}
PLC = {"hard": "#bdbdbd", "bayes": "#3182bd", "probs": "#fdae6b", "probs_t": "#e6550d", "cvar": "#31a354"}
VISC = ["#6a51a3", "#9e9ac8", "#fdd0a2"]
SENS = ["main32", "matrix", "floe3", "floe12", "thr25_75", "thr45_90", "cost148", "cost1212", "range2",
        "block1", "block6", "timesplit32", "timesplit_matrix", "dmi70"]


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "figure.dpi": 100})
    return plt


def boot_ep(df, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = df.groupby("episode")[col].mean().values
    if len(g) == 0:
        return np.nan, np.nan
    return tuple(np.percentile([rng.choice(g, len(g)).mean() for _ in range(n)], [2.5, 97.5]))


def latest(pattern):
    c = sorted(T.RUNS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def ep_weights(d):
    return 1.0 / d.groupby("episode")["episode"].transform("size")


# ============================================================
def fig_track(oof, out, t, plt, log):
    import collocate_ps131 as C
    ep = oof[oof["vis_label"] >= 0].groupby("episode").agg(lat=("lat", "median"), lon=("lon", "median"),
                                                          vis=("vis_label", lambda s: int(s.mode().iat[0])),
                                                          time=("time", "min"))
    mid = pd.to_datetime(oof["time"]).median().normalize()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.set_facecolor("#bdbdbd")      # land and no data in grey, so as not to confuse them with 100% ice
    grid = C.load_grid()
    for d in [mid + pd.Timedelta(days=k) for k in (0, 1, -1, 2, -2)]:
        p = next((C.AMSR_DIR / f"asi-AMSR2-n6250-{d:%Y%m%d}-v5.4.{e}" for e in ("hdf", "tif")
                  if (C.AMSR_DIR / f"asi-AMSR2-n6250-{d:%Y%m%d}-v5.4.{e}").exists()), None)
        if p is not None and grid is not None:
            a = C.read_amsr(p).astype(float)
            la, lo = grid["lat"], grid["lon"]
            m = (la > 68) & (la < 86) & (lo > -35) & (lo < 30)
            rr, cc = np.where(m)
            sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
            raw = a[sl]
            scale = 100.0 if np.nanmax(np.where(raw <= 100, raw, np.nan)) > 1.5 else 1.0
            v = raw / scale
            v = np.where((v >= 0) & (v <= 1.0001), v, np.nan)      # land/flags -> NaN (grey background)
            pc = ax.pcolormesh(lo[sl], la[sl], np.ma.masked_invalid(v), cmap="Blues_r", vmin=0, vmax=1,
                               shading="auto", rasterized=True)
            cb = fig.colorbar(pc, ax=ax, shrink=0.75)
            cb.set_label(f"{t['sic']} ({d:%d.%m.%Y})")
            break
    # frames exist only within hourly episodes -> the order of the episodes in time is shown with a dashed line
    epo = ep.sort_values("time")
    ax.plot(epo["lon"], epo["lat"], "--", lw=0.8, color="#e6550d", alpha=0.9, label=t["track"], zorder=2)
    for v in range(3):
        s = ep[ep["vis"] == v]
        ax.scatter(s["lon"], s["lat"], s=28, color=VISC[v], edgecolor="k", lw=0.5, zorder=3,
                   label=f"{t['vis_t']} {t['vis'][v]} ({len(s)})")
    ax.set_xlabel(t["lon"])
    ax.set_ylabel(t["lat"])
    lo_, la_ = oof["lon"].dropna(), oof["lat"].dropna()
    ax.set_xlim(lo_.min() - 4, lo_.max() + 4)
    ax.set_ylim(la_.min() - 1, la_.max() + 1)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig1_track_map.png")
    plt.close(fig)


def fig_frames(oof, out, t, plt, log):
    from PIL import Image
    d = oof[(oof["ice_label"] >= 0) & (oof["vis_label"] >= 0)]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 5.8))
    for i in range(3):
        for v in range(3):
            ax = axes[i, v]
            ax.axis("off")
            s = d[(d["ice_label"] == i) & (d["vis_label"] == v)]
            if i == 0:
                ax.set_title(f"{t['vis_t']} {t['vis'][v]}", fontsize=9)
            if s.empty:
                ax.text(0.5, 0.5, "—", ha="center", va="center")
                continue
            # frame from a "typical" episode: middle of the episode with the most frames
            e = s["episode"].value_counts().index[rng.integers(min(3, s["episode"].nunique()))]
            r = s[s["episode"] == e].iloc[len(s[s["episode"] == e]) // 2]
            try:
                im = Image.open(r["path"])
                im.thumbnail((640, 480))
                ax.imshow(im)
            except Exception as ex:
                ax.text(0.5, 0.5, f"no file\n{Path(str(r['path'])).name}", ha="center", va="center", fontsize=6)
            if v == 0:
                ax.text(-0.04, 0.5, t["ice"][i], transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_frames_grid.png")
    plt.close(fig)


def fig_confusion(oof, out, t, plt, log):
    d = oof[oof["ice_label"] >= 0].copy()
    d["w"] = ep_weights(d)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, m in zip(axes, ("amsr", "cam", "concat")):
        M = np.zeros((3, 3))
        for (a, b), w in d.groupby(["ice_label", f"pred_{m}"])["w"].sum().items():
            if b >= 0:
                M[int(a), int(b)] += w
        M = M / M.sum(1, keepdims=True)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] > 0.55 else "k")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["≤3", "4–8", "≥9"])
        ax.set_yticklabels(["≤3", "4–8", "≥9"])
        ax.set_xlabel(t["pred"] + " (/10)")
        ax.set_title(t["src"][m])
    axes[0].set_ylabel(t["true"] + " (/10)")
    fig.tight_layout()
    fig.savefig(out / "fig3_confusion.png")
    plt.close(fig)


def temp_cv(oof, m):
    """Temperature fitted on OTHER episodes (5 groups) -> calibrated OOF probabilities."""
    import route_ps131_v3 as V
    d = oof[oof["ice_label"] >= 0].copy()
    P = np.nan_to_num(d[[f"p_{m}_{k}" for k in range(3)]].values, nan=1 / 3)
    y = d["ice_label"].values.astype(int)
    eps = d["episode"].unique()
    grp = {e: i % 5 for i, e in enumerate(np.random.default_rng(0).permutation(eps))}
    g = d["episode"].map(grp).values
    w = ep_weights(d).values
    Pc = P.copy()
    Ts = []
    for k in range(5):
        tr = g != k
        Tk = V.fit_temperature(P[tr], y[tr], w[tr])
        Ts.append(Tk)
        Pc[~tr] = V.apply_temperature(P[~tr], Tk)
    return y, P, Pc, w, float(np.mean(Ts))


def calib_stats(y, P, w, n_bins=10):
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, pts = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            ww = w[s]
            a, c = np.average(acc[s], weights=ww), np.average(conf[s], weights=ww)
            ece += ww.sum() / w.sum() * abs(a - c)
            if ww.sum() / w.sum() >= 0.02:        # plot only bins with >=2% of the weight (otherwise noise of 1-2 episodes)
                pts.append((c, a, ww.sum()))
    brier = float(np.average(((P - np.eye(3)[y]) ** 2).sum(1), weights=w))
    nll = float(np.average(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)), weights=w))
    return ece, brier, nll, float(np.average(conf, weights=w)), float(np.average(acc, weights=w)), pts


def fig_reliability(oof, out, t, plt, log):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for m in ("amsr", "cam", "concat", "stack", "gated"):
        if f"p_{m}_0" not in oof:
            continue
        y, P, Pc, w, Tm = temp_cv(oof, m)
        for ax, PP, tag in ((axes[0], P, "raw"), (axes[1], Pc, "cal")):
            ece, br, nll, mc, ma, pts = calib_stats(y, PP, w)
            rows.append({"model": t["src"][m], "calibration": t[tag], "ECE": ece, "Brier": br, "NLL": nll,
                         "mean_conf": mc, "accuracy": ma, "T": Tm if tag == "cal" else 1.0})
            if m in ("amsr", "cam", "concat"):
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3.5, color=COL[m], label=f"{t['src'][m]}: ECE {ece:.3f}")
    for ax, tag in zip(axes, ("raw", "cal")):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label=t["ideal"])
        ax.set_xlim(0.3, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(t["conf"])
        ax.set_title(t[tag])
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel(t["acc"])
    fig.tight_layout()
    fig.savefig(out / "fig4_reliability.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "table2_calibration.csv", index=False, float_format="%.3f", encoding="utf-8-sig")


def table_classification(out, t, log):
    d = latest("cvrep_*")
    if d is None or not (d / "cv_repeat.csv").exists():
        log("  [table 1] no runs\\cvrep_* - skipped")
        return
    R = pd.read_csv(d / "cv_repeat.csv")
    rows = []
    for m in ("amsr", "cam", "concat"):
        rows.append({"model": t["src"][m], "macro_F1_mean": R[f"f1_{m}"].mean(), "macro_F1_sd": R[f"f1_{m}"].std(),
                     "macro_F1_min": R[f"f1_{m}"].min(), "macro_F1_max": R[f"f1_{m}"].max(),
                     "recall_close_mean": R[f"recall1_{m}"].mean(), "recall_close_sd": R[f"recall1_{m}"].std(),
                     "repeats": len(R)})
    for a, b in (("concat", "amsr"), ("concat", "cam"), ("cam", "amsr")):
        dd = R[f"f1_{a}"] - R[f"f1_{b}"]
        rows.append({"model": f"{t['src'][a]} − {t['src'][b]}", "macro_F1_mean": dd.mean(), "macro_F1_sd": dd.std(),
                     "macro_F1_min": dd.min(), "macro_F1_max": dd.max(), "repeats": int((dd > 0).sum())})
    pd.DataFrame(rows).to_csv(out / "table1_classification.csv", index=False, float_format="%.3f",
                              encoding="utf-8-sig")
    log(f"  table 1 from {d.name}")


def load_trials(tag):
    d = latest(f"routev3_{tag}_2*")
    if d is None or not (d / "trials.csv").exists():
        return None, None
    return pd.read_csv(d / "trials.csv"), d.name


def fig_regret(out, t, plt, log):
    df, name = load_trials("main32")
    if df is None:
        log("  [fig. 5] no runs\\routev3_main_* - skipped")
        return
    rows = []
    srcs, pls = ["amsr", "fusion", "camera"], [p for p in ("hard", "bayes", "probs", "probs_t", "cvar") if p in set(df["planner"])]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for k, (ax, vsel) in enumerate(zip(axes, [None, 0, 1, 2])):
        dv = df if vsel is None else df[df["vis"] == vsel]
        x = np.arange(len(srcs))
        w = 0.8 / len(pls)
        for j, pl in enumerate(pls):
            means, los, his = [], [], []
            for s in srcs:
                ss = dv[(dv["source"] == s) & (dv["planner"] == pl)]
                m = 100 * ss["regret"].mean()
                lo, hi = boot_ep(ss.assign(r=100 * ss["regret"]), "r")
                means.append(m)
                los.append(m - lo)
                his.append(hi - m)
                rows.append({"visibility": t["all"] if vsel is None else t["vis"][vsel], "source": t["src"][s],
                             "planner": t["pl"][pl], "regret_%": m, "ci_lo": lo, "ci_hi": hi,
                             "heavy_km": ss["heavy_km"].mean(), "length_km": ss["dist_km" if "dist_km" in ss else "len_km"].mean(),
                             "episodes": ss["episode"].nunique()})
            ax.bar(x + (j - (len(pls) - 1) / 2) * w, means, w, yerr=[los, his], color=PLC[pl], capsize=1.5,
                   error_kw={"lw": 0.7}, label=t["pl"][pl])
        ax.set_xticks(x)
        ax.set_xticklabels([t["src"][s] for s in srcs])
        ax.set_title(t["all"] if vsel is None else f"{t['vis_t']} {t['vis'][vsel]}")
    axes[0].set_ylabel(t["regret"])
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig5_regret.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "table3_routing.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    log(f"  fig. 5 / table 3 from {name}")


def fig_forest(out, t, plt, log):
    rows = []
    for tag in SENS:
        df, name = load_trials(tag)
        if df is None:
            continue
        w = df[df["planner"] == "bayes"].pivot_table(index=["episode", "vis", "trial"], columns="source",
                                                     values="regret").reset_index()
        w["d"] = 100 * (w["fusion"] - w["amsr"])
        w["d_hb"] = np.nan
        for vsel in (None, 0, 1, 2):
            s = w if vsel is None else w[w["vis"] == vsel]
            lo, hi = boot_ep(s, "d")
            rows.append({"run": t["runs"][tag], "tag": tag, "visibility": "all" if vsel is None else vsel,
                         "diff_pp": s["d"].mean(), "ci_lo": lo, "ci_hi": hi})
        # bayes - hard for fusion (for the table)
        h = df[df["source"] == "fusion"].pivot_table(index=["episode", "vis", "trial"], columns="planner",
                                                     values="regret").reset_index()
        h["d"] = 100 * (h["bayes"] - h["hard"])
        lo, hi = boot_ep(h, "d")
        rows.append({"run": t["runs"][tag], "tag": tag, "visibility": "fusion bayes−hard",
                     "diff_pp": h["d"].mean(), "ci_lo": lo, "ci_hi": hi})
    if not rows:
        log("  [fig. 6] no routev3 runs - skipped")
        return
    R = pd.DataFrame(rows)
    R.to_csv(out / "table4_sensitivity.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    F = R[R["visibility"] != "fusion bayes−hard"]
    tags = [x for x in SENS if x in set(F["tag"])]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(tags) + 1.6))
    vis_keys = ["all", 0, 1, 2]
    names = [t["all"]] + t["vis"]
    cols = ["k"] + VISC
    off = np.linspace(-0.27, 0.27, 4)
    for j, (vk, nm, c) in enumerate(zip(vis_keys, names, cols)):
        s = F[F["visibility"].astype(str) == str(vk)].set_index("tag").reindex(tags)
        y = np.arange(len(tags)) + off[j]
        ax.errorbar(s["diff_pp"], y, xerr=[s["diff_pp"] - s["ci_lo"], s["ci_hi"] - s["diff_pp"]], fmt="o",
                    ms=4, color=c, mec="k" if c != "k" else c, mew=0.4, capsize=2, lw=0.9, label=nm)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([t["runs"][x] for x in tags])
    ax.invert_yaxis()
    ax.set_xlabel(t["diff"])
    ax.text(0.02, 1.01, t["better"], transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
    ax.text(0.98, 1.01, t["worse"], transform=ax.transAxes, fontsize=7, ha="right", va="bottom")
    ax.legend(fontsize=7, title=t["vis_t"], title_fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig6_forest.png")
    plt.close(fig)
    log(f"  fig. 6 / table 4: runs {len(tags)}")


def run_route_path(V, R, true_cls, land, start, goal, far, near, r):
    true_cost = np.where(land, np.inf, R.CLASS_COST[true_cls])
    belief = far.astype(float).copy()
    seen = np.zeros(true_cls.shape, bool)
    pos, path, steps, route, k, total = start, [start], 0, None, 1, 0.0
    while pos != goal and steps < 4 * R.N:
        new = None
        if near is not None:
            new = R.disk(pos, r) & ~seen
            if new.any():
                belief[new] = near[new]
                seen |= new
        if route is None or steps % R.REPLAN_EVERY == 0 or (new is not None and new.any()):
            route = R.plan(np.where(land, 1e6, belief), pos, goal)
            k = 1
        nxt = tuple(route[k]) if k < len(route) else goal
        k += 1
        sl = np.sqrt(2) if nxt[0] != pos[0] and nxt[1] != pos[1] else 1.0
        total += sl * true_cost[nxt]
        pos = nxt
        path.append(pos)
        steps += 1
    return np.array(path), total


def fig_route_example(oof, out, t, plt, log):
    import route_ice_sources_ps131 as R
    import route_ps131_v3 as V
    eps = (oof[oof["vis_label"] == 2].groupby("episode")
           .agg(time=("time", "min"), lat=("lat", "median"), lon=("lon", "median"),
                ice1=("ice_label", lambda s: (s == 1).mean())).reset_index().sort_values("ice1", ascending=False))
    if eps.empty:
        return
    e = eps.iloc[0]
    excl = {e["episode"]}
    sub = oof[~oof["episode"].isin(excl)]
    conf = R.confusion_tables(sub, "pred_concat", lambda *a: None)
    ct = {p: R.cost_tables(conf, 2, p) for p in ("hard", "bayes")}
    pool_t = V.FramePool(oof, ["ice_label", "vis_label"], excl)
    pool_ta = V.FramePool(oof, ["ice_label", "vis_label", "pred_amsr"], excl)
    probs = {s: np.nan_to_num(oof[[f"p_{m}_{k}" for k in range(3)]].values, nan=1 / 3)
             for s, m in (("amsr", "amsr"), ("fusion", "concat"))}
    preds = {"amsr": oof["pred_amsr"].values, "fusion": oof["pred_concat"].values}
    macro = R.macro_sic_from_amsr(e["lat"], e["lon"], pd.to_datetime(e["time"]), log)
    rng = np.random.default_rng(11)
    if macro is None:
        macro = R.macro_sic_synthetic(rng)
    sic, land = macro
    best = None
    for _ in range(12):          # a scenario where the planner choice is visible
        tc = V.true_classes(sic, land, rng, 6.0, (0.35, 0.85))
        ends = R.pick_endpoints(land, rng)
        if ends is None:
            continue
        am, _ = V.resample_obs(tc, 6, rng, pool_t, None, probs["amsr"], preds["amsr"], 2)
        fu, _ = V.resample_obs(tc, 3, rng, pool_ta, None, probs["fusion"], preds["fusion"], 2, cond=am)
        r = R.CAM_RANGE[2]
        po, co = run_route_path(V, R, tc, land, *ends, R.CLASS_COST[tc], None, r)
        ph, ch = run_route_path(V, R, tc, land, *ends, ct["hard"]["amsr"][am], ct["hard"]["fusion"][am, fu], r)
        pb, cb = run_route_path(V, R, tc, land, *ends, ct["bayes"]["amsr"][am], ct["bayes"]["fusion"][am, fu], r)
        pa, ca = run_route_path(V, R, tc, land, *ends, ct["bayes"]["amsr"][am], None, r)
        score = (ch - cb) / co
        if best is None or score > best[0]:
            best = (score, tc, am, land, {t["oracle"]: (po, co), f"AMSR2, {t['pl']['bayes']}": (pa, ca),
                                          f"fusion, {t['pl']['hard']}": (ph, ch),
                                          f"fusion, {t['pl']['bayes']}": (pb, cb)})
    _, tc, am, land, paths = best
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1f4e79", "#9ecae1", "#f7f7f7"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    cols = ["k", "#7f7f7f", "#fdae6b", "#C44E52"]
    for ax, img, ttl in ((axes[0], tc, t["route_t"]), (axes[1], am, "AMSR2")):
        ax.imshow(np.where(land, np.nan, img), cmap=cmap, vmin=0, vmax=2)
        for (lbl, (p, c)), cc in zip(paths.items(), cols):
            reg = c / paths[t["oracle"]][1] - 1
            ax.plot(p[:, 1], p[:, 0], color=cc, lw=1.6, label=f"{lbl}" + ("" if lbl == t["oracle"] else f" (+{100 * reg:.0f}%)"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=9)
    from matplotlib.patches import Patch
    h1, l1 = axes[0].get_legend_handles_labels()
    h2 = [Patch(facecolor=c, edgecolor="k", lw=0.4) for c in ("#1f4e79", "#9ecae1", "#f7f7f7")]
    fig.legend(h1 + h2, l1 + t["ice"], fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig7_route_example.png")
    plt.close(fig)
    log(f"  fig. 7: episode {e['episode']}")


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--lang", default="en", choices=["en"])
    ap.add_argument("--only", default="", help="e.g. 1,5,6 - only these figures")
    args = ap.parse_args()
    t = L[args.lang]
    out = T.RUNS / f"article3_figs_{args.lang}"
    out.mkdir(parents=True, exist_ok=True)
    log = T.Log(out / "log.txt")
    plt = plt_setup()
    oof = pd.read_csv(args.oof, low_memory=False)
    jobs = [("1", "track map", lambda: fig_track(oof, out, t, plt, log)),
            ("2", "frames", lambda: fig_frames(oof, out, t, plt, log)),
            ("3", "confusion matrices", lambda: fig_confusion(oof, out, t, plt, log)),
            ("4", "calibration + table 2", lambda: fig_reliability(oof, out, t, plt, log)),
            ("t1", "table 1", lambda: table_classification(out, t, log)),
            ("5", "regret + table 3", lambda: fig_regret(out, t, plt, log)),
            ("6", "forest + table 4", lambda: fig_forest(out, t, plt, log)),
            ("7", "example routes", lambda: fig_route_example(oof, out, t, plt, log))]
    only = set(args.only.split(",")) - {""}
    for key, name, fn in jobs:
        if only and key not in only:
            continue
        log(f"[{key}] {name}")
        try:
            fn()
        except Exception as e:
            log(f"  ERROR in '{name}': {e}")
            (out / f"error_{key}.txt").write_text(traceback.format_exc(), encoding="utf-8")
    log(f"DONE: {out}")


if __name__ == "__main__":
    main()
