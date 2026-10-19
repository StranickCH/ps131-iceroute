"""
Fusion of camera + AMSR2 taking the imaging conditions into account (article 3). Minutes, no GPU needed
(the ConvNeXt features are taken from the cache ps131_features\\, computed by train_ps131_cv.py --mode probe).

Idea: the camera is good in clear weather and at the ice edge, AMSR2 - in open water and in compact pack.
Trust in the camera should depend on what the ship knows IN REAL TIME:
  - frame quality (sharpness, contrast, brightness...)  - computed from the frame itself
  - ERA5 / shipborne weather station (dew point, cloud cover, precipitation)
  - NOT the observer's visibility (not available in real time - used only for the analysis)

Compared models (all in the same 5 folds by episode as in train_ps131_cv.py):
  amsr      - AMSR2, learned thresholds (logistic regression on SIC)
  cam       - camera: ConvNeXt features + logistic regression
  concat    - early fusion: PCA-64 of the camera features + SIC (as in probe)
  stack     - late fusion: meta-model on the camera and AMSR2 probabilities
  gated     - late fusion with GATES: the meta-model also sees frame quality/weather
              and their products with the camera probabilities -> the camera weight changes with conditions
The base-model probabilities used to train the meta-model are out-of-fold within the training part
(nested cross-validation by episode), no leakage.

RUN:      python fusion_ps131.py
OUTPUT:   runs\\fusion_<date>\\ : log.txt, fusion_metrics.txt/.json, fusion_oof.csv (per-frame probabilities -
          input to the routing and DRL experiments), fusion_by_visibility.png
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T   # noqa: E402
import train_ps131_cv as CV      # noqa: E402

QUAL = ["brightness", "contrast", "sharpness", "white_frac", "sat_mean", "top_bottom"]
MET = ["dpd", "lcc", "tcc", "tp", "wind"]
SIC = ["sic_px", "sic_3x3", "sic_7x7"]


def mat(df, cols, log_sharp=True):
    have = [c for c in cols if c in df]
    M = df[have].astype(float).copy()
    if log_sharp and "sharpness" in M:
        M["sharpness"] = np.log10(M["sharpness"].clip(lower=1e-3))
    return M.fillna(M.median()).values


def logreg(Xtr, ytr, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(Xtr, ytr)


def proba3(model, X):
    p = np.zeros((len(X), 3))
    p[:, model.classes_] = model.predict_proba(X)
    return p


def inner_oof(X, y, groups, n_inner, C, seed):
    """Out-of-fold probabilities within the training part (by episode)."""
    from sklearn.model_selection import GroupKFold
    P = np.zeros((len(y), 3))
    for tr, te in GroupKFold(n_splits=n_inner).split(X, y, groups):
        P[te] = proba3(logreg(X[tr], y[tr], C), X[te])
    return P


def meta_features(p_cam, p_amsr, Q, gated):
    lc = np.log(np.clip(p_cam, 1e-4, 1))
    la = np.log(np.clip(p_amsr, 1e-4, 1))
    F = [lc, la]
    if gated:
        F += [Q] + [lc * Q[:, [j]] for j in range(Q.shape[1])]
    return np.hstack(F)


def main():
    args = type("A", (), dict(max_dt=180, folds=5, seed=42, arch="convnext_tiny", img="448x336", workers=0))()
    run_dir = T.RUNS / f"fusion_{datetime.now():%Y%m%d_%H%M}"
    log = T.Log(run_dir / "log.txt")
    df = CV.load_all(args.max_dt)
    df["amsr_ice3"] = df["ice_class"].map(T.AMSR_TO_3).fillna(-1).astype(int)
    folds, _ = CV.make_folds(df, args.folds, 3000, args.seed)
    df["fold"] = df["episode"].map(folds)
    X = CV.extract_features(df, args, log)
    Xs, Xq, Xm = mat(df, SIC), mat(df, QUAL), mat(df, MET)
    Q = np.hstack([Xq, Xm])
    Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)       # scale only; the weights themselves are learned by the meta-model
    y = df["ice_label"].values
    ep = df["episode"].values
    log(f"Frames with an ice label: {(y >= 0).sum()}, episodes {df.loc[y >= 0, 'episode'].nunique()}; "
        f"condition features: {QUAL + MET}")

    names = ["amsr", "cam", "concat", "stack", "gated"]
    P = {n: np.full((len(df), 3), np.nan) for n in names}
    for f in range(args.folds):
        te = (df["fold"] == f).values
        trm = (~te) & (y >= 0)
        Xp = CV.fold_pca(X, ~te, 64)
        # base models on the whole training part -> prediction on the test
        m_cam, m_amsr = logreg(X[trm], y[trm], 0.01), logreg(Xs[trm], y[trm], 1.0)
        P["cam"][te], P["amsr"][te] = proba3(m_cam, X[te]), proba3(m_amsr, Xs[te])
        Xc = np.hstack([Xp, Xs])
        P["concat"][te] = proba3(logreg(Xc[trm], y[trm], 0.1), Xc[te])
        # meta-models: trained on the nested OOF probabilities of the training part
        idx = np.where(trm)[0]
        pc_in = inner_oof(X[idx], y[idx], ep[idx], 4, 0.01, args.seed)
        pa_in = inner_oof(Xs[idx], y[idx], ep[idx], 4, 1.0, args.seed)
        for name, gated in (("stack", False), ("gated", True)):
            meta = logreg(meta_features(pc_in, pa_in, Q[idx], gated), y[idx], 0.1)
            P[name][te] = proba3(meta, meta_features(P["cam"][te], P["amsr"][te], Q[te], gated))
        log(f"  fold {f + 1}/{args.folds} done")

    for n in names:
        df[f"pred_{n}"] = np.where(np.isnan(P[n][:, 0]), -1, np.nan_to_num(P[n]).argmax(1)).astype(int)
        for k in range(3):
            df[f"p_{n}_{k}"] = P[n][:, k]

    out, res = [], {}
    log("\n" + "=" * 70 + "\nICE, out-of-fold over all episodes")
    titles = {"amsr": "AMSR2 (learned thresholds)", "cam": "camera", "concat": "early fusion (camera PCA + SIC)",
              "stack": "late fusion (stacking)", "gated": "fusion gated by imaging conditions"}
    for n in names:
        r, lines = CV.report_task(f"ICE: {titles[n]}", df, "ice_label", f"pred_{n}", T.ICE_NAMES, log)
        res[n] = r
        out += lines

    log("\n" + "=" * 70 + "\nPAIRWISE macro-F1 DIFFERENCES (95% CI, episode bootstrap)")
    out.append("\nPAIRWISE macro-F1 DIFFERENCES")
    for a, b in (("gated", "amsr"), ("gated", "cam"), ("gated", "stack"), ("gated", "concat"),
                 ("stack", "amsr"), ("concat", "amsr")):
        d, lo, hi, p0 = CV.bootstrap_diff(df, "ice_label", f"pred_{a}", f"pred_{b}")
        line = (f"  {a:7s} - {b:7s} {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                f"{'SIGNIFICANT' if lo > 0 or hi < 0 else 'not significant'}; share <= 0: {p0:.3f}")
        log(line)
        out.append(line)
        res[f"diff {a}-{b}"] = {"diff": d, "ci95": [lo, hi], "p_le0": p0}

    log("\n" + "=" * 70 + "\nBY OBSERVER VISIBILITY CLASS (analysis only, not a model input)")
    out.append("\nBY VISIBILITY: macro-F1 / recall of close ice")
    vis_rows = []
    for v, vn in enumerate(T.VIS_NAMES):
        s = df[(df["vis_label"] == v) & (df["ice_label"] >= 0)]
        row = {"visibility": vn, "episodes": s["episode"].nunique(), "frames": len(s)}
        for n in names:
            m = T.metrics(s["ice_label"].values, s[f"pred_{n}"].values)
            row[n] = (m.get("macro_f1", np.nan), m["recall"][1] if m.get("n") else np.nan)
        vis_rows.append(row)
        line = f"  {vn:14s} ep. {row['episodes']:2d} | " + "  ".join(
            f"{n} {row[n][0]:.2f}/{row[n][1]:.2f}" for n in names)
        log(line)
        out.append(line)

    # camera weight: how far gated deviates from AMSR2 towards the camera, by visibility
    agree = []
    for v, vn in enumerate(T.VIS_NAMES):
        s = df[(df["vis_label"] == v) & (df["ice_label"] >= 0)]
        dc = (s["pred_gated"] == s["pred_cam"]).mean()
        da = (s["pred_gated"] == s["pred_amsr"]).mean()
        agree.append((vn, dc, da))
        line = f"  {vn:14s} gated agrees with the camera {dc:.0%}, with AMSR2 {da:.0%}"
        log(line)
        out.append(line)

    (run_dir / "fusion_metrics.txt").write_text("\n".join(out), encoding="utf-8")
    (run_dir / "fusion_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                                 encoding="utf-8")
    keep = ["path", "episode", "time", "lat", "lon", "fold", "vis_label", "ice_label", "sic_3x3"] + \
           [c for c in df.columns if c.startswith(("pred_", "p_"))]
    df[[c for c in keep if c in df]].to_csv(run_dir / "fusion_oof.csv", index=False, float_format="%.4f")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(T.VIS_NAMES))
        w = 0.16
        colors = {"amsr": "#8c8c8c", "cam": "#4C72B0", "concat": "#55A868", "stack": "#DD8452", "gated": "#C44E52"}
        for i, n in enumerate(names):
            axes[0].bar(x + (i - 2) * w, [r[n][0] for r in vis_rows], w, label=titles[n], color=colors[n])
            axes[1].bar(x + (i - 2) * w, [r[n][1] for r in vis_rows], w, color=colors[n])
        for ax, t in zip(axes, ("macro-F1, ice", "recall: close ice 4-8/10 (ice edge)")):
            ax.set_xticks(x)
            ax.set_xticklabels([f"{vn}\n({r['episodes']} ep.)" for vn, r in zip(T.VIS_NAMES, vis_rows)])
            ax.set_title(t)
            ax.set_ylim(0, 1)
            ax.set_xlabel("visibility from the observer")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(run_dir / "fusion_by_visibility.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"[plot skipped: {e}]")
    log(f"DONE: {run_dir}")


if __name__ == "__main__":
    main()
