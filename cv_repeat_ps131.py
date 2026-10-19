"""
Checks for article 3 (items 2 and 3 of the plan):
  A) REPEATED cross-validation by episode: R different splits into 5 folds -> ice macro-F1
     for AMSR2 / camera / fusion (concat) as mean ± spread, and the fusion - AMSR2 difference over the repeats.
  B) CALIBRATION of the probabilities (ECE, Brier, reliability diagram) for the same models
     + for stack/gated from fusion_oof.csv. Needed to justify the planner "by model probabilities".
The ConvNeXt features are taken from the cache (ps131_features), no GPU needed. Time: ~5-15 min.

RUN:     python cv_repeat_ps131.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --repeats 10
OUTPUT:  runs\\cvrep_<date>\\ : log.txt, cv_repeat.csv, cv_repeat_summary.txt, calibration.txt, reliability.png
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T   # noqa: E402
import train_ps131_cv as CV      # noqa: E402
import fusion_ps131 as F         # noqa: E402

METHODS = ["amsr", "cam", "concat"]


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    m = (y >= 0) & (p >= 0)
    return f1_score(y[m], p[m], average="macro", labels=[0, 1, 2])


def ep_weighted_macro_f1(d, pcol):
    """macro-F1 with episode weight = 1 (the frames of an episode share the weight) - more robust to long episodes."""
    from sklearn.metrics import f1_score
    w = 1.0 / d.groupby("episode")["episode"].transform("size")
    return f1_score(d["ice_label"], d[pcol], average="macro", labels=[0, 1, 2], sample_weight=w)


def calib(y, P, n_bins=10):
    """ECE (on the maximum probability), Brier, mean confidence vs accuracy."""
    m = (y >= 0) & np.isfinite(P).all(1)
    y, P = y[m], P[m]
    conf = P.max(1)
    pred = P.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            ece += s.mean() * abs(acc[s].mean() - conf[s].mean())
            rows.append((lo, hi, int(s.sum()), conf[s].mean(), acc[s].mean()))
    onehot = np.eye(3)[y]
    brier = float(((P - onehot) ** 2).sum(1).mean())
    return {"ece": float(ece), "brier": brier, "mean_conf": float(conf.mean()), "acc": float(acc.mean()), "bins": rows}


def run_once(df, X, Xs, seed, log):
    folds, _ = CV.make_folds(df, 5, 3000, seed)
    fold = df["episode"].map(folds).values
    y = df["ice_label"].values
    P = {n: np.full((len(df), 3), np.nan) for n in METHODS}
    for f in range(5):
        te = fold == f
        trm = (~te) & (y >= 0)
        Xp = CV.fold_pca(X, ~te, 64)
        P["cam"][te] = F.proba3(F.logreg(X[trm], y[trm], 0.01), X[te])
        P["amsr"][te] = F.proba3(F.logreg(Xs[trm], y[trm], 1.0), Xs[te])
        Xc = np.hstack([Xp, Xs])
        P["concat"][te] = F.proba3(F.logreg(Xc[trm], y[trm], 0.1), Xc[te])
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="fusion_oof.csv (for the calibration of stack/gated)")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    a2 = type("A", (), dict(max_dt=180, arch="convnext_tiny", img="448x336", workers=0))()
    run_dir = T.RUNS / f"cvrep_{datetime.now():%Y%m%d_%H%M}"
    log = T.Log(run_dir / "log.txt")
    log(f"Repeated CV + calibration | {vars(args)}")
    df = CV.load_all(a2.max_dt)
    X = CV.extract_features(df, a2, log)
    Xs = F.mat(df, F.SIC)
    lab = df["ice_label"].values >= 0
    log(f"Frames with an ice label {lab.sum()}, episodes {df.loc[lab, 'episode'].nunique()}")

    rows, Pfirst = [], None
    t0 = time.time()
    for r in range(args.repeats):
        P = run_once(df, X, Xs, 1000 + r, log)
        if Pfirst is None:
            Pfirst = P
        d = df.loc[lab, ["episode", "ice_label"]].copy()
        row = {"repeat": r}
        for n in METHODS:
            d[f"pred_{n}"] = P[n][lab].argmax(1)
            row[f"f1_{n}"] = macro_f1(d["ice_label"].values, d[f"pred_{n}"].values)
            row[f"f1ep_{n}"] = ep_weighted_macro_f1(d, f"pred_{n}")
            row[f"recall1_{n}"] = float((d.loc[d["ice_label"] == 1, f"pred_{n}"] == 1).mean())
        rows.append(row)
        el = time.time() - t0
        log(f"  repeat {r + 1}/{args.repeats}: macro-F1 amsr {row['f1_amsr']:.3f}  cam {row['f1_cam']:.3f}  "
            f"concat {row['f1_concat']:.3f}   (~{el / (r + 1) * (args.repeats - r - 1) / 60:.1f} min left)")
    R = pd.DataFrame(rows)
    R.to_csv(run_dir / "cv_repeat.csv", index=False, float_format="%.4f")

    out = ["REPEATED CV BY EPISODE (ice, 3 classes): mean ± SD [min, max] over the repeats"]
    for key, title in (("f1", "macro-F1 (frames)"), ("f1ep", "macro-F1 (episode weight = 1)"),
                       ("recall1", "recall of close ice 4-8/10")):
        out.append(f"\n{title}")
        for n in METHODS:
            v = R[f"{key}_{n}"]
            out.append(f"  {n:7s} {v.mean():.3f} ± {v.std():.3f}  [{v.min():.3f}, {v.max():.3f}]")
    for a, b in (("concat", "amsr"), ("concat", "cam"), ("cam", "amsr")):
        dlt = R[f"f1_{a}"] - R[f"f1_{b}"]
        out.append(f"\n{a} - {b}: macro-F1 {dlt.mean():+.3f} ± {dlt.std():.3f}; >0 in {int((dlt > 0).sum())}/{len(dlt)} repeats")
    for l in out:
        log(l)
    (run_dir / "cv_repeat_summary.txt").write_text("\n".join(out), encoding="utf-8")

    # ---------- calibration ----------
    cal_out = ["CALIBRATION of the out-of-fold probabilities (ice). ECE - expected calibration error (0 = perfect)"]
    y = df["ice_label"].values
    curves = {}
    for n in METHODS:
        c = calib(y, Pfirst[n])
        curves[n] = c
        cal_out.append(f"  {n:7s} (repeat 1)      ECE {c['ece']:.3f}  Brier {c['brier']:.3f}  "
                       f"mean confidence {c['mean_conf']:.2f} vs accuracy {c['acc']:.2f}")
    oof = pd.read_csv(args.oof, low_memory=False)
    oy = oof["ice_label"].values
    for n in ("amsr", "cam", "concat", "stack", "gated"):
        cols = [f"p_{n}_{k}" for k in range(3)]
        if all(c in oof for c in cols):
            c = calib(oy, oof[cols].values)
            curves[f"{n} (fusion_oof)"] = c
            cal_out.append(f"  {n:7s} (fusion_oof.csv) ECE {c['ece']:.3f}  Brier {c['brier']:.3f}  "
                           f"mean confidence {c['mean_conf']:.2f} vs accuracy {c['acc']:.2f}")
    cal_out.append("\nConfidence > accuracy = the model is overconfident; then a planner using the probabilities "
                   "needs recalibration (temperature) or confusion matrices.")
    for l in cal_out:
        log(l)
    (run_dir / "calibration.txt").write_text("\n".join(cal_out), encoding="utf-8")
    (run_dir / "calibration.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "bins"}
                                                          for k, v in curves.items()}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for n, c in curves.items():
            if "fusion_oof" in n and not n.startswith(("stack", "gated")):
                continue
            b = np.array([(r[3], r[4]) for r in c["bins"]])
            ax.plot(b[:, 0], b[:, 1], "o-", ms=4, label=f"{n}: ECE {c['ece']:.3f}")
        ax.set_xlabel("model confidence (max. probability)")
        ax.set_ylabel("fraction correct")
        ax.set_title("Reliability diagram, ice")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run_dir / "reliability.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"[plot skipped: {e}]")
    log(f"DONE: {run_dir}")


if __name__ == "__main__":
    main()
