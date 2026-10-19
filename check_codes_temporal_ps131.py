"""
Two checks for article 3 (minutes, no GPU needed; ConvNeXt features from the cache):

A) WMO CODES. A check against WMO-No.306 (tables 4377 VV, 4677 ww, 0639 ci) showed that the visibility and weather
   codes are decoded correctly, while the ice code (0639) has AMBIGUOUS values that the labelling script mapped to classes by convention:
     1 = ship in an open lead >1 n mile OR in fast ice with no visible boundary  -> mapped to "<=3/10"
     6 = strips and patches of pack ice with open water between              -> mapped to ">=9/10"
     7 = strips of close/very close pack ice with lesser concentration between -> mapped to ">=9/10"
     8, 9 = fast ice (with very open / close pack ice to seaward)          -> mapped to ">=9/10"
   The script counts the frames/episodes with such codes and repeats the classification WITHOUT them.
B) TEMPORAL HOLD-OUT (instead of a second cruise): training on the first half of the cruise, test on the second
   (other regions and another melt phase), and vice versa. Shows whether the model transfers to "new" conditions.

RUN:     python check_codes_temporal_ps131.py
OUTPUT:  runs\\checks_<date>\\ : log.txt, codes_ci.csv, codes_vv.csv, codes_ww.csv, checks_summary.txt
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T   # noqa: E402
import train_ps131_cv as CV      # noqa: E402
import fusion_ps131 as F         # noqa: E402
import cv_repeat_ps131 as CR     # noqa: E402

AMBIG_CI = {1, 6, 7, 8, 9}
CI_TEXT = {0: "no sea ice in sight", 1: "open lead >1 n mile or fast ice without boundary",
           2: "<3/10, very open", 3: "4-6/10, open", 4: "7-8/10, close", 5: "9/10+, very close",
           6: "strips/patches of pack, water between", 7: "strips of close pack, lesser concentration between",
           8: "fast ice + very open pack to seaward", 9: "fast ice + close pack to seaward"}


def boot_f1(y, p, ep, n=1000, seed=0):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    E = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in E}
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[e] for e in rng.choice(E, len(E))])
        vals.append(f1_score(y[ii], p[ii], average="macro", labels=[0, 1, 2]))
    return np.percentile(vals, [2.5, 97.5])


def fit_pred(X, Xs, tr, te, y):
    Xp = CV.fold_pca(X, tr, 64)
    out = {}
    trm = tr & (y >= 0)
    out["amsr"] = F.proba3(F.logreg(Xs[trm], y[trm], 1.0), Xs[te]).argmax(1)
    out["cam"] = F.proba3(F.logreg(X[trm], y[trm], 0.01), X[te]).argmax(1)
    Xc = np.hstack([Xp, Xs])
    out["concat"] = F.proba3(F.logreg(Xc[trm], y[trm], 0.1), Xc[te]).argmax(1)
    return out


def main():
    a2 = type("A", (), dict(max_dt=180, arch="convnext_tiny", img="448x336", workers=0))()
    run_dir = T.RUNS / f"checks_{datetime.now():%Y%m%d_%H%M}"
    log = T.Log(run_dir / "log.txt")
    out = []
    say = lambda s: (log(s), out.append(s))
    df = CV.load_all(a2.max_dt)
    X = CV.extract_features(df, a2, log)
    Xs = F.mat(df, F.SIC)
    y = df["ice_label"].values.astype(int)
    lab = y >= 0

    # ---------------- A: codes ----------------
    say("=" * 70 + "\nA) WMO CODES in the labelled frames (observer report nearest to the frame)")
    for col, name in (("ci", "ci (0639, ice)"), ("vv_code", "VV (4377, visibility)"), ("ww", "ww (4677, weather)")):
        if col not in df:
            say(f"  no column {col}")
            continue
        g = df.groupby(df[col].astype("Int64")).agg(frames=("episode", "size"), episodes=("episode", "nunique"))
        if col == "ci":
            g["meaning"] = [CI_TEXT.get(int(k), "?") if pd.notna(k) else "" for k in g.index]
            g["class_used"] = [int(df.loc[df[col] == k, "ice_label"].mode().iat[0]) if (df[col] == k).any() else -1
                               for k in g.index]
        g.to_csv(run_dir / f"codes_{col.split('_')[0]}.csv", encoding="utf-8-sig")
        say(f"\n{name}:\n" + g.to_string())
    if "ci" in df:
        amb = df["ci"].isin(AMBIG_CI) & lab
        say(f"\nAmbiguous ice codes {sorted(AMBIG_CI)}: frames {int(amb.sum())} "
            f"({amb.sum() / lab.sum():.1%} of labelled), episodes {df.loc[amb, 'episode'].nunique()}")
        # classification without the ambiguous frames: 5-fold CV by episode, 5 repeats
        keep = ~amb.values
        if amb.any():
            res = {"all": [], "clean": []}
            for r in range(5):
                for tag, sel in (("all", np.ones(len(df), bool)), ("clean", keep)):
                    d = df[sel].reset_index(drop=True)
                    P = CR.run_once(d, X[sel], Xs[sel], 2000 + r, log)
                    yy = d["ice_label"].values
                    m = yy >= 0
                    res[tag].append({k: CR.macro_f1(yy[m], P[k][m].argmax(1)) for k in CR.METHODS})
            say("\nmacro-F1 (5 CV repeats), all frames vs without ambiguous ci codes:")
            for tag in ("all", "clean"):
                R = pd.DataFrame(res[tag])
                say(f"  {tag:5s} " + "  ".join(f"{k} {R[k].mean():.3f}±{R[k].std():.3f}" for k in CR.METHODS))
        else:
            say("  no ambiguous codes - the ice labels are unambiguous")

    # ---------------- B: temporal hold-out ----------------
    say("\n" + "=" * 70 + "\nB) TEMPORAL HOLD-OUT (episodes by date)")
    ep_t = df[lab].groupby("episode")["time"].min().sort_values()
    cut = ep_t.index[len(ep_t) // 2]
    early = set(ep_t[ep_t < ep_t[cut]].index)
    say(f"  boundary: {ep_t[cut]:%d.%m.%Y %H:%M}; episodes before {len(early)}, after {len(ep_t) - len(early)}")
    for name, tr_eps in (("training: 1st half -> test: 2nd", early),
                         ("training: 2nd half -> test: 1st", set(ep_t.index) - early)):
        tr = df["episode"].isin(tr_eps).values
        te = (~tr) & lab
        dist_tr = np.bincount(y[tr & lab], minlength=3) / max(1, (tr & lab).sum())
        dist_te = np.bincount(y[te], minlength=3) / max(1, te.sum())
        pr = fit_pred(X, Xs, tr, te, y)
        say(f"\n  {name}: classes in training {np.round(dist_tr, 2)}, in test {np.round(dist_te, 2)}")
        for k in CR.METHODS:
            f1 = CR.macro_f1(y[te], pr[k])
            lo, hi = boot_f1(y[te], pr[k], df.loc[te, "episode"].values)
            rec1 = float((pr[k][y[te] == 1] == 1).mean()) if (y[te] == 1).any() else np.nan
            say(f"    {k:7s} macro-F1 {f1:.3f} [{lo:.3f}, {hi:.3f}]   recall of close ice {rec1:.2f}")
    say("\nCompare with the repeated CV (shuffled episodes): AMSR2 0.67, camera 0.65, fusion 0.70. "
        "A strong drop under the temporal split = the model transfers poorly to new conditions.")
    (run_dir / "checks_summary.txt").write_text("\n".join(out), encoding="utf-8")
    log(f"DONE: {run_dir}")


if __name__ == "__main__":
    main()
