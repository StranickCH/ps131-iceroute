"""
Cross-validation by episode for the PS131 camera model (article 3).

Why: the label within an hourly episode is almost constant (one report per 3 h), so the
effective sample size is the number of EPISODES (~40), not frames. A single split gives a test
of 5-7 episodes and huge confidence intervals. Here every episode in turn goes
to the test (K folds grouped by episode); the final metrics are computed over
all out-of-fold predictions.

MODES
  --mode probe     (~15-25 min)
      1) computes the ConvNeXt-Tiny features (ImageNet, no fine-tuning) once for all labelled frames
      2) logistic regression on the features, K folds (seconds)
      3) BASELINES in the same folds: logistic regression on 6 frame metrics (brightness, sharpness...),
         for ice - AMSR2
  --mode finetune  (OVERNIGHT: ~1-3 h depending on --seeds)
      partial fine-tuning (early ConvNeXt layers frozen), K folds x seeds

RUN
    python train_ps131_cv.py --mode probe
    python train_ps131_cv.py --mode finetune --seeds 3          # overnight
    python train_ps131_cv.py --mode finetune --smoke            # 5-minute check before the night run
Options: --max-dt 180 (min to the synoptic report), --folds 5

OUTPUT: runs\\cv_<mode>_<date>\\ : log.txt, cv_metrics.txt/.json, oof_predictions.csv, folds.csv
"""

import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T          # noqa: E402  (FrameDS, model, metrics, log)
import prepare_ps131_dataset as P       # noqa: E402  (labels, cache path)

FEAT_DIR = HERE / "ps131_features"
HANDCRAFTED = ["brightness", "contrast", "sharpness", "white_frac", "sat_mean", "top_bottom"]


# ============================================================
# Data and folds
# ============================================================
def load_all(max_dt):
    df = pd.read_csv(P.LABELED, low_memory=False)
    df["time"] = pd.to_datetime(df["time"])
    df = P.make_labels(df, max_dt)
    from pathlib import PureWindowsPath
    df["cache_path"] = [str(P.CACHE_ROOT / str(e) / PureWindowsPath(p).name) for e, p in zip(df["episode"], df["path"])]
    df = df[(df["vis_label"] >= 0) | (df["ice_label"] >= 0)].reset_index(drop=True)
    ok = df["cache_path"].map(lambda x: Path(x).exists())
    if not ok.all():
        print(f"[data] {int((~ok).sum())} frames not in the cache - skipping them")
    df = df[ok].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No labelled frames in the cache {P.CACHE_ROOT}. Run prepare_ps131_dataset.py --cache first")
    return df


def make_folds(df, k, trials, seed):
    """Episodes -> K folds: all classes of both tasks in every fold where possible,
    class distributions of the folds as close as possible to the overall one."""
    rng = np.random.default_rng(seed)
    eps = df["episode"].unique()
    ref_v, ref_i = P.dist(df, "vis_label"), P.dist(df, "ice_label")
    best, best_cost = None, np.inf
    for _ in range(trials):
        perm = rng.permutation(eps)
        assign = {e: i % k for i, e in enumerate(perm)}
        f = df["episode"].map(assign)
        cost = 0.0
        for j in range(k):
            sub = df[f == j]
            dv, di = P.dist(sub, "vis_label"), P.dist(sub, "ice_label")
            cost += np.abs(dv - ref_v).sum() + np.abs(di - ref_i).sum() + 2.0 * ((dv == 0).sum() + (di == 0).sum())
        if cost < best_cost:
            best, best_cost = assign, cost
    return best, best_cost


# ============================================================
# Out-of-fold metrics
# ============================================================
def report_task(name, d, ycol, pcol, names, log):
    d = d[(d[ycol] >= 0) & (d[pcol] >= 0)]
    m = T.metrics(d[ycol].values, d[pcol].values)
    ci = T.bootstrap_by_episode(d, ycol, pcol)
    # vote per episode: the class the model most often assigns to the frames of the episode
    ep = d.groupby("episode").agg(y=(ycol, lambda s: s.mode().iat[0]), p=(pcol, lambda s: s.mode().iat[0]))
    ep_acc = float((ep["y"] == ep["p"]).mean())
    lines = [f"\n{name}: {m['n']} frames / {ci['n_episodes']} episodes",
             f"  frames:   accuracy {m['acc']:.3f}  95% CI {ci['acc_ci95']}   macro-F1 {m['macro_f1']:.3f}  95% CI {ci['macro_f1_ci95']}",
             f"  episodes: accuracy by vote {ep_acc:.3f} ({int((ep['y'] == ep['p']).sum())}/{len(ep)})",
             "  recall: " + ", ".join(f"{n} {r:.2f}" for n, r in zip(names, m["recall"])),
             "  matrix (rows - truth):"]
    lines += ["    " + f"{n:>14s} " + " ".join(f"{v:7d}" for v in row) for n, row in zip(names, m["cm"])]
    for l in lines:
        log(l)
    return {**m, **ci, "episode_vote_acc": ep_acc}, lines


# ============================================================
# PROBE: ConvNeXt features + logistic regression
# ============================================================
def extract_features(df, args, log):
    import torch
    from torch.utils.data import DataLoader
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    W, H = map(int, args.img.split("x"))
    fpath = FEAT_DIR / f"{args.arch}_{W}x{H}.npz"
    have = {}
    if fpath.exists():
        z = np.load(fpath, allow_pickle=True)
        have = dict(zip(z["paths"], z["feats"]))
    todo = df[~df["cache_path"].isin(have)]
    log(f"[features] in the feature cache {len(have)}, to compute {len(todo)} frames")
    if len(todo):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bb, dim = T.get_backbone(args.arch, log, pretrained=True)
        bb = bb.to(device).eval().to(memory_format=torch.channels_last)
        ds = T.FrameDS(todo.reset_index(drop=True), (W, H), train=False)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers, persistent_workers=False)
        feats, t0, n = [], time.time(), 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            for b, (x, _, _, _) in enumerate(dl):
                feats.append(bb(x.to(device).to(memory_format=torch.channels_last)).float().cpu().numpy())
                n += len(x)
                if (b + 1) % 50 == 0:
                    ips = n / (time.time() - t0)
                    log(f"  features {n}/{len(todo)}  {ips:.0f} frames/s  ~{T.fmt_dur((len(todo) - n) / ips)} left")
        feats = np.concatenate(feats)
        have.update(zip(todo["cache_path"].values, feats))
        np.savez(fpath, paths=np.array(list(have.keys()), dtype=object), feats=np.stack(list(have.values())))
        log(f"[features] saved: {fpath}")
    return np.stack([have[p] for p in df["cache_path"].values])


def fit_predict_logreg(Xtr, ytr, Xva, yva, Xte):
    """Logistic regression with standardisation; C is chosen on a validation fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, best_f1 = None, -1
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
        f1 = T.metrics(yva, clf.predict(Xva))["macro_f1"] if len(yva) else 0
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best.predict(Xte), best.predict_proba(Xte)


def fold_pca(X, tr_mask, n=64):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr_mask])
    p = PCA(n_components=min(n, X.shape[1]), random_state=0).fit(sc.transform(X[tr_mask]))
    return p.transform(sc.transform(X))


def cols_matrix(df, cols):
    have = [c for c in cols if c in df]
    if not have:
        return None, []
    M = df[have].astype(float)
    return M.fillna(M.median()).values, have


def bootstrap_diff(df, ycol, p1, p2, n_boot=1000, seed=0):
    """macro-F1(p1) - macro-F1(p2) on THE SAME frames, episode bootstrap."""
    rng = np.random.default_rng(seed)
    d = df[(df[ycol] >= 0) & (df[p1] >= 0) & (df[p2] >= 0)]
    groups = {e: g for e, g in d.groupby("episode")}
    eps = list(groups)
    f = lambda s, p: T.metrics(s[ycol].values, s[p].values)["macro_f1"]
    point = f(d, p1) - f(d, p2)
    diffs = []
    for _ in range(n_boot):
        s = pd.concat([groups[e] for e in rng.choice(eps, size=len(eps), replace=True)])
        diffs.append(f(s, p1) - f(s, p2))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi, float(np.mean(np.array(diffs) <= 0))


def run_probe(df, folds, args, log):
    X = extract_features(df, args, log)
    Xh = df[HANDCRAFTED].copy()
    Xh["sharpness"] = np.log10(Xh["sharpness"].clip(lower=1e-3))
    Xh = Xh.fillna(Xh.median()).values
    Xsic, sic_cols = cols_matrix(df, ["sic_px", "sic_3x3", "sic_7x7"])
    Xmet, met_cols = cols_matrix(df, ["t2m", "dpd", "lcc", "tcc", "wind", "tp", "msl"])
    log(f"[fusion] satellite: {sic_cols}; ERA5 weather: {met_cols}")
    k = args.folds
    preds = ["cnn_vis", "base_vis", "met_vis", "fus_vis", "cnn_ice", "base_ice", "amsrlr_ice", "fus_ice"]
    for col in preds:
        df[col] = -1
    for f in range(k):
        te = (df["fold"] == f).values
        va = (df["fold"] == (f + 1) % k).values
        tr = ~(te | va)
        Xp = fold_pca(X, tr, 64)  # 64 principal components of the camera features (PCA on the training fold only)
        sets = {
            "vis": {"cnn": X, "base": Xh,
                    "met": Xmet,
                    "fus": np.hstack([m for m in (Xp, Xh, Xmet) if m is not None])},
            "ice": {"cnn": X, "base": Xh,
                    "amsrlr": Xsic,
                    "fus": np.hstack([m for m in (Xp, Xsic) if m is not None])},
        }
        for task, ycol in (("vis", "vis_label"), ("ice", "ice_label")):
            y = df[ycol].values
            m_tr, m_va = tr & (y >= 0), va & (y >= 0)
            if m_tr.sum() == 0 or len(np.unique(y[m_tr])) < 2:
                continue
            for name, M in sets[task].items():
                if M is None:
                    continue
                p, _ = fit_predict_logreg(M[m_tr], y[m_tr], M[m_va], y[m_va], M[te])
                df.loc[te, f"{name}_{task}"] = p
        log(f"  fold {f + 1}/{k} done")

    tasks = [("VISIBILITY: camera (ConvNeXt) + logistic regression", "vis_label", "cnn_vis", T.VIS_NAMES),
             ("VISIBILITY: baseline, 6 frame metrics", "vis_label", "base_vis", T.VIS_NAMES),
             ("VISIBILITY: ERA5 only", "vis_label", "met_vis", T.VIS_NAMES),
             ("VISIBILITY: FUSION camera + metrics + ERA5", "vis_label", "fus_vis", T.VIS_NAMES),
             ("ICE: camera (ConvNeXt) + logistic regression", "ice_label", "cnn_ice", T.ICE_NAMES),
             ("ICE: baseline, 6 frame metrics", "ice_label", "base_ice", T.ICE_NAMES),
             ("ICE: AMSR2, fixed C_ice thresholds", "ice_label", "amsr_ice3", T.ICE_NAMES),
             ("ICE: AMSR2, learned thresholds (logistic regression on SIC)", "ice_label", "amsrlr_ice", T.ICE_NAMES),
             ("ICE: FUSION camera + AMSR2", "ice_label", "fus_ice", T.ICE_NAMES)]
    pairs = [("ICE: fusion - AMSR2 (learned)", "ice_label", "fus_ice", "amsrlr_ice"),
             ("ICE: fusion - camera", "ice_label", "fus_ice", "cnn_ice"),
             ("ICE: camera - AMSR2 (learned)", "ice_label", "cnn_ice", "amsrlr_ice"),
             ("VISIBILITY: camera - baseline", "vis_label", "cnn_vis", "base_vis"),
             ("VISIBILITY: fusion - baseline", "vis_label", "fus_vis", "base_vis"),
             ("VISIBILITY: fusion - ERA5 only", "vis_label", "fus_vis", "met_vis")]
    return tasks, pairs


# ============================================================
# FINETUNE: partial fine-tuning per fold
# ============================================================
def freeze_partial(model):
    feats = getattr(model.backbone, "features", None)
    if feats is None:
        return
    n_freeze = 6 if len(feats) == 8 else max(1, len(feats) - 2)   # convnext: train stage4 + head
    for i, blk in enumerate(feats):
        if i < n_freeze:
            for p in blk.parameters():
                p.requires_grad = False


def run_finetune(df, folds, args, log):
    import torch
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = map(int, args.img.split("x"))
    k = args.folds
    pin = False
    seeds = list(range(args.seeds))
    for s in seeds:
        df[f"ft_vis_s{s}"] = -1
        df[f"ft_ice_s{s}"] = -1
    t_all, done_units, total_units = time.time(), 0, k * len(seeds)
    for s in seeds:
        for f in range(k):
            t_unit = time.time()
            te, va = df["fold"] == f, df["fold"] == (f + 1) % k
            tr = ~(te | va)
            dtr, dva, dte = df[tr].reset_index(drop=True), df[va].reset_index(drop=True), df[te].reset_index(drop=True)
            dva = T.stride_subset(dva, 3).reset_index(drop=True)
            torch.manual_seed(1000 * s + f)
            model = T.build_model(args.arch, log, pretrained=True).to(device).to(memory_format=torch.channels_last)
            freeze_partial(model)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
            sampler = T.EpochSampler(dtr, args.per_episode, 1000 * s + f)
            dl_tr = DataLoader(T.FrameDS(dtr, (W, H), True), sampler=sampler, batch_size=args.batch,
                               num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
            ev = dict(batch_size=args.batch, num_workers=max(0, args.workers // 2), persistent_workers=False)
            dl_va = DataLoader(T.FrameDS(dva, (W, H), False), shuffle=False, **ev)
            dl_te = DataLoader(T.FrameDS(dte, (W, H), False), shuffle=False, **ev)
            steps = math.ceil(len(sampler) / args.batch) * args.epochs
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            cv_ = torch.nn.CrossEntropyLoss(weight=T.class_weights(dtr["vis_label"].values).to(device),
                                            ignore_index=-1, label_smoothing=0.1)
            ci_ = torch.nn.CrossEntropyLoss(weight=T.class_weights(dtr["ice_label"].values).to(device),
                                            ignore_index=-1, label_smoothing=0.1)
            log(f"=== seed {s + 1}/{len(seeds)}, fold {f + 1}/{k}: train {dtr['episode'].nunique()} ep., "
                f"val {dva['episode'].nunique()}, test {dte['episode'].nunique()} ===")
            best, bad, best_state = -1, 0, None
            timing = {}
            for ep in range(1, args.epochs + 1):
                sampler.set_epoch(ep)
                T.train_epoch(model, dl_tr, opt, sched, scaler, cv_, ci_, device, log, ep, args.epochs,
                              max_batches=args.max_batches, timing=timing)
                pv, pi, ix, ev_ips = T.run_eval(model, dl_va, device, max_batches=args.max_batches)
                sc = (T.metrics(dva["vis_label"].values[ix], pv.argmax(1)).get("macro_f1", 0) +
                      T.metrics(dva["ice_label"].values[ix], pi.argmax(1)).get("macro_f1", 0)) / 2
                if sc > best:
                    best, bad = sc, 0
                    best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    bad += 1
                log(f"  epoch {ep}: val score {sc:.4f}{'  <- best' if bad == 0 else f'  ({bad}/{args.patience})'}")
                if bad >= args.patience:
                    break
            model.load_state_dict(best_state)
            pv, pi, ix, _ = T.run_eval(model, dl_te, device, max_batches=args.max_batches)
            idx = df.index[te][ix]
            df.loc[idx, f"ft_vis_s{s}"] = pv.argmax(1)
            df.loc[idx, f"ft_ice_s{s}"] = pi.argmax(1)
            del model, opt, dl_tr
            torch.cuda.empty_cache()
            done_units += 1
            el = time.time() - t_all
            left = el / done_units * (total_units - done_units)
            log(f"  fold took {T.fmt_dur(time.time() - t_unit)}; done {done_units}/{total_units}, "
                f"~{T.fmt_dur(left)} left (finish ~{(datetime.now() + timedelta(seconds=left)):%H:%M})")
            if args.smoke:
                tr_ips = timing.get("train_ips", 1e-6)
                sizes = []
                for ff in range(k):
                    te_, va_ = df["fold"] == ff, df["fold"] == (ff + 1) % k
                    tr_ = df[~(te_ | va_)]
                    n_tr = len(T.EpochSampler(tr_, args.per_episode, 0))
                    n_va = len(T.stride_subset(df[va_], 3))
                    sizes.append((n_tr, n_va, int(te_.sum())))
                n_tr, n_va, n_te = (np.mean([x[i] for x in sizes]) for i in range(3))
                t_ep = n_tr / tr_ips + n_va / ev_ips
                per_fold_min = (args.full_patience + 1) * t_ep + n_te / ev_ips + 30
                per_fold_max = args.full_epochs * t_ep + n_te / ev_ips + 30
                units = k * args.full_seeds
                lines = ["=" * 70,
                         f"ESTIMATE OF THE FULL RUN: {k} folds x {args.full_seeds} seed = {units} trainings",
                         f"  speed: training {tr_ips:.0f} frames/s, evaluation {ev_ips:.0f} frames/s",
                         f"  epoch ~{T.fmt_dur(t_ep)} ({n_tr:.0f} training frames + {n_va:.0f} val)",
                         f"  total: from ~{T.fmt_dur(units * per_fold_min)} (early stopping) to ~{T.fmt_dur(units * per_fold_max)} "
                         f"(all {args.full_epochs} epochs)",
                         f"  if started now, finishes between {(datetime.now() + timedelta(seconds=units * per_fold_min)):%H:%M} "
                         f"and {(datetime.now() + timedelta(seconds=units * per_fold_max)):%H:%M}",
                         "=" * 70, "SMOKE PASSED."]
                for l in lines:
                    log(l)
                return None
    tasks = []
    for s in seeds:
        tasks += [(f"VISIBILITY: fine-tuning, seed {s + 1}", "vis_label", f"ft_vis_s{s}", T.VIS_NAMES),
                  (f"ICE: fine-tuning, seed {s + 1}", "ice_label", f"ft_ice_s{s}", T.ICE_NAMES)]
    tasks.append(("ICE: AMSR2 (satellite)", "ice_label", "amsr_ice3", T.ICE_NAMES))
    return tasks


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "finetune"], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-dt", type=float, default=180)
    ap.add_argument("--arch", default="convnext_tiny")
    ap.add_argument("--img", default="448x336")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--per-episode", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="finetune: 1 fold, 40 batches each, + time estimate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.max_batches = None
    args.full_seeds, args.full_epochs, args.full_patience = args.seeds, args.epochs, args.patience
    if args.smoke:
        args.seeds, args.max_batches = 1, 40
        args.epochs, args.patience = 1, 1

    run_dir = T.RUNS / f"cv_{args.mode}{'_smoke' if args.smoke else ''}_{datetime.now():%Y%m%d_%H%M}"
    log = T.Log(run_dir / "log.txt")
    log(f"Cross-validation by episode | mode {args.mode} | {vars(args)}")
    df = load_all(args.max_dt)
    df["amsr_ice3"] = df["ice_class"].map(T.AMSR_TO_3).fillna(-1).astype(int) if "ice_class" in df else -1
    folds, cost = make_folds(df, args.folds, 3000, args.seed)
    df["fold"] = df["episode"].map(folds)
    pd.Series(folds, name="fold").rename_axis("episode").to_csv(run_dir / "folds.csv")
    log(f"Data (max-dt {args.max_dt:.0f} min): {len(df)} frames, {df['episode'].nunique()} episodes; "
        f"with a visibility label {int((df['vis_label'] >= 0).sum())}, ice {int((df['ice_label'] >= 0).sum())}")
    for f in range(args.folds):
        sub = df[df["fold"] == f]
        log(f"  fold {f + 1}: {sub['episode'].nunique()} ep.; visibility {P.dist(sub, 'vis_label').round(2).tolist()}, "
            f"ice {P.dist(sub, 'ice_label').round(2).tolist()}")

    pairs = []
    if args.mode == "probe":
        tasks, pairs = run_probe(df, folds, args, log)
    else:
        tasks = run_finetune(df, folds, args, log)
        if tasks is None:
            return

    log("\n" + "=" * 70 + "\nRESULT (out-of-fold over all episodes; CI - episode bootstrap)")
    res, out = {}, []
    for name, y, p, names in tasks:
        r, lines = report_task(name, df, y, p, names, log)
        res[name] = r
        out += lines
    if pairs:
        log("\n" + "=" * 70 + "\nPAIRWISE macro-F1 DIFFERENCES (same frames; 95% CI by episode bootstrap)")
        out.append("\nPAIRWISE macro-F1 DIFFERENCES (95% CI, episode bootstrap)")
        for name, y, p1, p2 in pairs:
            if (df[p1] < 0).all() or (df[p2] < 0).all():
                continue
            d, lo, hi, p_le0 = bootstrap_diff(df, y, p1, p2)
            sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
            line = f"  {name:34s} {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  ({sig}; share of bootstrap draws <= 0: {p_le0:.3f})"
            log(line)
            out.append(line)
            res[f"diff: {name}"] = {"diff": d, "ci95": [lo, hi], "p_le0": p_le0}
    (run_dir / "cv_metrics.txt").write_text("\n".join(out), encoding="utf-8")
    (run_dir / "cv_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    keep = ["path", "episode", "time", "fold", "vis_label", "ice_label", "amsr_ice3", "sic_3x3"] + \
           [c for c in df.columns if c.startswith(("cnn_", "base_", "ft_", "met_", "fus_", "amsrlr_"))]
    df[[c for c in keep if c in df]].to_csv(run_dir / "oof_predictions.csv", index=False, float_format="%.4f")
    log(f"DONE: {run_dir}")


if __name__ == "__main__":
    main()
