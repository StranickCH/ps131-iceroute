"""
Routing v3 - checks for article 3 (items 1, 3, 4 of the plan). Uses the functions of route_ice_sources_ps131.py.

1) NO LEAKAGE (--matrix-cv 5): the sensor confusion matrices and planner tables for an episode are estimated
   from OTHER episodes (5 groups of episodes, stratified by visibility).
3) MODEL PROBABILITIES (--obs-model resample): the observation of a cell = a real out-of-fold PS131 frame
   with the same truth (and visibility, and AMSR2 observation for fusion), taken from OTHER episodes.
   Both the class (argmax) and the model's probability vector are taken from the frame. The planners are
   compared on THE SAME observations:
       hard  - cost of the observed class
       bayes - expected cost from the confusion matrices (as in v2)
       probs - expected cost from the model's own probabilities: sum_k p_k c_k
       probs_t - the same after temperature calibration (T is fitted on OTHER episodes)
   (--obs-model matrix = sampling from the matrices, as before; the probs planner is then unavailable)
4) SENSITIVITY TO THE "TRUE" FIELD: --floe-sigma (scale of the floe clusters, 250 m cells; was 6)
   and --thr (class concentration thresholds; was 0.35,0.85).

RUN (example):
  python route_ps131_v3.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --matrix-cv 5 --obs-model resample --trials 20
OUTPUT: runs\\routev3_<tag>_<date>\\ : log.txt, args.json, trials.csv, summary.txt, summary.json
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

VERSION = "v3.5 (Oct 16: --cam-donor - camera/fusion frames from one donor episode)"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T          # noqa: E402
import route_ice_sources_ps131 as R     # noqa: E402

N = R.N
SOURCES = {"amsr": "pred_amsr", "camera": "pred_cam"}       # the fusion column is set by --fusion-model
PLANNERS = ["hard", "bayes", "probs", "probs_t", "cvar"]
KN_KMH = 1.852


# ============================================================
# True field with sensitivity parameters
# ============================================================
def true_classes(sic, land, rng, sigma, thr):
    from scipy.ndimage import gaussian_filter, zoom
    g = gaussian_filter(rng.standard_normal((N * R.SUB, N * R.SUB)), sigma)
    u = (np.argsort(np.argsort(g.ravel())) / g.size).reshape(g.shape)
    ice = u < zoom(sic, R.SUB, order=1)
    conc = ice.reshape(N, R.SUB, N, R.SUB).mean(axis=(1, 3))
    return np.where(conc < thr[0], 0, np.where(conc < thr[1], 1, 2))


def align_to_dmi(w, sic, land):
    """AMSR2 field (its own polar stereographic grid) to the DMI window: of the 4 reflections the one that
    correlates best with the DMI concentration on cells with data is chosen. Returns (sic, land, index, r)."""
    ok = np.isfinite(w) & ~land
    best = None
    for k, (s_, l_) in enumerate(((sic, land), (sic[::-1], land[::-1]), (sic[:, ::-1], land[:, ::-1]),
                                  (sic[::-1, ::-1], land[::-1, ::-1]))):
        m = ok & ~l_
        r = np.corrcoef(w[m], s_[m])[0, 1] if m.sum() > 10 and np.std(s_[m]) > 0 and np.std(w[m]) > 0 else -np.inf
        if best is None or r > best[3]:
            best = (s_, l_, k, r)
    return best


def cvar_tables(conf, v, alpha):
    """Risk-averse planner: cell cost = CVaR_alpha of the cost under the posterior class distribution
    (the same confusion matrices and prior frequencies as bayes) - the mean cost over the worst (1 - alpha) outcomes."""
    c = R.CLASS_COST
    p0 = conf["prior"]["all"]
    Ma, Mc, J = conf["amsr"][v], conf["camera"][v], conf["fusion"][v]

    def cv(P):
        P = P / np.clip(P.sum(0, keepdims=True), 1e-12, None)
        tail = 1.0 - alpha
        out, rem = np.zeros(P.shape[1:]), np.full(P.shape[1:], tail)
        for t in np.argsort(c)[::-1]:              # from the most expensive class
            take = np.minimum(P[t], rem)
            out += take * c[t]
            rem -= take
        return out / tail
    return {"amsr": cv(p0[:, None] * Ma), "camera": cv(p0[:, None] * Mc),
            "fusion": cv(p0[:, None, None] * Ma[:, :, None] * J), "unknown": float(cv(p0[:, None])[0])}


# ============================================================
# Observations from real frames (resample)
# ============================================================
class FramePool:
    """Frame pools by key (truth, visibility[, AMSR2 observation]); selection: first an episode, then a frame."""

    def __init__(self, oof, keys, exclude_eps):
        d = oof.assign(_row=np.arange(len(oof)))
        d = d[(d["ice_label"] >= 0) & ~d["episode"].isin(exclude_eps)]
        self.idx, self.by_ep = {}, {}
        for k, g in d.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            # row positions in oof (iloc), grouped by episode
            gi = g.groupby("episode").indices
            self.idx[tuple(int(x) for x in k)] = [g["_row"].values[ix] for ix in gi.values()]
            self.by_ep[tuple(int(x) for x in k)] = {ep: g["_row"].values[ix] for ep, ix in gi.items()}
        self.keys = keys
        self.eps_vis = {v: sorted(d.loc[d["vis_label"] == v, "episode"].unique()) for v in range(3)}
        self.n_donor, self.n_draw = 0, 0

    def draw(self, rng, key, fallbacks, donor=None):
        for k in [key] + fallbacks:
            lst = self.idx.get(tuple(k))
            if lst:
                e = lst[rng.integers(len(lst))]      # rng is always called: the random stream is the same as without a donor
                self.n_draw += donor is not None
                if donor is not None and donor in self.by_ep[tuple(k)]:   # frame from the donor episode
                    self.n_donor += 1
                    e = self.by_ep[tuple(k)][donor]
                return int(e[rng.integers(len(e))])
        return None


def resample_obs(true_cls, block, rng, pool, src_cols, oof_p, oof_pred, vis, cond=None, donor=None):
    """Block observation: one real frame per block x block block. Returns the class (N,N) and p (N,N,3)."""
    t_mode = R.block_mode(true_cls, block)
    cls = np.zeros((N, N), int)
    P = np.zeros((N, N, 3))
    nb = int(np.ceil(N / block))
    for bi in range(nb):
        for bj in range(nb):
            y0, x0 = bi * block, bj * block
            t = int(t_mode[y0, x0])
            if cond is None:
                key, fb = (t, vis), [(t, (vis + 1) % 3), (t, (vis + 2) % 3)]
            else:
                a = int(cond[y0, x0])
                key, fb = (t, vis, a), [(t, (vis + 1) % 3, a), (t, (vis + 2) % 3, a)]
            j = pool.draw(rng, key, fb, donor)
            if j is None:          # no frames at all with this combination - class as the truth (rare)
                c, p = t, np.eye(3)[t]
            else:
                c, p = oof_pred[j], oof_p[j]
            cls[y0:y0 + block, x0:x0 + block] = c
            P[y0:y0 + block, x0:x0 + block] = p
    return cls, P


# ============================================================
# Strategy: far cost map (AMSR2/no data) + near map (what the camera has revealed)
# ============================================================
def run_route(true_cls, land, start, goal, far, near, r):
    true_cost = np.where(land, np.inf, R.CLASS_COST[true_cls])
    belief = far.astype(float).copy()
    seen = np.zeros((N, N), bool)
    pos, path, total, heavy, steps, route, k = start, [start], 0.0, 0.0, 0, None, 1
    dist = 0.0
    km = np.zeros(3)
    while pos != goal and steps < 4 * N:
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
        sl = R.CELL_KM * (np.sqrt(2) if nxt[0] != pos[0] and nxt[1] != pos[1] else 1.0)
        c = true_cost[nxt]
        total += sl * c if np.isfinite(c) else 1e3
        heavy += sl * (true_cls[nxt] == 2)
        dist += sl
        km[true_cls[nxt]] += sl
        pos = nxt
        path.append(pos)
        steps += 1
    return total, heavy, dist, pos == goal, km


def fit_temperature(P, y, w):
    """Scalar temperature: minimum of the weighted NLL on a grid."""
    lp = np.log(np.clip(P, 1e-6, 1))
    best, bt = np.inf, 1.0
    for T_ in np.exp(np.linspace(np.log(0.5), np.log(6), 60)):
        z = lp / T_
        z = z - z.max(1, keepdims=True)
        q = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        nll = -(w * np.log(np.clip(q[np.arange(len(y)), y], 1e-9, 1))).sum() / w.sum()
        if nll < best:
            best, bt = nll, T_
    return float(bt)


def apply_temperature(P, T_):
    z = np.log(np.clip(P, 1e-6, 1)) / T_
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def boot_ci(df, col, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = df.groupby("episode")[col].mean().values
    return np.percentile([rng.choice(g, len(g)).mean() for _ in range(n)], [2.5, 97.5])


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--fusion-model", default="concat", choices=["concat", "stack", "gated"])
    ap.add_argument("--obs-model", default="resample", choices=["resample", "matrix"])
    ap.add_argument("--matrix-cv", type=int, default=5, help="0 = matrices from all episodes (as in v2)")
    ap.add_argument("--time-split", action="store_true",
                    help="groups = halves of the cruise in time: matrices/frames/temperature for an episode come from the OTHER half")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--floe-sigma", type=float, default=6.0)
    ap.add_argument("--thr", default="0.35,0.85")
    ap.add_argument("--cam-block", type=int, default=3)
    ap.add_argument("--class-cost", default="1,2.5,8", help="cost of 1 km: open,close,heavy")
    ap.add_argument("--cam-range-scale", type=float, default=1.0, help="multiplier of the camera view radius")
    ap.add_argument("--class-speed", default="12,6,3",
                    help="speed, kn: open, close, heavy - only for the transit time (does not affect the route)")
    ap.add_argument("--cvar-alpha", type=float, default=0.8, help="CVaR level of the cvar planner")
    ap.add_argument("--max-label-dt", type=float, default=None,
                    help="min: use only frames whose bridge report is no further than this (default 180, as in labelling)")
    ap.add_argument("--labels", default=None, help="ps131_labeled.csv (for --max-label-dt)")
    ap.add_argument("--truth", default="amsr", choices=["amsr", "dmi"],
                    help="true field: amsr = AMSR2 + floe structure; dmi = 100x100 km window of the DMI ice chart")
    ap.add_argument("--dmi-windows", default=None, help="dmi_windows_ps131.npz (check_dmi_coverage_ps131.py)")
    ap.add_argument("--dmi-min-cover", type=float, default=0.7, help="min. fraction of filled cells in the DMI window")
    ap.add_argument("--dmi-nodata", default="blocked", choices=["blocked", "water", "heavy", "amsr"],
                    help="DMI cells without data: blocked = impassable (as before); water / heavy = open water / heavy ice; "
                         "amsr = classes from AMSR2 + floe structure; with water/heavy/amsr only AMSR2 land is impassable")
    ap.add_argument("--cam-donor", action="store_true",
                    help="camera and fusion: all frames of a trial come from one random donor episode with the same "
                         "visibility (carries the between-episode systematic error); if there is no frame of the needed class - "
                         "from a random episode, as without the flag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--smoke", action="store_true", help="3 episodes x 2 routes + time estimate")
    args = ap.parse_args()
    thr = tuple(float(x) for x in args.thr.split(","))
    R.CAM_BLOCK = args.cam_block
    R.CLASS_COST = np.array([float(x) for x in args.class_cost.split(",")])
    R.CAM_RANGE = {k_: max(1, int(round(r_ * args.cam_range_scale))) for k_, r_ in R.CAM_RANGE.items()}

    run_dir = T.RUNS / f"routev3_{args.tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = T.Log(run_dir / "log.txt")
    (run_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Routing {VERSION} | {vars(args)} | cost {R.CLASS_COST.tolist()} | radius {R.CAM_RANGE}")

    oof = pd.read_csv(args.oof, low_memory=False)
    oof["time"] = pd.to_datetime(oof["time"])
    if args.max_label_dt is not None:
        lp = Path(args.labels) if args.labels else HERE / "ps131_index" / "ps131_labeled.csv"
        lab = pd.read_csv(lp, low_memory=False, usecols=["path", "dt_ci_min", "dt_vv_code_min"])
        oof = oof.merge(lab, on="path", how="left")
        far_ice = ~(oof["dt_ci_min"] <= args.max_label_dt)
        far_vis = ~(oof["dt_vv_code_min"] <= args.max_label_dt)
        oof.loc[far_ice, "ice_label"] = -1
        oof.loc[far_vis, "vis_label"] = -1
        log(f"Label window {args.max_label_dt:.0f} min: ice label removed from {int(far_ice.sum())} frames, "
            f"visibility label from {int(far_vis.sum())}")
    fcol = f"pred_{args.fusion_model}"
    cols = {"amsr": "amsr", "camera": "cam", "fusion": args.fusion_model}
    probs = {s: oof[[f"p_{m}_{k}" for k in range(3)]].values for s, m in cols.items()}
    preds = {s: oof[f"pred_{m}"].values for s, m in cols.items()}
    for s in probs:     # rows without probabilities (no label) are not used
        probs[s] = np.nan_to_num(probs[s], nan=1 / 3)

    eps = (oof[oof["vis_label"] >= 0].groupby("episode")
           .agg(time=("time", "min"), lat=("lat", "median"), lon=("lon", "median"),
                vis=("vis_label", lambda s: int(s.mode().iat[0]))).reset_index())
    # groups of episodes for leakage-free matrix estimation (stratified by visibility)
    rng0 = np.random.default_rng(args.seed + 7)
    eps["grp"] = -1
    if args.time_split:
        cut = eps["time"].sort_values().iloc[len(eps) // 2]
        eps["grp"] = (eps["time"] >= cut).astype(int)
        log(f"Time split: boundary {cut:%d.%m.%Y %H:%M}, episodes {eps['grp'].value_counts().sort_index().tolist()}")
    elif args.matrix_cv > 1:
        for v in range(3):
            ix = eps.index[eps["vis"] == v].values
            ix = rng0.permutation(ix)
            eps.loc[ix, "grp"] = np.arange(len(ix)) % args.matrix_cv
    dmi = None
    if args.truth == "dmi":   # episodes are selected AFTER the group split: tables still come from other episodes
        wp = Path(args.dmi_windows) if args.dmi_windows else HERE / "08_icecharts_2022" / "dmi_windows_ps131.npz"
        dmi = dict(np.load(wp))
        cover = eps["episode"].astype(str).map(lambda k: float(np.isfinite(dmi[k]).mean()) if k in dmi else 0.0)
        eps = eps[cover >= args.dmi_min_cover].reset_index(drop=True)
        log(f"Truth = DMI charts: episodes with window coverage >= {args.dmi_min_cover:.0%}: {len(eps)}")
    if args.smoke:
        eps = eps.groupby("vis").head(1).reset_index(drop=True)
        args.trials = 2
    groups = sorted(eps["grp"].unique())
    log(f"Episodes {len(eps)}; groups for matrices: {len(groups)} ({'no leakage' if args.matrix_cv > 1 else 'all episodes'})")

    # matrices, tables, frame pools - per group
    G = {}
    for g in groups:
        excl = set(eps.loc[eps["grp"] == g, "episode"]) if g >= 0 else set()
        sub = oof[~oof["episode"].isin(excl)]
        conf = R.confusion_tables(sub, fcol, lambda *a: None)
        G[g] = {"conf": conf,
                "ct": dict({p: {v: R.cost_tables(conf, v, p) for v in range(3)} for p in ("hard", "bayes")},
                           cvar={v: cvar_tables(conf, v, args.cvar_alpha) for v in range(3)}),
                "pool_t": FramePool(oof, ["ice_label", "vis_label"], excl),
                "pool_ta": FramePool(oof, ["ice_label", "vis_label", "pred_amsr"], excl)}
        lab = (oof["ice_label"] >= 0).values & ~oof["episode"].isin(excl).values
        w = (1.0 / oof.loc[lab].groupby("episode")["episode"].transform("size")).values
        G[g]["temp"] = {s_: fit_temperature(probs[s_][lab], oof.loc[lab, "ice_label"].values.astype(int), w)
                        for s_ in probs}
        log(f"  group {g}: calibration temperature " + ", ".join(f"{k} {v:.2f}" for k, v in G[g]["temp"].items()))
        # OBSERVATIONS: by default from the same "other" episodes. With --time-split the sensors behave as in THEIR OWN
        # half of the cruise (frames/matrices of their half), while the planner knows only the other half -> a fair shift.
        if args.time_split and g >= 0:
            own = set(eps.loc[eps["grp"] == g, "episode"])
            oexcl = set(oof["episode"].unique()) - own
            G[g]["oconf"] = R.confusion_tables(oof[oof["episode"].isin(own)], fcol, lambda *a: None)
            G[g]["opool_t"] = FramePool(oof, ["ice_label", "vis_label"], oexcl)
            G[g]["opool_ta"] = FramePool(oof, ["ice_label", "vis_label", "pred_amsr"], oexcl)
        else:
            G[g]["oconf"], G[g]["opool_t"], G[g]["opool_ta"] = G[g]["conf"], G[g]["pool_t"], G[g]["pool_ta"]
    strategies = [(s, p) for s in ("amsr", "fusion", "camera") for p in PLANNERS
                  if not (p.startswith("probs") and args.obs_model == "matrix")]
    log("Strategies: " + ", ".join(f"{s}|{p}" for s, p in strategies))

    rows = []
    rng_master = np.random.default_rng(args.seed)
    rng_don = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    for i, e in eps.iterrows():
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        gi = G[e["grp"]]
        v = int(e["vis"])
        if dmi is not None:      # DMI concentration, %: <=30 -> 0, 40-80 -> 1, >=90 -> 2; no data/land - impassable
            w = dmi[str(e["episode"])]
            land = ~np.isfinite(w)
            tc_dmi = np.where(land, 0, np.where(w <= 30, 0, np.where(w <= 80, 1, 2))).astype(int)
            fill = None
            if args.dmi_nodata != "blocked":
                macro = R.macro_sic_from_amsr(e["lat"], e["lon"], e["time"], log)
                if macro is None:
                    log(f"  {e['episode']}: no AMSR2 - cells without data stay impassable")
                else:
                    sic, land_a, k, rr = align_to_dmi(w, *macro)
                    fill, land = land & ~land_a, land & land_a
                    log(f"  {e['episode']}: filled {fill.mean():.0%} of the window, land {land.mean():.0%}, "
                        f"AMSR2 orientation {k}, r = {rr:.2f}")
                    if args.dmi_nodata in ("water", "heavy"):
                        tc_dmi = np.where(fill, 0 if args.dmi_nodata == "water" else 2, tc_dmi)
        else:
            macro = R.macro_sic_from_amsr(e["lat"], e["lon"], e["time"], log)
            if macro is None:
                macro = R.macro_sic_synthetic(rng)
            sic, land = macro
        for t in range(args.trials):
            tc = tc_dmi if dmi is not None else true_classes(sic, land, rng, args.floe_sigma, thr)
            if dmi is not None and fill is not None and args.dmi_nodata == "amsr":
                tc = np.where(fill, true_classes(sic, land, rng, args.floe_sigma, thr), tc_dmi)
            ends = R.pick_endpoints(land, rng)
            if ends is None:
                continue
            s, gl = ends
            # observations
            if args.obs_model == "matrix":
                am = R.sample_obs(tc, gi["oconf"]["amsr"][v], rng, block=6)
                fu = R.sample_obs(tc, gi["oconf"]["fusion"][v], rng, block=args.cam_block, cond=am)
                ca = R.sample_obs(tc, gi["oconf"]["camera"][v], rng, block=args.cam_block)
                Pam = Pfu = Pca = None
            else:
                don = None
                if args.cam_donor:      # separate generator: the true fields and routes are the same as without the flag
                    ev = gi["opool_t"].eps_vis[v]
                    don = ev[rng_don.integers(len(ev))] if ev else None
                am, Pam = resample_obs(tc, 6, rng, gi["opool_t"], None, probs["amsr"], preds["amsr"], v)
                fu, Pfu = resample_obs(tc, args.cam_block, rng, gi["opool_ta"], None, probs["fusion"],
                                       preds["fusion"], v, cond=am, donor=don)
                ca, Pca = resample_obs(tc, args.cam_block, rng, gi["opool_t"], None, probs["camera"],
                                       preds["camera"], v, donor=don)
            r = R.CAM_RANGE[v]
            oracle = run_route(tc, land, s, gl, R.CLASS_COST[tc], None, r)[0]
            for src, pl in strategies:
                if pl in ("hard", "bayes", "cvar"):
                    ct = gi["ct"][pl][v]
                    far_am = ct["amsr"][am]
                    near = {"amsr": None, "fusion": ct["fusion"][am, fu], "camera": ct["camera"][ca]}[src]
                    far = np.full((N, N), ct["unknown"]) if src == "camera" else far_am
                else:
                    tf = (lambda P_, s_: apply_temperature(P_, gi["temp"][s_])) if pl == "probs_t" else (lambda P_, s_: P_)
                    unk = gi["ct"]["bayes"][v]["unknown"]
                    far = np.full((N, N), unk) if src == "camera" else tf(Pam, "amsr") @ R.CLASS_COST
                    near = {"amsr": None, "fusion": tf(Pfu, "fusion") @ R.CLASS_COST,
                            "camera": tf(Pca, "camera") @ R.CLASS_COST}[src]
                c, h, L, ok, km = run_route(tc, land, s, gl, far, near, r)
                rows.append({"episode": e["episode"], "vis": v, "trial": t, "source": src, "planner": pl,
                             "regret": c / oracle - 1, "heavy_km": h, "dist_km": L, "reached": ok,
                             "km0": km[0], "km1": km[1], "km2": km[2]})
        el = time.time() - t0
        log(f"  episode {i + 1}/{len(eps)}  (~{el / (i + 1) * (len(eps) - i - 1) / 60:.0f} min left)")
    if args.smoke:
        per_ep = (time.time() - t0) / len(eps) / max(args.trials, 1)
        log(f"ESTIMATE: ~{per_ep:.1f} s per route -> 42 episodes x 20 routes ≈ {per_ep * 42 * 20 / 60:.0f} min per run")

    if args.cam_donor:
        nd = sum(p.n_donor for g_ in G.values() for p in {id(g_["opool_t"]): g_["opool_t"], id(g_["opool_ta"]): g_["opool_ta"]}.values())
        na = sum(p.n_draw for g_ in G.values() for p in {id(g_["opool_t"]): g_["opool_t"], id(g_["opool_ta"]): g_["opool_ta"]}.values())
        log(f"Donor: camera/fusion blocks {na}, from the donor episode {nd} ({nd / max(na, 1):.0%}; for the rest the donor has no frames of the needed class)")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "trials.csv", index=False, float_format="%.4f")
    summarize(df, run_dir, log, [float(x) for x in args.class_speed.split(",")])
    log(f"DONE: {run_dir}")


def summarize(df, run_dir, log, speed_kn=(12.0, 6.0, 3.0)):
    out, res = [], {}
    if "km0" in df:
        df = df.assign(time_h=sum(df[f"km{k}"] / (speed_kn[k] * KN_KMH) for k in range(3)))
    say = lambda s_: (log(s_), out.append(s_))
    say("\n" + "=" * 70 + "\nREGRET, % [95% CI by episode]")
    for vi, vn in enumerate(["all"] + list(T.VIS_NAMES)):
        dv = df if vi == 0 else df[df["vis"] == vi - 1]
        if dv.empty:
            continue
        say(f"\n{vn} ({dv['episode'].nunique()} ep.)")
        for (src, pl), s in dv.groupby(["source", "planner"], sort=False):
            m = s["regret"].mean()
            lo, hi = boot_ci(s, "regret")
            say(f"  {src:7s} {pl:6s} {100 * m:6.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]   "
                f"heavy ice {s['heavy_km'].mean():5.1f} km  path {s['dist_km'].mean():5.1f} km  "
                + (f"time {s['time_h'].mean():5.2f} h  " if "time_h" in s else "") + f"reached {s['reached'].mean():.0%}")
            res[f"{vn}|{src}|{pl}"] = {"regret": m, "ci95": [lo, hi]}
    say("\n" + "=" * 70 + "\nPAIRWISE DIFFERENCES, pp (<0 = first is better)")
    w = df.pivot_table(index=["episode", "vis", "trial"], columns=["source", "planner"], values="regret")
    pairs = []
    for src in ("amsr", "fusion", "camera"):
        pairs += [((src, "bayes"), (src, "hard")), ((src, "probs"), (src, "bayes")),
                  ((src, "probs_t"), (src, "bayes")), ((src, "probs_t"), (src, "probs")),
                  ((src, "cvar"), (src, "bayes")), ((src, "cvar"), (src, "hard"))]
    for pl in PLANNERS:
        pairs += [(("fusion", pl), ("amsr", pl)), (("camera", pl), ("amsr", pl))]
    for vi, vn in enumerate(["all"] + list(T.VIS_NAMES)):
        wv = w if vi == 0 else w[w.index.get_level_values("vis") == vi - 1]
        if wv.empty:
            continue
        for a, b in pairs:
            if a not in wv.columns or b not in wv.columns:
                continue
            x = pd.DataFrame({"episode": wv.index.get_level_values("episode"), "d": (wv[a] - wv[b]).values}).dropna()
            lo, hi = boot_ci(x, "d")
            sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
            say(f"  {vn:14s} {a[0]}|{a[1]} - {b[0]}|{b[1]}: {100 * x['d'].mean():+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]  {sig}")
            res[f"diff|{vn}|{a[0]}|{a[1]}-{b[0]}|{b[1]}"] = {"diff": x["d"].mean(), "ci95": [lo, hi]}
    say("\n" + "=" * 70 + "\nPATH LENGTH hard - bayes, km [95% CI by episode]")
    wl = df.pivot_table(index=["episode", "vis", "trial"], columns=["source", "planner"], values="dist_km")
    for src in ("amsr", "fusion", "camera"):
        if (src, "hard") in wl.columns and (src, "bayes") in wl.columns:
            x = pd.DataFrame({"episode": wl.index.get_level_values("episode"),
                              "d": (wl[(src, "hard")] - wl[(src, "bayes")]).values}).dropna()
            lo, hi = boot_ci(x, "d")
            say(f"  {src:7s} hard - bayes: {x['d'].mean():+6.2f} km [{lo:+.2f}, {hi:+.2f}]")
            res[f"dist|{src}|hard-bayes"] = {"diff": x["d"].mean(), "ci95": [lo, hi]}
    if "time_h" in df:
        say("\n" + "=" * 70 + f"\nTRANSIT TIME, h (speed {speed_kn} kn); differences over all episodes [95% CI]")
        wt = df.pivot_table(index=["episode", "vis", "trial"], columns=["source", "planner"], values="time_h")
        for a, b in pairs:
            if a in wt.columns and b in wt.columns:
                x = pd.DataFrame({"episode": wt.index.get_level_values("episode"), "d": (wt[a] - wt[b]).values}).dropna()
                lo, hi = boot_ci(x, "d")
                sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
                say(f"  {a[0]}|{a[1]} - {b[0]}|{b[1]}: {x['d'].mean():+6.3f} h [{lo:+.3f}, {hi:+.3f}]  {sig}")
                res[f"time|{a[0]}|{a[1]}-{b[0]}|{b[1]}"] = {"diff": x["d"].mean(), "ci95": [lo, hi]}
    (run_dir / "summary.txt").write_text("\n".join(out), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
