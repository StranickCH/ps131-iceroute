"""
Effect of the ice information source on the route (article 3, step 2).
Question: how do the errors of AMSR2 / camera / fusion MEASURED on PS131, and the drop of camera quality
in poor weather, affect the cost and safety of a route in ice.

SETUP (empirically calibrated simulation)
  * Scenario = a labelled hourly PS131 episode: its date, position (GPS) and the observer's visibility.
  * True ice field 100x100 km (1 km cells) around the ship:
      macro concentration = the real AMSR2 Bremen map of that day (6.25 km, official grid),
      inside it a floe structure (thresholded Gaussian field with local concentration = AMSR2),
      cell class from concentration: 0 open/very open (<0.35), 1 close (0.35-0.85), 2 heavy (>=0.85).
  * Sensors report a cell class drawn from THEIR REAL confusion matrices P(observation | truth)
    from cross-validation on PS131 (fusion_oof.csv), separately for each visibility class:
      amsr   - the whole map, one error per 6x6 km pixel (like a real AMSR2 pixel)
      camera - only within the view radius r(visibility), no information beyond it
      fusion - the whole AMSR2 map + the fusion-model observation within the camera view radius
      oracle - the true map (upper bound)
  * Router: cost map C_ice -> shortest path (MCP, 8-connectivity), with REPLANNING
    as the camera reveals new cells (receding horizon). The path cost is computed
    on the TRUE field. This is where the DRL agent of the dissertation plugs in later (same map interface).

METRICS (per scenario): regret = cost / oracle cost - 1; km in heavy ice; path length.
Aggregated by visibility class; 95% CI by episode bootstrap; pairwise differences between sources.

RUN
    python route_ice_sources_ps131.py --oof runs\\fusion_YYYYMMDD_HHMM\\fusion_oof.csv
    python route_ice_sources_ps131.py --oof ... --fusion-model concat --trials 20
    python route_ice_sources_ps131.py --synthetic --oof ...      # without AMSR2 files (check)
OUTPUT: runs\\route_<date>\\ : log.txt, route_trials.csv, route_summary.txt/.json, route_by_visibility.png,
       route_examples.png, confusion_used.json (confusion matrices used)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T   # noqa: E402  (Log, RUNS, class names)

CELL_KM = 1.0
N = 100                          # 100 x 100 km
SUB = 4                          # 250 m sub-grid for the floe structure
CLASS_COST = np.array([1.0, 2.5, 8.0])   # relative cost of 1 km: open/very open, close, heavy (Arc4)
CAM_RANGE = {0: 1, 1: 3, 2: 5}           # camera view radius, cells (km), by visibility class
REPLAN_EVERY = 3
# Camera errors are spatially correlated: a frame sees a whole patch, and if the model confuses
# close ice with heavy ice, it does so over the whole patch. So the error is drawn per CAM_BLOCK x CAM_BLOCK km block.
CAM_BLOCK = 3
VIS_NAMES = T.VIS_NAMES
ICE_NAMES = T.ICE_NAMES


# ============================================================
# Confusion matrices from out-of-fold predictions
# ============================================================
SHRINK_EP = 2.0   # strength of shrinkage towards the pooled matrix, in episodes


def _counts(s, cols, shape):
    """Weighted counts (episode weight = 1, shared among its frames)."""
    C = np.zeros(shape)
    for key, w in s.groupby(cols)["w"].sum().items():
        key = key if isinstance(key, tuple) else (key,)
        if all(k >= 0 for k in key):
            C[tuple(int(k) for k in key)] += w
    return C


def _norm(C, prior):
    """Rows -> probabilities; a row without data is taken from the prior."""
    out = prior.copy()
    tot = C.sum(-1, keepdims=True)
    ok = tot[..., 0] > 0
    out[ok] = C[ok] / tot[ok]
    return out


def confusion_tables(oof, fusion_col, log):
    """P(obs | true) by visibility class, shrunk towards the pooled matrix (empirical Bayes):
    M_v = (N_v + k*M_all) / (n_v + k), k = SHRINK_EP episodes. For fusion - conditional on the AMSR2 observation:
    P(fusion | true, amsr), because fusion itself uses AMSR2 and their errors are strongly linked."""
    d = oof[(oof["ice_label"] >= 0)].copy()
    d["w"] = 1.0 / d.groupby("episode")["episode"].transform("size")
    for c in ("pred_amsr", "pred_cam", fusion_col):
        if c not in d:
            raise SystemExit(f"no column {c} in {fusion_col} - run fusion_ps131.py first")
    eye = np.eye(3) * 0.9 + 0.1 / 3
    out = {"amsr": {}, "camera": {}, "fusion": {}}

    def shrunk(sub_all, sub_v, cols, shape, base_prior):
        C_all = _counts(sub_all, cols, shape)
        M_all = _norm(C_all + 1e-3, base_prior)
        C_v = _counts(sub_v, cols, shape)
        n_v = C_v.sum(-1, keepdims=True)
        return (C_v + SHRINK_EP * M_all) / (n_v + SHRINK_EP), M_all

    for src, col in (("amsr", "pred_amsr"), ("camera", "pred_cam")):
        _, M_all = shrunk(d, d, ["ice_label", col], (3, 3), eye)
        out[src]["all"] = M_all
        for v in (0, 1, 2):
            if src == "amsr":                      # AMSR2 does not physically depend on visibility
                out[src][v] = M_all
            else:
                out[src][v], _ = shrunk(d, d[d["vis_label"] == v], ["ice_label", col], (3, 3), eye)
    prior3 = np.broadcast_to(eye[:, None, :], (3, 3, 3)).copy()
    for v in (0, 1, 2):
        out["fusion"][v], J_all = shrunk(d, d[d["vis_label"] == v], ["ice_label", "pred_amsr", fusion_col],
                                         (3, 3, 3), prior3)
    out["fusion"]["all"] = J_all
    # prior class frequency (episode weight = 1) - for the Bayesian planner
    pr = _counts(d, ["ice_label"], (3,))
    out["prior"] = {"all": pr / pr.sum()}

    # check: marginal accuracy of fusion = sum_a P(a|t) P(f=t|t,a)
    for v in (0, 1, 2):
        marg = np.einsum("ta,taf->tf", out["amsr"][v], out["fusion"][v])
        for src, M in (("amsr", out["amsr"][v]), ("camera", out["camera"][v]), ("fusion", marg)):
            n_ep = d[d["vis_label"] == v].groupby("ice_label")["episode"].nunique().to_dict()
            log(f"  {src:7s} visibility {VIS_NAMES[v]:14s} P(correct | truth): "
                + ", ".join(f"{ICE_NAMES[k]} {M[k, k]:.2f}" for k in range(3))
                + (f"   (episodes per class: {n_ep})" if src == "amsr" else ""))
    return out


def cost_tables(conf, v, planner):
    """Cost of a 1 km cell assumed by the planner, given the observation.
    hard  - cost of the observed class (the label is trusted as is);
    bayes - expected cost E[c | observation] = sum_t P(t | observation) c_t, where P(t | observation)
            follows from Bayes' rule with the prior frequencies and the same confusion matrices. A false "water"
            does not lure the route if the sensor often makes this error."""
    c = CLASS_COST
    if planner == "hard":
        return {"amsr": c.copy(), "camera": c.copy(), "fusion": np.tile(c, (3, 1)), "unknown": c[1]}
    p0 = conf["prior"]["all"]
    Ma, Mc, J = conf["amsr"][v], conf["camera"][v], conf["fusion"][v]
    post_a = p0[:, None] * Ma                              # [t, a]
    post_c = p0[:, None] * Mc                              # [t, o]
    post_f = p0[:, None, None] * Ma[:, :, None] * J        # [t, a, f]
    e = lambda P: np.tensordot(c, P, axes=(0, 0)) / np.clip(P.sum(0), 1e-12, None)
    return {"amsr": e(post_a), "camera": e(post_c), "fusion": e(post_f), "unknown": float(p0 @ c)}


# ============================================================
# True field
# ============================================================
def macro_sic_from_amsr(lat, lon, date, log_once):
    """16x16 AMSR2 pixel window (100 km) around the ship, upsampled to 1 km. NaN = land/no data."""
    import collocate_ps131 as C
    from scipy.ndimage import zoom
    fmt = "hdf" if (C.AMSR_DIR / f"asi-AMSR2-n6250-{date:%Y%m%d}-v5.4.hdf").exists() else "tif"
    p = C.AMSR_DIR / f"asi-AMSR2-n6250-{date:%Y%m%d}-v5.4.{fmt}"
    if not p.exists():
        return None
    grid = C.load_grid()
    if grid is None:
        return None
    a = C.read_amsr(p)
    r, c, _ = C.rowcol_from_grid(grid, [lat], [lon])
    r, c = int(r[0]), int(c[0])
    h = 8
    if r - h < 0 or c - h < 0 or r + h > a.shape[0] or c + h > a.shape[1]:
        return None
    w = a[r - h:r + h, c - h:c + h].astype(float)
    land = np.isnan(w)
    w_f = np.where(land, np.nanmean(w) if np.isfinite(w).any() else 0.0, w)
    sic = np.clip(zoom(w_f, N / w.shape[0], order=1), 0, 1)
    land_hi = zoom(land.astype(float), N / w.shape[0], order=0) > 0.5
    return sic, land_hi


def macro_sic_synthetic(rng):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((N, N)), 15)
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    return np.clip(1.3 * f - 0.15, 0, 1), np.zeros((N, N), bool)


def true_classes(sic, land, rng):
    """Floe structure at 250 m with local concentration = sic, then the class of each 1 km cell."""
    from scipy.ndimage import gaussian_filter, zoom
    g = gaussian_filter(rng.standard_normal((N * SUB, N * SUB)), 6)   # floe clusters ~1-3 km
    u = (np.argsort(np.argsort(g.ravel())) / g.size).reshape(g.shape)     # uniform distribution
    sic_hi = zoom(sic, SUB, order=1)
    ice = u < sic_hi
    conc = ice.reshape(N, SUB, N, SUB).mean(axis=(1, 3))
    cls = np.where(conc < 0.35, 0, np.where(conc < 0.85, 1, 2))
    return cls, conc


# ============================================================
# Sensors
# ============================================================
def block_uniform(rng, block):
    """Uniform numbers shared by a block x block km block -> spatially correlated errors."""
    nb = int(np.ceil(N / block))
    return np.kron(rng.random((nb, nb)), np.ones((block, block)))[:N, :N]


def block_mode(true_cls, block):
    """Dominant true class of the block, spread back to the cells (what a pixel "sees")."""
    if block == 1:
        return true_cls
    nb = int(np.ceil(N / block))
    pad = np.pad(true_cls, ((0, nb * block - N), (0, nb * block - N)), mode="edge")
    b = pad.reshape(nb, block, nb, block).transpose(0, 2, 1, 3).reshape(nb, nb, -1)
    mode = np.apply_along_axis(lambda x: np.bincount(x, minlength=3).argmax(), -1, b)
    return np.kron(mode, np.ones((block, block), int))[:N, :N]


def sample_obs(true_cls, M, rng, block=1, cond=None):
    """Observed class: P(obs | true) or, if cond is given, P(obs | true, cond)."""
    t = block_mode(true_cls, block)
    probs = M[t] if cond is None else M[t, cond]
    u = block_uniform(rng, block)
    return (u[..., None] > probs.cumsum(-1)).sum(-1).clip(0, 2)


def disk(pos, r):
    yy, xx = np.ogrid[:N, :N]
    return (yy - pos[0]) ** 2 + (xx - pos[1]) ** 2 <= r * r


# ============================================================
# Routing with replanning
# ============================================================
def plan(cost, start, goal):
    from skimage.graph import MCP_Geometric
    m = MCP_Geometric(cost)
    m.find_costs([start], [goal])
    return m.traceback(goal)


def run_strategy(name, true_cls, land, start, goal, obs, vis, rng, ct):
    """obs: dictionary of observations drawn in advance; the camera 'sees' a cell when the ship is within range.
    ct: cost tables by observation (cost_tables); belief - the 1 km cost map seen by the planner."""
    true_cost = np.where(land, np.inf, CLASS_COST[true_cls])
    if name == "oracle":
        belief = CLASS_COST[true_cls].astype(float)
    elif name in ("amsr", "fusion"):
        belief = ct["amsr"][obs["amsr"]].astype(float)
    else:                                  # camera: no information -> prior cost
        belief = np.full((N, N), ct["unknown"], float)
    seen = np.zeros((N, N), bool)
    r = CAM_RANGE[vis]
    pos = start
    path, total, heavy, steps = [start], 0.0, 0, 0
    route = None
    while pos != goal and steps < 4 * N:
        if name in ("camera", "fusion"):
            new = disk(pos, r) & ~seen
            if new.any():
                if name == "camera":
                    belief[new] = ct["camera"][obs["camera"][new]]
                else:
                    belief[new] = ct["fusion"][obs["amsr"][new], obs["fusion"][new]]
                seen |= new
        if route is None or steps % REPLAN_EVERY == 0 or name in ("camera", "fusion") and new.any():
            bcost = np.where(land, 1e6, belief)
            route = plan(bcost, pos, goal)
            k = 1
        nxt = tuple(route[k]) if k < len(route) else goal
        k += 1
        step_len = CELL_KM * (np.sqrt(2) if nxt[0] != pos[0] and nxt[1] != pos[1] else 1.0)
        c = true_cost[nxt]
        if not np.isfinite(c):
            total += 1e3        # entering land (should not happen)
        else:
            total += step_len * c
        heavy += step_len * (true_cls[nxt] == 2)
        pos = nxt
        path.append(pos)
        steps += 1
    return {"cost": total, "heavy_km": heavy, "len_km": len(path) - 1, "reached": pos == goal, "path": path}


def pick_endpoints(land, rng):
    for _ in range(200):
        side = rng.integers(2)
        if side == 0:
            s, g = (rng.integers(10, 90), 2), (rng.integers(10, 90), N - 3)
        else:
            s, g = (2, rng.integers(10, 90)), (N - 3, rng.integers(10, 90))
        if rng.random() < 0.5:
            s, g = g, s
        if not land[s] and not land[g]:
            return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))
    return None


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="fusion_oof.csv from fusion_ps131.py")
    ap.add_argument("--fusion-model", default="gated", choices=["gated", "stack", "concat"])
    ap.add_argument("--trials", type=int, default=10, help="random routes/realisations per episode")
    ap.add_argument("--synthetic", action="store_true", help="synthetic macro concentration instead of AMSR2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cam-block", type=int, default=3, help="size of the correlated camera error block, km")
    ap.add_argument("--planner", default="bayes", choices=["bayes", "hard"],
                    help="bayes: expected cost from P(truth | observation); hard: cost of the observed class")
    ap.add_argument("--class-cost", default="1,2.5,8", help="cost of 1 km: open,close,heavy")
    ap.add_argument("--cam-range-scale", type=float, default=1.0, help="multiplier of the camera view radius")
    ap.add_argument("--tag", default="", help="run tag for the summary table")
    args = ap.parse_args()

    global CAM_BLOCK, CLASS_COST, CAM_RANGE
    CAM_BLOCK = args.cam_block
    CLASS_COST = np.array([float(x) for x in args.class_cost.split(",")])
    CAM_RANGE = {k: max(1, int(round(r * args.cam_range_scale))) for k, r in CAM_RANGE.items()}
    run_dir = T.RUNS / f"route_{args.planner}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(dict(vars(args), cam_range=CAM_RANGE), ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    log = T.Log(run_dir / "log.txt")
    log(f"Routing by ice source | {vars(args)}")
    oof = pd.read_csv(args.oof, low_memory=False)
    oof["time"] = pd.to_datetime(oof["time"])
    conf = confusion_tables(oof, f"pred_{args.fusion_model}", log)
    cts = {v: cost_tables(conf, v, args.planner) for v in (0, 1, 2)}
    for v in (0, 1, 2):
        ct = cts[v]
        log(f"  planner {args.planner}, {VIS_NAMES[v]}: cost by observation "
            f"amsr {np.round(ct['amsr'], 2).tolist()}  camera {np.round(ct['camera'], 2).tolist()}  "
            f"fusion[amsr, f] {np.round(ct['fusion'], 2).tolist()}  no data {ct['unknown']:.2f}")
    json.dump({s: {str(v): np.round(m, 4).tolist() for v, m in d.items()} for s, d in conf.items()},
              open(run_dir / "confusion_used.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    eps = (oof[oof["vis_label"] >= 0].groupby("episode")
           .agg(time=("time", "min"), lat=("lat", "median"), lon=("lon", "median"),
                vis=("vis_label", lambda s: int(s.mode().iat[0]))).reset_index())
    log(f"Scenario episodes: {len(eps)}; by visibility: "
        + ", ".join(f"{VIS_NAMES[v]} {int((eps['vis'] == v).sum())}" for v in range(3)))

    rng_master = np.random.default_rng(args.seed)
    rows, examples = [], []
    strategies = ["oracle", "amsr", "camera", "fusion"]
    for i, e in eps.iterrows():
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        macro = None if args.synthetic else macro_sic_from_amsr(e["lat"], e["lon"], e["time"], log)
        if macro is None:
            if not args.synthetic:
                log(f"  {e['episode']}: no AMSR2 window (grid edge/file) - synthetic macro concentration")
            macro = macro_sic_synthetic(rng)
        sic, land = macro
        for t in range(args.trials):
            true_cls, _ = true_classes(sic, land, rng)
            ends = pick_endpoints(land, rng)
            if ends is None:
                continue
            s, g = ends
            v = int(e["vis"])
            am = sample_obs(true_cls, conf["amsr"][v], rng, block=6)
            obs = {"amsr": am,
                   "camera": sample_obs(true_cls, conf["camera"][v], rng, block=CAM_BLOCK),
                   "fusion": sample_obs(true_cls, conf["fusion"][v], rng, block=CAM_BLOCK, cond=am)}
            res = {k: run_strategy(k, true_cls, land, s, g, obs, v, rng, cts[v]) for k in strategies}
            base = res["oracle"]["cost"]
            for k in strategies:
                rows.append({"episode": e["episode"], "vis": v, "trial": t, "strategy": k,
                             "cost": res[k]["cost"], "regret": res[k]["cost"] / base - 1,
                             "heavy_km": res[k]["heavy_km"], "len_km": res[k]["len_km"],
                             "reached": res[k]["reached"],
                             "ice_share_true": float((true_cls > 0).mean())})
            if t == 0 and len(examples) < 3 and (true_cls == 1).mean() > 0.15:
                examples.append((e["episode"], v, true_cls, obs["amsr"], land, {k: res[k]["path"] for k in strategies}))
        if (i + 1) % 5 == 0:
            log(f"  episodes processed {i + 1}/{len(eps)}")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "route_trials.csv", index=False, float_format="%.4f")
    summarize(df, run_dir, log)
    plot_examples(examples, run_dir)
    log(f"DONE: {run_dir}")


def boot_ep(df, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = {e: x for e, x in df.groupby("episode")}
    E = list(g)
    vals = [fn(pd.concat([g[e] for e in rng.choice(E, len(E))])) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def summarize(df, run_dir, log):
    out, res = [], {}
    say = lambda s: (log(s), out.append(s))
    say("\n" + "=" * 70 + "\nMEANS OVER SCENARIOS (regret = cost/oracle - 1; 95% CI by episode)")
    for vname, sub in [("all", df)] + [(VIS_NAMES[v], df[df["vis"] == v]) for v in range(3)]:
        if sub.empty:
            continue
        say(f"\nVisibility: {vname} ({sub['episode'].nunique()} ep., {sub['trial'].nunique() * sub['episode'].nunique()} routes)")
        for k in ["amsr", "camera", "fusion"]:
            s = sub[sub["strategy"] == k]
            m = s["regret"].mean()
            lo, hi = boot_ep(s, lambda x: x["regret"].mean())
            say(f"  {k:7s} regret {100 * m:6.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]   "
                f"heavy ice {s['heavy_km'].mean():5.1f} km   length {s['len_km'].mean():5.1f} km   "
                f"reached {s['reached'].mean():.0%}")
            res[f"{vname}|{k}"] = {"regret": m, "ci95": [lo, hi], "heavy_km": s["heavy_km"].mean()}
        o = sub[sub["strategy"] == "oracle"]
        say(f"  oracle  heavy ice {o['heavy_km'].mean():5.1f} km   length {o['len_km'].mean():5.1f} km")

    say("\n" + "=" * 70 + "\nPAIRWISE regret DIFFERENCES (same routes; <0 = first is better)")
    wide = df.pivot_table(index=["episode", "vis", "trial"], columns="strategy", values="regret").reset_index()
    for vname, sub in [("all", wide)] + [(VIS_NAMES[v], wide[wide["vis"] == v]) for v in range(3)]:
        for a, b in (("fusion", "amsr"), ("fusion", "camera"), ("camera", "amsr")):
            if sub.empty:
                continue
            d = (sub[a] - sub[b]).mean()
            lo, hi = boot_ep(sub, lambda x: (x[a] - x[b]).mean())
            sig = "SIGNIFICANT" if hi < 0 or lo > 0 else "not significant"
            say(f"  {vname:14s} {a} - {b}: {100 * d:+6.2f} pp [{100 * lo:+.2f}, {100 * hi:+.2f}]  {sig}")
            res[f"diff|{vname}|{a}-{b}"] = {"diff": d, "ci95": [lo, hi]}
    (run_dir / "route_summary.txt").write_text("\n".join(out), encoding="utf-8")
    (run_dir / "route_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                                encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = {"amsr": "#8c8c8c", "camera": "#4C72B0", "fusion": "#C44E52"}
        x = np.arange(3)
        for i, k in enumerate(["amsr", "camera", "fusion"]):
            s = df[df["strategy"] == k]
            reg = [100 * s[s["vis"] == v]["regret"].mean() for v in range(3)]
            hv = [s[s["vis"] == v]["heavy_km"].mean() for v in range(3)]
            axes[0].bar(x + (i - 1) * 0.27, reg, 0.27, label=k, color=colors[k])
            axes[1].bar(x + (i - 1) * 0.27, hv, 0.27, color=colors[k])
        o = df[df["strategy"] == "oracle"]
        axes[1].plot(x, [o[o["vis"] == v]["heavy_km"].mean() for v in range(3)], "k_", ms=30, label="oracle")
        for ax, t in zip(axes, ("regret, % vs optimal route", "km in heavy ice (>=9/10)")):
            ax.set_xticks(x)
            ax.set_xticklabels(VIS_NAMES)
            ax.set_xlabel("visibility (PS131 observer)")
            ax.set_title(t)
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(run_dir / "route_by_visibility.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"[plot skipped: {e}]")


def plot_examples(examples, run_dir):
    if not examples:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#1f4e79", "#9ecae1", "#ffffff"])
        fig, axes = plt.subplots(len(examples), 2, figsize=(9, 4.3 * len(examples)), squeeze=False)
        colors = {"oracle": "k", "amsr": "#8c8c8c", "camera": "#4C72B0", "fusion": "#C44E52"}
        for r, (ep, v, tc, am, land, paths) in enumerate(examples):
            for c, (img, title) in enumerate(((tc, "true field"), (am, "what AMSR2 sees"))):
                ax = axes[r, c]
                ax.imshow(np.where(land, np.nan, img), cmap=cmap, vmin=0, vmax=2)
                for k, p in paths.items():
                    p = np.array(p)
                    ax.plot(p[:, 1], p[:, 0], color=colors[k], lw=1.6, label=k)
                ax.set_title(f"episode {ep}, visibility {VIS_NAMES[v]}: {title}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[r, 0].legend(fontsize=7, loc="lower right")
        fig.suptitle("dark blue - open/very open, light blue - close, white - heavy ice", fontsize=9)
        fig.tight_layout()
        fig.savefig(run_dir / "route_examples.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
