# -*- coding: utf-8 -*-
"""Additional analyses for the review (article 3, PS131) - only from finished results, no new runs.
1) DMI chart time lag and the <= 24 h subset; 2) tail metrics; 3) Holm correction;
4) time offsets of the labels and 60 vs 180 min on common episodes; 5) autocorrelation of camera errors along the track.
Run:  python reviewer_analyses_ps131.py .   (folder with runs, 08_icecharts_2022, ps131_index)
"""
import sys, glob, os
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RUNS = os.path.join(ROOT, "runs")
NB = 10000
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def run_dir(tag):
    d = sorted(glob.glob(os.path.join(RUNS, f"routev3_{tag}_2*")))
    return d[-1]


def load(tag):
    return pd.read_csv(os.path.join(run_dir(tag), "trials.csv"))


def boot(x_ep, rng):
    """x_ep - episode means; returns bootstrap means"""
    idx = rng.integers(0, len(x_ep), (NB, len(x_ep)))
    return x_ep[idx].mean(1)


def pair(df, a, b, eps=None, seed=0):
    """regret difference a-b (source, planner): mean over trials, 95% CI and p (episode bootstrap)"""
    w = df.pivot_table(index=["episode", "vis", "trial"], columns=["source", "planner"], values="regret")
    x = pd.DataFrame({"episode": w.index.get_level_values("episode"), "d": (w[a] - w[b]).values}).dropna()
    if eps is not None:
        x = x[x["episode"].isin(eps)]
    g = x.groupby("episode")["d"].mean().values
    bs = boot(g, np.random.default_rng(seed))
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * x["d"].mean(), 100 * lo, 100 * hi, p, len(g)


KEY = [(("amsr", "bayes"), ("amsr", "hard")), (("fusion", "bayes"), ("fusion", "hard")),
       (("camera", "bayes"), ("camera", "hard")),
       (("amsr", "cvar"), ("amsr", "bayes")), (("fusion", "cvar"), ("fusion", "bayes")),
       (("camera", "cvar"), ("camera", "bayes")),
       (("fusion", "hard"), ("amsr", "hard")), (("fusion", "bayes"), ("amsr", "bayes")),
       (("fusion", "cvar"), ("amsr", "cvar"))]


def fmt(a, b):
    return f"{a[0]}|{a[1]} - {b[0]}|{b[1]}"


def show_pairs(df, title, eps=None):
    say(f"\n--- {title}")
    for a, b in KEY:
        m, lo, hi, p, n = pair(df, a, b, eps)
        say(f"  {fmt(a, b):32s} {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]  p={p:.4f}  ep.={n}")


# ---------------------------------------------------------------- 1) DMI: time lag
say("=" * 70 + "\n1) TIME LAG OF THE DMI CHARTS RELATIVE TO THE EPISODE")
cov = pd.read_csv(os.path.join(ROOT, "08_icecharts_2022", "dmi_coverage.csv"))
dmi = load("dmi70")
e70 = sorted(dmi["episode"].unique())
c70 = cov[cov["episode"].isin(e70)]
q = np.percentile(c70["dt_h"], [0, 25, 50, 75, 100])
say(f"Episodes in dmi70: {len(e70)}; |dt|, h: min {q[0]:.1f}, Q1 {q[1]:.1f}, median {q[2]:.1f}, Q3 {q[3]:.1f}, max {q[4]:.1f}")
for t in (6, 12, 24, 36):
    say(f"  |dt| <= {t:2d} h: {int((c70['dt_h'].abs() <= t).sum())} ep.")
say("  by chart type: " + ", ".join(f"{k} {v}" for k, v in c70["set"].value_counts().items()))
e24 = c70.loc[c70["dt_h"].abs() <= 24, "episode"].tolist()
for tag in ("dmi70", "dmi70amsr"):
    d = load(tag)
    show_pairs(d, f"{tag}: all {d['episode'].nunique()} ep.")
    show_pairs(d, f"{tag}: only |dt| <= 24 h", e24)

# ---------------------------------------------------------------- 2) tails
say("\n" + "=" * 70 + "\n2) TAIL METRICS (over all trials)")
for tag in ("main32", "dmi70"):
    d = load(tag)
    say(f"\n--- {tag}: regret, %  |  heavy ice along the route, km")
    say(f"  {'':7s}{'':8s}{'mean':>7s}{'p90':>7s}{'p95':>7s}{'CVaR90':>8s}   {'km mean':>8s}{'km p95':>8s}")
    for (s, pl), g in d.groupby(["source", "planner"]):
        if pl not in ("hard", "bayes", "cvar"):
            continue
        r = 100 * g["regret"].values
        p90, p95 = np.percentile(r, [90, 95])
        say(f"  {s:7s}{pl:8s}{r.mean():7.1f}{p90:7.1f}{p95:7.1f}{r[r >= p90].mean():8.1f}   "
            f"{g['heavy_km'].mean():8.1f}{np.percentile(g['heavy_km'], 95):8.1f}")

# ---------------------------------------------------------------- 3) Holm
say("\n" + "=" * 70 + "\n3) HOLM CORRECTION (main32, family of 9 key comparisons)")
d = load("main32")
rows = [(fmt(a, b),) + pair(d, a, b) for a, b in KEY]
ps = np.array([r[4] for r in rows])
order = np.argsort(ps)
adj = np.empty_like(ps)
run = 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (len(ps) - k) * ps[i]))
    adj[i] = run
for r, pa in zip(rows, adj):
    say(f"  {r[0]:32s} {r[1]:+6.2f}  p={r[4]:.4f}  p_Holm={pa:.4f}  {'sig.' if pa < 0.05 else 'n.s.'}")
say(f"  (smallest p resolvable by the bootstrap = 1/{NB} -> 2e-4)")

# ---------------------------------------------------------------- 4) labels
say("\n" + "=" * 70 + "\n4) TIME OFFSET OF THE LABELS (frame - nearest bridge report)")
lab = pd.read_csv(os.path.join(ROOT, "ps131_index", "ps131_labeled.csv"), low_memory=False,
                  usecols=["episode", "dt_ci_min", "dt_vv_code_min", "ice_obs_class"])
for col, nm in (("dt_ci_min", "ice (ci)"), ("dt_vv_code_min", "visibility (VV)")):
    x = lab[col].dropna().abs()
    q = np.percentile(x, [25, 50, 75, 90])
    say(f"  {nm:15s} frames {len(x):6d}: Q1 {q[0]:.0f}, median {q[1]:.0f}, Q3 {q[2]:.0f}, p90 {q[3]:.0f} min; "
        f"<=30: {100 * (x <= 30).mean():.0f}%, <=60: {100 * (x <= 60).mean():.0f}%, <=120: {100 * (x <= 120).mean():.0f}%")
m180, m60 = load("main32"), load("lab60")
com = sorted(set(m180["episode"]) & set(m60["episode"]))
say(f"\n  episodes: window 180 min {m180['episode'].nunique()}, window 60 min {m60['episode'].nunique()}, common {len(com)}")
show_pairs(m180, "window 180 min, common episodes", com)
show_pairs(m60, "window 60 min, common episodes", com)

# ---------------------------------------------------------------- 5) autocorrelation of camera errors
say("\n" + "=" * 70 + "\n5) AUTOCORRELATION OF CAMERA ERRORS ALONG THE TRACK (fusion_oof, frames with an ice label)")
oof = pd.read_csv(os.path.join(sorted(glob.glob(os.path.join(RUNS, "fusion_2*")))[-1], "fusion_oof.csv"),
                  usecols=["episode", "time", "lat", "lon", "ice_label", "pred_cam", "pred_concat"])
oof = oof[oof["ice_label"] >= 0].copy()
oof["time"] = pd.to_datetime(oof["time"])
bins = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30])
for col, nm in (("pred_cam", "camera"), ("pred_concat", "fusion")):
    num = np.zeros(len(bins) - 1)
    cnt = np.zeros(len(bins) - 1)
    e_all = []
    for ep, g in oof.groupby("episode"):
        g = g.sort_values("time")
        if len(g) < 3:
            continue
        lat, lon = np.radians(g["lat"].values), np.radians(g["lon"].values)
        dx = 6371 * np.cos(lat[:-1]) * np.diff(lon)
        dy = 6371 * np.diff(lat)
        s = np.r_[0, np.cumsum(np.hypot(dx, dy))]
        e = (g[col].values != g["ice_label"].values).astype(float)
        e_all.append((s, e))
    mu = np.concatenate([e for _, e in e_all]).mean()
    var = np.concatenate([e for _, e in e_all]).var()
    for s, e in e_all:
        z = e - mu
        D = np.abs(s[:, None] - s[None, :])
        P = z[:, None] * z[None, :]
        iu = np.triu_indices(len(s), 1)
        k = np.digitize(D[iu], bins) - 1
        ok = (k >= 0) & (k < len(bins) - 1)
        np.add.at(num, k[ok], P[iu][ok])
        np.add.at(cnt, k[ok], 1)
    rho = num / np.maximum(cnt, 1) / var
    say(f"  {nm}: error rate {100 * mu:.1f}%; error correlation vs distance along the track:")
    say("    pooled (incl. between-episode):  " + "  ".join(f"{bins[i]:g}-{bins[i + 1]:g} km: {rho[i]:+.2f}" for i in range(len(rho))))
    # within episode: deviations from the episode's own error rate
    num_w, cnt_w, ss, nn = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), 0.0, 0
    for s, e in e_all:
        if e.std() == 0:
            continue
        z = e - e.mean()
        ss, nn = ss + (z ** 2).sum(), nn + len(z)
        D = np.abs(s[:, None] - s[None, :])
        iu = np.triu_indices(len(s), 1)
        k = np.digitize(D[iu], bins) - 1
        ok = (k >= 0) & (k < len(bins) - 1)
        np.add.at(num_w, k[ok], (z[:, None] * z[None, :])[iu][ok])
        np.add.at(cnt_w, k[ok], 1)
    rw = num_w / np.maximum(cnt_w, 1) / (ss / nn)
    say("    within episode:                  " + "  ".join(f"{bins[i]:g}-{bins[i + 1]:g} km: {rw[i]:+.2f}" for i in range(len(rw))))
    epm = np.array([e.mean() for _, e in e_all])
    say(f"    error rate by episode: median {100 * np.median(epm):.0f}%, episodes with rate > 50%: "
        f"{int((epm > 0.5).sum())} of {len(epm)}")

with open(os.path.join(ROOT, "reviewer_analyses_ps131.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
print("\nwritten reviewer_analyses_ps131.txt")
