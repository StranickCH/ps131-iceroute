# -*- coding: utf-8 -*-
"""Summary of PPO over several training seeds (article 3, Table 8 and Fig. S1).
Uses test_results.csv from runs\\drlv3_20261012_2324 (seed 0 of the paper) and runs\\drlv3_s*_* (seeds 0, 1, 2, ...).
Run:  python drl_seeds_summary.py            -> drl_seeds_summary.txt
"""
import glob
import os
import numpy as np
import pandas as pd

RUNS = "runs"
dirs = {0: os.path.join(RUNS, "drlv3_20261012_2324")} if os.path.isdir(os.path.join(RUNS, "drlv3_20261012_2324")) else {}  # seed-0 run of the paper (before --train-seed)
for d in sorted(glob.glob(os.path.join(RUNS, "drlv3_s*_2*"))):
    if os.path.basename(d).startswith("drlv3_smoke"):
        continue
    s = int(os.path.basename(d).split("_")[1][1:])
    if os.path.exists(os.path.join(d, "test_results.csv")):
        dirs[s] = d                      # if a seed was run twice, the latest run is used
out = []


def say(s=""):
    print(s)
    out.append(s)


def boot(x_ep, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    return x_ep[rng.integers(0, len(x_ep), (n, len(x_ep)))].mean(1)


df = []
for s, d in dirs.items():
    t = pd.read_csv(os.path.join(d, "test_results.csv"))
    df.append(t.assign(seed=s))
    say(f"seed {s}: {d}, rows {len(t)}")
df = pd.concat(df)
seeds = sorted(dirs)

# MCP does not depend on the training seed - check that the test scenarios are identical
m = df[df["method"] == "mcp"].pivot_table(index=["source", "planner", "scen"], columns="seed", values="regret")
say(f"MCP identical for all seeds: {bool(np.allclose(m.values, m.values[:, :1], atol=1e-4))}")

say("\nTEST REGRET, % (13 episodes x 20 scenarios)")
say(f"{'variant':14s}{'MCP':>7s}" + "".join(f"{'start s' + str(s):>11s}{'PPO s' + str(s):>9s}" for s in seeds)
    + f"{'PPO mean':>9s}{'PPO SD':>8s}")
for (src, pl), g in df.groupby(["source", "planner"]):
    mcp = 100 * g[(g["method"] == "mcp") & (g["seed"] == seeds[0])]["regret"].mean()
    row, ppo = "", []
    for s in seeds:
        gs = g[g["seed"] == s]
        i, p = 100 * gs[gs["method"] == "init"]["regret"].mean(), 100 * gs[gs["method"] == "ppo"]["regret"].mean()
        ppo.append(p)
        row += f"{i:11.1f}{p:9.1f}"
    say(f"{src + '-' + pl:14s}{mcp:7.1f}{row}{np.mean(ppo):9.1f}{np.std(ppo, ddof=1) if len(ppo) > 1 else 0:8.2f}")

say("\nPPO - MCP, pp [95% CI, episode bootstrap]; 'mean' - PPO averaged over the seeds in each scenario")
for (src, pl), g in df.groupby(["source", "planner"]):
    w = g.pivot_table(index=["episode", "scen"], columns=["seed", "method"], values="regret")
    for s in seeds + ["mean"]:
        ppo = w.xs("ppo", axis=1, level="method").mean(axis=1) if s == "mean" else w[(s, "ppo")]
        d = (ppo - w[(seeds[0], "mcp")]).rename("d").reset_index()
        e = d.groupby("episode")["d"].mean().values
        bs = boot(e)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
        say(f"  {src + '-' + pl:13s} {('seed ' + str(s)) if s != 'mean' else 'mean':9s} "
            f"{100 * d['d'].mean():+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]  {sig}")

say("\nTRAINING CURVES: best validation regret relative to the start, pp")
for s, d in dirs.items():
    f = os.path.join(d, "curves.csv")
    if not os.path.exists(f):
        continue
    c = pd.read_csv(f)
    tagc = [k for k in c.columns if k in ("tag", "agent", "variant")]
    valc = [k for k in c.columns if "val" in k]
    stepc = [k for k in c.columns if "step" in k]
    if not (tagc and valc):
        continue
    for tag, g in c.groupby(tagc[0]):
        v = g[valc[0]].values
        say(f"  seed {s} {tag:13s} start {100 * v[0]:5.1f}  best {100 * v.min():5.1f}  "
            f"({100 * (v.min() - v[0]):+.2f})  steps {int(g[stepc[0]].max()) if stepc else -1}")

with open("drl_seeds_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
print("\nwritten drl_seeds_summary.txt")
