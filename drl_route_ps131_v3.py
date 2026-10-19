"""
DRL agent (PPO) in the v3 routing simulation (article 3, Sect. 5.7) - the same setup as in Sects. 5.3-5.6.

Differences from drl_route_ps131.py (v2):
  1) NO LEAKAGE. The planner tables (hard / bayes) and frame pools for an episode are built from OTHER episodes:
     test episodes (30 %, the same as in v2) - from all training episodes; training episodes - from the training episodes
     of other groups (5 groups, stratified by visibility). Test episodes are used nowhere except the test.
  2) OBSERVATIONS - real out-of-fold PS131 frames (as --obs-model resample in route_ps131_v3.py):
     AMSR2 in 6 km blocks, fusion in 3 km blocks conditional on the AMSR2 observation.
  3) TRUE FIELD, oracle and MCP planner - functions of route_ps131_v3.py (run_route), i.e. the MCP regret
     on the test is computed exactly as in Table 5a.
Agent, PPO and critic warm-up - unchanged (taken from drl_route_ps131.py).

RUN
    python drl_route_ps131_v3.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --smoke     # ~5 min + time estimate
    python drl_route_ps131_v3.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --hours 6   # overnight
OUTPUT: runs\\drlv3_s<seed>_<date>\\ : log.txt, summary.txt, test_results.csv, curves.csv, policy_*.pt
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ps131_camera as T          # noqa: E402
import route_ice_sources_ps131 as R     # noqa: E402
import route_ps131_v3 as V              # noqa: E402
import drl_route_ps131 as D             # noqa: E402

VERSION = "drl v3.1 (Oct 16: --train-seed - training seed with the same split and the same scenarios)"
N = R.N
PLANNERS = ("hard", "bayes")
THR = (0.35, 0.85)
FLOE_SIGMA = 6.0


# ============================================================
# Episode groups: leakage-free planner tables and frame pools
# ============================================================
def make_groups(oof, eps, test_eps, n_groups, seed, log):
    rng = np.random.default_rng(seed)
    eps = eps.copy()
    test = set(test_eps)
    eps["grp"] = -1                                     # -1 = test
    tr = eps[~eps["episode"].isin(test)]
    for v in range(3):
        ix = rng.permutation(tr.index[tr["vis"] == v].values)
        eps.loc[ix, "grp"] = np.arange(len(ix)) % n_groups
    G = {}
    for g in sorted(eps["grp"].unique()):
        excl = test | set(eps.loc[eps["grp"] == g, "episode"])
        conf = R.confusion_tables(oof[~oof["episode"].isin(excl)], "pred_concat", lambda *a: None)
        G[g] = {"ct": {p: {v: R.cost_tables(conf, v, p) for v in range(3)} for p in PLANNERS},
                "pool_t": V.FramePool(oof, ["ice_label", "vis_label"], excl),
                "pool_ta": V.FramePool(oof, ["ice_label", "vis_label", "pred_amsr"], excl)}
        log(f"  group {'test' if g < 0 else g}: episodes {int((eps['grp'] == g).sum())}, "
            f"tables and frames from {oof['episode'].nunique() - len(excl)} other episodes")
    return eps, G


# ============================================================
# Scenario bank: truth, land, start/goal, hard and bayes cost maps (far AMSR2, near fusion)
# ============================================================
def build_bank(eps, G, probs, preds, k_per_ep, seed, synthetic, log):
    rng_master = np.random.default_rng(seed)
    keys = ["true", "land", "s", "g", "vis", "ep", "oracle"] + [f"{a}_{p}" for a in ("far", "near") for p in PLANNERS]
    B = {k: [] for k in keys}
    t0, dropped = time.time(), 0
    for _, e in eps.iterrows():
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        gi, v = G[e["grp"]], int(e["vis"])
        macro = None if synthetic else R.macro_sic_from_amsr(e["lat"], e["lon"], e["time"], log)
        if macro is None:
            macro = R.macro_sic_synthetic(rng)
        sic, land = macro
        for _ in range(k_per_ep):
            tc = V.true_classes(sic, land, rng, FLOE_SIGMA, THR)
            ends = R.pick_endpoints(land, rng)
            if ends is None:
                continue
            s, gl = ends
            if D.ctg_map(np.where(land, 1e6, R.CLASS_COST[tc]), gl)[s] >= 1e5:   # goal cut off by land
                dropped += 1
                continue
            am, _ = V.resample_obs(tc, 6, rng, gi["pool_t"], None, probs["amsr"], preds["amsr"], v)
            fu, _ = V.resample_obs(tc, R.CAM_BLOCK, rng, gi["pool_ta"], None, probs["fusion"], preds["fusion"],
                                   v, cond=am)
            for p in PLANNERS:
                ct = gi["ct"][p][v]
                B[f"far_{p}"].append(ct["amsr"][am].astype(np.float32))
                B[f"near_{p}"].append(ct["fusion"][am, fu].astype(np.float32))
            B["oracle"].append(V.run_route(tc, land, s, gl, R.CLASS_COST[tc], None, R.CAM_RANGE[v])[0])
            for k, x in (("true", tc.astype(np.int8)), ("land", land), ("s", s), ("g", gl), ("vis", v),
                         ("ep", e["episode"])):
                B[k].append(x)
    out = {k: np.array(x) for k, x in B.items()}
    log(f"  scenarios {len(out['vis'])} from {len(eps)} episodes in {time.time() - t0:.0f} s "
        f"(dropped {dropped}: goal cut off by land)")
    return out


# ============================================================
# Environment: as in v2, but the cost maps come from the bank (own group of episodes)
# ============================================================
class EnvV3(D.RouteEnv):
    def __init__(self, bank, idx, source, planner, rng):
        super().__init__(bank, idx, source, None, rng)
        self.planner = planner

    def reset(self, j=None):
        b = self.b
        self.j = int(self.rng.choice(self.idx)) if j is None else int(j)
        j = self.j
        self.vis = int(b["vis"][j])
        self.true = b["true"][j].astype(int)
        self.land = b["land"][j]
        self.pos, self.goal = tuple(b["s"][j]), tuple(b["g"][j])
        self.true_cost = R.CLASS_COST[self.true]
        self.belief = b[f"far_{self.planner}"][j].astype(float)
        self.near = b[f"near_{self.planner}"][j]
        self.seen = np.zeros((N, N), bool)
        self.r = R.CAM_RANGE[self.vis]
        self.steps, self.cost, self.heavy = 0, 0.0, 0.0
        self.reveal()
        self._replan()
        return self.obs()

    def reveal(self):
        if self.source != "fusion":
            return False
        new = R.disk(self.pos, self.r) & ~self.seen
        if new.any():
            self.belief[new] = self.near[new]
            self.seen |= new
            return True
        return False


def mcp_result(bank, j, source, planner):
    """MCP planner exactly as in route_ps131_v3.py (Table 5a)."""
    near = bank[f"near_{planner}"][j] if source == "fusion" else None
    return V.run_route(bank["true"][j].astype(int), bank["land"][j], tuple(bank["s"][j]), tuple(bank["g"][j]),
                       bank[f"far_{planner}"][j], near, R.CAM_RANGE[int(bank["vis"][j])])[:4]


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--smoke", action="store_true", help="quick check + time estimate of the overnight run")
    ap.add_argument("--hours", type=float, default=6.0, help="total PPO budget for the 4 agents, hours")
    ap.add_argument("--max-steps", type=int, default=10 ** 9)
    ap.add_argument("--k-train", type=int, default=40)
    ap.add_argument("--k-test", type=int, default=20)
    ap.add_argument("--groups", type=int, default=5, help="groups of training episodes for leakage-free tables")
    ap.add_argument("--bc-steps", type=int, default=40000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--synthetic", action="store_true", help="synthetic field instead of AMSR2 (debugging only)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the episode split and the scenario banks")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="training seed (network weights, PPO, training scenarios); default = --seed")
    args = ap.parse_args()
    ts = args.seed if args.train_seed is None else args.train_seed

    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(ts)
    np.random.seed(ts)

    run_dir = T.RUNS / f"drlv3_{'smoke_' if args.smoke else ''}s{ts}_{datetime.now():%Y%m%d_%H%M}"
    log = T.Log(run_dir / "log.txt")
    log(f"DRL on PS131 ice maps | {VERSION} | device {dev} | {vars(args)}")
    if args.smoke:
        args.k_train, args.k_test, args.bc_steps, args.n_val, args.val_every, args.rollout = 3, 2, 2000, 6, 1, 512
    variants = D.VARIANTS if args.variants == "all" else [tuple(v.split("-")) for v in args.variants.split(",")]

    oof = pd.read_csv(args.oof, low_memory=False)
    oof["time"] = pd.to_datetime(oof["time"])
    cols = {"amsr": "amsr", "fusion": "concat"}
    probs = {s: np.nan_to_num(oof[[f"p_{m}_{k}" for k in range(3)]].values, nan=1 / 3) for s, m in cols.items()}
    preds = {s: oof[f"pred_{m}"].values for s, m in cols.items()}
    eps = (oof[oof["vis_label"] >= 0].groupby("episode")
           .agg(time=("time", "min"), lat=("lat", "median"), lon=("lon", "median"),
                vis=("vis_label", lambda s: int(s.mode().iat[0]))).reset_index())
    # held-out episodes: 30 % in each visibility class (the same choice as in v2)
    rng = np.random.default_rng(args.seed)
    test_eps = []
    for v in range(3):
        e = eps.loc[eps["vis"] == v, "episode"].values
        test_eps += list(rng.choice(e, max(1, int(round(0.3 * len(e)))), replace=False))
    log("Planner tables and frame pools (no leakage):")
    eps, G = make_groups(oof, eps, test_eps, args.groups, args.seed + 7, log)
    tr, te = eps[eps["grp"] >= 0], eps[eps["grp"] < 0]
    if args.smoke:
        tr = tr.groupby("vis").head(1)
        te = te.groupby("vis").head(1)
    log(f"Episodes: training {len(tr)}, test {len(te)} (test by visibility: "
        f"{te['vis'].value_counts().sort_index().to_dict()})")

    t_bank = time.time()
    log("Scenario bank: training")
    bank_tr = build_bank(tr, G, probs, preds, args.k_train, args.seed + 1, args.synthetic, log)
    log("Scenario bank: test")
    bank_te = build_bank(te, G, probs, preds, args.k_test, args.seed + 2, args.synthetic, log)
    rng_v = np.random.default_rng(args.seed + 3)
    val_idx = rng_v.choice(len(bank_tr["vis"]), min(args.n_val, len(bank_tr["vis"])), replace=False)
    train_idx = np.setdiff1d(np.arange(len(bank_tr["vis"])), val_idx)
    te_idx = np.arange(len(bank_te["vis"]))
    t_bank = time.time() - t_bank
    log(f"Bank and oracle ready in {t_bank / 60:.1f} min")

    rows, curves, speeds = [], [], {}
    budget = 45 if args.smoke else args.hours * 3600 / len(variants)
    for source, planner in variants:
        tag = f"{source}-{planner}"
        log("\n" + "=" * 70 + f"\nAGENT {tag}: source {source}, input {planner}")
        env_fn = lambda s, idx=train_idx: EnvV3(bank_tr, idx, source, planner,
                                                np.random.default_rng(ts * 1000 + s))
        val_env = EnvV3(bank_tr, val_idx, source, planner, np.random.default_rng(0))

        def val_fn(pol):
            pol.eval()
            r = [D.run_policy(pol, torch, dev, val_env, j)[0] / bank_tr["oracle"][int(j)] - 1 for j in val_idx]
            pol.train()
            return float(np.mean(np.clip(r, 0, 3)))

        pol = D.make_policy(torch, nn).to(dev)
        t = time.time()
        D.behaviour_clone(env_fn, pol, torch, dev, args.bc_steps, args.gamma, log)
        t_bc = time.time() - t
        log(f"    critic warm-up in {t_bc / 60:.1f} min")
        bc_state = {k: v.detach().clone() for k, v in pol.state_dict().items()}
        _, sps, _ = D.ppo_train(env_fn, pol, torch, dev, budget, args.max_steps, args, val_fn, log, tag, curves)
        speeds[tag] = (sps, t_bc)
        torch.save(pol.state_dict(), run_dir / f"policy_{tag}.pt")

        t = time.time()
        te_env = EnvV3(bank_te, te_idx, source, planner, np.random.default_rng(0))
        bc_pol = D.make_policy(torch, nn).to(dev)
        bc_pol.load_state_dict(bc_state)
        for p in (pol, bc_pol):
            p.eval()
        for j in te_idx:
            j = int(j)
            base = dict(episode=bank_te["ep"][j], vis=int(bank_te["vis"][j]), scen=j, source=source, planner=planner)
            for meth, res in (("mcp", mcp_result(bank_te, j, source, planner)),
                              ("init", D.run_policy(bc_pol, torch, dev, te_env, j)),
                              ("ppo", D.run_policy(pol, torch, dev, te_env, j))):
                c, h, L, ok = res
                rows.append(dict(base, method=meth, cost=c, regret=c / bank_te["oracle"][j] - 1, heavy_km=h,
                                 len_km=L, reached=ok))
        log(f"    test of {len(te_idx)} scenarios in {(time.time() - t) / 60:.1f} min")
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "test_results.csv", index=False, float_format="%.4f")
        pd.DataFrame(curves).to_csv(run_dir / "curves.csv", index=False, float_format="%.4f")
        d = df[(df["source"] == source) & (df["planner"] == planner)]
        for meth in ("mcp", "init", "ppo"):
            s = d[d["method"] == meth]
            lo, hi = D.boot_ci(s, "regret")
            log(f"    {meth:4s} regret {100 * s['regret'].mean():5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  "
                f"heavy ice {s['heavy_km'].mean():5.1f} km  reached {s['reached'].mean():.0%}")

    D.summarize(pd.DataFrame(rows), run_dir, log)
    if args.smoke:
        log("\n" + "=" * 70 + "\nESTIMATED TIME OF THE FULL RUN")
        n_tr, n_te = int((eps["grp"] >= 0).sum()), int((eps["grp"] < 0).sum())
        scale = (n_tr * 40 + n_te * 20) / max(1, len(bank_tr["vis"]) + len(bank_te["vis"]))
        bank_full = t_bank * scale
        bc_full = sum(t_bc * 40000 / args.bc_steps for _, t_bc in speeds.values())
        hours = 6.0
        log(f"  scenario bank + oracle: ~{bank_full / 60:.0f} min")
        log(f"  critic warm-up (4 agents): ~{bc_full / 60:.0f} min")
        for tag, (sps, _) in speeds.items():
            log(f"  {tag:13s} PPO ~{sps:4.0f} steps/s -> in {hours / 4:.2f} h ~{sps * hours / 4 * 3600 / 1e6:.2f} M steps")
        total = bank_full + bc_full + hours * 3600 + 40 * 60
        log(f"  TOTAL with --hours {hours}: ~{total / 3600:.1f} h (test ~40 min, rough)")
    log(f"DONE: {run_dir}")


if __name__ == "__main__":
    main()
