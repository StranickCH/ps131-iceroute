"""
DRL agent (PPO) on the PS131 ice maps (article 3, step 3 - link to the dissertation).

QUESTION: what should be given to the DRL agent - hard sensor labels (hard) or the expected cost from
class probabilities (bayes)? And can the agent beat the classical planner (MCP) on the same data?

ENVIRONMENT = the same as in route_ice_sources_ps131.py (true field from the AMSR2 map of the episode day + floe structure,
sensors with errors from the PS131 OOF matrices by visibility, fusion = concat, the camera reveals cells within r(visibility)).
At every step the agent sees a 31x31 km window around the ship:
   1) the cost map given by the sensor (hard: class cost; bayes: E[c | observation])
   2) cost-to-go to the goal on this map (dynamic cost map, as in the dissertation architecture)
   3) where the camera has already looked   4) land/map edge
 + vector: direction and distance to the goal, visibility class.
Action: 8 directions (invalid ones are masked). Reward: -step cost on the TRUE field (/8)
+ potential-based shaping with the cost-to-go. Starts from behaviour cloning of the MCP planner, then PPO.

4 agents: source {amsr, fusion} x input {hard, bayes}.
Evaluation - on HELD-OUT PS131 episodes (30%, stratified by visibility), regret to the oracle,
95% CI by episode bootstrap, comparison with MCP on the same scenarios.

RUN
    python drl_route_ps131.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --smoke        # 3-5 min: check + time estimate
    python drl_route_ps131.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv --hours 7      # overnight
OUTPUT: runs\\drl_<date>\\ : log.txt, summary.txt, test_results.csv, curves.csv, policy_*.pt
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
import train_ps131_camera as T          # noqa: E402
import route_ice_sources_ps131 as R     # noqa: E402

N = R.N
W = 15                                   # half window -> 31x31
MAX_STEPS = 3 * N
DIRS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])
STEP_LEN = np.where(np.abs(DIRS).sum(1) == 2, np.sqrt(2), 1.0)
CNORM = 8.0
VERSION = "v2 (Oct 6: land filter, actor = MCP, robust validation)"
VARIANTS = [("amsr", "hard"), ("amsr", "bayes"), ("fusion", "hard"), ("fusion", "bayes")]


# ============================================================
# Scenario bank
# ============================================================
def build_bank(oof, conf, eps, k_per_ep, seed, synthetic, log):
    rng_master = np.random.default_rng(seed)
    B = {"true": [], "am": [], "fus": [], "land": [], "s": [], "g": [], "vis": [], "ep": []}
    t0 = time.time()
    dropped = 0
    for i, e in eps.iterrows():
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        macro = None if synthetic else R.macro_sic_from_amsr(e["lat"], e["lon"], e["time"], log)
        if macro is None:
            macro = R.macro_sic_synthetic(rng)
        sic, land = macro
        v = int(e["vis"])
        for _ in range(k_per_ep):
            tc, _ = R.true_classes(sic, land, rng)
            ends = R.pick_endpoints(land, rng)
            if ends is None:
                continue
            # the goal must be reachable by water (otherwise the path goes only through land -> garbage rewards and metrics)
            if ctg_map(np.where(land, 1e6, R.CLASS_COST[tc]), ends[1])[ends[0]] >= 1e5:
                dropped += 1
                continue
            am = R.sample_obs(tc, conf["amsr"][v], rng, block=6)
            fus = R.sample_obs(tc, conf["fusion"][v], rng, block=R.CAM_BLOCK, cond=am)
            for k, x in (("true", tc), ("am", am), ("fus", fus)):
                B[k].append(x.astype(np.int8))
            B["land"].append(land)
            B["s"].append(ends[0])
            B["g"].append(ends[1])
            B["vis"].append(v)
            B["ep"].append(e["episode"])
    out = {k: np.array(v) for k, v in B.items()}
    log(f"  scenarios {len(out['vis'])} from {len(eps)} episodes in {time.time() - t0:.0f} s "
        f"(dropped {dropped}: goal cut off by land)")
    return out


# ============================================================
# Environment
# ============================================================
def ctg_map(bcost, goal):
    from skimage.graph import MCP_Geometric
    costs, _ = MCP_Geometric(bcost).find_costs([goal])
    return costs


class RouteEnv:
    def __init__(self, bank, idx, source, ct_by_vis, rng):
        self.b, self.idx, self.source, self.cts, self.rng = bank, np.asarray(idx), source, ct_by_vis, rng

    def reset(self, j=None):
        b = self.b
        self.j = int(self.rng.choice(self.idx)) if j is None else int(j)
        j = self.j
        self.vis = int(b["vis"][j])
        self.ct = self.cts[self.vis]
        self.true = b["true"][j].astype(int)
        self.am = b["am"][j].astype(int)
        self.fus = b["fus"][j].astype(int)
        self.land = b["land"][j]
        self.pos, self.goal = tuple(b["s"][j]), tuple(b["g"][j])
        self.true_cost = R.CLASS_COST[self.true]
        self.belief = self.ct["amsr"][self.am].astype(float)
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
            self.belief[new] = self.ct["fusion"][self.am[new], self.fus[new]]
            self.seen |= new
            return True
        return False

    def _replan(self):
        self.bcost = np.where(self.land, 1e6, self.belief)
        self.ctg = ctg_map(self.bcost, self.goal)

    def valid(self):
        p = np.array(self.pos) + DIRS
        ok = (p >= 0).all(1) & (p < N).all(1)
        ok[ok] = ~self.land[p[ok, 0], p[ok, 1]]
        return ok

    def qvals(self):
        """Planner score for the 8 actions: step cost on the map + cost-to-go from the neighbour."""
        ok = self.valid()
        p = np.array(self.pos) + DIRS
        q = np.full(8, np.inf)
        y, x = p[ok, 0], p[ok, 1]
        q[ok] = STEP_LEN[ok] * (self.bcost[self.pos] + self.bcost[y, x]) / 2 + self.ctg[y, x]
        return q

    def teacher(self):
        """Step of the MCP planner on the current map (as in route_ice_sources_ps131.py)."""
        return int(np.argmin(self.qvals()))

    def obs(self):
        y, x = self.pos
        pad = lambda a, v: np.pad(a, W, constant_values=v)[y:y + 2 * W + 1, x:x + 2 * W + 1]
        c0 = self.ctg[self.pos]
        rel = np.clip((self.ctg - c0) / (CNORM * 3), -5, 5)
        img = np.stack([pad(np.where(self.land, 1.0, self.belief / CNORM), 1.0),
                        pad(rel, 5.0),
                        pad(self.seen.astype(float), 0.0),
                        pad(self.land.astype(float), 1.0)]).astype(np.float32)
        dy, dx = (self.goal[0] - y) / N, (self.goal[1] - x) / N
        q = self.qvals()
        qrel = np.where(np.isfinite(q), np.clip((q - q[np.isfinite(q)].min()) / CNORM, 0, 5), 5.0)
        vec = np.array([dy, dx, np.hypot(dy, dx), min(c0, 1e4) / (CNORM * N)] +
                       [float(self.vis == v) for v in range(3)] + list(qrel), np.float32)
        return img, vec, self.valid()

    def step(self, a, gamma):
        ctg_old = self.ctg
        old = self.pos
        nxt = (old[0] + DIRS[a][0], old[1] + DIRS[a][1])
        c = STEP_LEN[a] * self.true_cost[nxt]
        self.cost += c
        self.heavy += STEP_LEN[a] * (self.true[nxt] == 2)
        self.pos = nxt
        self.steps += 1
        r = -c / CNORM + (gamma * (-min(ctg_old[nxt], 1e4)) - (-min(ctg_old[old], 1e4))) / CNORM
        done = nxt == self.goal
        trunc = self.steps >= MAX_STEPS and not done
        if self.reveal():
            self._replan()
        if trunc:
            r -= min(self.ctg[nxt], 1e4) / CNORM
        return r, done or trunc

    def final_cost(self):
        """Path cost; if the goal was not reached - plus the true cost-to-go from the stopping point."""
        if self.pos == self.goal:
            return self.cost, True
        rest = ctg_map(np.where(self.land, 1e6, self.true_cost.astype(float)), self.goal)[self.pos]
        return self.cost + rest, False


# ============================================================
# Policy
# ============================================================
def make_policy(torch, nn):
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten())
            self.fc = nn.Sequential(nn.Linear(64 * 4 * 4 + 15, 256), nn.ReLU())
            self.pi = nn.Linear(256, 8)
            self.v = nn.Linear(256, 1)
            nn.init.zeros_(self.pi.weight)
            nn.init.zeros_(self.pi.bias)
            self.log_beta = nn.Parameter(torch.tensor(np.log(30.0), dtype=torch.float32))

        def forward(self, img, vec, mask):
            # residual policy over the planner: logits = -beta * (MCP score of the action) + learned correction.
            # At the start the agent = MCP; PPO learns when to deviate from it.
            h = self.fc(torch.cat([self.conv(img), vec], 1))
            logits = (self.pi(h) - self.log_beta.exp() * vec[:, 7:15]).masked_fill(~mask, -1e9)
            return logits, self.v(h).squeeze(-1)
    return Policy()


def to_t(torch, dev, img, vec, mask):
    return (torch.as_tensor(np.asarray(img), device=dev), torch.as_tensor(np.asarray(vec), device=dev),
            torch.as_tensor(np.asarray(mask), device=dev))


# ============================================================
# Behaviour cloning of the planner
# ============================================================
def behaviour_clone(env_fn, pol, torch, dev, steps, gamma, log):
    envs = [env_fn(i) for i in range(8)]
    obs = [e.reset() for e in envs]
    I, V, M, A, Rw, D, Ep = [], [], [], [], [], [], []
    for t in range(steps // len(envs)):
        for k, e in enumerate(envs):
            img, vec, mask = obs[k]
            a = e.teacher()
            r, done = e.step(a, gamma)
            I.append(img); V.append(vec); M.append(mask); A.append(a); Rw.append(r); D.append(done); Ep.append(k)
            obs[k] = e.reset() if done else e.obs()
    # discounted returns per environment (for the critic)
    Rw, D, Ep = np.array(Rw), np.array(D), np.array(Ep)
    G = np.zeros(len(Rw), np.float32)
    for k in range(len(envs)):
        ids = np.where(Ep == k)[0]
        g = 0.0
        for i in ids[::-1]:
            g = Rw[i] + gamma * g * (1 - D[i])
            G[i] = g
    I, V, M, A = np.array(I), np.array(V), np.array(M), np.array(A)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    n = len(A)
    for epoch in range(6):
        perm = np.random.permutation(n)
        tot, acc = 0.0, 0.0
        for s in range(0, n, 512):
            b = perm[s:s + 512]
            img, vec, mask = to_t(torch, dev, I[b], V[b], M[b])
            logits, v = pol(img, vec, mask)
            at = torch.as_tensor(A[b], device=dev)
            # the actor already equals the planner by construction (residual policy with zero correction),
            # so only the critic is trained here - on the planner's returns
            loss = ((v - torch.as_tensor(G[b], device=dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            acc += (logits.argmax(1) == at).float().sum().item()
        log(f"    critic warm-up, epoch {epoch + 1}/6: MSE {tot / n:.3f}, actor agreement with the planner {acc / n:.1%}")


# ============================================================
# PPO
# ============================================================
def ppo_train(env_fn, pol, torch, dev, time_budget_s, max_steps, args, val_fn, log, tag, curves):
    n_env, roll = 8, args.rollout
    envs = [env_fn(100 + i) for i in range(n_env)]
    obs = [e.reset() for e in envs]
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    t0, steps, it = time.time(), 0, 0
    best_val, best_state = val_fn(pol), {k: v.detach().clone() for k, v in pol.state_dict().items()}
    log(f"    [{tag}] PPO start: validation (agent = MCP) regret {100 * best_val:.1f}%")
    ep_ret, ep_rets = np.zeros(n_env), []
    while time.time() - t0 < time_budget_s and steps < max_steps:
        I, V, M, A, LP, VAL, RW, DN = [], [], [], [], [], [], [], []
        for t in range(roll // n_env):
            img, vec, mask = zip(*obs)
            with torch.no_grad():
                logits, val = pol(*to_t(torch, dev, img, vec, mask))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            a_np = a.cpu().numpy()
            rs, ds = [], []
            for k, e in enumerate(envs):
                r, d = e.step(int(a_np[k]), args.gamma)
                ep_ret[k] += r
                if d:
                    ep_rets.append(ep_ret[k])
                    ep_ret[k] = 0
                rs.append(r)
                ds.append(d)
                obs[k] = e.reset() if d else e.obs()
            I.append(img); V.append(vec); M.append(mask); A.append(a_np)
            LP.append(lp.cpu().numpy()); VAL.append(val.cpu().numpy()); RW.append(rs); DN.append(ds)
        with torch.no_grad():
            img, vec, mask = zip(*obs)
            _, last_v = pol(*to_t(torch, dev, img, vec, mask))
            last_v = last_v.cpu().numpy()
        RW, DN, VAL = np.array(RW), np.array(DN, float), np.array(VAL)
        adv = np.zeros_like(RW)
        gae = np.zeros(n_env)
        for t in reversed(range(len(RW))):
            nv = last_v if t == len(RW) - 1 else VAL[t + 1]
            delta = RW[t] + args.gamma * nv * (1 - DN[t]) - VAL[t]
            gae = delta + args.gamma * args.lam * (1 - DN[t]) * gae
            adv[t] = gae
        ret = adv + VAL
        f = lambda x: np.asarray(x).reshape(-1, *np.asarray(x).shape[2:])
        I, V, M = f(I), f(V), f(M)
        A, LP, ADV, RET = f(A), f(LP), adv.reshape(-1), ret.reshape(-1)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(A)
        for _ in range(args.ppo_epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, 256):
                b = perm[s:s + 256]
                logits, v = pol(*to_t(torch, dev, I[b], V[b], M[b]))
                dist = torch.distributions.Categorical(logits=logits)
                at = torch.as_tensor(A[b], device=dev)
                ratio = torch.exp(dist.log_prob(at) - torch.as_tensor(LP[b], device=dev))
                ad = torch.as_tensor(ADV[b], device=dev, dtype=torch.float32)
                pl = -torch.min(ratio * ad, torch.clamp(ratio, 0.8, 1.2) * ad).mean()
                vl = ((v - torch.as_tensor(RET[b], device=dev, dtype=torch.float32)) ** 2).mean()
                loss = pl + 0.5 * vl - args.ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        steps += n
        it += 1
        if it % args.val_every == 0 or time.time() - t0 >= time_budget_s:
            vr = val_fn(pol)
            if vr < best_val:
                best_val = vr
                best_state = {k: v.detach().clone() for k, v in pol.state_dict().items()}
            el = time.time() - t0
            curves.append({"variant": tag, "steps": steps, "minutes": el / 60, "val_regret": vr,
                           "train_return": float(np.mean(ep_rets[-50:])) if ep_rets else np.nan})
            log(f"    [{tag}] step {steps:>8d}  {steps / el:5.0f} steps/s  validation regret {100 * vr:5.1f}% "
                f"(best {100 * best_val:.1f}%)  {max(0, time_budget_s - el) / 60:4.0f} min left")
    pol.load_state_dict(best_state)
    return steps, steps / max(time.time() - t0, 1e-9), best_val


# ============================================================
# Evaluation
# ============================================================
def run_policy(pol, torch, dev, env, j):
    env.reset(j)
    while True:
        img, vec, mask = env.obs()
        with torch.no_grad():
            logits, _ = pol(*to_t(torch, dev, [img], [vec], [mask]))
        _, done = env.step(int(logits.argmax(1).item()), 0.99)
        if done:
            break
    cost, reached = env.final_cost()
    return cost, env.heavy, env.steps, reached


def oracle_costs(bank, idx, cts):
    out = {}
    for j in idx:
        j = int(j)
        o = R.run_strategy("oracle", bank["true"][j].astype(int), bank["land"][j], tuple(bank["s"][j]),
                           tuple(bank["g"][j]), None, int(bank["vis"][j]), None, cts["hard"][int(bank["vis"][j])])
        out[j] = o["cost"]
    return out


def mcp_result(bank, j, source, ct):
    obs = {"amsr": bank["am"][j].astype(int), "fusion": bank["fus"][j].astype(int),
           "camera": bank["fus"][j].astype(int)}
    o = R.run_strategy(source, bank["true"][j].astype(int), bank["land"][j], tuple(bank["s"][j]),
                       tuple(bank["g"][j]), obs, int(bank["vis"][j]), None, ct)
    return o["cost"], o["heavy_km"], o["len_km"], o["reached"]


def boot_ci(df, col, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = df.groupby("episode")[col].mean().values
    vals = [rng.choice(g, len(g)).mean() for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--smoke", action="store_true", help="quick check + time estimate of the overnight run")
    ap.add_argument("--hours", type=float, default=7.0, help="total PPO budget for the 4 agents, hours")
    ap.add_argument("--max-steps", type=int, default=10 ** 9, help="limit of PPO steps per agent")
    ap.add_argument("--k-train", type=int, default=40, help="scenarios per training episode")
    ap.add_argument("--k-test", type=int, default=20, help="scenarios per test episode")
    ap.add_argument("--bc-steps", type=int, default=40000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=150, help="validation scenarios (from training episodes)")
    ap.add_argument("--variants", default="all", help="e.g. fusion-bayes,fusion-hard")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = T.RUNS / f"drl_{'smoke_' if args.smoke else ''}{datetime.now():%Y%m%d_%H%M}"
    log = T.Log(run_dir / "log.txt")
    log(f"DRL on PS131 ice maps | {VERSION} | device {dev} | {vars(args)}")
    if args.smoke:
        args.k_train, args.k_test, args.bc_steps, args.n_val, args.val_every, args.rollout = 3, 2, 2000, 6, 1, 512
    variants = VARIANTS if args.variants == "all" else [tuple(v.split("-")) for v in args.variants.split(",")]

    oof = pd.read_csv(args.oof, low_memory=False)
    oof["time"] = pd.to_datetime(oof["time"])
    conf = R.confusion_tables(oof, "pred_concat", lambda *a: None)
    cts = {p: {v: R.cost_tables(conf, v, p) for v in (0, 1, 2)} for p in ("hard", "bayes")}
    eps = (oof[oof["vis_label"] >= 0].groupby("episode")
           .agg(time=("time", "min"), lat=("lat", "median"), lon=("lon", "median"),
                vis=("vis_label", lambda s: int(s.mode().iat[0]))).reset_index())
    # held-out episodes: 30% in each visibility class
    rng = np.random.default_rng(args.seed)
    test_eps = []
    for v in range(3):
        e = eps.loc[eps["vis"] == v, "episode"].values
        test_eps += list(rng.choice(e, max(1, int(round(0.3 * len(e)))), replace=False))
    tr, te = eps[~eps["episode"].isin(test_eps)], eps[eps["episode"].isin(test_eps)]
    if args.smoke:
        tr = tr.groupby("vis").head(1)
        te = te.groupby("vis").head(1)
    log(f"Episodes: training {len(tr)}, test {len(te)} (test by visibility: "
        f"{te['vis'].value_counts().sort_index().to_dict()})")

    t_bank = time.time()
    log("Scenario bank: training")
    bank_tr = build_bank(oof, conf, tr, args.k_train, args.seed + 1, args.synthetic, log)
    log("Scenario bank: test")
    bank_te = build_bank(oof, conf, te, args.k_test, args.seed + 2, args.synthetic, log)
    rng_v = np.random.default_rng(args.seed + 3)
    val_idx = rng_v.choice(len(bank_tr["vis"]), min(args.n_val, len(bank_tr["vis"])), replace=False)
    train_idx = np.setdiff1d(np.arange(len(bank_tr["vis"])), val_idx)
    or_val = oracle_costs(bank_tr, val_idx, cts)
    te_idx = np.arange(len(bank_te["vis"]))
    or_te = oracle_costs(bank_te, te_idx, cts)
    t_bank = time.time() - t_bank
    log(f"Bank and oracle ready in {t_bank / 60:.1f} min")

    rows, curves, speeds = [], [], {}
    budget = args.hours * 3600 / len(variants)
    if args.smoke:
        budget = 45
    for source, planner in variants:
        tag = f"{source}-{planner}"
        log("\n" + "=" * 70 + f"\nAGENT {tag}: source {source}, input {planner}")
        ct = cts[planner]
        env_fn = lambda s, idx=train_idx: RouteEnv(bank_tr, idx, source, ct, np.random.default_rng(args.seed * 1000 + s))
        val_env = RouteEnv(bank_tr, val_idx, source, ct, np.random.default_rng(0))

        def val_fn(pol):
            pol.eval()
            r = [run_policy(pol, torch, dev, val_env, j)[0] / or_val[int(j)] - 1 for j in val_idx]
            pol.train()
            return float(np.mean(np.clip(r, 0, 3)))     # robust to single failures

        pol = make_policy(torch, nn).to(dev)
        t = time.time()
        behaviour_clone(env_fn, pol, torch, dev, args.bc_steps, args.gamma, log)
        t_bc = time.time() - t
        log(f"    critic warm-up in {t_bc / 60:.1f} min")
        bc_state = {k: v.detach().clone() for k, v in pol.state_dict().items()}
        steps, sps, best = ppo_train(env_fn, pol, torch, dev, budget, args.max_steps, args, val_fn, log, tag, curves)
        speeds[tag] = (sps, t_bc)
        torch.save(pol.state_dict(), run_dir / f"policy_{tag}.pt")

        # test: MCP, agent after BC, agent after PPO - the same scenarios
        t = time.time()
        te_env = RouteEnv(bank_te, te_idx, source, ct, np.random.default_rng(0))
        bc_pol = make_policy(torch, nn).to(dev)
        bc_pol.load_state_dict(bc_state)
        for p in (pol, bc_pol):
            p.eval()
        for j in te_idx:
            j = int(j)
            base = dict(episode=bank_te["ep"][j], vis=int(bank_te["vis"][j]), scen=j, source=source, planner=planner)
            for meth, res in (("mcp", mcp_result(bank_te, j, source, ct[int(bank_te["vis"][j])])),
                              ("init", run_policy(bc_pol, torch, dev, te_env, j)),
                              ("ppo", run_policy(pol, torch, dev, te_env, j))):
                c, h, L, ok = res
                rows.append(dict(base, method=meth, cost=c, regret=c / or_te[j] - 1, heavy_km=h, len_km=L,
                                 reached=ok))
        log(f"    test of {len(te_idx)} scenarios in {(time.time() - t) / 60:.1f} min")
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "test_results.csv", index=False, float_format="%.4f")
        pd.DataFrame(curves).to_csv(run_dir / "curves.csv", index=False, float_format="%.4f")
        d = df[(df["source"] == source) & (df["planner"] == planner)]
        for meth in ("mcp", "init", "ppo"):
            s = d[d["method"] == meth]
            lo, hi = boot_ci(s, "regret")
            log(f"    {meth:4s} regret {100 * s['regret'].mean():5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  "
                f"heavy ice {s['heavy_km'].mean():5.1f} km  reached {s['reached'].mean():.0%}")

    summarize(pd.DataFrame(rows), run_dir, log)
    if args.smoke:
        log("\n" + "=" * 70 + "\nESTIMATED TIME OF THE FULL RUN")
        n_tr_full = int(round(len(eps) * 0.7))
        scale = (n_tr_full * 40 + (len(eps) - n_tr_full) * 20) / max(1, len(bank_tr["vis"]) + len(bank_te["vis"]))
        bank_full = t_bank * scale
        bc_full = sum(t_bc * 40000 / args.bc_steps for _, t_bc in speeds.values())
        log(f"  scenario bank + oracle: ~{bank_full / 60:.0f} min")
        log(f"  behaviour cloning (4 agents): ~{bc_full / 60:.0f} min")
        for tag, (sps, _) in speeds.items():
            log(f"  {tag:13s} PPO ~{sps:4.0f} steps/s -> in {args.hours / 4:.2f} h ~{sps * args.hours / 4 * 3600 / 1e6:.2f} M steps")
        log(f"  test (4 agents x 3 methods, ~{len(eps) - n_tr_full} ep. x 20): ~{4 * 10:.0f} min (rough)")
        total = bank_full + bc_full + args.hours * 3600 + 40 * 60
        log(f"  TOTAL with --hours {args.hours}: ~{total / 3600:.1f} h")
    log(f"DONE: {run_dir}")


def summarize(df, run_dir, log):
    if df.empty:
        return
    out = []
    say = lambda s: (log(s), out.append(s))
    say("\n" + "=" * 70 + "\nTEST (held-out episodes): regret, % [95% CI by episode]")
    vis_names = ["all"] + list(T.VIS_NAMES)
    for (src, pl), d in df.groupby(["source", "planner"]):
        say(f"\n{src}-{pl}")
        for vi, vn in enumerate(vis_names):
            dv = d if vi == 0 else d[d["vis"] == vi - 1]
            if dv.empty:
                continue
            parts = []
            for meth in ("mcp", "init", "ppo"):
                s = dv[dv["method"] == meth]
                parts.append(f"{meth} {100 * s['regret'].mean():5.1f}")
            say(f"  {vn:14s} " + "   ".join(parts))
    say("\nPAIRWISE regret DIFFERENCES, pp (same scenarios; <0 = first is better)")
    w = df.pivot_table(index=["episode", "vis", "scen"], columns=["source", "planner", "method"],
                       values="regret")

    def diff(a, b, name):
        if a not in w.columns or b not in w.columns:
            return
        x = pd.DataFrame({"episode": w.index.get_level_values("episode"), "d": (w[a] - w[b]).values}).dropna()
        lo, hi = boot_ci(x, "d")
        sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
        say(f"  {name:45s} {100 * x['d'].mean():+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]  {sig}")

    for src in ("amsr", "fusion"):
        diff((src, "bayes", "ppo"), (src, "hard", "ppo"), f"PPO {src}: bayes - hard")
        for pl in ("hard", "bayes"):
            diff((src, pl, "ppo"), (src, pl, "mcp"), f"{src}-{pl}: PPO - MCP")
            diff((src, pl, "ppo"), (src, pl, "init"), f"{src}-{pl}: PPO - start")
    for pl in ("hard", "bayes"):
        diff(("fusion", pl, "ppo"), ("amsr", pl, "ppo"), f"PPO {pl}: fusion - amsr")
    (run_dir / "summary.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
