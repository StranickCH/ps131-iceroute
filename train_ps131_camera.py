"""
Training of the PS131 camera model (article 3): shared encoder + two heads
    vis: visibility  0 poor <1 km (fog) | 1 moderate 1-10 km | 2 good >10 km
    ice: ice in frame 0 open/very open <=3/10 | 1 close 4-8/10      | 2 heavy >=9/10
Labels - Polarstern observers (PANGAEA 987909). A frame lacking one of the labels trains only the other head.

MODES
    python train_ps131_camera.py --smoke      # ~5 min: check of the whole path + ESTIMATE of the full training time
    python train_ps131_camera.py              # full training (overnight), then test
    python train_ps131_camera.py --resume     # continue after a crash from the last checkpoint
    python train_ps131_camera.py --test-only  # only test the best model

Each log line: [hours:min:sec | elapsed since start]. At the end of each epoch - ETA to the end of training.

OUTPUT: runs\\<name>\\
    log.txt                 - full log with time stamps
    last.pt / best.pt       - checkpoints (last - every epoch, best - by val macro-F1)
    history.csv             - metrics by epoch
    test_metrics.txt/.json  - accuracy, macro-F1, recall by class, confusion matrices,
                              95% CI by EPISODE bootstrap, ice comparison: camera vs AMSR2
    test_predictions.csv    - probabilities for each test frame (for the fusion model and analysis)
    smoke_estimate.txt      - (with --smoke) estimated duration of the full training
"""

import os
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
DATA = HERE / "ps131_dataset"
RUNS = HERE / "runs"

VIS_NAMES = ["poor<1km", "moderate1-10km", "good>10km"]
ICE_NAMES = ["open<=3/10", "close4-8/10", "heavy>=9/10"]
AMSR_TO_3 = {0: 0, 1: 0, 2: 1, 3: 2}  # AMSR2 ice_class (4 classes) -> the same 3 classes


# ============================================================
# Log with time stamps
# ============================================================
class Log:
    def __init__(self, path):
        self.t0 = time.time()
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg=""):
        el = timedelta(seconds=int(time.time() - self.t0))
        line = f"[{datetime.now():%H:%M:%S} | +{el}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def fmt_dur(sec):
    sec = int(max(0, sec))
    h, m = divmod(sec // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {sec % 60:02d} s"


# ============================================================
# Data
# ============================================================
class FrameDS:
    """Reads a frame from the 1024x768 cache. Defined at module level - needed for DataLoader on Windows."""

    def __init__(self, df, size, train):
        self.paths = df["cache_path"].values
        self.vis = df["vis_label"].values.astype(np.int64)
        self.ice = df["ice_label"].values.astype(np.int64)
        self.size = size  # (W, H)
        self.train = train
        self._tf = None

    def _build_tf(self):
        import torchvision.transforms as T
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        w, h = self.size
        if self.train:
            # mild augmentations: strong brightness/contrast changes would distort the visibility itself
            return T.Compose([
                T.RandomResizedCrop((h, w), scale=(0.6, 1.0), ratio=(1.2, 1.45)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.1, hue=0.02),
                T.ToTensor(), norm])
        return T.Compose([T.Resize((h, w)), T.ToTensor(), norm])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        if self._tf is None:
            self._tf = self._build_tf()
        with Image.open(self.paths[i]) as im:
            x = self._tf(im.convert("RGB"))
        return x, self.vis[i], self.ice[i], i


def load_split(name):
    df = pd.read_csv(DATA / f"labels_{name}.csv", low_memory=False)
    df = df[(df["vis_label"] >= 0) | (df["ice_label"] >= 0)]
    df = df[df["cache_path"].notna() & (df["cache_path"] != "")]
    return df.reset_index(drop=True)


def sample_epoch(df, per_episode, rng):
    """Frames of one episode (5 s step) are nearly identical - every epoch takes random per_episode frames from each."""
    idx = []
    for _, g in df.groupby("episode"):
        k = min(per_episode, len(g))
        idx.extend(rng.choice(g.index.values, size=k, replace=False))
    return np.array(idx)


class EpochSampler:
    """Every epoch - new random per_episode frames from each episode.
    Allows the training DataLoader to be created ONCE (on Windows every new set of
    loader processes costs several GB of virtual memory)."""

    def __init__(self, df, per_episode, seed):
        self.groups = [g.index.values for _, g in df.groupby("episode")]
        self.per_episode, self.seed, self.epoch = per_episode, seed, 0

    def set_epoch(self, ep):
        self.epoch = ep

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = np.concatenate([rng.choice(g, size=min(self.per_episode, len(g)), replace=False)
                              for g in self.groups])
        rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return int(sum(min(self.per_episode, len(g)) for g in self.groups))


def stride_subset(df, stride):
    d = df.sort_values("time")
    return d[d.groupby("episode").cumcount() % stride == 0]


# ============================================================
# Model
# ============================================================
def get_backbone(name, log, pretrained=True):
    import torch.nn as nn
    import torchvision.models as tvm
    if name == "convnext_tiny":
        ctor, wts, dim = tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.DEFAULT, 768
    elif name == "efficientnet_b0":
        ctor, wts, dim = tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT, 1280
    else:
        raise ValueError(name)
    m = None
    if pretrained:
        for proxy in (None, os.environ.get("PS131_PROXY")):
            if proxy:
                os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"] = proxy
                log(f"[model] trying to download the weights via proxy {proxy}")
            try:
                m = ctor(weights=wts)
                log(f"[model] {name}: pretrained ImageNet weights loaded")
                break
            except Exception as e:
                log(f"[model] could not load the weights: {str(e)[:150]}")
        if m is None:
            raise SystemExit("No ImageNet weights. Check the internet connection (or set PS131_PROXY) and run again. "
                             "The weights are cached, they are downloaded once (~110 MB).")
    else:
        m = ctor(weights=None)
    if name.startswith("convnext"):
        m.classifier[2] = nn.Identity()
    else:
        m.classifier = nn.Identity()
    return m, dim


def build_model(name, log, pretrained=True):
    import torch.nn as nn

    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, dim = get_backbone(name, log, pretrained)
            self.drop = nn.Dropout(0.3)
            self.vis = nn.Linear(dim, 3)
            self.ice = nn.Linear(dim, 3)

        def forward(self, x):
            f = self.drop(self.backbone(x))
            return self.vis(f), self.ice(f)

    return TwoHead()


def class_weights(labels, n=3):
    import torch
    lab = labels[labels >= 0]
    cnt = np.bincount(lab, minlength=n).astype(float) + 1
    w = (cnt.sum() / cnt) ** 0.5  # inverse square-root frequency: helps rare classes without overcorrection
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ============================================================
# Metrics
# ============================================================
def metrics(y, p, n=3):
    m = y >= 0
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0}
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-9), 0)
    present = cm.sum(1) > 0
    return {"n": int(len(y)), "acc": float((y == p).mean()), "macro_f1": float(f1[present].mean()),
            "recall": rec.round(3).tolist(), "precision": prec.round(3).tolist(), "cm": cm.tolist()}


def bootstrap_by_episode(df, ycol, pcol, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = df[df[ycol] >= 0]
    eps = d["episode"].unique()
    groups = {e: g for e, g in d.groupby("episode")}
    accs, f1s = [], []
    for _ in range(n_boot):
        s = pd.concat([groups[e] for e in rng.choice(eps, size=len(eps), replace=True)])
        mm = metrics(s[ycol].values, s[pcol].values)
        accs.append(mm["acc"])
        f1s.append(mm["macro_f1"])
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"acc_ci95": q(accs), "macro_f1_ci95": q(f1s), "n_episodes": int(len(eps))}


# ============================================================
# Loops
# ============================================================
def run_eval(model, loader, device, max_batches=None):
    import torch
    model.eval()
    out_v, out_i, idxs, n_img = [], [], [], 0
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for b, (x, _, _, ix) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x = x.to(device).to(memory_format=torch.channels_last)
            lv, li = model(x)
            out_v.append(torch.softmax(lv.float(), 1).cpu().numpy())
            out_i.append(torch.softmax(li.float(), 1).cpu().numpy())
            idxs.append(ix.numpy())
            n_img += len(x)
    dt = time.time() - t
    return np.concatenate(out_v), np.concatenate(out_i), np.concatenate(idxs), n_img / max(dt, 1e-6)


def train_epoch(model, loader, opt, sched, scaler, crit_v, crit_i, device, log, ep, n_ep,
                max_batches=None, timing=None):
    import torch
    model.train()
    nb = min(len(loader), max_batches) if max_batches else len(loader)
    t_ep = time.time()
    t_meas, n_meas = None, 0
    tot_loss, tot_n = 0.0, 0
    for b, (x, yv, yi, _) in enumerate(loader):
        if b >= nb:
            break
        if b == 5:  # the first batches warm up cudnn and the loaders, not used for the speed measurement
            t_meas, n_meas = time.time(), 0
        x = x.to(device).to(memory_format=torch.channels_last)
        yv, yi = yv.to(device), yi.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lv, li = model(x)
            loss = 0.0
            if (yv >= 0).any():
                loss = loss + crit_v(lv.float(), yv)
            if (yi >= 0).any():
                loss = loss + crit_i(li.float(), yi)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot_loss += loss.item() * len(x)
        tot_n += len(x)
        if t_meas is not None:
            n_meas += len(x)
        if (b + 1) % 50 == 0 or b + 1 == nb:
            ips = n_meas / max(time.time() - t_meas, 1e-6) if t_meas and n_meas else 0
            left = (nb - b - 1) * loader.batch_size / ips if ips else 0
            log(f"  epoch {ep}/{n_ep} batch {b + 1}/{nb}  loss {tot_loss / tot_n:.4f}  "
                f"{ips:.0f} frames/s  to the end of the epoch ~{fmt_dur(left)}")
    if timing is not None and t_meas:
        timing["train_ips"] = n_meas / max(time.time() - t_meas, 1e-6)
    return tot_loss / max(tot_n, 1), time.time() - t_ep


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="quick check + time estimate")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--run", default=None, help="run folder name (default - date/time)")
    ap.add_argument("--arch", default="convnext_tiny", choices=["convnext_tiny", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7, help="early stopping on val macro-F1")
    ap.add_argument("--per-episode", type=int, default=300, help="frames per episode per epoch")
    ap.add_argument("--val-stride", type=int, default=3, help="every N-th val frame")
    ap.add_argument("--batch", type=int, default=0, help="0 = automatic from GPU memory")
    ap.add_argument("--img", default="448x336", help="WxH of the network input")
    ap.add_argument("--workers", type=int, default=4, help="training loaders (val/test - half as many)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="without ImageNet weights (debugging only)")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    W, H = map(int, args.img.split("x"))
    run_name = args.run or ("smoke" if args.smoke else f"ps131_cam_{datetime.now():%Y%m%d_%H%M}")
    if args.resume or args.test_only:
        cands = sorted([p for p in RUNS.glob("ps131_cam_*") if (p / "last.pt").exists()])
        if args.run:
            cands = [RUNS / args.run]
        if not cands:
            raise SystemExit("No runs with a checkpoint for --resume/--test-only")
        run_dir = cands[-1]
    else:
        run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Log(run_dir / "log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Run: {run_dir.name} | mode: {'SMOKE' if args.smoke else 'TEST' if args.test_only else 'RESUME' if args.resume else 'FULL'}")
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                   "  [WARNING: without a GPU this will be very slow]"))
    log(f"Parameters: {vars(args)}")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        log(f"GPU memory: {vram:.1f} GB")
        if args.batch == 0:
            args.batch = 32 if vram >= 10 else 16 if vram >= 5.5 else 8
    elif args.batch == 0:
        args.batch = 8
    log(f"Batch size: {args.batch}")

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    va_sub = stride_subset(va, args.val_stride).reset_index(drop=True)
    log(f"Data: train {len(tr)} frames / {tr['episode'].nunique()} ep., val {len(va_sub)} (stride {args.val_stride}), "
        f"test {len(te)} / {te['episode'].nunique()} ep.")

    model = build_model(args.arch, log, pretrained=not args.no_pretrained).to(device).to(memory_format=torch.channels_last)
    crit_v = torch.nn.CrossEntropyLoss(weight=class_weights(tr["vis_label"].values).to(device), ignore_index=-1)
    crit_i = torch.nn.CrossEntropyLoss(weight=class_weights(tr["ice_label"].values).to(device), ignore_index=-1)
    head_params = list(model.vis.parameters()) + list(model.ice.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr * 0.25},
                             {"params": head_params, "lr": args.lr}], weight_decay=0.05)

    ds_tr = FrameDS(tr, (W, H), train=True)
    ds_va = FrameDS(va_sub, (W, H), train=False)
    ds_te = FrameDS(te, (W, H), train=False)
    # pin_memory on Windows crashes the CUDA caching host allocator - disabled
    pin = device.type == "cuda" and os.name != "nt"
    ev_workers = max(0, args.workers // 2)
    sampler = EpochSampler(tr, args.per_episode, args.seed)
    # training: one DataLoader for the whole run, its processes live all the time
    dl_tr = DataLoader(ds_tr, sampler=sampler, batch_size=args.batch, num_workers=args.workers,
                       pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    # val/test: fewer processes, and they exit after the pass (saves memory on Windows)
    ev_kw = dict(batch_size=args.batch, num_workers=ev_workers, pin_memory=pin, persistent_workers=False)
    dl_va = DataLoader(ds_va, shuffle=False, **ev_kw)
    dl_te = DataLoader(ds_te, shuffle=False, **ev_kw)

    n_train_epoch = len(sampler)
    steps_per_epoch = math.ceil(n_train_epoch / args.batch)
    total_steps = steps_per_epoch * (1 if args.smoke else args.epochs)
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_ep, best, bad, hist = 1, -1.0, 0, []
    if (args.resume or args.test_only) and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best, bad, hist = ck["epoch"] + 1, ck["best"], ck["bad"], ck["hist"]
        log(f"Resuming from epoch {start_ep} (best val macro-F1 = {best:.4f})")

    # ---------------- SMOKE: speed measurement and estimate of the full time ----------------
    if args.smoke:
        timing = {}
        sampler.set_epoch(0)
        log("SMOKE 1/3: training on 60 batches (speed measurement)")
        train_epoch(model, dl_tr, opt, sched, scaler, crit_v, crit_i, device, log, 1, 1,
                    max_batches=60, timing=timing)
        log("SMOKE 2/3: validation on 25 batches (speed measurement)")
        pv, pi, ix, eval_ips = run_eval(model, dl_va, device, max_batches=25)
        m = metrics(va_sub["vis_label"].values[ix], pv.argmax(1))
        log(f"  val (after 60 batches, not trained yet): vis acc {m.get('acc', 0):.3f}")
        log("SMOKE 3/3: check of the test, metrics and checkpoint saving")
        torch.save({"model": model.state_dict()}, run_dir / "smoke_check.pt")
        pv, pi, ix, _ = run_eval(model, dl_te, device, max_batches=5)
        _ = bootstrap_by_episode(te.iloc[ix].assign(pv=pv.argmax(1)), "vis_label", "pv", n_boot=50)
        (run_dir / "smoke_check.pt").unlink(missing_ok=True)

        if device.type == "cuda":
            log(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
        tr_ips = timing.get("train_ips", 0)
        t_train = n_train_epoch / max(tr_ips, 1e-6)
        t_val = len(va_sub) / max(eval_ips, 1e-6)
        t_epoch = t_train + t_val + 5
        t_test = len(te) / max(eval_ips, 1e-6) + 30
        full = args.epochs * t_epoch + t_test
        early = max(args.patience + 8, int(args.epochs * 0.6)) * t_epoch + t_test
        lines = [
            "=" * 70, "ESTIMATED FULL TRAINING TIME (measured on this computer)",
            f"  training speed:  {tr_ips:.0f} frames/s   |  validation speed: {eval_ips:.0f} frames/s",
            f"  epoch: {n_train_epoch} training frames ({fmt_dur(t_train)}) + {len(va_sub)} val frames "
            f"({fmt_dur(t_val)}) = ~{fmt_dur(t_epoch)}",
            f"  {args.epochs} epochs + test: ~{fmt_dur(full)}   (if early stopping triggers sooner: ~{fmt_dur(early)})",
            f"  start now -> finish at about {(datetime.now() + timedelta(seconds=full)):%H:%M} "
            f"(with early stopping ~{(datetime.now() + timedelta(seconds=early)):%H:%M})",
            "  Rough estimate: it becomes more accurate after the 1st epoch of the full run (ETA is printed every epoch).",
            "=" * 70, "SMOKE PASSED: data are read, the model trains, test and metrics work.",
            "Full run (overnight):  python train_ps131_camera.py",
        ]
        for l in lines:
            log(l)
        (run_dir / "smoke_estimate.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    # ---------------- FULL TRAINING ----------------
    if not args.test_only:
        ep_times = []
        for ep in range(start_ep, args.epochs + 1):
            t_ep0 = time.time()
            sampler.set_epoch(ep)
            log(f"=== Epoch {ep}/{args.epochs} ===")
            loss, t_tr = train_epoch(model, dl_tr, opt, sched, scaler, crit_v, crit_i, device, log, ep, args.epochs)
            pv, pi, ix, _ = run_eval(model, dl_va, device)
            mv = metrics(va_sub["vis_label"].values[ix], pv.argmax(1))
            mi = metrics(va_sub["ice_label"].values[ix], pi.argmax(1))
            score = (mv["macro_f1"] + mi["macro_f1"]) / 2
            hist.append({"epoch": ep, "loss": loss, "val_vis_acc": mv["acc"], "val_vis_f1": mv["macro_f1"],
                         "val_ice_acc": mi["acc"], "val_ice_f1": mi["macro_f1"], "score": score,
                         "time": datetime.now().isoformat(timespec="seconds")})
            pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
            improved = score > best
            if improved:
                best, bad = score, 0
                torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args)}, run_dir / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "best": best, "bad": bad, "hist": hist},
                       run_dir / "last.pt")
            ep_times.append(time.time() - t_ep0)
            avg = np.mean(ep_times[-3:])
            left = (args.epochs - ep) * avg
            log(f"Epoch {ep}: loss {loss:.4f} | val visibility acc {mv['acc']:.3f} F1 {mv['macro_f1']:.3f} | "
                f"val ice acc {mi['acc']:.3f} F1 {mi['macro_f1']:.3f} | score {score:.4f}"
                f"{'  <- best, saved' if improved else f'  (no improvement {bad}/{args.patience})'}")
            log(f"  epoch took {fmt_dur(ep_times[-1])}; ~{fmt_dur(left)} left to {args.epochs} epochs "
                f"(finish ~{(datetime.now() + timedelta(seconds=left)):%H:%M})")
            if bad >= args.patience:
                log(f"Early stopping: {args.patience} epochs without improvement")
                break

    # ---------------- TEST ----------------
    log("=== TEST (best model on val) ===")
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    pv, pi, ix, _ = run_eval(model, dl_te, device)
    t = te.iloc[ix].copy()
    for k in range(3):
        t[f"p_vis_{k}"], t[f"p_ice_{k}"] = pv[:, k], pi[:, k]
    t["pred_vis"], t["pred_ice"] = pv.argmax(1), pi.argmax(1)
    t["amsr_ice3"] = t["ice_class"].map(AMSR_TO_3).fillna(-1).astype(int) if "ice_class" in t else -1
    t.to_csv(run_dir / "test_predictions.csv", index=False, float_format="%.4f")

    res = {"best_epoch": ck.get("epoch")}
    out = [f"TEST: best epoch {ck.get('epoch')}, episodes {t['episode'].nunique()}, frames {len(t)}"]
    for task, y, p, names in (("visibility (camera)", "vis_label", "pred_vis", VIS_NAMES),
                              ("ice (camera)", "ice_label", "pred_ice", ICE_NAMES),
                              ("ice (AMSR2, satellite) - baseline", "ice_label", "amsr_ice3", ICE_NAMES)):
        d = t[(t[y] >= 0) & (t[p] >= 0)]
        m = metrics(d[y].values, d[p].values)
        ci = bootstrap_by_episode(d, y, p)
        res[task] = {**m, **ci}
        out += [f"\n{task}: n={m['n']} frames / {ci['n_episodes']} ep.",
                f"  accuracy {m['acc']:.3f}  95% CI {ci['acc_ci95']}   macro-F1 {m['macro_f1']:.3f}  95% CI {ci['macro_f1_ci95']}",
                "  recall by class: " + ", ".join(f"{n} {r:.2f}" for n, r in zip(names, m["recall"])),
                "  confusion matrix (rows - truth, columns - prediction):"]
        out += ["    " + f"{n:>14s} " + " ".join(f"{v:7d}" for v in row) for n, row in zip(names, m["cm"])]
    out.append("\nCI - episode bootstrap (1000 resamples): more honest than by frame, since frames of an episode are correlated.")
    for l in out:
        log(l)
    (run_dir / "test_metrics.txt").write_text("\n".join(out), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE. Everything in {run_dir}")


if __name__ == "__main__":
    main()
