"""Train a TrackNet-style ball detector on SoccerNet tracking sequences.

Input:  3 consecutive frames (t-1, t, t+1) stacked as 9 channels, resized to --size.
Output: a heatmap for frame t that peaks at the ball centre (all zeros if no ball).

Split is by GAME (gameID in gameinfo.ini), so validation clips never come from a
match the network trained on. Validation uses soccernet.evaluate (ball within 15 px
of GT in full-res pixels), so the score is directly comparable to non-nn.py.

First run resizes every frame once into --cache (a few GB less than the originals,
and much faster to load than 1080p JPEGs).

Examples:
    python train_tracknet.py                         # defaults
    python train_tracknet.py --epochs 30 --batch 8
    python train_tracknet.py --base 64               # full TrackNetV2 width (slower)
    python train_tracknet.py --limit-seqs 4 --epochs 1   # quick smoke test
"""
import argparse
import configparser
import glob
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import soccernet as sn

FULL_W, FULL_H = 1920, 1080


# ----------------------------------------------------------------------------- args

ON_KAGGLE = os.path.isdir("/kaggle/working")


def default_root():
    """Locally: soccernet.ROOT. On Kaggle: the folder under /kaggle/input that holds the SNMOT-* dirs."""
    if ON_KAGGLE:
        hits = sorted(glob.glob("/kaggle/input/**/SNMOT-*", recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    return sn.ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    work = "/kaggle/working" if ON_KAGGLE else os.path.expanduser("~/ball_tracker")
    p.add_argument("--root", default=None, help="folder with SNMOT-* sequences (auto-found on Kaggle)")
    p.add_argument("--cache", default=os.path.join(work, "data/soccernet/tracknet_cache"))
    p.add_argument("--size", default="640x360", help="network input WxH, both divisible by 8")
    p.add_argument("--sigma", type=float, default=2.5, help="Gaussian radius of the target (px at --size)")
    p.add_argument("--base", type=int, default=32, help="channels of first block. 64 = TrackNetV2 paper size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-games", type=int, default=2, help="number of games held out for validation")
    p.add_argument("--val-stride", type=int, default=2, help="validate on every n-th frame (speed)")
    p.add_argument("--thr", type=float, default=0.5, help="heatmap peak below this = no ball")
    p.add_argument("--neg-frac", type=float, default=0.3,
                   help="max fraction of training samples without a ball (rest are dropped)")
    p.add_argument("--limit-seqs", type=int, default=0, help="only use the first N sequences (debugging)")
    p.add_argument("--device", default="auto", help="auto | cuda | xpu | mps | cpu")
    p.add_argument("--out", default=os.path.join(work, "checkpoints/tracknet.pt" if ON_KAGGLE else "code/checkpoints/tracknet.pt"))
    p.add_argument("--resume", default="", help="checkpoint to continue from")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.root = a.root or default_root()
    a.W, a.H = map(int, a.size.lower().split("x"))
    assert a.W % 8 == 0 and a.H % 8 == 0, "--size must be divisible by 8 (3 poolings)"
    return a


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():      # Intel GPU
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    """fp16 mixed precision on CUDA (uses the tensor cores, e.g. on a T4); no-op elsewhere."""
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


# ----------------------------------------------------------------------------- data

def seq_length(seq, root):
    c = configparser.ConfigParser()
    c.read(os.path.join(root, seq, "seqinfo.ini"))
    return int(c["Sequence"]["seqLength"])


def game_id(seq, root):
    for line in open(os.path.join(root, seq, "gameinfo.ini")):
        if line.startswith("gameID="):
            return int(line.split("=")[1])
    return -1


def _resize_seq(job):
    seq, root, cache, W, H = job
    src = os.path.join(root, seq, "img1")
    dst = os.path.join(cache, f"{W}x{H}", seq)
    os.makedirs(dst, exist_ok=True)
    done = 0
    for name in sorted(os.listdir(src)):
        out = os.path.join(dst, name)
        if os.path.exists(out):
            continue
        img = cv2.imread(os.path.join(src, name))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1
    return seq, done


def build_cache(seqs, a):
    jobs = [(s, a.root, a.cache, a.W, a.H) for s in seqs]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for seq, done in pool.imap_unordered(_resize_seq, jobs):
            if done:
                print(f"  cached {seq}: {done} frames")


class BallFrames(Dataset):
    """One sample = frames (t-1, t, t+1) of a sequence + heatmap target for t."""

    def __init__(self, seqs, a, train):
        self.a, self.train = a, train
        self.dir = {s: os.path.join(a.cache, f"{a.W}x{a.H}", s) for s in seqs}
        self.len = {s: seq_length(s, a.root) for s in seqs}
        self.gt = {}
        for s in seqs:                                           # full-res centres -> network size
            self.gt[s] = {f: (x * a.W / FULL_W, y * a.H / FULL_H) for f, (x, y) in sn.load_ball_gt(s, a.root).items()}

        pos = [(s, f) for s in seqs for f in range(1, self.len[s] + 1) if f in self.gt[s]]
        neg = [(s, f) for s in seqs for f in range(1, self.len[s] + 1) if f not in self.gt[s]]
        if train:
            # keep some no-ball frames so it learns to output nothing, but not too many
            rng = random.Random(a.seed)
            max_neg = int(len(pos) * a.neg_frac / (1 - a.neg_frac))
            neg = rng.sample(neg, min(len(neg), max_neg))
        self.items = pos + neg
        ys, xs = np.mgrid[0:a.H, 0:a.W]
        self.xs, self.ys = xs.astype(np.float32), ys.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def load(self, seq, f):
        f = min(max(f, 1), self.len[seq])                        # repeat edge frames
        img = cv2.imread(os.path.join(self.dir[seq], f"{f:06d}.jpg"))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        seq, f = self.items[i]
        frames = np.concatenate([self.load(seq, f + d) for d in (-1, 0, 1)], axis=2)   # H x W x 9
        frames = frames.astype(np.float32) / 255.0

        ball = self.gt[seq].get(f)
        heat = np.zeros((self.a.H, self.a.W), np.float32)
        if ball is not None:
            cx, cy = ball
            heat = np.exp(-((self.xs - cx) ** 2 + (self.ys - cy) ** 2) / (2 * self.a.sigma ** 2))

        if self.train:
            if random.random() < 0.5:                            # horizontal flip
                frames, heat = frames[:, ::-1], heat[:, ::-1]
            # same brightness / contrast change for all 3 frames
            frames = frames * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
            frames = np.clip(frames, 0, 1)

        x = torch.from_numpy(np.ascontiguousarray(frames.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(heat))[None]
        return x, y, i


def worker_init(_):
    cv2.setNumThreads(0)                                         # don't fight the DataLoader workers
    random.seed(torch.initial_seed() % 2**32)


# ----------------------------------------------------------------------------- model

def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def block(cin, cout, n):
    return nn.Sequential(conv(cin, cout), *[conv(cout, cout) for _ in range(n - 1)])


class TrackNet(nn.Module):
    """TrackNetV2-style U-Net: 3 poolings down, 3 upsamplings back with skip connections."""

    def __init__(self, in_ch=9, base=32):
        super().__init__()
        b = base
        self.e1 = block(in_ch, b, 2)
        self.e2 = block(b, 2 * b, 2)
        self.e3 = block(2 * b, 4 * b, 3)
        self.mid = block(4 * b, 8 * b, 3)
        self.d3 = block(8 * b + 4 * b, 4 * b, 3)
        self.d2 = block(4 * b + 2 * b, 2 * b, 2)
        self.d1 = block(2 * b + b, b, 2)
        self.head = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.up(m), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return self.head(d1)                                     # logits, sigmoid -> heatmap


def wbce_loss(logits, target):
    """TrackNetV2 weighted BCE: down-weights the easy background pixels (almost all of them)."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    loss = -((1 - p) ** 2 * target * torch.log(p) + p ** 2 * (1 - target) * torch.log(1 - p))
    return loss.mean()


# ----------------------------------------------------------------------------- inference / eval

def heatmap_to_xy(heat, thr):
    """Peak of one heatmap (H x W, 0..1) -> (x, y) in full-res pixels, or None if below thr."""
    H, W = heat.shape
    idx = int(np.argmax(heat))
    py, px = divmod(idx, W)
    if heat[py, px] < thr:
        return None
    # sub-pixel: weighted centre of the 5x5 neighbourhood around the peak
    y0, y1, x0, x1 = max(0, py - 2), min(H, py + 3), max(0, px - 2), min(W, px + 3)
    w = heat[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx, cy = (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()
    return cx * FULL_W / W, cy * FULL_H / H


@torch.no_grad()
def evaluate_seqs(model, seqs, a, device):
    """Run on validation sequences and score with soccernet.evaluate (15 px in full res)."""
    model.eval()
    ds = BallFrames(seqs, a, train=False)
    keep = [i for i, (s, f) in enumerate(ds.items) if f % a.val_stride == 0]
    loader = DataLoader(torch.utils.data.Subset(ds, keep), batch_size=a.batch, num_workers=a.workers,
                        worker_init_fn=worker_init)
    preds = defaultdict(dict)
    for x, _, idx in loader:
        with autocast(device):
            logits = model(x.to(device))
        heat = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        for h, i in zip(heat, idx.tolist()):
            seq, f = ds.items[i]
            xy = heatmap_to_xy(h, a.thr)
            if xy is not None:
                preds[seq][f] = xy

    results = []
    for s in seqs:
        gt = {f: xy for f, xy in sn.load_ball_gt(s, a.root).items() if f % a.val_stride == 0}
        # only frames we actually evaluated count; renumber so evaluate() walks just those
        frames = [f for f in range(1, ds.len[s] + 1) if f % a.val_stride == 0]
        remap = {f: k + 1 for k, f in enumerate(frames)}
        results.append(sn.evaluate({remap[f]: p for f, p in preds[s].items()},
                                   {remap[f]: g for f, g in gt.items()}, len(frames)))
    model.train()
    return sn.summarize(results)


# ----------------------------------------------------------------------------- main

def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = pick_device(a.device)
    print(f"device: {device} | data: {a.root}")

    seqs = sn.list_seqs(a.root)
    if a.limit_seqs:
        seqs = seqs[:a.limit_seqs]

    # split by game, so val never sees a match from training
    by_game = defaultdict(list)
    for s in seqs:
        by_game[game_id(s, a.root)].append(s)
    games = sorted(by_game)
    if len(games) > 1:
        val_games = games[-min(a.val_games, len(games) - 1):]
        val_seqs = [s for g in val_games for s in by_game[g]]
    else:                                                        # only one game (e.g. --limit-seqs): hold out last seq
        val_games, val_seqs = games, seqs[-1:]
    train_seqs = [s for s in seqs if s not in val_seqs]
    print(f"{len(games)} games | train {len(train_seqs)} seqs | val {len(val_seqs)} seqs (games {val_games})")

    print("resizing frames into cache (first run only)...")
    build_cache(seqs, a)

    train_ds = BallFrames(train_seqs, a, train=True)
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    print(f"train samples: {len(train_ds)}")

    model = TrackNet(base=a.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader))
    start_epoch, best_f1 = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"resumed from epoch {ck['epoch']}, best F1 {best_f1:.3f}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if a.resume and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    last_path = a.out.replace(".pt", "_last.pt")

    for epoch in range(start_epoch, a.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, (x, y, _) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                logits = model(x)
            loss = wbce_loss(logits.float(), y)                  # loss in fp32 for stability
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 50 == 0 or step == len(loader):
                rate = step * a.batch / (time.time() - t0)
                print(f"epoch {epoch + 1}/{a.epochs} step {step}/{len(loader)} "
                      f"loss {running / step:.5f}  {rate:.1f} img/s", flush=True)

        m = evaluate_seqs(model, val_seqs, a, device)
        print(f"== epoch {epoch + 1} val: F1 {m['f1']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  (tp {m['tp']} fp {m['fp']} fn {m['fn']})", flush=True)

        ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                  epoch=epoch, best_f1=max(best_f1, m["f1"]), args=vars(a), val=m)
        torch.save(ck, last_path)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(ck, a.out)
            print(f"   new best -> {a.out}")

    print(f"done. best val F1 {best_f1:.3f}")


if __name__ == "__main__":
    main()
