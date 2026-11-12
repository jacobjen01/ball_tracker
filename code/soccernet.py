"""Shared helpers: load ball ground truth + evaluate a ball track.

Used by both the non-NN and the NN method so they are scored identically.
"""
import glob
import os
import re

import numpy as np

ROOT = os.path.expanduser("~/Perception/data/soccernet/train")


def list_seqs(root=ROOT):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "SNMOT-*")))


def seq_dir(seq, root=ROOT):
    return os.path.join(root, seq)


def frame_source(seq_or_video, root=ROOT):
    """Path that cv2.VideoCapture can open: SoccerNet image sequence or a video file."""
    if os.path.isfile(seq_or_video):
        return seq_or_video
    return os.path.join(seq_dir(seq_or_video, root), "img1", "%06d.jpg")


def load_ball_boxes(seq, root=ROOT):
    """SoccerNet MOT gt -> {frame: (x, y, w, h)} for the ball track(s). Frames are 1-based."""
    d = seq_dir(seq, root)
    info = open(os.path.join(d, "gameinfo.ini")).read()
    ball_ids = {int(m) for m in re.findall(r"trackletID_(\d+)=\s*ball", info)}
    boxes = {}
    for line in open(os.path.join(d, "gt", "gt.txt")):
        p = line.split(",")
        if int(p[1]) in ball_ids:
            boxes[int(p[0])] = tuple(map(float, p[2:6]))
    return boxes


def load_ball_gt(seq, root=ROOT):
    """{frame: (cx, cy)} ball centres."""
    return {f: (x + w / 2, y + h / 2) for f, (x, y, w, h) in load_ball_boxes(seq, root).items()}


def load_csv_gt(path):
    """Own annotations (test clip): CSV with header frame,x,y,visible. Frames 1-based."""
    gt = {}
    for line in open(path).read().splitlines()[1:]:
        f, x, y, vis = line.split(",")[:4]
        if int(vis):
            gt[int(f)] = (float(x), float(y))
    return gt


def evaluate(pred, gt, n_frames, thr=15.0):
    """Frame-wise centre-distance evaluation.

    pred: {frame: (x, y)} -- frames without a prediction are simply absent (or NaN).
    gt:   {frame: (x, y)} -- frames where the ball is not visible are absent.
    A prediction is a true positive if it is within `thr` px of the GT centre.
    A prediction that is too far away counts as both a FP and a FN.
    """
    tp = fp = fn = 0
    errs = []
    for f in range(1, n_frames + 1):
        p = pred.get(f)
        if p is not None and np.isnan(p[0]):
            p = None
        g = gt.get(f)
        if p is not None and g is not None:
            e = float(np.hypot(p[0] - g[0], p[1] - g[1]))
            if e <= thr:
                tp += 1
                errs.append(e)
            else:
                fp += 1
                fn += 1
        elif p is not None:
            fp += 1
        elif g is not None:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                mean_err=float(np.mean(errs)) if errs else float("nan"))


def summarize(results):
    """Pool per-clip evaluate() dicts (micro average over all frames)."""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)
