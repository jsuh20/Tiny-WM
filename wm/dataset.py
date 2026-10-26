"""Turn logged episodes into (scene, action, next scene) transitions + slice tags.

Episodes are split by seed: seed % 5 == 0 -> test, == 1 -> validation, else train.
"""
import pickle

import numpy as np

from wm.common import (K, delta_mask, scene_from_road, select_neighbours, to_delta,
                       to_input)

CAR_LEN, CAR_W = 5.0, 2.0
DENSE_RADIUS = 50.0     # [m] count vehicles this close to ego for the density slice
NEAR_TTC = 1.5          # [s]
NEAR_GAP = 3.0          # [m] bumper-to-bumper
CRASH_SOON_STEPS = 10   # 2 s


def load_episodes(path="data/episodes.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_of(seed):
    return "test" if seed % 5 == 0 else "val" if seed % 5 == 1 else "train"


def risk_tags(S_t):
    """Local density, min time-to-collision and min bumper gap for the ego at one timestep."""
    ego, oth = S_t[0], S_t[1:]
    dx = oth[:, 0] - ego[0]
    dy = oth[:, 1] - ego[1]
    n_close = int((np.hypot(dx, dy) < DENSE_RADIUS).sum())
    overlap = np.abs(dy) < CAR_W + 0.5
    gap = np.abs(dx) - CAR_LEN
    closing = np.where(dx > 0, ego[2] - oth[:, 2], oth[:, 2] - ego[2])
    ttc = np.where(overlap & (closing > 0.1), np.maximum(gap, 0) / np.maximum(closing, 0.1), np.inf)
    min_gap = gap[overlap].min() if overlap.any() else np.inf
    return n_close, float(ttc.min()) if len(ttc) else np.inf, float(min_gap)


def build_transitions(episodes, split):
    """One-step dataset for one split. Returns dict of arrays."""
    X, A, D, W, meta = [], [], [], [], []
    for ei, ep in enumerate(episodes):
        if split_of(ep["seed"]) != split:
            continue
        S, Acts = ep["S"].astype(np.float64), ep["A"]
        T = len(Acts)
        for t in range(T):
            ids = select_neighbours(S[t])
            E, Nb, M = scene_from_road(S[t:t + 2], ids)
            X.append(to_input(E[:1], Nb[:1], M[:1])[0])
            D.append(to_delta(E[:1], Nb[:1], E[1:], Nb[1:])[0])
            W.append(delta_mask(M[:1])[0])
            A.append(Acts[t])
            n_close, ttc, gap = risk_tags(S[t])
            crash_soon = ep["crashed"] and t >= T - CRASH_SOON_STEPS
            meta.append((ei, t, n_close, ttc, gap, crash_soon, ep["density"]))
    meta = np.array(meta, dtype=np.float64)
    return {
        "x": np.array(X, np.float32), "a": np.array(A, np.float32),
        "d": np.array(D, np.float32), "w": np.array(W, np.float32),
        "episode": meta[:, 0].astype(int), "t": meta[:, 1].astype(int),
        "n_close": meta[:, 2], "ttc": meta[:, 3], "gap": meta[:, 4],
        "crash_soon": meta[:, 5].astype(bool), "density_cfg": meta[:, 6],
    }


def near_collision(ds):
    return (ds["ttc"] < NEAR_TTC) | (ds["gap"] < NEAR_GAP) | ds["crash_soon"]


def norm_stats(train):
    w = train["w"]
    d_mean = (train["d"] * w).sum(0) / np.maximum(w.sum(0), 1)
    d_var = (((train["d"] - d_mean) * w) ** 2).sum(0) / np.maximum(w.sum(0), 1)
    return {
        "x_mean": train["x"].mean(0), "x_std": train["x"].std(0) + 1e-3,
        "d_mean": d_mean, "d_std": np.sqrt(d_var) + 1e-4,
    }


__all__ = ["K", "load_episodes", "build_transitions", "near_collision", "norm_stats", "split_of"]
