"""Step 2: train the dynamics MLP and the action-blind baseline on the same data.

Usage: python -m wm.train
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from wm.common import DynamicsMLP
from wm.dataset import build_transitions, load_episodes, norm_stats


def masked_mse(pred, target, w):
    return (((pred - target) ** 2) * w).sum() / w.sum()


def train_model(use_action, train, val, stats, epochs, seed=0, batch=512, lr=2e-3):
    torch.manual_seed(seed)
    model = DynamicsMLP(stats, use_action=use_action)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((len(train["x"]) + batch - 1) // batch))
    T = {k: torch.as_tensor(train[k]) for k in ("x", "a", "d", "w")}
    V = {k: torch.as_tensor(val[k]) for k in ("x", "a", "d", "w")}
    d_mean, d_std = model.d_mean, model.d_std
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(T["x"]))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            pred = model.forward_norm(T["x"][idx], T["a"][idx])
            loss = masked_mse(pred, (T["d"][idx] - d_mean) / d_std, T["w"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = masked_mse(model.forward_norm(V["x"], V["a"]), (V["d"] - d_mean) / d_std, V["w"]).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [{'full' if use_action else 'blind'}] epoch {ep:3d}  train {loss.item():.4f}  val {vl:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/episodes.pkl")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--models", default="models")
    args = p.parse_args()
    torch.set_num_threads(8)

    t0 = time.time()
    eps = load_episodes(args.data)
    train, val = build_transitions(eps, "train"), build_transitions(eps, "val")
    print(f"train {len(train['x'])}  val {len(val['x'])} transitions ({time.time() - t0:.0f}s to build)")
    stats = norm_stats(train)
    os.makedirs(args.models, exist_ok=True)
    with open(f"{args.models}/stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    for use_action, name in [(True, "full"), (False, "blind")]:
        model, best = train_model(use_action, train, val, stats, args.epochs)
        torch.save(model.state_dict(), f"{args.models}/{name}.pt")
        print(f"saved {args.models}/{name}.pt  best val (normalised MSE) {best:.4f}")


def load_model(name, models_dir="models"):
    with open(f"{models_dir}/stats.pkl", "rb") as f:
        stats = pickle.load(f)
    m = DynamicsMLP(stats, use_action=(name == "full"))
    m.load_state_dict(torch.load(f"{models_dir}/{name}.pt"))
    return m.eval()


if __name__ == "__main__":
    main()
