"""Step 1: run the simulator with the noisy driver and log full road states + actions.

Each episode is saved as:
  S  [T+1, N, 5]  every vehicle's x, y, vx, vy, heading (ego is row 0)
  A  [T, 2]       normalised ego action (accel, steer)
  crashed         whether the episode ended in an ego crash
Traffic density is randomised per episode so we can slice dense vs sparse later.

Usage: python -m wm.collect --transitions 100000
"""
import argparse
import os
import pickle
import time
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np

from wm.common import Driver, make_env, road_state


def run_episode(seed, noisy=True):
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    density = float(rng.uniform(0.5, 2.5))
    count = int(rng.integers(15, 55))
    env = make_env(density=density, vehicles_count=count)
    env.reset(seed=int(seed))
    driver = Driver(rng, noisy=noisy)
    S, A = [road_state(env)], []
    while True:
        a = driver.act(S[-1])
        _, _, term, trunc, _ = env.step(a)
        S.append(road_state(env))
        A.append(a)
        if term or trunc:
            break
    v = env.unwrapped.vehicle
    env.close()
    return {"seed": int(seed), "density": density, "vehicles_count": count,
            "S": np.array(S, dtype=np.float32), "A": np.array(A, dtype=np.float32),
            "crashed": bool(v.crashed), "offroad": bool(not v.on_road)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--out", default="data/episodes.pkl")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--clean", action="store_true", help="noise-free driver (actions predictable from state)")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    episodes, n, t0 = [], 0, time.time()
    seeds = iter(range(args.seed_offset, args.seed_offset + 10**7))
    with Pool(args.workers) as pool:
        while n < args.transitions:
            batch = [next(seeds) for _ in range(args.workers * 4)]
            for ep in pool.imap_unordered(partial(run_episode, noisy=not args.clean), batch):
                episodes.append(ep)
                n += len(ep["A"])
            print(f"{len(episodes)} episodes, {n} transitions, {time.time() - t0:.0f}s", flush=True)
    episodes.sort(key=lambda e: e["seed"])
    with open(args.out, "wb") as f:
        pickle.dump(episodes, f)
    crashes = sum(e["crashed"] for e in episodes)
    print(f"saved {args.out}: {len(episodes)} episodes, {n} transitions, {crashes} crashes")


if __name__ == "__main__":
    main()
