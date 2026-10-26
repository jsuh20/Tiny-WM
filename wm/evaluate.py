"""Steps 3-5: compounding error, counterfactual action test, and slice metrics.

Models compared:
  full   - MLP that sees the ego action
  blind  - same MLP, same data, never sees the action
  cv     - constant-velocity physics baseline (no learning, no action)

Usage: python -m wm.evaluate   (writes results/ plots + results.json)
       python -m wm.evaluate --data data/episodes_clean.pkl --models models_clean --out results_clean --clean
"""
import argparse
import copy
import json
import os
import warnings
from functools import partial
from multiprocessing import Pool

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wm.common import (DT, Driver, constant_velocity_step, make_env, road_state,  # noqa: E402
                       scene_from_road, select_neighbours)
from wm.dataset import (CRASH_SOON_STEPS, build_transitions, load_episodes,  # noqa: E402
                        near_collision, risk_tags, split_of)
from wm.train import load_model  # noqa: E402

HORIZONS = [1, 5, 10, 20]
MAX_H = 20
CF_H = 10                 # counterfactual horizon (2 s)
CF_STARTS = 400
DENSE_MIN, SPARSE_MAX = 8, 3   # vehicles within 50 m of ego

# Validated categorical slots 1-3 (dataviz reference palette, light mode)
COL = {"full": "#2a78d6", "blind": "#eb6834", "cv": "#1baf7a"}
LABEL = {"full": "Full model (sees action)", "blind": "Action-blind model", "cv": "Constant velocity"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "axes.axisbelow": True,
})


def rollout(step_fn, E0, Nb0, M0, actions):
    """Roll a model forward on its own predictions. actions [B, H, 2] -> E [B,H+1,5], Nb [B,H+1,K,5]."""
    E, Nb, M = (torch.as_tensor(v, dtype=torch.float32) for v in (E0, Nb0, M0))
    acts = torch.as_tensor(actions, dtype=torch.float32)
    Es, Ns = [E], [Nb]
    with torch.no_grad():
        for h in range(acts.shape[1]):
            E, Nb = step_fn(E, Nb, M, acts[:, h])
            Es.append(E)
            Ns.append(Nb)
    return torch.stack(Es, 1).numpy(), torch.stack(Ns, 1).numpy()


def step_fns(models_dir):
    full, blind = load_model("full", models_dir), load_model("blind", models_dir)
    return {"full": full.step, "blind": blind.step, "cv": constant_velocity_step}


def pos_errors(E_pred, N_pred, E_true, N_true, M):
    """Per-sample ego and mean-neighbour position error [m] at every horizon."""
    ego = np.linalg.norm(E_pred[..., :2] - E_true[..., :2], axis=-1)
    nb = np.linalg.norm(N_pred[..., :2] - N_true[..., :2], axis=-1)       # [B, H, K]
    m = M[:, None, :]
    nb = (nb * m).sum(-1) / np.maximum(m.sum(-1), 1)
    return ego, nb


# ---------------------------------------------------------------- 1-step + compounding

def rollout_windows(episodes, stride=5):
    """Fixed-identity windows of MAX_H steps from test episodes, with slice tags at the start."""
    E, Nb, M, A, tags = [], [], [], [], []
    for ep in episodes:
        if split_of(ep["seed"]) != "test":
            continue
        S, Acts = ep["S"].astype(np.float64), ep["A"]
        T = len(Acts)
        for t0 in range(0, T - MAX_H + 1, stride):
            ids = select_neighbours(S[t0])
            e, n, m = scene_from_road(S[t0:t0 + MAX_H + 1], ids)
            E.append(e)
            Nb.append(n)
            M.append(m[0])
            A.append(Acts[t0:t0 + MAX_H])
            n_close, ttc, gap = risk_tags(S[t0])
            crash_soon = ep["crashed"] and t0 >= T - MAX_H - CRASH_SOON_STEPS
            tags.append((n_close, ttc, gap, crash_soon))
    tags = np.array(tags, dtype=np.float64)
    return (np.array(E), np.array(Nb), np.array(M), np.array(A),
            {"n_close": tags[:, 0], "ttc": tags[:, 1], "gap": tags[:, 2], "crash_soon": tags[:, 3] > 0})


def slices(tags):
    near = near_collision(tags)
    return {
        "all": np.ones_like(near),
        "sparse (<=3 cars in 50 m)": tags["n_close"] <= SPARSE_MAX,
        "dense (>=8 cars in 50 m)": tags["n_close"] >= DENSE_MIN,
        "normal": ~near,
        "near-collision": near,
    }


def one_step_eval(episodes, models):
    full = models["full"]
    """One-step error on every test transition, in physical units and as normalised MSE."""
    test = build_transitions(episodes, "test")
    x, a, d, w = (torch.as_tensor(test[k]) for k in ("x", "a", "d", "w"))
    preds = {}
    with torch.no_grad():
        for name in ("full", "blind"):
            preds[name] = models[name].predict_delta(x, a).numpy()
    # constant velocity, expressed as a delta in the same representation
    cv = np.zeros_like(test["d"])
    xn = test["x"]
    cv[:, 0], cv[:, 1] = xn[:, 1] * DT, xn[:, 2] * DT        # ego vx, vy -> dx, dy
    nb = xn[:, 5:].reshape(len(xn), -1, 7)
    rel_dpos = (nb[..., 3:5] - xn[:, None, 1:3]) * DT * nb[..., :1]   # (v_nb - v_ego) * dt
    cvn = cv[:, 6:].reshape(len(xn), -1, 6)
    cvn[..., :2] = rel_dpos
    cv[:, 6:] = cvn.reshape(len(xn), -1)
    preds["cv"] = cv

    D, W = test["d"], test["w"]
    dn = (D - full.d_mean.numpy()) / full.d_std.numpy()
    out = {}
    for name, P in preds.items():
        pn = (P - full.d_mean.numpy()) / full.d_std.numpy()
        sq = (pn - dn) ** 2 * W
        ego_pos = np.linalg.norm(P[:, :2] - D[:, :2], axis=-1)
        ego_vel = np.linalg.norm(P[:, 2:4] - D[:, 2:4], axis=-1)
        Pn, Dn = P[:, 6:].reshape(len(P), -1, 6), D[:, 6:].reshape(len(D), -1, 6)
        m = W[:, 6::6]
        nb_pos = (np.linalg.norm(Pn[..., :2] - Dn[..., :2], axis=-1) * m).sum(-1) / np.maximum(m.sum(-1), 1)
        out[name] = {"per_sample_nmse": sq.sum(-1) / W.sum(-1), "ego_pos": ego_pos,
                     "ego_vel": ego_vel, "nb_pos": nb_pos}
    return test, out


# ---------------------------------------------------------------- counterfactuals

BRANCHES = {
    "coast": lambda h: [0.0, 0.0],
    "brake": lambda h: [-1.0, 0.0],
    "accelerate": lambda h: [1.0, 0.0],
    # a jink: steer one way for 3 steps, back for 3, then straight
    "swerve +y": lambda h: [0.0, 0.3 if h < 3 else -0.3 if h < 6 else 0.0],
    "swerve -y": lambda h: [0.0, -0.3 if h < 3 else 0.3 if h < 6 else 0.0],
}


def cf_start(seed, noisy=True):
    """Drive a fresh (never-seen) episode for a bit, then branch it with each scripted action."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    env = make_env(density=float(rng.uniform(0.5, 2.5)), vehicles_count=int(rng.integers(15, 55)))
    env.reset(seed=int(seed))
    driver = Driver(rng, noisy=noisy)
    for _ in range(int(rng.integers(10, 40))):
        _, _, term, trunc, _ = env.step(driver.act(road_state(env)))
        if term or trunc:
            return None
    S0 = road_state(env)
    ids = select_neighbours(S0)
    out = {"ids": ids, "S0": S0}
    for name, act in BRANCHES.items():
        b = copy.deepcopy(env)
        traj, acts = [S0], []
        for h in range(CF_H):
            a = np.array(act(h))
            b.step(a)
            traj.append(road_state(b))
            acts.append(a)
        out[name] = (np.array(traj), np.array(acts), bool(b.unwrapped.vehicle.crashed))
    env.close()
    return out


def counterfactual_eval(fns, noisy=True):
    seeds = range(10**6, 10**6 + int(CF_STARTS * 1.3))
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        starts = [s for s in pool.map(partial(cf_start, noisy=noisy), seeds) if s is not None][:CF_STARTS]
    res = {"n_starts": len(starts)}
    true, pred = {}, {m: {} for m in fns}
    for br in BRANCHES:
        Es, Ns, Ms, As = [], [], [], []
        for s in starts:
            traj, acts, _ = s[br]
            e, n, m = scene_from_road(traj, s["ids"])
            Es.append(e)
            Ns.append(n)
            Ms.append(m[0])
            As.append(acts)
        E, Nb, M, A = map(np.array, (Es, Ns, Ms, As))
        true[br] = (E, Nb, M)
        for mname, fn in fns.items():
            pred[mname][br] = rollout(fn, E[:, 0], Nb[:, 0], M, A)

    # action effect = where you end up under action X minus where you end up when coasting
    effects = {}
    for br in BRANCHES:
        if br == "coast":
            continue
        E_t, N_t, M = true[br]
        E_c, N_c, _ = true["coast"]
        eff_true = E_t[:, -1, :2] - E_c[:, -1, :2]
        nb_true = N_t[:, -1, :, :2] - N_c[:, -1, :, :2]
        axis = 0 if br in ("brake", "accelerate") else 1
        effects[br] = {"true": eff_true[:, axis].tolist()}
        row = {"true_effect_m": float(np.linalg.norm(eff_true, axis=-1).mean())}
        crashed = np.array([s[br][2] for s in starts])
        row["ego_crashed_in_branch"] = int(crashed.sum())
        # neighbours that actually react (moved > 0.5 m differently because of the ego's choice)
        react = (np.linalg.norm(nb_true, axis=-1) > 0.5) & (M > 0)
        row["reacting_neighbours"] = int(react.sum())
        row["reacting_true_effect_m"] = float(np.linalg.norm(nb_true, axis=-1)[react].mean()) if react.any() else 0.0
        for mname in fns:
            E_p, N_p = pred[mname][br]
            E_pc, N_pc = pred[mname]["coast"]
            eff = E_p[:, -1, :2] - E_pc[:, -1, :2]
            nb_eff = N_p[:, -1, :, :2] - N_pc[:, -1, :, :2]
            effects[br][mname] = eff[:, axis].tolist()
            err = np.linalg.norm(eff - eff_true, axis=-1)
            row[mname] = {
                "effect_error_m": float(err.mean()),
                "effect_captured": float(1 - err.mean() / np.linalg.norm(eff_true, axis=-1).mean()),
                "sign_agreement": float((np.sign(eff[:, axis]) == np.sign(eff_true[:, axis])).mean()),
                "final_ego_pos_error_m": float(np.linalg.norm(E_p[:, -1, :2] - E_t[:, -1, :2], axis=-1).mean()),
                "final_ego_err_if_crashed_m": float(np.linalg.norm(E_p[:, -1, :2] - E_t[:, -1, :2], axis=-1)[crashed].mean())
                if crashed.any() else None,
                "final_ego_err_if_not_crashed_m": float(np.linalg.norm(E_p[:, -1, :2] - E_t[:, -1, :2], axis=-1)[~crashed].mean()),
                "reacting_nb_effect_captured": float(
                    1 - np.linalg.norm(nb_eff - nb_true, axis=-1)[react].mean() / row["reacting_true_effect_m"])
                if react.any() else None,
            }
        res[br] = row
    return res, effects


# ---------------------------------------------------------------- plots

def plot_compounding(ego, nb, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hs = np.arange(1, MAX_H + 1)
    for ax, errs, title in [(axes[0], ego, "Ego car"), (axes[1], nb, "Nearby cars (mean)")]:
        for name in ("full", "blind", "cv"):
            y = errs[name][:, 1:].mean(0)
            ax.plot(hs, y, color=COL[name], lw=2, label=LABEL[name])
            ax.plot(HORIZONS, y[np.array(HORIZONS) - 1], "o", color=COL[name], ms=5,
                    mec=SURF, mew=1.5)
            nudge = {"full": -7, "blind": 0, "cv": 7}[name] if title.startswith("Nearby") else 0
            ax.annotate(f"{y[-1]:.1f} m", (hs[-1], y[-1]), xytext=(6, nudge), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        ax.set_yscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, MAX_H + 3.5)
        ax.set_xlabel("Rollout horizon (steps of 0.2 s)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Position error [m], log scale")
    axes[0].legend(loc="upper left")
    fig.suptitle("Error compounds when the model rolls forward on its own predictions",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_counterfactual(effects, path):
    brs = [b for b in BRANCHES if b != "coast"]
    fig, axes = plt.subplots(1, len(brs), figsize=(13, 3.6))
    for ax, br in zip(axes, brs):
        t = np.array(effects[br]["true"])
        lim = np.abs(t).max() * 1.15
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--", zorder=1)
        for name in ("blind", "full"):
            ax.scatter(t, effects[br][name], s=14, color=COL[name], alpha=0.7, lw=0.4,
                       edgecolor=SURF, label=LABEL[name], zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        axis = "x (along road)" if br in ("brake", "accelerate") else "y (across road)"
        ax.set_title(f"{br} vs coast", loc="left")
        ax.set_xlabel(f"True shift in {axis} [m]")
    axes[0].set_ylabel("Predicted shift [m]")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Counterfactual test: how much does the ego's final position (after {CF_H * DT:.0f} s) "
                 "move because of the action? Dashed line = perfect.",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slices(table, path):
    names = list(table["one_step"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "one_step", "One-step ego position error"),
                           (axes[1], "h10", "Ego position error after 10 steps (2 s)")]:
        x = np.arange(len(names))
        wbar = 0.26
        for i, m in enumerate(("full", "blind", "cv")):
            vals = [table[key][s][m] for s in names]
            ax.bar(x + (i - 1) * wbar, vals, wbar * 0.92, color=COL[m], label=LABEL[m])
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(").replace(" in 50 m", "") for n in names], fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Mean error [m]")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/episodes.pkl")
    p.add_argument("--models", default="models")
    p.add_argument("--out", default="results")
    p.add_argument("--clean", action="store_true", help="use the noise-free driver to reach counterfactual start states")
    args = p.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    episodes = load_episodes(args.data)
    fns = step_fns(args.models)
    models = {"full": load_model("full", args.models), "blind": load_model("blind", args.models)}
    R = {}

    # 1-step
    test, one = one_step_eval(episodes, models)
    tslices = slices(test)
    R["one_step"] = {s: {m: {k: float(v[mask].mean()) for k, v in one[m].items()}
                         for m in one} | {"n": int(mask.sum())}
                     for s, mask in tslices.items()}

    # compounding
    E, Nb, M, A, tags = rollout_windows(episodes)
    ego_err, nb_err = {}, {}
    for name, fn in fns.items():
        Ep, Np = rollout(fn, E[:, 0], Nb[:, 0], M, A)
        ego_err[name], nb_err[name] = pos_errors(Ep, Np, E, Nb, M)
    R["compounding"] = {"n_windows": len(E), **{
        name: {f"h{h}": {"ego_pos": float(ego_err[name][:, h].mean()),
                         "nb_pos": float(nb_err[name][:, h].mean())} for h in HORIZONS}
        for name in fns}}
    plot_compounding(ego_err, nb_err, f"{out}/compounding_error.png")

    wslices = slices(tags)
    R["rollout_slices"] = {s: {"n": int(mask.sum()), **{
        name: {f"h{h}": float(ego_err[name][mask, h].mean()) for h in HORIZONS} for name in fns}}
        for s, mask in wslices.items()}
    slice_table = {"one_step": {s: {m: R["one_step"][s][m]["ego_pos"] for m in fns} for s in tslices},
                   "h10": {s: {m: R["rollout_slices"][s][m]["h10"] for m in fns} for s in wslices}}
    plot_slices(slice_table, f"{out}/slices.png")

    # counterfactual
    R["counterfactual"], effects = counterfactual_eval(fns, noisy=not args.clean)
    plot_counterfactual(effects, f"{out}/counterfactual.png")

    with open(f"{out}/results.json", "w") as f:
        json.dump(R, f, indent=2)
    print_summary(R)


def print_summary(R):
    print("\n=== One-step error on test set ===")
    print(f"{'slice':32s} {'n':>6s} | {'model':6s} {'norm MSE':>9s} {'ego pos m':>10s} {'ego vel m/s':>11s} {'nbr pos m':>10s}")
    for s, row in R["one_step"].items():
        for m in ("full", "blind", "cv"):
            r = row[m]
            print(f"{s:32s} {row['n']:6d} | {m:6s} {r['per_sample_nmse']:9.4f} {r['ego_pos']:10.4f} "
                  f"{r['ego_vel']:11.4f} {r['nb_pos']:10.4f}")
    print(f"\n=== Compounding error ({R['compounding']['n_windows']} rollouts) — ego / neighbour position error [m] ===")
    for m in ("full", "blind", "cv"):
        print(f"{m:6s} " + "  ".join(f"h{h:<2d} {R['compounding'][m][f'h{h}']['ego_pos']:6.2f} / "
                                     f"{R['compounding'][m][f'h{h}']['nb_pos']:5.2f}" for h in HORIZONS))
    print("\n=== Rollout ego error by slice [m] ===")
    for s, row in R["rollout_slices"].items():
        print(f"{s:32s} n={row['n']:5d} " + " ".join(
            f"{m}: " + "/".join(f"{row[m][f'h{h}']:.2f}" for h in HORIZONS) for m in ("full", "blind", "cv")))
    cf = R["counterfactual"]
    print(f"\n=== Counterfactual ({cf['n_starts']} start states, {CF_H} steps) ===")
    for br, row in cf.items():
        if br == "n_starts":
            continue
        print(f"{br:11s} true effect {row['true_effect_m']:5.2f} m; reacting nbrs {row['reacting_neighbours']}; "
              f"ego crashed in {row['ego_crashed_in_branch']} branches")
        for m in ("full", "blind", "cv"):
            r = row[m]
            nbc = r["reacting_nb_effect_captured"]
            print(f"   {m:6s} effect captured {r['effect_captured']:6.1%}  sign agree {r['sign_agreement']:5.1%}  "
                  f"final ego err {r['final_ego_pos_error_m']:5.2f} m  nbr reaction captured "
                  f"{'n/a' if nbc is None else f'{nbc:.1%}'}")
            if r["final_ego_err_if_crashed_m"] is not None:
                print(f"          final ego err: crashed {r['final_ego_err_if_crashed_m']:.2f} m vs "
                      f"no crash {r['final_ego_err_if_not_crashed_m']:.2f} m")


if __name__ == "__main__":
    main()
