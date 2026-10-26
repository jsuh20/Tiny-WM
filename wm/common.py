"""Shared pieces: env config, data-collection driver, state representation, model.

State representation (per timestep, "scene"):
  E  [B, 5]     ego in world coords: x, y, vx, vy, heading
  N  [B, K, 5]  K tracked neighbours in world coords (same fields)
  M  [B, K]     1 if the neighbour slot is filled

Model input (translation-invariant along the road, so ego x is dropped):
  ego:        y, vx, vy, cos h, sin h                         (5)
  neighbour:  present, dx, dy, vx, vy, cos h, sin h  (dx,dy relative to ego)  (7 each)
Model output (the change over one step, dt = 0.2 s):
  ego:        dx, dy, dvx, dvy, dcos, dsin                    (6)
  neighbour:  d(rel x), d(rel y), dvx, dvy, dcos, dsin         (6 each)
"""
import numpy as np
import torch
import torch.nn as nn

K = 8                 # neighbours tracked
NEIGHBOUR_RANGE = 100.0  # [m] only vehicles this close are tracked
POLICY_HZ = 5
DT = 1.0 / POLICY_HZ
STEER_MAX = 0.2       # [rad] steering at action = +/-1
ACC_MAX = 5.0         # [m/s^2] acceleration at action = +/-1
LANE_W = 4.0
N_LANES = 4

IN_DIM = 5 + 7 * K
OUT_DIM = 6 + 6 * K
ACT_DIM = 2


def make_env(density=1.0, vehicles_count=40, duration=20):
    import gymnasium as gym
    import highway_env  # noqa: F401  (registers envs)
    return gym.make("highway-v0", config={
        "action": {"type": "ContinuousAction",
                   "acceleration_range": [-ACC_MAX, ACC_MAX],
                   "steering_range": [-STEER_MAX, STEER_MAX]},
        "observation": {"type": "Kinematics"},  # unused; we read the road directly
        "policy_frequency": POLICY_HZ,
        "simulation_frequency": 15,
        "lanes_count": N_LANES,
        "vehicles_count": vehicles_count,
        "vehicles_density": density,
        "duration": duration,
        "offroad_terminal": True,
    })


def road_state(env):
    """All vehicles as [N, 5] (x, y, vx, vy, heading); ego is row 0."""
    u = env.unwrapped
    ego = u.vehicle
    vs = [ego] + [v for v in u.road.vehicles if v is not ego]
    return np.array([[v.position[0], v.position[1], v.velocity[0], v.velocity[1], v.heading]
                     for v in vs], dtype=np.float64)


class Driver:
    """Noisy lane-keeping driver used to collect data.

    Tracks a target lane and target speed, re-picks them at random, adds
    Ornstein-Uhlenbeck noise, and sometimes fires short bursts of random
    actions so the data covers hard braking, hard acceleration and swerves.
    """

    def __init__(self, rng, noisy=True):
        self.rng = rng
        self.noisy = noisy
        self.lane = None          # set to the current lane on the first call
        self.speed = rng.uniform(20, 30)
        self.noise = np.zeros(2)
        self.burst = 0
        self.burst_action = np.zeros(2)

    def _gap(self, S, lane):
        """Distance and speed of the nearest car ahead (and gap behind) in a lane."""
        ego = S[0]
        others = S[1:]
        in_lane = np.abs(others[:, 1] - lane * LANE_W) < LANE_W / 2
        dx = others[:, 0] - ego[0]
        ahead = in_lane & (dx > 0)
        behind = in_lane & (dx <= 0)
        lead_gap, lead_v = np.inf, None
        if ahead.any():
            i = np.argmin(np.where(ahead, dx, np.inf))
            lead_gap, lead_v = dx[i] - 5.0, others[i, 2]
        back_gap = -dx[behind].max() - 5.0 if behind.any() else np.inf
        return lead_gap, lead_v, back_gap

    def act(self, S):
        """S: road_state() rows, ego first. Returns normalised [accel, steer]."""
        r = self.rng
        x, y, vx, vy, h = S[0]
        cur = int(np.clip(round(y / LANE_W), 0, N_LANES - 1))
        if self.lane is None:
            self.lane = cur
        if r.random() < 0.03:
            cand = int(np.clip(self.lane + r.choice([-1, 1]), 0, N_LANES - 1))
            lg, _, bg = self._gap(S, cand)
            if lg > 15 and bg > 10:
                self.lane = cand
        if r.random() < 0.04:
            self.speed = r.uniform(18, 32)
        if self.noisy and self.burst == 0 and r.random() < 0.05:
            self.burst = int(r.integers(1, 4))
            self.burst_action = np.array([r.uniform(-1, 1), r.uniform(-0.4, 0.4)])
        if self.burst > 0:
            self.burst -= 1
            return self.burst_action.copy()

        # IDM-style longitudinal control toward the target speed, braking for the leader
        speed = np.hypot(vx, vy)
        acc = 3.0 * (1 - (speed / self.speed) ** 4)
        lg, lv, _ = self._gap(S, cur)
        if lv is not None:
            s_star = 5 + speed * 1.2 + speed * (speed - lv) / (2 * np.sqrt(3.0 * 5.0))
            acc -= 3.0 * (max(s_star, 0) / max(lg, 0.5)) ** 2
        acc /= ACC_MAX

        y_err = self.lane * LANE_W - y
        h_des = np.clip(0.03 * y_err, -0.08, 0.08)
        steer = (1.5 * (h_des - h) - 0.3 * h) / STEER_MAX
        if self.noisy:
            self.noise += -0.3 * self.noise + r.normal(0, [0.12, 0.08])
        return np.clip(np.array([acc, steer]) + self.noise, -1, 1)


# ---------------------------------------------------------------- scenes

def select_neighbours(S_t):
    """Pick ids (rows of S_t, excluding ego row 0) of the K nearest vehicles in range."""
    rel = S_t[1:, :2] - S_t[0, :2]
    d = np.hypot(rel[:, 0], rel[:, 1])
    order = np.argsort(d)
    order = order[d[order] < NEIGHBOUR_RANGE][:K]
    return order + 1


def scene_from_road(S, ids):
    """S [T, N, 5] or [N, 5]; ids: neighbour row indices (len <= K). Returns E, N, M."""
    S = np.asarray(S)
    single = S.ndim == 2
    if single:
        S = S[None]
    T = S.shape[0]
    E = S[:, 0, :].copy()
    Nb = np.zeros((T, K, 5))
    M = np.zeros((T, K))
    Nb[:, :len(ids)] = S[:, ids, :]
    M[:, :len(ids)] = 1
    if single:
        return E[0], Nb[0], M[0]
    return E, Nb, M


def _xp(a):
    return torch if isinstance(a, torch.Tensor) else np


def sort_slots(E, Nb, M):
    """Permutation that orders neighbour slots by distance (absent slots last)."""
    xp = _xp(E)
    d = xp.hypot(Nb[..., 0] - E[..., None, 0], Nb[..., 1] - E[..., None, 1])
    d = xp.where(M > 0, d, d * 0 + 1e9)
    return xp.argsort(d, axis=-1) if xp is np else torch.argsort(d, dim=-1)


def gather_slots(X, perm):
    """X [B, K, F] reordered along K by perm [B, K]."""
    if isinstance(X, torch.Tensor):
        return torch.gather(X, 1, perm[..., None].expand(-1, -1, X.shape[-1]))
    return np.take_along_axis(X, perm[..., None], axis=1)


def to_input(E, Nb, M):
    """Scene -> model input [B, IN_DIM]. Slots must already be sorted."""
    xp = _xp(E)
    cat = (lambda xs: torch.cat(xs, -1)) if xp is torch else (lambda xs: np.concatenate(xs, -1))
    stack = (lambda xs: torch.stack(xs, -1)) if xp is torch else (lambda xs: np.stack(xs, -1))
    ego = stack([E[:, 1], E[:, 2], E[:, 3], xp.cos(E[:, 4]), xp.sin(E[:, 4])])
    m = M
    nb = stack([m,
                (Nb[..., 0] - E[:, None, 0]) * m,
                (Nb[..., 1] - E[:, None, 1]) * m,
                Nb[..., 2] * m, Nb[..., 3] * m,
                xp.cos(Nb[..., 4]) * m, xp.sin(Nb[..., 4]) * m])
    return cat([ego, nb.reshape(nb.shape[0], -1)])


def to_delta(E, Nb, E2, Nb2):
    """Change from scene t to t+1 (same slot order) -> [B, OUT_DIM]."""
    ego = np.stack([E2[:, 0] - E[:, 0], E2[:, 1] - E[:, 1], E2[:, 2] - E[:, 2], E2[:, 3] - E[:, 3],
                    np.cos(E2[:, 4]) - np.cos(E[:, 4]), np.sin(E2[:, 4]) - np.sin(E[:, 4])], -1)
    rel = Nb[..., :2] - E[:, None, :2]
    rel2 = Nb2[..., :2] - E2[:, None, :2]
    nb = np.concatenate([rel2 - rel, Nb2[..., 2:4] - Nb[..., 2:4],
                         (np.cos(Nb2[..., 4]) - np.cos(Nb[..., 4]))[..., None],
                         (np.sin(Nb2[..., 4]) - np.sin(Nb[..., 4]))[..., None]], -1)
    return np.concatenate([ego, nb.reshape(len(E), -1)], -1)


def apply_delta(E, Nb, D):
    """Inverse of to_delta: scene t + predicted change -> scene t+1 (torch)."""
    de = D[:, :6]
    dn = D[:, 6:].reshape(-1, K, 6)
    ch = torch.cos(E[:, 4]) + de[:, 4]
    sh = torch.sin(E[:, 4]) + de[:, 5]
    E2 = torch.stack([E[:, 0] + de[:, 0], E[:, 1] + de[:, 1], E[:, 2] + de[:, 2],
                      E[:, 3] + de[:, 3], torch.atan2(sh, ch)], -1)
    rel = Nb[..., :2] - E[:, None, :2] + dn[..., :2]
    nch = torch.cos(Nb[..., 4]) + dn[..., 4]
    nsh = torch.sin(Nb[..., 4]) + dn[..., 5]
    Nb2 = torch.cat([E2[:, None, :2] + rel, Nb[..., 2:4] + dn[..., 2:4],
                     torch.atan2(nsh, nch)[..., None]], -1)
    return E2, Nb2


def delta_mask(M):
    """Which output dims count in the loss: ego always, neighbours only if present."""
    xp = _xp(M)
    ones = xp.ones_like(M[:, :1]).repeat(6, 1) if xp is torch else np.ones((len(M), 6))
    nb = M[:, :, None].repeat(1, 1, 6) if xp is torch else np.repeat(M[:, :, None], 6, -1)
    nb = nb.reshape(len(M), -1)
    return torch.cat([ones, nb], -1) if xp is torch else np.concatenate([ones, nb], -1)


# ---------------------------------------------------------------- model

class DynamicsMLP(nn.Module):
    """Predicts the one-step change in the scene. use_action=False is the action-blind baseline."""

    def __init__(self, stats, use_action=True, hidden=256, layers=3):
        super().__init__()
        self.use_action = use_action
        d_in = IN_DIM + (ACT_DIM if use_action else 0)
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, OUT_DIM))
        self.net = nn.Sequential(*mods)
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def forward_norm(self, x, a):
        """Normalised input -> normalised delta."""
        xn = (x - self.x_mean) / self.x_std
        if self.use_action:
            xn = torch.cat([xn, a], -1)
        return self.net(xn)

    def predict_delta(self, x, a):
        return self.forward_norm(x, a) * self.d_std + self.d_mean

    def step(self, E, Nb, M, a):
        """One model step on a scene (re-sorts slots by distance first, like the training data)."""
        perm = sort_slots(E, Nb, M)
        Nb_s, M_s = gather_slots(Nb, perm), torch.gather(M, 1, perm)
        D = self.predict_delta(to_input(E, Nb_s, M_s), a)
        E2, Nb2_s = apply_delta(E, Nb_s, D)
        inv = torch.argsort(perm, dim=1)
        return E2, gather_slots(Nb2_s, inv)


def constant_velocity_step(E, Nb, M, a):
    """Physics baseline: everyone keeps their current velocity."""
    E2 = E.clone()
    E2[:, :2] += E[:, 2:4] * DT
    Nb2 = Nb.clone()
    Nb2[..., :2] += Nb[..., 2:4] * DT
    return E2, Nb2
