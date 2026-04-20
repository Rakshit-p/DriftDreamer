#!/usr/bin/env python3
"""Trainer for the tugbot RSSM world model.

Loads a trajectory dataset (``states``, ``actions``, ``next_states`` +
optional ``episode_starts``), trains a GRU-based RSSM with an MLP encoder
and decoder, and optionally a CarDreamer-style reward head. Validation
measures both 1-step reconstruction and open-loop multi-step imagination
error, so we can save the checkpoint with the best 12-step rollout.
"""

from __future__ import annotations

import collections
import collections.abc

for _name in collections.abc.__all__:
    setattr(collections, _name, getattr(collections.abc, _name))

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from attrdict import AttrDict
from torch.utils.tensorboard import SummaryWriter

from dreamer.modules.model import RSSM
from dreamer.utils.utils import (
    build_network,
    create_normal_dist,
    horizontal_forward,
)

from cardreamer_reward import (
    RewardConfig,
    compute_reward,
    relative_goal,
    sample_synthetic_goal,
)


def _to_attrdict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


class MLPEncoder(nn.Module):
    """Maps ``(B, T, obs_dim) → (B, T, embedded_state_size)``."""

    def __init__(self, obs_dim: int, config: AttrDict):
        super().__init__()
        p = config.parameters.dreamer.encoder
        out = config.parameters.dreamer.embedded_state_size
        self.obs_dim = obs_dim
        self.network = build_network(obs_dim, p.hidden_size, p.num_layers, p.activation, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return horizontal_forward(
            self.network, x, input_shape=(self.obs_dim,), output_shape=(-1,))


class VectorDecoder(nn.Module):
    """Gaussian reconstruction of the full observation from ``(posterior || deterministic)``."""

    def __init__(self, obs_dim: int, config: AttrDict):
        super().__init__()
        p = config.parameters.dreamer.decoder
        self.obs_dim = obs_dim
        st = config.parameters.dreamer.stochastic_size
        det = config.parameters.dreamer.deterministic_size
        self.network = build_network(
            st + det, p.hidden_size, p.num_layers, p.activation, obs_dim * 2)
        self.min_std = p.min_std

    def forward(self, posterior: torch.Tensor, deterministic: torch.Tensor):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(self.obs_dim * 2,))
        return create_normal_dist(x, min_std=self.min_std, event_shape=1)


class GoalRewardHead(nn.Module):
    """Reward predictor conditioned on RSSM latents + the goal in the robot's body frame."""

    GOAL_DIM = 3

    def __init__(self, config: AttrDict):
        super().__init__()
        p = config.parameters.dreamer.reward_head
        st = config.parameters.dreamer.stochastic_size
        det = config.parameters.dreamer.deterministic_size
        self.network = build_network(
            st + det + self.GOAL_DIM, p.hidden_size, p.num_layers, p.activation, 1)

    def forward(
        self,
        posterior: torch.Tensor,
        deterministic: torch.Tensor,
        rel_goal: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([posterior, deterministic, rel_goal], dim=-1)
        return self.network(x).squeeze(-1)


def load_yaml_config(path: Path) -> AttrDict:
    with open(path) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    cfg = _to_attrdict(raw)
    cfg.parameters.dreamer.use_continue_flag = False
    return cfg


def pick_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sequences(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(obs[N, 2, D], act[N, 2, A])`` from an i.i.d. transitions dataset."""
    n, _ = states.shape
    _, ad = actions.shape
    obs = np.stack([states, next_states], axis=1).astype(np.float32)
    act = np.zeros((n, 2, ad), dtype=np.float32)
    act[:, 0, :] = actions.astype(np.float32)
    return obs, act


def build_windows_from_episodes(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    episode_starts: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a ``window``-long window inside each episode without crossing boundaries.

    Returns ``(obs[M, window, D], act[M, window, A], episode_ids[M])``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    n = states.shape[0]
    d = states.shape[-1]
    a = actions.shape[-1]

    starts = np.asarray(episode_starts, dtype=np.int64)
    ends = np.append(starts[1:], n).astype(np.int64)

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    ep_ids: list[int] = []

    for ep_idx, (s_row, e_row) in enumerate(zip(starts, ends)):
        ep_len = int(e_row - s_row)
        if ep_len < window:
            continue
        for i in range(s_row, e_row - window + 1):
            obs_window = np.empty((window, d), dtype=np.float32)
            obs_window[0] = states[i]
            obs_window[1:] = next_states[i : i + window - 1]

            act_window = np.zeros((window, a), dtype=np.float32)
            act_window[:-1] = actions[i : i + window - 1]

            obs_list.append(obs_window)
            act_list.append(act_window)
            ep_ids.append(ep_idx)

    if not obs_list:
        raise RuntimeError(
            f"No episode is at least {window} steps long. Collect longer episodes or "
            f"lower --batch-length.")

    return (
        np.stack(obs_list, axis=0).astype(np.float32),
        np.stack(act_list, axis=0).astype(np.float32),
        np.asarray(ep_ids, dtype=np.int64),
    )


def load_transition_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load the dataset; returns ``episode_starts=None`` for legacy .npz files."""
    d = np.load(path)
    return (
        d["states"], d["actions"], d["next_states"],
        d["episode_starts"] if "episode_starts" in d.files else None,
    )


def normalize(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = obs.reshape(-1, obs.shape[-1])
    mean = flat.mean(0, keepdims=True)
    std = flat.std(0, keepdims=True) + 1e-6
    return (obs - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def dynamic_losses(
    rssm: RSSM,
    encoder: MLPEncoder,
    decoder: VectorDecoder,
    obs: torch.Tensor,
    act: torch.Tensor,
    config: AttrDict,
    device: torch.device,
    reward_head: GoalRewardHead | None = None,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
    reward_cfg: RewardConfig | None = None,
    goal_rng: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruction + KL (+ open-loop recon + reward MSE when enabled)."""
    p = config.parameters.dreamer
    batch_length = p.batch_length
    bsz = obs.shape[0]

    prior, deterministic = rssm.recurrent_model_input_init(bsz)
    prior = prior.to(device)
    deterministic = deterministic.to(device)
    embedded = encoder(obs)

    post_list, det_list, prior_list = [], [], []
    prior_means, prior_stds = [], []
    post_means, post_stds = [], []

    for t in range(1, batch_length):
        deterministic = rssm.recurrent_model(prior, act[:, t - 1], deterministic)
        prior_dist, prior = rssm.transition_model(deterministic)
        post_dist, posterior = rssm.representation_model(embedded[:, t], deterministic)
        post_list.append(posterior)
        prior_list.append(prior)
        det_list.append(deterministic)
        prior_means.append(prior_dist.mean)
        prior_stds.append(prior_dist.scale)
        post_means.append(post_dist.mean)
        post_stds.append(post_dist.scale)
        prior = posterior

    post = torch.stack(post_list, dim=1)
    prior_stack = torch.stack(prior_list, dim=1)
    det = torch.stack(det_list, dim=1)
    prior_m = torch.stack(prior_means, dim=1)
    prior_s = torch.stack(prior_stds, dim=1)
    post_m = torch.stack(post_means, dim=1)
    post_s = torch.stack(post_stds, dim=1)

    recon_dist = decoder(post, det)
    recon_loss = -recon_dist.log_prob(obs[:, 1:]).mean()

    ol_recon_dist = decoder(prior_stack, det)
    ol_recon_loss = -ol_recon_dist.log_prob(obs[:, 1:]).mean()

    prior_dist = create_normal_dist(prior_m, prior_s, event_shape=1)
    post_dist = create_normal_dist(post_m, post_s, event_shape=1)
    kl = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
    kl = torch.max(torch.tensor(p.free_nats, device=device, dtype=kl.dtype), kl)

    ol_weight = float(getattr(p, "open_loop_weight", 0.0))
    loss = p.kl_divergence_scale * kl + recon_loss + ol_weight * ol_recon_loss
    metrics = {"loss": loss, "recon": recon_loss, "kl": kl, "ol_recon": ol_recon_loss}

    if reward_head is not None:
        if state_mean is None or state_std is None or reward_cfg is None:
            raise ValueError("reward_head training requires state_mean, state_std, reward_cfg")

        obs_world = obs * state_std.view(1, 1, -1) + state_mean.view(1, 1, -1)
        s_world = obs_world[:, 0]
        sp_world = obs_world[:, 1]

        rh_cfg = config.parameters.dreamer.reward_head
        radius_range = tuple(rh_cfg.goal_radius_range)
        forward_bias = float(rh_cfg.forward_bias)

        with torch.no_grad():
            goal = sample_synthetic_goal(
                s_world, radius_range=radius_range,
                forward_bias=forward_bias, generator=goal_rng)
            r_label = compute_reward(s_world, sp_world, goal, reward_cfg)
            g_rel = relative_goal(sp_world, goal)

        post_t = post[:, 0]
        det_t = det[:, 0]
        r_pred = reward_head(post_t, det_t, g_rel)
        r_loss = torch.mean((r_pred - r_label) ** 2)

        loss = loss + float(rh_cfg.scale) * r_loss
        metrics["loss"] = loss
        metrics["reward"] = r_loss

    return metrics


def train_step(
    rssm: RSSM,
    encoder: MLPEncoder,
    decoder: VectorDecoder,
    obs: torch.Tensor,
    act: torch.Tensor,
    config: AttrDict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reward_head: GoalRewardHead | None = None,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
    reward_cfg: RewardConfig | None = None,
    goal_rng: torch.Generator | None = None,
) -> dict[str, float]:
    metrics = dynamic_losses(
        rssm, encoder, decoder, obs, act, config, device,
        reward_head=reward_head, state_mean=state_mean, state_std=state_std,
        reward_cfg=reward_cfg, goal_rng=goal_rng,
    )
    p = config.parameters.dreamer
    optimizer.zero_grad()
    metrics["loss"].backward()
    params = (list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters()))
    if reward_head is not None:
        params += list(reward_head.parameters())
    nn.utils.clip_grad_norm_(params, p.clip_grad, norm_type=p.grad_norm_type)
    optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in metrics.items()}


@torch.no_grad()
def eval_recon(
    rssm: RSSM,
    encoder: MLPEncoder,
    decoder: VectorDecoder,
    obs: torch.Tensor,
    act: torch.Tensor,
    config: AttrDict,
    device: torch.device,
    reward_head: GoalRewardHead | None = None,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
    reward_cfg: RewardConfig | None = None,
    goal_rng: torch.Generator | None = None,
) -> dict[str, float]:
    metrics = dynamic_losses(
        rssm, encoder, decoder, obs, act, config, device,
        reward_head=reward_head, state_mean=state_mean, state_std=state_std,
        reward_cfg=reward_cfg, goal_rng=goal_rng,
    )
    return {k: float(v.cpu()) for k, v in metrics.items()}


@torch.no_grad()
def eval_multistep_imagination(
    rssm: RSSM,
    encoder: MLPEncoder,
    decoder: VectorDecoder,
    obs: torch.Tensor,
    act: torch.Tensor,
    device: torch.device,
    teacher_steps: int = 1,
    open_steps: int = 12,
    probe_ks: tuple[int, ...] = (1, 4, 12),
) -> dict[str, float]:
    """Measure prior-rollout reconstruction MSE at k = 1, 4, 12 steps after teacher-forcing."""
    T = obs.shape[1]
    if T < teacher_steps + open_steps + 1:
        return {}

    bsz = obs.shape[0]
    prior, deterministic = rssm.recurrent_model_input_init(bsz)
    prior = prior.to(device)
    deterministic = deterministic.to(device)
    embedded = encoder(obs)

    latent = prior
    for t in range(1, teacher_steps + 1):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        _, _prior = rssm.transition_model(deterministic)
        _, posterior = rssm.representation_model(embedded[:, t], deterministic)
        latent = posterior

    recon_errs: list[torch.Tensor] = []
    for t in range(teacher_steps + 1, teacher_steps + open_steps + 1):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        _, prior_sample = rssm.transition_model(deterministic)
        latent = prior_sample
        x_hat = decoder(latent.unsqueeze(1), deterministic.unsqueeze(1)).mean.squeeze(1)
        mse = ((x_hat - obs[:, t]) ** 2).mean(dim=-1)
        recon_errs.append(mse)

    errs = torch.stack(recon_errs, dim=1)
    out: dict[str, float] = {}
    for k in probe_ks:
        if 1 <= k <= open_steps:
            out[f"multistep_mse_k{k}"] = float(errs[:, k - 1].mean().cpu())
    out["multistep_mse_mean"] = float(errs.mean().cpu())
    return out


def load_world_model_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RSSM, MLPEncoder, VectorDecoder, torch.Tensor, torch.Tensor, AttrDict]:
    """Load (encoder, rssm, decoder, mean, std, config) from a checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_path = Path(ckpt["config_path"])
    config = load_yaml_config(cfg_path)
    config["operation"]["device"] = str(device)
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])
    mean = torch.as_tensor(np.asarray(ckpt["mean"]), dtype=torch.float32, device=device)
    std = torch.as_tensor(np.asarray(ckpt["std"]), dtype=torch.float32, device=device)
    rssm = RSSM(action_dim, config).to(device)
    encoder = MLPEncoder(obs_dim, config).to(device)
    decoder = VectorDecoder(obs_dim, config).to(device)
    rssm.load_state_dict(ckpt["rssm"])
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    rssm.eval()
    encoder.eval()
    decoder.eval()
    return rssm, encoder, decoder, mean, std, config


def load_world_model_with_reward(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RSSM, MLPEncoder, VectorDecoder, GoalRewardHead, torch.Tensor, torch.Tensor, AttrDict]:
    """Load everything plus a GoalRewardHead; raises KeyError if the head is missing."""
    rssm, encoder, decoder, mean, std, config = load_world_model_bundle(checkpoint_path, device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if "reward_head" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'reward_head' — retrain with --train-reward-head.")
    head = GoalRewardHead(config).to(device)
    head.load_state_dict(ckpt["reward_head"])
    head.eval()
    return rssm, encoder, decoder, head, mean, std, config


@torch.no_grad()
def imagine_with_latents(
    rssm: RSSM,
    encoder: MLPEncoder,
    decoder: VectorDecoder,
    s0_norm: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll ``actions`` through the RSSM prior; return ``(preds, z_seq, d_seq)``.

    ``preds`` is normalised decoder output, ``z_seq`` is the prior stochastic
    latent at each step, ``d_seq`` is the GRU hidden state at each step.
    """
    K, H, _ = actions.shape
    obs_dim = s0_norm.shape[-1]
    obs_in = s0_norm.view(K, 1, obs_dim)
    emb = encoder(obs_in)
    z, d = rssm.recurrent_model_input_init(K)
    z = z.to(device)
    d = d.to(device)
    zero_a = torch.zeros(K, 2, device=device, dtype=actions.dtype)
    d = rssm.recurrent_model(z, zero_a, d)
    _, z = rssm.representation_model(emb[:, 0], d)

    preds: list[torch.Tensor] = []
    z_seq: list[torch.Tensor] = []
    d_seq: list[torch.Tensor] = []
    for t in range(H):
        a = actions[:, t]
        d = rssm.recurrent_model(z, a, d)
        prior_dist, _ = rssm.transition_model(d)
        z = prior_dist.mean
        dec = decoder(z.unsqueeze(1), d.unsqueeze(1))
        preds.append(dec.mean.squeeze(1))
        z_seq.append(z)
        d_seq.append(d)
    return (
        torch.stack(preds, dim=1),
        torch.stack(z_seq, dim=1),
        torch.stack(d_seq, dim=1),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_cfg = repo_root / "dreamer" / "configs" / "tugbot-worldmodel.yml"
    default_data = repo_root.parent / "transitions.npz"

    parser = argparse.ArgumentParser(description="Train the tugbot RSSM world model.")
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--save", type=Path,
                        default=repo_root / "checkpoints" / "tugbot_world_model.pt")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-length", type=int, default=None,
                        help="Override dreamer.batch_length (set to e.g. 16 for sequence training).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--free-nats", type=float, default=None)
    parser.add_argument("--open-loop-weight", type=float, default=0.0,
                        help="Weight of the open-loop reconstruction loss (0.2–0.5 helps imagination).")
    parser.add_argument("--save-best", action="store_true",
                        help="Also save the lowest-imag[k12] checkpoint as <save>.best.pt.")
    parser.add_argument("--multistep-val-every", type=int, default=500,
                        help="Run open-loop multistep validation every N steps (0 to disable).")
    parser.add_argument("--logdir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--train-reward-head", action="store_true",
                        help="Also train the GoalRewardHead (required for learned-cost MPC).")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="Initialise encoder/rssm/decoder from this checkpoint.")
    parser.add_argument("--freeze-dynamics", action="store_true",
                        help="Freeze encoder/rssm/decoder and train only the reward head "
                             "(requires --train-reward-head and --warm-start).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_yaml_config(args.config)
    dreamer_cfg = config["parameters"]["dreamer"]
    if args.train_steps is not None:
        dreamer_cfg["train_steps"] = args.train_steps
    if args.batch_length is not None:
        dreamer_cfg["batch_length"] = args.batch_length
    if args.batch_size is not None:
        dreamer_cfg["batch_size"] = args.batch_size
    if args.free_nats is not None:
        dreamer_cfg["free_nats"] = args.free_nats
    dreamer_cfg["open_loop_weight"] = float(args.open_loop_weight)
    device = pick_device(args.device)
    config["operation"]["device"] = str(device)

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    states_np, actions_np, next_states_np, episode_starts_np = load_transition_dataset(args.data)
    batch_length = int(config.parameters.dreamer.batch_length)

    if episode_starts_np is not None and batch_length > 2:
        obs, act, ep_ids = build_windows_from_episodes(
            states_np, actions_np, next_states_np, episode_starts_np, window=batch_length)
        print(f"[data] trajectory mode: {obs.shape[0]} windows of length {batch_length} "
              f"from {len(episode_starts_np)} episodes ({states_np.shape[0]} raw steps)")
        num_episodes = int(ep_ids.max()) + 1
        rng = np.random.RandomState(args.seed)
        ep_perm = rng.permutation(num_episodes)
        n_val_ep = max(1, int(num_episodes * args.val_fraction))
        val_episodes = set(ep_perm[:n_val_ep].tolist())
        val_mask = np.array([eid in val_episodes for eid in ep_ids])
        tr_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        print(f"[split] {len(tr_idx)} train windows  /  {len(val_idx)} val windows "
              f"({num_episodes - n_val_ep} train eps / {n_val_ep} val eps)")
    else:
        if batch_length > 2 and episode_starts_np is None:
            print("[data] WARNING: --batch-length > 2 but dataset has no 'episode_starts' — "
                  "falling back to batch_length=2.")
            dreamer_cfg["batch_length"] = 2
            batch_length = 2
        obs, act = build_sequences(states_np, actions_np, next_states_np)
        print(f"[data] legacy 1-step mode: {obs.shape[0]} i.i.d. transitions")
        n = obs.shape[0]
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * args.val_fraction))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

    if len(tr_idx) < 2:
        raise RuntimeError("Not enough training rows after split.")

    obs_tr, mean, std = normalize(obs[tr_idx])
    obs_val = (obs[val_idx] - mean) / std

    obs_dim = obs_tr.shape[-1]
    action_dim = act.shape[-1]

    rssm = RSSM(action_dim, config).to(device)
    encoder = MLPEncoder(obs_dim, config).to(device)
    decoder = VectorDecoder(obs_dim, config).to(device)

    reward_head: GoalRewardHead | None = None
    if args.train_reward_head:
        reward_head = GoalRewardHead(config).to(device)

    if args.warm_start is not None:
        if not args.warm_start.is_file():
            raise FileNotFoundError(f"--warm-start checkpoint not found: {args.warm_start}")
        try:
            ws = torch.load(args.warm_start, map_location=device, weights_only=False)
        except TypeError:
            ws = torch.load(args.warm_start, map_location=device)
        encoder.load_state_dict(ws["encoder"])
        rssm.load_state_dict(ws["rssm"])
        decoder.load_state_dict(ws["decoder"])
        if reward_head is not None and "reward_head" in ws:
            reward_head.load_state_dict(ws["reward_head"])
        print(f"[warm-start] loaded encoder/rssm/decoder from {args.warm_start}")

    if args.freeze_dynamics:
        if reward_head is None:
            raise SystemExit("--freeze-dynamics requires --train-reward-head.")
        if args.warm_start is None:
            raise SystemExit("--freeze-dynamics requires --warm-start (nothing to freeze otherwise).")
        for mod in (encoder, rssm, decoder):
            for prm in mod.parameters():
                prm.requires_grad_(False)
            mod.eval()
        params = list(reward_head.parameters())
        print("[freeze-dynamics] encoder/rssm/decoder frozen; training reward_head only.")
    else:
        params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters())
        if reward_head is not None:
            params += list(reward_head.parameters())
    optimizer = torch.optim.Adam(params, lr=config.parameters.dreamer.model_learning_rate)

    obs_tr_t = torch.from_numpy(obs_tr).to(device)
    act_tr_t = torch.from_numpy(act[tr_idx]).to(device)
    obs_val_t = torch.from_numpy(obs_val).to(device)
    act_val_t = torch.from_numpy(act[val_idx]).to(device)

    mean_t = torch.as_tensor(mean.reshape(-1), dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std.reshape(-1), dtype=torch.float32, device=device)
    reward_cfg = RewardConfig()
    goal_rng: torch.Generator | None = None
    if reward_head is not None and device.type in ("cpu", "cuda"):
        goal_rng = torch.Generator(device=device)
        goal_rng.manual_seed(args.seed)

    steps = int(config.parameters.dreamer.train_steps)
    batch = int(config.parameters.dreamer.batch_size)
    n_tr = obs_tr_t.shape[0]
    n_val = obs_val_t.shape[0]

    args.save.parent.mkdir(parents=True, exist_ok=True)

    writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        logdir = args.logdir
        if logdir is None:
            logdir = repo_root / "runs" / (
                "tugbot_worldmodel_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(logdir))
        print(f"TensorBoard log_dir={logdir}")

    log_every = max(1, int(args.log_every))

    print(f"device={device}  train={n_tr}  val={n_val}  obs_dim={obs_dim}  "
          f"action_dim={action_dim}  reward_head={'on' if reward_head is not None else 'off'}")

    def _build_payload() -> dict:
        payload: dict = {
            "encoder": encoder.state_dict(),
            "rssm": rssm.state_dict(),
            "decoder": decoder.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "mean": mean,
            "std": std,
            "config_path": str(args.config),
        }
        if reward_head is not None:
            payload["reward_head"] = reward_head.state_dict()
        return payload

    best_k12 = float("inf")
    best_step = -1
    best_path = args.save.with_name(args.save.stem + ".best" + args.save.suffix)

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_tr, (batch,), device=device)
        metrics = train_step(
            rssm, encoder, decoder, obs_tr_t[idx], act_tr_t[idx], config, optimizer, device,
            reward_head=reward_head,
            state_mean=mean_t, state_std=std_t,
            reward_cfg=reward_cfg, goal_rng=goal_rng,
        )

        if writer is not None and (step % log_every == 0 or step == 1):
            writer.add_scalar("train/loss", metrics["loss"], step)
            writer.add_scalar("train/recon", metrics["recon"], step)
            writer.add_scalar("train/kl", metrics["kl"], step)
            if "reward" in metrics:
                writer.add_scalar("train/reward_mse", metrics["reward"], step)

        run_val = (step % 200 == 0 or step == 1 or step == steps)
        run_multistep = (
            args.multistep_val_every > 0
            and (step % args.multistep_val_every == 0 or step == 1 or step == steps)
            and batch_length >= 4
        )
        if run_val or run_multistep:
            rssm.eval()
            encoder.eval()
            decoder.eval()
            if reward_head is not None:
                reward_head.eval()

            val_metrics: dict[str, float] = {}
            if run_val:
                with torch.no_grad():
                    vidx = torch.randint(0, n_val, (min(batch, n_val),), device=device)
                    val_metrics = eval_recon(
                        rssm, encoder, decoder, obs_val_t[vidx], act_val_t[vidx], config, device,
                        reward_head=reward_head,
                        state_mean=mean_t, state_std=std_t,
                        reward_cfg=reward_cfg, goal_rng=goal_rng,
                    )

            multistep_metrics: dict[str, float] = {}
            if run_multistep:
                with torch.no_grad():
                    vidx_ms = torch.randint(0, n_val, (min(batch, n_val),), device=device)
                    obs_ms = obs_val_t[vidx_ms]
                    act_ms = act_val_t[vidx_ms]
                    max_open = obs_ms.shape[1] - 2
                    if max_open >= 1:
                        open_steps = min(12, max_open)
                        multistep_metrics = eval_multistep_imagination(
                            rssm, encoder, decoder, obs_ms, act_ms, device,
                            teacher_steps=1, open_steps=open_steps,
                            probe_ks=(1, 4, min(12, open_steps)),
                        )

            if not args.freeze_dynamics:
                rssm.train()
                encoder.train()
                decoder.train()
            if reward_head is not None:
                reward_head.train()

            if writer is not None:
                if "recon" in val_metrics:
                    writer.add_scalar("val/recon", val_metrics["recon"], step)
                if "reward" in val_metrics:
                    writer.add_scalar("val/reward_mse", val_metrics["reward"], step)
                for k, v in multistep_metrics.items():
                    writer.add_scalar(f"val/{k}", v, step)

            extra = (f"  reward_mse={metrics.get('reward', float('nan')):.4f}"
                     if "reward" in metrics else "")
            val_extra = (f"  val_reward={val_metrics.get('reward', float('nan')):.4f}"
                         if "reward" in val_metrics else "")
            ms_extra = ""
            if multistep_metrics:
                ms_bits = [f"{k.replace('multistep_mse_', '')}={v:.4f}"
                           for k, v in multistep_metrics.items()
                           if k.startswith("multistep_mse_k")]
                if ms_bits:
                    ms_extra = "  imag[" + " ".join(ms_bits) + "]"
            val_recon_str = (f"  val_recon={val_metrics['recon']:.4f}"
                             if "recon" in val_metrics else "")
            ol_extra = (f"  ol_recon={float(metrics['ol_recon']):.4f}"
                        if "ol_recon" in metrics else "")
            print(f"step {step:5d}  train loss={metrics['loss']:.4f}  "
                  f"recon={metrics['recon']:.4f}  kl={metrics['kl']:.4f}"
                  f"{ol_extra}{extra}{val_recon_str}{val_extra}{ms_extra}")

            if args.save_best and multistep_metrics:
                current_k12 = multistep_metrics.get(
                    "multistep_mse_k12",
                    multistep_metrics.get("multistep_mse_mean", float("inf")))
                if current_k12 < best_k12:
                    best_k12 = float(current_k12)
                    best_step = step
                    torch.save(_build_payload(), best_path)
                    print(f"  ↳ new best imag[k12]={best_k12:.4f} → {best_path}")

    torch.save(_build_payload(), args.save)
    print(f"Saved checkpoint → {args.save}")
    if args.save_best and best_step > 0:
        print(f"Best checkpoint (imag[k12]={best_k12:.4f} @ step {best_step}) → {best_path}")
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
