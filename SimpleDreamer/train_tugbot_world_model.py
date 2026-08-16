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
import math
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

from obs_layout import COSSIN, COSSIN_STD, RAW, ObsLayout, layout_from_checkpoint
from reward import (
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
    """Reward predictor conditioned on RSSM latents + the goal in the robot's body frame.

    ``goal_dim`` follows the observation layout: 3 for a raw heading error,
    4 when the heading error is encoded as ``(cos, sin)``.
    """

    GOAL_DIM = 3

    def __init__(self, config: AttrDict, goal_dim: int | None = None):
        super().__init__()
        p = config.parameters.dreamer.reward_head
        st = config.parameters.dreamer.stochastic_size
        det = config.parameters.dreamer.deterministic_size
        self.goal_dim = int(goal_dim if goal_dim is not None else self.GOAL_DIM)
        self.network = build_network(
            st + det + self.goal_dim, p.hidden_size, p.num_layers, p.activation, 1)

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


def normalize(
    obs: np.ndarray,
    layout: ObsLayout | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score observations, leaving any ``(cos, sin)`` heading pair on a
    shared constant scale so the unit circle is not distorted."""
    layout = layout or ObsLayout()
    mean, std = layout.norm_stats(obs)
    return (obs - mean) / std, mean, std


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
    layout: ObsLayout | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruction + KL (+ open-loop recon + reward MSE when enabled)."""
    layout = layout or ObsLayout()
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
                forward_bias=forward_bias, generator=goal_rng, layout=layout)
            r_label = compute_reward(s_world, sp_world, goal, reward_cfg, layout=layout)
            g_rel = relative_goal(sp_world, goal, layout=layout)

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
    layout: ObsLayout | None = None,
) -> dict[str, float]:
    metrics = dynamic_losses(
        rssm, encoder, decoder, obs, act, config, device,
        reward_head=reward_head, state_mean=state_mean, state_std=state_std,
        reward_cfg=reward_cfg, goal_rng=goal_rng, layout=layout,
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
    layout: ObsLayout | None = None,
) -> dict[str, float]:
    metrics = dynamic_losses(
        rssm, encoder, decoder, obs, act, config, device,
        reward_head=reward_head, state_mean=state_mean, state_std=state_std,
        reward_cfg=reward_cfg, goal_rng=goal_rng, layout=layout,
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
    layout: ObsLayout | None = None,
) -> dict[str, float]:
    """Measure prior-rollout reconstruction MSE at each ``probe_ks`` step.

    Besides the total MSE over all dims, this also reports the MSE
    restricted to the pose block and to the lidar block, so pose drift can
    be tracked separately from lidar prediction quality.
    """
    pose_dim = (layout or ObsLayout()).pose_dim
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

    per_dim_errs: list[torch.Tensor] = []
    for t in range(teacher_steps + 1, teacher_steps + open_steps + 1):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        _, prior_sample = rssm.transition_model(deterministic)
        latent = prior_sample
        x_hat = decoder(latent.unsqueeze(1), deterministic.unsqueeze(1)).mean.squeeze(1)
        per_dim_errs.append((x_hat - obs[:, t]) ** 2)

    errs_per_dim = torch.stack(per_dim_errs, dim=1)
    errs = errs_per_dim.mean(dim=-1)
    obs_dim = errs_per_dim.shape[-1]
    pose_errs = errs_per_dim[..., :pose_dim].mean(dim=-1) if pose_dim > 0 else None
    lidar_errs = errs_per_dim[..., pose_dim:].mean(dim=-1) if obs_dim > pose_dim else None

    out: dict[str, float] = {}
    for k in probe_ks:
        if 1 <= k <= open_steps:
            out[f"multistep_mse_k{k}"] = float(errs[:, k - 1].mean().cpu())
            if pose_errs is not None:
                out[f"multistep_mse_pose_k{k}"] = float(pose_errs[:, k - 1].mean().cpu())
            if lidar_errs is not None:
                out[f"multistep_mse_lidar_k{k}"] = float(lidar_errs[:, k - 1].mean().cpu())
    out["multistep_mse_mean"] = float(errs.mean().cpu())
    if pose_errs is not None:
        out["multistep_mse_pose_mean"] = float(pose_errs.mean().cpu())
    if lidar_errs is not None:
        out["multistep_mse_lidar_mean"] = float(lidar_errs.mean().cpu())
    return out


def resolve_config_path(stored: str | Path) -> Path:
    """Resolve a checkpoint's ``config_path``, tolerating a foreign machine.

    Checkpoints record an absolute config path from the machine that trained
    them. When that path is missing (different user, container, or clone
    location) fall back to the same file name inside this repo's
    ``dreamer/configs`` directory.
    """
    stored_path = Path(stored)
    if stored_path.is_file():
        return stored_path
    local = Path(__file__).resolve().parent / "dreamer" / "configs" / stored_path.name
    if local.is_file():
        return local
    raise FileNotFoundError(
        f"Config not found at checkpoint path {stored_path} nor at {local}. "
        f"Pass the config explicitly or re-save the checkpoint."
    )


def load_world_model_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RSSM, MLPEncoder, VectorDecoder, torch.Tensor, torch.Tensor, AttrDict, ObsLayout]:
    """Load (rssm, encoder, decoder, mean, std, config, layout) from a checkpoint.

    ``layout`` records how the checkpoint encodes heading; checkpoints
    written before that distinction existed are treated as ``raw``.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_path = resolve_config_path(ckpt["config_path"])
    config = load_yaml_config(cfg_path)
    config["operation"]["device"] = str(device)
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])
    layout = layout_from_checkpoint(ckpt)
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
    return rssm, encoder, decoder, mean, std, config, layout


def load_world_model_with_reward(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RSSM, MLPEncoder, VectorDecoder, GoalRewardHead, torch.Tensor, torch.Tensor, AttrDict, ObsLayout]:
    """Load everything plus a GoalRewardHead; raises KeyError if the head is missing."""
    rssm, encoder, decoder, mean, std, config, layout = load_world_model_bundle(
        checkpoint_path, device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if "reward_head" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'reward_head' — retrain with --train-reward-head.")
    head = GoalRewardHead(config, goal_dim=layout.rel_goal_dim).to(device)
    head.load_state_dict(ckpt["reward_head"])
    head.eval()
    return rssm, encoder, decoder, head, mean, std, config, layout


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


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi)``."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@torch.no_grad()
def imagine_with_latents_hybrid(
    rssm: RSSM,
    encoder: MLPEncoder,
    decoder: VectorDecoder,
    s0_norm: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    action_dt: float,
    pose_mode: str = "position",
    yaw_mode: str = "mid",
    v_mode: str = "mid",
    layout: ObsLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hybrid RSSM + exact-kinematics rollout (physics-grounded position).

    Two splice variants are supported:

    ``pose_mode="position"`` (default, "A-prime")
        Integrate only ``x, y`` with the exact unicycle update, driven by the
        RSSM's own *decoded* ``yaw`` and ``v``. The ``yaw, v, w`` dimensions
        are left untouched, so the learned command-to-realised actuator
        mapping the RSSM already captured is preserved. This avoids feeding
        the exact integrator a biased commanded-velocity input.

    ``pose_mode="full"``
        Overwrite all five pose dims using the *commanded* action and an
        analytically integrated yaw. Kept for A/B comparison -- it assumes
        the command is realised instantly, which is wrong on a real robot
        and makes the exact integrator compound that bias.

    ``yaw_mode`` selects which yaw drives ``cos/sin`` over the interval:
    ``"mid"`` (midpoint of previous and decoded next yaw, 2nd-order),
    ``"prev"`` (forward Euler) or ``"next"``. ``v_mode`` likewise picks
    ``"mid"`` or ``"next"`` for the integrated speed.

    Returns ``(preds_norm, z_seq, d_seq)`` with identical shapes to
    :func:`imagine_with_latents`, so it is a drop-in replacement.
    """
    if pose_mode not in ("position", "full"):
        raise ValueError(f"pose_mode must be 'position' or 'full', got {pose_mode!r}")
    if yaw_mode not in ("mid", "prev", "next"):
        raise ValueError(f"yaw_mode must be 'mid', 'prev' or 'next', got {yaw_mode!r}")
    if v_mode not in ("mid", "next"):
        raise ValueError(f"v_mode must be 'mid' or 'next', got {v_mode!r}")

    layout = layout or ObsLayout()
    pose_dim = layout.pose_dim
    K, H, _ = actions.shape
    obs_dim = s0_norm.shape[-1]
    if obs_dim < pose_dim:
        raise ValueError(f"obs_dim={obs_dim} must be >= pose_dim={pose_dim}")
    if mean.shape[-1] != obs_dim or std.shape[-1] != obs_dim:
        raise ValueError(
            f"mean/std last-dim must equal obs_dim={obs_dim}; "
            f"got {tuple(mean.shape)} / {tuple(std.shape)}"
        )

    mean_v = mean.view(1, -1)
    std_v = std.view(1, -1)
    mean_pose = mean_v[:, :pose_dim]
    std_pose = std_v[:, :pose_dim]

    s0_world = s0_norm * std_v + mean_v
    x_prev = s0_world[:, 0].clone()
    y_prev = s0_world[:, 1].clone()
    yaw_prev = layout.get_yaw(s0_world).clone()
    v_prev = layout.get_v(s0_world).clone()

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
        z_prior = prior_dist.mean
        dec = decoder(z_prior.unsqueeze(1), d.unsqueeze(1))
        pred_norm = dec.mean.squeeze(1)
        corrected_norm = pred_norm.clone()

        if pose_mode == "full":
            v_cmd = a[:, 0]
            w_cmd = a[:, 1]
            x_new = x_prev + v_cmd * action_dt * torch.cos(yaw_prev)
            y_new = y_prev + v_cmd * action_dt * torch.sin(yaw_prev)
            yaw_new = wrap_pi(yaw_prev + w_cmd * action_dt)
            pose_new = layout.build_pose(x_new, y_new, yaw_new, v_cmd, w_cmd)
            corrected_norm[:, :pose_dim] = (pose_new - mean_pose) / std_pose
            yaw_prev = yaw_new
            v_prev = v_cmd
        else:
            pred_world = pred_norm * std_v + mean_v
            yaw_next = layout.get_yaw(pred_world)
            v_next = layout.get_v(pred_world)

            if yaw_mode == "mid":
                yaw_int = yaw_prev + 0.5 * wrap_pi(yaw_next - yaw_prev)
            elif yaw_mode == "prev":
                yaw_int = yaw_prev
            else:
                yaw_int = yaw_next

            v_int = 0.5 * (v_prev + v_next) if v_mode == "mid" else v_next

            x_new = x_prev + v_int * action_dt * torch.cos(yaw_int)
            y_new = y_prev + v_int * action_dt * torch.sin(yaw_int)

            corrected_norm[:, 0] = (x_new - mean_v[0, 0]) / std_v[0, 0]
            corrected_norm[:, 1] = (y_new - mean_v[0, 1]) / std_v[0, 1]
            yaw_prev = yaw_next
            v_prev = v_next

        emb_corr = encoder(corrected_norm.unsqueeze(1)).squeeze(1)
        post_dist, _ = rssm.representation_model(emb_corr, d)
        z = post_dist.mean

        preds.append(corrected_norm)
        z_seq.append(z)
        d_seq.append(d)
        x_prev = x_new
        y_prev = y_new

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
    parser.add_argument("--yaw-encoding", choices=(RAW, COSSIN), default=RAW,
                        help="How heading enters the observation. 'raw' keeps the angle and has "
                             "a discontinuity at +-pi that MSE cannot fit; 'cossin' expands it "
                             "to (cos, sin), widening obs_dim by one.")
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

    layout = ObsLayout(args.yaw_encoding)
    if layout.is_cossin:
        obs = layout.encode(obs)
        print(f"[layout] heading encoded as (cos, sin): obs_dim {obs.shape[-1] - 1} -> "
              f"{obs.shape[-1]}; the pair shares std={COSSIN_STD:.4f} and is not z-scored")
    else:
        print("[layout] heading kept as a raw angle (has a discontinuity at +-pi)")

    obs_tr, mean, std = normalize(obs[tr_idx], layout)
    obs_val = (obs[val_idx] - mean) / std

    obs_dim = obs_tr.shape[-1]
    action_dim = act.shape[-1]

    rssm = RSSM(action_dim, config).to(device)
    encoder = MLPEncoder(obs_dim, config).to(device)
    decoder = VectorDecoder(obs_dim, config).to(device)

    reward_head: GoalRewardHead | None = None
    if args.train_reward_head:
        reward_head = GoalRewardHead(config, goal_dim=layout.rel_goal_dim).to(device)

    if args.warm_start is not None:
        if not args.warm_start.is_file():
            raise FileNotFoundError(f"--warm-start checkpoint not found: {args.warm_start}")
        try:
            ws = torch.load(args.warm_start, map_location=device, weights_only=False)
        except TypeError:
            ws = torch.load(args.warm_start, map_location=device)
        ws_layout = layout_from_checkpoint(ws)
        if ws_layout.yaw_encoding != layout.yaw_encoding:
            raise SystemExit(
                f"--warm-start checkpoint uses yaw_encoding={ws_layout.yaw_encoding!r} but this "
                f"run uses {layout.yaw_encoding!r}; the observation widths differ so the weights "
                f"are incompatible. Retrain from scratch or match the encoding."
            )
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
            "yaw_encoding": layout.yaw_encoding,
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
            reward_cfg=reward_cfg, goal_rng=goal_rng, layout=layout,
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
                        reward_cfg=reward_cfg, goal_rng=goal_rng, layout=layout,
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
                            layout=layout,
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
