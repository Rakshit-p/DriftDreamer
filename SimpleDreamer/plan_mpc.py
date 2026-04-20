#!/usr/bin/env python3
"""Receding-horizon CEM-MPC planner for the tugbot.

Given the current state and a goal, this module searches for the best
12-step action sequence using the Cross-Entropy Method, scoring each
rollout either through the learned RSSM + reward head or through a
pure-kinematic unicycle model. A squared-hinge obstacle penalty can be
added on top when the observation carries 16-D lidar rays.

Only the first action of the best sequence is executed; the caller
re-plans every tick so the loop is closed around the real sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import train_tugbot_world_model as wm
from cardreamer_reward import (
    RewardConfig,
    compute_reward,
    relative_goal,
)


POSE_DIM = 5


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@torch.no_grad()
def diff_drive_rollout(
    s0_world: torch.Tensor,
    actions: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Unicycle kinematic rollout; `(K, 5)` start → `(K, H, 5)` future poses."""
    K, H, _ = actions.shape
    x = s0_world[:, 0].clone()
    y = s0_world[:, 1].clone()
    yaw = s0_world[:, 2].clone()
    out = torch.zeros(K, H, POSE_DIM, device=actions.device, dtype=actions.dtype)
    for t in range(H):
        v = actions[:, t, 0]
        w = actions[:, t, 1]
        x = x + v * dt * torch.cos(yaw)
        y = y + v * dt * torch.sin(yaw)
        yaw = _wrap_pi(yaw + w * dt)
        out[:, t, 0] = x
        out[:, t, 1] = y
        out[:, t, 2] = yaw
        out[:, t, 3] = v
        out[:, t, 4] = w
    return out


@dataclass
class MPCState:
    """CEM warm-start state carried between consecutive plan calls."""

    mu: torch.Tensor | None = None
    sig: torch.Tensor | None = None
    horizon: int = 0
    _plan_idx: int = field(default=0)

    def warm_start_next(self, horizon: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, sig)`` for the next plan, shifted by one timestep."""
        if self.mu is None or self.mu.shape[0] != horizon or self.horizon != horizon:
            mu = torch.zeros(horizon, 2, device=device, dtype=dtype)
            sig = torch.ones(horizon, 2, device=device, dtype=dtype) * 0.25
        else:
            mu = torch.cat([self.mu[1:], self.mu[-1:]], dim=0).to(device=device, dtype=dtype)
            sig = torch.cat([self.sig[1:], self.sig[-1:]], dim=0).to(device=device, dtype=dtype)
        return mu, sig

    def update(self, mu: torch.Tensor, sig: torch.Tensor, horizon: int) -> None:
        self.mu = mu.detach().clone()
        self.sig = sig.detach().clone()
        self.horizon = horizon
        self._plan_idx += 1


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    g.manual_seed(seed)
    return g


@torch.no_grad()
def rollout_and_score(
    rssm,
    encoder,
    decoder,
    reward_head,
    mean: torch.Tensor,
    std: torch.Tensor,
    s0_norm: torch.Tensor,
    actions: torch.Tensor,
    goal_xyyaw: torch.Tensor,
    device: torch.device,
    discount: float = 0.97,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll `actions` through the RSSM and score every rollout with the reward head."""
    preds_norm, z_seq, d_seq = wm.imagine_with_latents(
        rssm, encoder, decoder, s0_norm, actions, device
    )
    preds_world = preds_norm * std + mean

    K, H, _ = preds_world.shape
    pose_preds = preds_world[..., :POSE_DIM]
    g_full = goal_xyyaw.view(1, 1, 3).expand(K, H, 3)
    g_rel = relative_goal(pose_preds, g_full)

    r_pred = reward_head(z_seq, d_seq, g_rel)

    if discount == 1.0:
        total = r_pred.sum(dim=1)
    else:
        gammas = torch.pow(
            torch.tensor(discount, device=device, dtype=r_pred.dtype),
            torch.arange(H, device=device, dtype=r_pred.dtype),
        )
        total = (r_pred * gammas.view(1, H)).sum(dim=1)
    return total, preds_world


@torch.no_grad()
def score_analytical(
    preds_world: torch.Tensor,
    s0_world: torch.Tensor,
    actions: torch.Tensor,
    goal_xyyaw: torch.Tensor,
    reward_cfg: RewardConfig,
    discount: float = 0.97,
) -> torch.Tensor:
    """Score rollouts with the CarDreamer reward formula on the decoded pose."""
    pose_preds = preds_world[..., :POSE_DIM]
    pose_s0 = s0_world[..., :POSE_DIM]

    K, H, _ = pose_preds.shape
    g_full = goal_xyyaw.view(1, 1, 3).expand(K, H, 3)
    prev = torch.cat(
        [pose_s0.view(1, 1, POSE_DIM).expand(K, 1, POSE_DIM), pose_preds[:, :-1]],
        dim=1,
    )
    r = compute_reward(prev, pose_preds, g_full, reward_cfg)
    if discount == 1.0:
        return r.sum(dim=1)
    gammas = torch.pow(
        torch.tensor(discount, device=r.device, dtype=r.dtype),
        torch.arange(H, device=r.device, dtype=r.dtype),
    )
    return (r * gammas.view(1, H)).sum(dim=1)


@torch.no_grad()
def score_obstacle_penalty(
    preds_world: torch.Tensor,
    *,
    safety_radius: float = 0.6,
    max_range: float = 5.0,
    discount: float = 0.97,
) -> torch.Tensor | None:
    """Squared-hinge penalty on decoded lidar rays that fall inside the safety radius."""
    obs_dim = preds_world.shape[-1]
    if obs_dim <= POSE_DIM:
        return None
    rays_norm = preds_world[..., POSE_DIM:].clamp(0.0, 1.0)
    rays_m = rays_norm * max_range
    violation = torch.clamp(safety_radius - rays_m, min=0.0)
    per_step_pen = (violation * violation).sum(dim=-1)

    H = per_step_pen.shape[-1]
    if discount == 1.0:
        return -per_step_pen.sum(dim=1)
    gammas = torch.pow(
        torch.tensor(discount, device=per_step_pen.device, dtype=per_step_pen.dtype),
        torch.arange(H, device=per_step_pen.device, dtype=per_step_pen.dtype),
    )
    return -(per_step_pen * gammas.view(1, H)).sum(dim=1)


@torch.no_grad()
def plan_cem_mpc(
    rssm,
    encoder,
    decoder,
    reward_head,
    mean: torch.Tensor,
    std: torch.Tensor,
    s0_world: np.ndarray,
    goal_xyyaw: np.ndarray,
    *,
    state: MPCState,
    horizon: int = 12,
    samples: int = 256,
    iters: int = 4,
    elite_frac: float = 0.15,
    lin_range: tuple[float, float] = (-0.25, 0.30),
    ang_range: tuple[float, float] = (-1.0, 1.0),
    discount: float = 0.97,
    device: torch.device,
    seed: int = 0,
    dynamics: str = "analytical",
    cost: str = "analytical",
    action_dt: float = 0.3,
    reward_cfg: RewardConfig | None = None,
    obstacle_weight: float = 0.0,
    obstacle_safety: float = 0.6,
    obstacle_max_range: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Run one CEM solve and return the first action plus diagnostics."""
    if reward_cfg is None:
        reward_cfg = RewardConfig()
    if dynamics not in ("rssm", "analytical"):
        raise ValueError(f"dynamics must be 'rssm' or 'analytical', got {dynamics!r}")
    if cost not in ("learned", "analytical", "blend"):
        raise ValueError(f"cost must be 'learned', 'analytical' or 'blend', got {cost!r}")
    if dynamics == "analytical" and cost != "analytical":
        raise ValueError("cost='learned'/'blend' requires dynamics='rssm' (reward head needs RSSM latents).")

    g = torch.as_tensor(goal_xyyaw, dtype=torch.float32, device=device)
    s0_full = torch.as_tensor(s0_world, dtype=torch.float32, device=device).view(1, -1)
    obs_dim = s0_full.shape[-1]
    if obs_dim < POSE_DIM:
        raise ValueError(f"s0_world must have at least {POSE_DIM} columns, got {obs_dim}.")
    s0_pose = s0_full[:, :POSE_DIM]

    rng = _make_generator(device, seed + state._plan_idx)
    lo = torch.tensor([lin_range[0], ang_range[0]], device=device, dtype=torch.float32)
    hi = torch.tensor([lin_range[1], ang_range[1]], device=device, dtype=torch.float32)

    mu, sig = state.warm_start_next(horizon, device=device, dtype=torch.float32)

    best_seq = mu.clone()
    best_r = -float("inf")
    last_preds = None

    s0_pose_batched = s0_pose.expand(samples, POSE_DIM).contiguous()
    if dynamics == "rssm":
        s0_full_norm = (s0_full - mean) / std
        s0n = s0_full_norm.expand(samples, obs_dim).contiguous()

    obs_has_rays = obs_dim > POSE_DIM and obstacle_weight > 0.0 and dynamics == "rssm"

    for _ in range(iters):
        eps = torch.randn(samples, horizon, 2, generator=rng, device=device, dtype=torch.float32)
        actions = mu.unsqueeze(0) + sig.unsqueeze(0) * eps
        actions = torch.max(torch.min(actions, hi), lo)

        if dynamics == "rssm":
            rewards_learned, preds_world = rollout_and_score(
                rssm, encoder, decoder, reward_head, mean, std, s0n, actions, g, device,
                discount=discount,
            )
        else:
            preds_world = diff_drive_rollout(s0_pose_batched, actions, action_dt)
            rewards_learned = None

        if cost == "learned":
            scores = rewards_learned
        elif cost == "analytical":
            scores = score_analytical(preds_world, s0_pose.squeeze(0), actions, g, reward_cfg, discount)
        else:
            ra = score_analytical(preds_world, s0_pose.squeeze(0), actions, g, reward_cfg, discount)
            scores = 0.5 * rewards_learned + 0.5 * ra

        if obs_has_rays:
            obs_pen = score_obstacle_penalty(
                preds_world,
                safety_radius=obstacle_safety,
                max_range=obstacle_max_range,
                discount=discount,
            )
            if obs_pen is not None:
                scores = scores + obstacle_weight * obs_pen

        elite_n = max(2, int(samples * elite_frac))
        elite_idx = torch.argsort(scores, descending=True)[:elite_n]
        elite = actions[elite_idx]
        mu = elite.mean(dim=0)
        sig = elite.std(dim=0).clamp(min=0.05)

        j = int(torch.argmax(scores).item())
        if float(scores[j].item()) > best_r:
            best_r = float(scores[j].item())
            best_seq = actions[j].clone()
            last_preds = preds_world[j].clone()

    state.update(mu, sig, horizon)

    info = {
        "plan": best_seq.cpu().numpy(),
        "preds_world": None if last_preds is None else last_preds.cpu().numpy(),
        "best_reward": best_r,
        "mu": mu.cpu().numpy(),
        "sig": sig.cpu().numpy(),
        "dynamics": dynamics,
        "cost": cost,
        "obstacle_weight": obstacle_weight,
    }
    return best_seq[0].cpu().numpy(), info
