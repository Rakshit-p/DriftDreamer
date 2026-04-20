"""CarDreamer-style waypoint-following reward for the tugbot.

Given a transition (s, s', goal) this module returns a scalar reward that
encourages moving toward the goal at the desired speed, staying on the
goal-heading lane, and finally stopping at the destination. The same
function is used both to label training data for the reward head and to
score imagined rollouts inside the CEM-MPC planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights tuned for the tugbot's ~0.25 m/s operating speed."""

    desired_speed: float = 0.25
    w_waypoint: float = 2.0
    w_speed: float = 2.0
    w_out_of_lane: float = 3.0
    w_destination: float = 20.0
    w_time: float = 0.0

    r_reach: float = 0.35
    lane_tolerance: float = 0.30
    goal_tol: float = 0.25

    perp_speed_cap: float = 0.20


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def relative_goal(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return (dx_ego, dy_ego, dyaw) — the goal in the robot's body frame."""
    sx, sy, syaw = state[..., 0], state[..., 1], state[..., 2]
    gx, gy, gyaw = goal[..., 0], goal[..., 1], goal[..., 2]
    c, s = torch.cos(syaw), torch.sin(syaw)
    dx = gx - sx
    dy = gy - sy
    dxe = c * dx + s * dy
    dye = -s * dx + c * dy
    dyaw = _wrap_pi(gyaw - syaw)
    return torch.stack([dxe, dye, dyaw], dim=-1)


def compute_reward(
    state: torch.Tensor,
    next_state: torch.Tensor,
    goal: torch.Tensor,
    cfg: RewardConfig = RewardConfig(),
) -> torch.Tensor:
    """Scalar reward for a single (state, next_state, goal) transition."""
    nx, ny, nyaw = next_state[..., 0], next_state[..., 1], next_state[..., 2]
    v_lin = next_state[..., 3]

    gx, gy, gyaw = goal[..., 0], goal[..., 1], goal[..., 2]
    tx, ty = torch.cos(gyaw), torch.sin(gyaw)
    ox, oy = gx - nx, gy - ny

    d_goal = torch.sqrt(ox * ox + oy * oy + 1e-12)
    o_par = ox * tx + oy * ty
    px = ox - o_par * tx
    py = oy - o_par * ty
    d_perp = torch.sqrt(px * px + py * py + 1e-12)

    vx = v_lin * torch.cos(nyaw)
    vy = v_lin * torch.sin(nyaw)
    v_par = vx * tx + vy * ty
    v_perp = torch.abs(vx * (-ty) + vy * tx)

    r_wp = cfg.w_waypoint * (d_goal < cfg.r_reach).to(state.dtype)
    r_speed = cfg.w_speed * (
        cfg.desired_speed
        - torch.abs(v_par - cfg.desired_speed)
        - 2.0 * torch.clamp(v_perp, max=cfg.perp_speed_cap)
    )
    r_oof = -cfg.w_out_of_lane * torch.clamp(d_perp - cfg.lane_tolerance, min=0.0)
    r_dest = cfg.w_destination * (d_goal < cfg.goal_tol).to(state.dtype)
    r_time = -cfg.w_time * torch.ones_like(d_goal)

    return r_wp + r_speed + r_oof + r_dest + r_time


def compute_reward_terms(
    state: torch.Tensor,
    next_state: torch.Tensor,
    goal: torch.Tensor,
    cfg: RewardConfig = RewardConfig(),
) -> dict[str, torch.Tensor]:
    """Same formula as ``compute_reward`` but returns every term for logging."""
    nx, ny, nyaw = next_state[..., 0], next_state[..., 1], next_state[..., 2]
    v_lin = next_state[..., 3]
    gx, gy, gyaw = goal[..., 0], goal[..., 1], goal[..., 2]
    tx, ty = torch.cos(gyaw), torch.sin(gyaw)
    ox, oy = gx - nx, gy - ny
    d_goal = torch.sqrt(ox * ox + oy * oy + 1e-12)
    o_par = ox * tx + oy * ty
    px = ox - o_par * tx
    py = oy - o_par * ty
    d_perp = torch.sqrt(px * px + py * py + 1e-12)
    vx = v_lin * torch.cos(nyaw)
    vy = v_lin * torch.sin(nyaw)
    v_par = vx * tx + vy * ty
    v_perp = torch.abs(vx * (-ty) + vy * tx)

    r_wp = cfg.w_waypoint * (d_goal < cfg.r_reach).to(state.dtype)
    r_speed = cfg.w_speed * (
        cfg.desired_speed
        - torch.abs(v_par - cfg.desired_speed)
        - 2.0 * torch.clamp(v_perp, max=cfg.perp_speed_cap)
    )
    r_oof = -cfg.w_out_of_lane * torch.clamp(d_perp - cfg.lane_tolerance, min=0.0)
    r_dest = cfg.w_destination * (d_goal < cfg.goal_tol).to(state.dtype)
    r_time = -cfg.w_time * torch.ones_like(d_goal)
    return {
        "r_waypoint": r_wp,
        "r_speed": r_speed,
        "r_out_of_lane": r_oof,
        "r_destination": r_dest,
        "r_time": r_time,
        "d_goal": d_goal,
        "d_perp": d_perp,
        "v_par": v_par,
        "v_perp": v_perp,
    }


def sample_synthetic_goal(
    states: torch.Tensor,
    *,
    radius_range: tuple[float, float] = (0.5, 3.0),
    forward_bias: float = 0.3,
    p_at_goal: float = 0.15,
    p_near_goal: float = 0.15,
    goal_tol: float = 0.25,
    r_reach: float = 0.35,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a random (gx, gy, gyaw) per state for reward-head training.

    Three buckets are drawn per sample:
      * with probability ``p_at_goal``  : radius in ``[0, goal_tol]``
        (the destination-bonus regime; the head must see it to learn the
        +w_destination spike),
      * with probability ``p_near_goal``: radius in ``[goal_tol, r_reach]``
        (the waypoint-reach regime, r_waypoint term),
      * otherwise                        : annulus ``radius_range``
        with a ``forward_bias`` chance of being forced ahead of the robot
        (the on-route regime the planner actually sees at runtime).
    """
    device = states.device
    dtype = states.dtype
    shape = states.shape[:-1]

    def _rand(*s: int) -> torch.Tensor:
        return torch.rand(*s, generator=generator, device=device, dtype=dtype)

    r_far = _rand(*shape) * (radius_range[1] - radius_range[0]) + radius_range[0]
    theta = _rand(*shape) * (2.0 * math.pi) - math.pi
    bias_mask = _rand(*shape) < forward_bias
    syaw = states[..., 2]
    forward_theta = syaw + (_rand(*shape) - 0.5) * math.pi
    theta = torch.where(bias_mask, forward_theta, theta)

    u = _rand(*shape)
    at_goal_mask = u < p_at_goal
    near_goal_mask = (u >= p_at_goal) & (u < p_at_goal + p_near_goal)

    r_at = _rand(*shape) * goal_tol
    r_near = _rand(*shape) * (r_reach - goal_tol) + goal_tol
    r = torch.where(at_goal_mask, r_at, torch.where(near_goal_mask, r_near, r_far))

    gx = states[..., 0] + r * torch.cos(theta)
    gy = states[..., 1] + r * torch.sin(theta)
    gyaw = _wrap_pi(_rand(*shape) * (2.0 * math.pi) - math.pi)
    return torch.stack([gx, gy, gyaw], dim=-1)


__all__ = [
    "RewardConfig",
    "relative_goal",
    "compute_reward",
    "compute_reward_terms",
    "sample_synthetic_goal",
]
