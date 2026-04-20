#!/usr/bin/env python3
"""Offline smoke-test for the receding-horizon MPC controller.

Runs the planner against a kinematic simulator (no Gazebo required), so we
can verify that the analytical CEM-MPC can follow the waypoints in
``route_clean.txt`` before trying it on the real simulator.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_SD = Path(__file__).resolve().parent
_ROOT = _SD.parent
for p in (str(_SD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import collections
import collections.abc
for _n in collections.abc.__all__:
    setattr(collections, _n, getattr(collections.abc, _n))

from plan_mpc import MPCState, plan_cem_mpc, diff_drive_rollout  # noqa: E402
from run_route import _parse_route  # noqa: E402


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def advance_goal(route_xy: np.ndarray, idx: int, x: float, y: float, advance_radius: float) -> int:
    """Skip past any waypoints already within ``advance_radius`` of the robot."""
    last = len(route_xy) - 1
    while idx < last:
        gx, gy = float(route_xy[idx, 0]), float(route_xy[idx, 1])
        if math.hypot(gx - x, gy - y) <= advance_radius:
            idx += 1
        else:
            break
    return idx


def main() -> None:
    route = _parse_route(_ROOT / "route_clean.txt")
    route_xy = route[:, :2]
    route_yaw = route[:, 2]
    print(f"route: {len(route)} waypoints, start=({route_xy[0, 0]:.2f},{route_xy[0, 1]:.2f}), "
          f"end=({route_xy[-1, 0]:.2f},{route_xy[-1, 1]:.2f})")

    device = torch.device("cpu")
    mean = torch.zeros(5, device=device)
    std = torch.ones(5, device=device)

    mpc = MPCState()
    state = np.array([route_xy[0, 0], route_xy[0, 1], route_yaw[0], 0.0, 0.0], dtype=np.float32)

    dt = 0.3
    advance_radius = 0.4
    goal_tol = 0.25
    goal_yaw_tol = 0.7
    max_steps = 800

    idx = 0
    for step in range(max_steps):
        d_last = math.hypot(route_xy[-1, 0] - state[0], route_xy[-1, 1] - state[1])
        yaw_err_last = abs(_wrap_pi(route_yaw[-1] - state[2]))
        if d_last <= goal_tol and yaw_err_last <= goal_yaw_tol:
            print(f"[step {step:4d}] REACHED destination.  "
                  f"pos=({state[0]:.2f},{state[1]:.2f},{state[2]:.2f})  d_last={d_last:.3f}")
            return

        idx = advance_goal(route_xy, idx, float(state[0]), float(state[1]), advance_radius)
        goal = np.array([route_xy[idx, 0], route_xy[idx, 1], route_yaw[idx]], dtype=np.float32)

        a0, info = plan_cem_mpc(
            None, None, None, None, mean, std,
            s0_world=state, goal_xyyaw=goal,
            state=mpc,
            horizon=12, samples=256, iters=4, elite_frac=0.15,
            lin_range=(-0.25, 0.30), ang_range=(-1.0, 1.0),
            discount=0.97,
            device=device, seed=0,
            dynamics="analytical", cost="analytical",
            action_dt=dt,
        )
        v, w = float(a0[0]), float(a0[1])

        s_t = torch.tensor(state, dtype=torch.float32).view(1, 5)
        a_t = torch.tensor([[v, w]], dtype=torch.float32).view(1, 1, 2)
        nxt = diff_drive_rollout(s_t, a_t, dt).view(-1).cpu().numpy()
        state = nxt.astype(np.float32)

        if step % 10 == 0 or step < 5:
            print(
                f"[step {step:4d}] pos=({state[0]:+6.2f},{state[1]:+6.2f},"
                f"{state[2]:+5.2f})  idx={idx:2d}/{len(route)-1}  "
                f"goal=({goal[0]:+6.2f},{goal[1]:+6.2f})  "
                f"d_goal={math.hypot(goal[0]-state[0], goal[1]-state[1]):.2f}  "
                f"d_last={d_last:.2f}  cmd=({v:+.2f},{w:+.2f})  R={info['best_reward']:+6.2f}"
            )

    print("TIMED OUT before reaching final waypoint.")


if __name__ == "__main__":
    main()
