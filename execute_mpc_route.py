#!/usr/bin/env python3
"""Closed-loop MPC route runner for the tugbot in Gazebo.

At each tick this script reads the live pose (and lidar, if the checkpoint
was trained with it) from the running Gazebo simulator, runs CEM on the
learned RSSM to pick the best 12-step action sequence, sends the first
action via ``cmd_vel``, and re-plans from the new observation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_SD = _ROOT / "SimpleDreamer"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SD) not in sys.path:
    sys.path.insert(0, str(_SD))

import collections
import collections.abc

for _n in collections.abc.__all__:
    setattr(collections, _n, getattr(collections.abc, _n))

os.environ.setdefault("GZ_CMD_VEL_USE_SUBPROCESS", "1")

import torch  # noqa: E402

import train_tugbot_world_model as wm  # noqa: E402
from cardreamer_reward import RewardConfig  # noqa: E402
from collect_transitions import (  # noqa: E402
    CMD_VEL_TOPIC,
    LidarReader,
    NUM_RAY_BINS,
    RAY_MAX_RANGE,
    StateReader,
    send_cmd_vel,
    unpause_sim,
)
from plan_mpc import POSE_DIM, MPCState, plan_cem_mpc  # noqa: E402
from run_route import _parse_route, _wrap_pi  # noqa: E402


def _pick_device(name: str) -> torch.device:
    """Return a torch.device, auto-selecting CUDA/CPU when ``name='auto'``."""
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _advance_goal(
    route_xy: np.ndarray,
    idx: int,
    x: float,
    y: float,
    advance_radius: float,
) -> int:
    """Skip the goal index past any waypoint already within ``advance_radius``."""
    last = len(route_xy) - 1
    while idx < last:
        gx, gy = float(route_xy[idx, 0]), float(route_xy[idx, 1])
        if math.hypot(gx - x, gy - y) <= advance_radius:
            idx += 1
        else:
            break
    return idx


def main() -> None:
    p = argparse.ArgumentParser(description="Receding-horizon MPC route runner")
    p.add_argument("--checkpoint", type=Path,
                   default=_SD / "checkpoints" / "tugbot_world_model_lidar_v4.best.pt")
    p.add_argument("--route-file", type=Path, default=_ROOT / "route_clean.txt")

    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--discount", type=float, default=0.97)
    p.add_argument("--lin-range", type=float, nargs=2, default=[-0.25, 0.30])
    p.add_argument("--ang-range", type=float, nargs=2, default=[-1.0, 1.0])
    p.add_argument("--action-duration", type=float, default=0.3)

    p.add_argument("--advance-radius", type=float, default=0.6)
    p.add_argument("--goal-tol", type=float, default=0.35)
    p.add_argument("--goal-yaw-tol", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=240.0)

    p.add_argument("--dynamics", choices=("analytical", "rssm"), default="analytical")
    p.add_argument("--cost", choices=("analytical", "learned", "blend"), default="analytical")

    p.add_argument("--obstacle-weight", type=float, default=0.0,
                   help="Weight of the obstacle penalty (try 1.0–10.0). Needs a lidar checkpoint.")
    p.add_argument("--safety-radius", type=float, default=0.6,
                   help="Desired clearance to obstacles in metres.")
    p.add_argument("--no-lidar", action="store_true",
                   help="Force-disable live lidar reads even for a lidar checkpoint.")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics-csv", type=Path, default=_ROOT / "mpc_route_metrics.csv")
    p.add_argument("--log-imag", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--greedy", action="store_true",
                   help="Execute one full CEM plan open-loop (baseline / A-B only).")
    args = p.parse_args()

    if args.cost in ("learned", "blend") and args.dynamics != "rssm":
        p.error("--cost learned/blend requires --dynamics rssm")

    route = _parse_route(args.route_file)
    route_xy = route[:, :2]
    route_yaw = route[:, 2]
    goal_xy_last = route_xy[-1]
    goal_yaw_last = float(route[-1, 2])
    print(f"route: {len(route)} points from {args.route_file}", flush=True)

    device = _pick_device(args.device)
    print(f"device={device}  dynamics={args.dynamics}  cost={args.cost}", flush=True)

    if args.cost in ("learned", "blend") or args.dynamics == "rssm":
        rssm, enc, dec, head, mean, std, _cfg = wm.load_world_model_with_reward(
            args.checkpoint, device)
    else:
        try:
            rssm, enc, dec, head, mean, std, _cfg = wm.load_world_model_with_reward(
                args.checkpoint, device)
        except (KeyError, FileNotFoundError):
            rssm = enc = dec = head = None
            mean = torch.zeros(POSE_DIM, device=device)
            std = torch.ones(POSE_DIM, device=device)
            print("no checkpoint loaded — running model-free analytical MPC.", flush=True)

    expected_obs_dim = int(mean.shape[-1])
    needs_lidar = expected_obs_dim > POSE_DIM and not args.no_lidar
    if expected_obs_dim > POSE_DIM and args.no_lidar:
        print(f"[WARN] checkpoint expects obs_dim={expected_obs_dim} but --no-lidar was set; "
              f"rays will be filled with 1.0 (clear).", flush=True)
    if args.obstacle_weight > 0.0 and expected_obs_dim == POSE_DIM:
        print("[WARN] --obstacle-weight > 0 but checkpoint has no ray columns; penalty is a no-op.",
              flush=True)

    reader = StateReader()
    lidar = LidarReader() if needs_lidar else None
    print("starting pose reader…", flush=True)
    unpause_sim()
    reader.start()
    if not reader.wait_ready(timeout=60.0):
        reader.stop()
        raise SystemExit("No pose data — check gz sim is running with GZ_IP=127.0.0.1 "
                         "GZ_DISCOVERY_LOCALHOST_ENABLED=1.")
    if lidar is not None:
        lidar.start()
        print(f"starting lidar reader (obs_dim={expected_obs_dim}, "
              f"{expected_obs_dim - POSE_DIM} ray bins)…", flush=True)
        if not lidar.wait_ready(timeout=10.0):
            reader.stop()
            lidar.stop()
            raise SystemExit("No scan data — enable the Sensors plugin in the SDF or pass --no-lidar.")
    print(f"connected.  cmd_vel topic: {CMD_VEL_TOPIC}", flush=True)

    reward_cfg = RewardConfig()
    mpc = MPCState()

    dt = float(args.action_duration)
    idx = 0
    rows: list[dict] = []
    reached = False
    t0 = time.monotonic()
    last_sent_pose: np.ndarray | None = None

    try:
        while True:
            s = reader.get_state()
            if s is None:
                time.sleep(0.05)
                continue
            x, y, yaw = float(s[0]), float(s[1]), float(s[2])

            d_last = math.hypot(goal_xy_last[0] - x, goal_xy_last[1] - y)
            yaw_err_last = abs(_wrap_pi(goal_yaw_last - yaw))
            if d_last <= args.goal_tol and yaw_err_last <= args.goal_yaw_tol:
                reached = True
                break
            elapsed = time.monotonic() - t0
            if elapsed > args.max_seconds:
                break

            idx = _advance_goal(route_xy, idx, x, y, args.advance_radius)
            goal = np.array(
                [float(route_xy[idx, 0]), float(route_xy[idx, 1]), float(route_yaw[idx])],
                dtype=np.float32,
            )

            pose_vec = np.array([x, y, yaw, float(s[3]), float(s[4])], dtype=np.float32)
            if expected_obs_dim > POSE_DIM:
                if lidar is not None:
                    rays_vec = lidar.get_rays().astype(np.float32)
                else:
                    rays_vec = np.ones(NUM_RAY_BINS, dtype=np.float32)
                s0 = np.concatenate([pose_vec, rays_vec], axis=0)
            else:
                s0 = pose_vec

            a0, info = plan_cem_mpc(
                rssm, enc, dec, head, mean, std,
                s0_world=s0, goal_xyyaw=goal,
                state=mpc,
                horizon=args.horizon, samples=args.samples, iters=args.iters,
                elite_frac=args.elite_frac,
                lin_range=tuple(args.lin_range), ang_range=tuple(args.ang_range),
                discount=args.discount,
                device=device, seed=args.seed,
                dynamics=args.dynamics, cost=args.cost,
                action_dt=dt, reward_cfg=reward_cfg,
                obstacle_weight=float(args.obstacle_weight),
                obstacle_safety=float(args.safety_radius),
                obstacle_max_range=float(RAY_MAX_RANGE),
            )

            v, w = float(a0[0]), float(a0[1])

            imag_err = float("nan")
            if args.log_imag and last_sent_pose is not None and info.get("preds_world") is not None:
                imag_err = float(math.hypot(last_sent_pose[0] - x, last_sent_pose[1] - y))

            send_cmd_vel(v, w)
            if info.get("preds_world") is not None:
                last_sent_pose = np.asarray(info["preds_world"][0], dtype=np.float32)

            if expected_obs_dim > POSE_DIM and len(s0) > POSE_DIM:
                ray_min_m = float(np.min(s0[POSE_DIM:])) * RAY_MAX_RANGE
            else:
                ray_min_m = float("nan")

            rows.append({
                "t_sec": round(elapsed, 3),
                "x": x, "y": y, "yaw": yaw,
                "v_lin_meas": float(s[3]), "v_ang_meas": float(s[4]),
                "goal_idx": idx,
                "goal_x": float(goal[0]), "goal_y": float(goal[1]), "goal_yaw": float(goal[2]),
                "d_goal": math.hypot(goal[0] - x, goal[1] - y),
                "d_final": d_last,
                "v_cmd": v, "w_cmd": w,
                "best_reward": float(info["best_reward"]),
                "imag_err_m": imag_err,
                "ray_min_m": ray_min_m,
                "obstacle_weight": float(args.obstacle_weight),
                "dynamics": args.dynamics, "cost": args.cost,
            })

            if args.verbose and len(rows) % 5 == 0:
                row = rows[-1]
                print(f"t={elapsed:6.1f}s  idx={idx:3d}/{len(route)-1}  "
                      f"pos=({x:+6.2f},{y:+6.2f},{yaw:+.2f})  "
                      f"goal=({goal[0]:+6.2f},{goal[1]:+6.2f})  "
                      f"d_goal={row['d_goal']:.2f}  cmd=({v:+.2f},{w:+.2f})  "
                      f"R={row['best_reward']:+6.2f}", flush=True)

            if args.greedy:
                plan = info["plan"]
                print(f"greedy: executing {len(plan)} planned actions open-loop.")
                for a in plan[1:]:
                    time.sleep(dt)
                    send_cmd_vel(float(a[0]), float(a[1]))
                break

            time.sleep(dt)
    finally:
        send_cmd_vel(0.0, 0.0)
        reader.stop()
        if lidar is not None:
            lidar.stop()

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_sec", "x", "y", "yaw", "v_lin_meas", "v_ang_meas",
        "goal_idx", "goal_x", "goal_y", "goal_yaw", "d_goal", "d_final",
        "v_cmd", "w_cmd", "best_reward", "imag_err_m", "ray_min_m",
        "obstacle_weight", "dynamics", "cost",
    ]
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.monotonic() - t0
    print(f"done.  reached={reached}  elapsed={elapsed:.1f}s  steps={len(rows)}  "
          f"metrics={args.metrics_csv}", flush=True)


if __name__ == "__main__":
    os.chdir(_ROOT)
    main()
