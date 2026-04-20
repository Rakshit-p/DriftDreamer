#!/usr/bin/env python3
"""
Deterministic route runner for tugbot in Gazebo.

Input route formats:
1) waypoints txt lines:
     --waypoint X Y YAW
2) JSON array:
     [[x, y, yaw], [x, y, yaw], ...]
     [[x, y], [x, y], ...]   # yaw optional

Outputs:
- live console progress
- CSV metrics (default: route_metrics.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collect_transitions import StateReader, send_cmd_vel, unpause_sim


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _parse_route(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON route must be a list")
        out = []
        for p in data:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            yaw = float(p[2]) if len(p) >= 3 else 0.0
            out.append([float(p[0]), float(p[1]), yaw])
        if len(out) < 2:
            raise ValueError("Need at least 2 route points")
        return np.asarray(out, dtype=np.float32)

    pat = re.compile(r"--waypoint\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if not m:
            continue
        out.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    if len(out) < 2:
        raise ValueError("Need at least 2 '--waypoint x y yaw' lines")
    return np.asarray(out, dtype=np.float32)


def _nearest_idx(route_xy: np.ndarray, x: float, y: float, start_idx: int) -> int:
    pts = route_xy[start_idx:]
    if len(pts) == 0:
        return len(route_xy) - 1
    d2 = np.sum((pts - np.asarray([x, y], dtype=np.float32)) ** 2, axis=1)
    return start_idx + int(np.argmin(d2))


def _lookahead_idx(route_xy: np.ndarray, from_idx: int, lookahead: float) -> int:
    if from_idx >= len(route_xy) - 1:
        return len(route_xy) - 1
    acc = 0.0
    i = from_idx
    prev = route_xy[from_idx]
    while i < len(route_xy):
        cur = route_xy[i]
        acc += float(np.linalg.norm(cur - prev))
        if acc >= lookahead:
            return i
        prev = cur
        i += 1
    return len(route_xy) - 1


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic route tracker with metrics")
    p.add_argument("--route-file", type=Path, default=Path("route_clean.txt"))
    p.add_argument("--lookahead", type=float, default=0.45)
    p.add_argument("--goal-tol", type=float, default=0.25)
    p.add_argument("--rate-hz", type=float, default=15.0)
    p.add_argument("--v-max", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.1)
    p.add_argument("--k-turn", type=float, default=1.9)
    p.add_argument("--k-slow", type=float, default=1.4)
    p.add_argument("--max-seconds", type=float, default=240.0)
    p.add_argument("--metrics-csv", type=Path, default=Path("route_metrics.csv"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    route = _parse_route(args.route_file)
    route_xy = route[:, :2]
    goal_xy = route_xy[-1]
    goal_yaw = float(route[-1, 2])
    print(f"Loaded route: {len(route)} points from {args.route_file}", flush=True)

    reader = StateReader()
    print("Starting pose reader …", flush=True)
    unpause_sim()
    reader.start()
    if not reader.wait_ready(timeout=60.0):
        reader.stop()
        raise SystemExit("No pose data. Check gz sim and GZ_IP/GZ_DISCOVERY_LOCALHOST_ENABLED.")

    dt = 1.0 / max(1.0, args.rate_hz)
    t0 = time.monotonic()
    idx_progress = 0
    rows = []
    reached = False
    try:
        while True:
            s = reader.get_state()
            if s is None:
                time.sleep(dt)
                continue
            x, y, yaw = float(s[0]), float(s[1]), float(s[2])
            d_goal = math.hypot(float(goal_xy[0] - x), float(goal_xy[1] - y))
            yaw_err_goal = abs(_wrap_pi(goal_yaw - yaw))
            if d_goal <= args.goal_tol and yaw_err_goal <= 0.6:
                reached = True
                break
            elapsed = time.monotonic() - t0
            if elapsed > args.max_seconds:
                break

            idx_near = _nearest_idx(route_xy, x, y, idx_progress)
            idx_progress = max(idx_progress, idx_near)
            idx_tgt = _lookahead_idx(route_xy, idx_near, args.lookahead)
            tx, ty = float(route_xy[idx_tgt, 0]), float(route_xy[idx_tgt, 1])

            heading_tgt = math.atan2(ty - y, tx - x)
            err = _wrap_pi(heading_tgt - yaw)

            slow = max(0.0, 1.0 - args.k_slow * min(1.0, abs(err) / math.pi))
            v = args.v_max * slow
            w = max(-args.w_max, min(args.w_max, args.k_turn * err))
            if abs(err) > 1.35:
                v = 0.0

            send_cmd_vel(v, w)
            rows.append(
                {
                    "t_sec": round(elapsed, 3),
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "target_idx": idx_tgt,
                    "target_x": tx,
                    "target_y": ty,
                    "heading_err": err,
                    "v_cmd": v,
                    "w_cmd": w,
                    "d_goal": d_goal,
                }
            )
            if args.verbose and len(rows) % 10 == 0:
                print(
                    f"t={elapsed:6.1f}s i={idx_tgt:3d} pos=({x:5.2f},{y:5.2f}) "
                    f"err={err:+.3f} v={v:+.3f} w={w:+.3f} d_goal={d_goal:.2f}",
                    flush=True,
                )
            time.sleep(dt)
    finally:
        send_cmd_vel(0.0, 0.0)
        reader.stop()

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_sec",
                "x",
                "y",
                "yaw",
                "target_idx",
                "target_x",
                "target_y",
                "heading_err",
                "v_cmd",
                "w_cmd",
                "d_goal",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.monotonic() - t0
    print(
        f"Done. reached_goal={reached}  elapsed={elapsed:.1f}s  "
        f"steps={len(rows)}  metrics={args.metrics_csv}",
        flush=True,
    )


if __name__ == "__main__":
    os.chdir(_ROOT)
    main()
