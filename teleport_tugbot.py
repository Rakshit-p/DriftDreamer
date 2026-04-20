"""Teleport the tugbot to a given (x, y, z, yaw) pose in the running Gazebo world.

Usage:
    python teleport_tugbot.py X Y [Z] [YAW_RAD]

Defaults:  Z=0.132, YAW=0.0

Requires the Gazebo server to be running with world name `world_demo` and a
model named `tugbot`. Uses `gz service` over the default partition so make
sure the server terminal exported:

    export GZ_IP=127.0.0.1
    export GZ_DISCOVERY_LOCALHOST_ENABLED=1
    unset GZ_PARTITION
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys

WORLD_NAME = "world_demo"
MODEL_NAME = "tugbot"


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def set_pose(x: float, y: float, z: float, yaw: float) -> None:
    if shutil.which("gz") is None:
        sys.exit("error: `gz` command not found in PATH")

    qx, qy, qz, qw = yaw_to_quat(yaw)
    req = (
        f'name: "{MODEL_NAME}", '
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}"
    )
    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD_NAME}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = (result.stdout or "").strip().lower()
    err = (result.stderr or "").strip()
    if result.returncode != 0 or "data: true" not in out:
        sys.exit(f"error: set_pose failed\n  stdout: {result.stdout}\n  stderr: {err}")
    print(f"set_pose ok  x={x} y={y} z={z} yaw={yaw}")


def main() -> None:
    p = argparse.ArgumentParser(description="Teleport the tugbot in Gazebo.")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("z", type=float, nargs="?", default=0.132)
    p.add_argument("yaw", type=float, nargs="?", default=0.0, help="yaw in radians")
    args = p.parse_args()
    set_pose(args.x, args.y, args.z, args.yaw)


if __name__ == "__main__":
    main()
