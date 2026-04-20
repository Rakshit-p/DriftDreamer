#!/usr/bin/env python3
"""Collect episodic tugbot trajectories from Gazebo.

Unlike a simple i.i.d. transition collector, this script drives the robot
in continuous episodes, mixing four motion modes (straight / spot-turn /
arc / random) so the RSSM can learn multi-step dynamics. When lidar is
enabled, each observation becomes ``[pose(5) | rays(16)]``. The output
.npz includes an ``episode_starts`` array so the trainer can build
sequence windows that never cross episode boundaries.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from collect_transitions import (  # noqa: E402
    ACTION_DURATION,
    ANGULAR_VEL_RANGE,
    LINEAR_VEL_RANGE,
    LidarReader,
    NUM_RAY_BINS,
    RESET_POSE,
    StateReader,
    _try_init_cmd_vel_transport,
    in_bounds,
    in_focus_bounds,
    reset_robot,
    send_cmd_vel,
    unpause_sim,
)

LIDAR_WAIT_TIMEOUT = 10.0

MODES = ("straight", "spot_turn", "arc", "random")


def _sample_episode_targets(
    mode: str,
    rng: np.random.Generator,
    forward_bias: float,
) -> tuple[float, float, float, float]:
    """Pick per-episode (v_mean, v_jitter, w_mean, w_jitter) for the given mode."""
    v_lo, v_hi = LINEAR_VEL_RANGE
    w_lo, w_hi = ANGULAR_VEL_RANGE

    if mode == "straight":
        v_mean = float(rng.uniform(0.10, v_hi))
        return v_mean, 0.015, 0.0, 0.05

    if mode == "spot_turn":
        w_mag = float(rng.uniform(0.5, abs(w_hi)))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return 0.0, 0.015, sign * w_mag, 0.05

    if mode == "arc":
        v_mean = float(rng.uniform(0.10, v_hi))
        w_mag = float(rng.uniform(0.10, 0.5))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return v_mean, 0.02, sign * w_mag, 0.08

    v_mean = float(rng.uniform(v_lo, v_hi))
    w_mean = float(rng.uniform(w_lo, w_hi))
    if forward_bias > 0.0:
        v_mean = max(v_mean, v_lo + (1.0 - forward_bias) * (v_hi - v_lo) * 0.5)
    return v_mean, (v_hi - v_lo) * 0.25, w_mean, (w_hi - w_lo) * 0.25


def _sample_action(
    rng: np.random.Generator,
    v_mean: float,
    v_jitter: float,
    w_mean: float,
    w_jitter: float,
) -> tuple[float, float]:
    """Sample one (v, w) action from the episode's current target distribution."""
    v_lo, v_hi = LINEAR_VEL_RANGE
    w_lo, w_hi = ANGULAR_VEL_RANGE
    v = float(np.clip(rng.normal(v_mean, v_jitter), v_lo, v_hi))
    omega = float(np.clip(rng.normal(w_mean, w_jitter), w_lo, w_hi))
    return v, omega


def _pick_mode(rng: np.random.Generator, weights: tuple[float, ...]) -> str:
    probs = np.array(weights, dtype=np.float64)
    probs = probs / probs.sum()
    return MODES[int(rng.choice(len(MODES), p=probs))]


def collect_trajectories(
    reader: StateReader,
    num_episodes: int,
    episode_length: int,
    persist_prob: float,
    persist_max: int,
    forward_bias: float,
    min_episode_length: int,
    mode_weights: tuple[float, float, float, float],
    seed: int = 42,
    progress: dict | None = None,
    lidar: LidarReader | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect ``num_episodes`` episodes and return stacked arrays.

    If ``progress`` is given, it is updated after every completed episode so
    a Ctrl-C signal handler can still save completed work.
    """
    rng = np.random.default_rng(seed=seed)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_starts: list[int] = []
    if progress is not None:
        progress["states"] = states
        progress["actions"] = actions
        progress["next_states"] = next_states
        progress["episode_starts"] = episode_starts

    _try_init_cmd_vel_transport()

    t_start = time.time()
    per_step = ACTION_DURATION + 0.02
    eta_min = num_episodes * episode_length * per_step / 60.0
    print(f"Target  : {num_episodes} episodes × {episode_length} steps "
          f"= {num_episodes * episode_length} total steps")
    print(f"Action  : {ACTION_DURATION}s hold, linear ∈ "
          f"[{LINEAR_VEL_RANGE[0]},{LINEAR_VEL_RANGE[1]}] (forward_bias={forward_bias}),  "
          f"angular ∈ [{ANGULAR_VEL_RANGE[0]},{ANGULAR_VEL_RANGE[1]}]")
    print(f"Persist : p={persist_prob}  max_repeats={persist_max}")
    w_norm = np.array(mode_weights, dtype=np.float64)
    w_norm = w_norm / w_norm.sum()
    print("Modes   : " + "  ".join(f"{m}={w_norm[i]:.2f}" for i, m in enumerate(MODES)))
    print(f"Estimate: ~{eta_min:.1f} min at ~{per_step:.2f}s / step\n")

    mode_counts = {m: 0 for m in MODES}
    for ep in range(num_episodes):
        send_cmd_vel(0.0, 0.0)
        reset_robot()
        time.sleep(0.2)

        s0 = reader.get_state()
        for _ in range(20):
            if s0 is not None and in_bounds(s0):
                break
            time.sleep(0.1)
            s0 = reader.get_state()
        if s0 is None or not in_bounds(s0):
            print(f"  [ep {ep:3d}] reader not ready, skipping")
            continue

        ep_start_row = len(states)
        prev_action: tuple[float, float] | None = None
        persist_count = 0
        steps_this_ep = 0
        terminated_early = False

        mode = _pick_mode(rng, mode_weights)
        v_mean, v_jit, w_mean, w_jit = _sample_episode_targets(mode, rng, forward_bias=forward_bias)
        mode_counts[mode] += 1

        for step in range(episode_length):
            state_before = reader.get_state()
            if state_before is None:
                time.sleep(0.05)
                continue
            if not in_bounds(state_before) or not in_focus_bounds(state_before):
                terminated_early = True
                break
            rays_before = (
                lidar.get_rays() if lidar is not None
                else np.ones(NUM_RAY_BINS, dtype=np.float32)
            )

            if (prev_action is not None
                    and persist_count < persist_max
                    and rng.random() < persist_prob):
                v, omega = prev_action
                persist_count += 1
            else:
                v, omega = _sample_action(rng, v_mean, v_jit, w_mean, w_jit)
                persist_count = 0
            prev_action = (v, omega)

            send_cmd_vel(v, omega)
            time.sleep(ACTION_DURATION)
            state_after = reader.get_state()
            if state_after is None:
                continue
            rays_after = (
                lidar.get_rays() if lidar is not None
                else np.ones(NUM_RAY_BINS, dtype=np.float32)
            )

            obs_before = np.concatenate(
                [state_before.astype(np.float32), rays_before.astype(np.float32)], axis=0)
            obs_after = np.concatenate(
                [state_after.astype(np.float32), rays_after.astype(np.float32)], axis=0)
            states.append(obs_before)
            actions.append(np.array([v, omega], dtype=np.float32))
            next_states.append(obs_after)
            steps_this_ep += 1

        send_cmd_vel(0.0, 0.0)

        if steps_this_ep < min_episode_length:
            del states[ep_start_row:]
            del actions[ep_start_row:]
            del next_states[ep_start_row:]
            print(f"  [ep {ep:3d}] discarded — only {steps_this_ep} valid steps "
                  f"(< min {min_episode_length})")
            continue

        episode_starts.append(ep_start_row)
        elapsed = time.time() - t_start
        total_rows = len(states)
        print(f"  [ep {ep:3d}] mode={mode:<9} {steps_this_ep:>3} steps  "
              f"{'(trunc)' if terminated_early else '       '}  "
              f"rows={total_rows:>6}  elapsed={elapsed/60:5.1f} min")

    send_cmd_vel(0.0, 0.0)
    print("\nMode tally: " + "  ".join(f"{m}={mode_counts[m]}" for m in MODES))

    if not states:
        raise RuntimeError("No usable episodes were collected.")

    return (
        np.stack(states, dtype=np.float32),
        np.stack(actions, dtype=np.float32),
        np.stack(next_states, dtype=np.float32),
        np.asarray(episode_starts, dtype=np.int64),
    )


def save_trajectories(
    save_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    episode_starts: np.ndarray,
) -> None:
    """Write the dataset to ``save_path`` (compressed .npz) and print a summary."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        states=states,
        actions=actions,
        next_states=next_states,
        episode_starts=episode_starts,
    )

    ep_lens = np.diff(np.append(episode_starts, len(states)))
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Saved {len(states):,} steps across {len(episode_starts)} episodes")
    print(f"  → {save_path}")
    print(sep)
    print(f"  Episode length  min / mean / max = "
          f"{ep_lens.min()} / {ep_lens.mean():.1f} / {ep_lens.max()}")
    obs_dim = int(states.shape[-1])
    if obs_dim == 5:
        print("  State columns  : x, y, yaw, v_linear, v_angular")
    else:
        print(f"  State columns  : x, y, yaw, v_linear, v_angular, + {obs_dim - 5} ray bins in [0, 1]")
    print(f"  Action columns : cmd_linear, cmd_angular")
    print(f"  Extra array    : episode_starts (shape {episode_starts.shape})")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect episodic tugbot trajectories.")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=48)
    parser.add_argument("--min-episode-length", type=int, default=8)
    parser.add_argument("--persist-prob", type=float, default=0.7)
    parser.add_argument("--persist-max", type=int, default=5)
    parser.add_argument("--forward-bias", type=float, default=0.3)
    parser.add_argument(
        "--mode-weights", type=float, nargs=4,
        metavar=("STRAIGHT", "SPOT_TURN", "ARC", "RANDOM"),
        default=[0.25, 0.25, 0.25, 0.25],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path,
                        default=Path("/Users/rakshitpradhan/Desktop/factory/trajectories.npz"))
    parser.add_argument("--append", action="store_true",
                        help="Merge new episodes into an existing .npz instead of overwriting.")
    parser.add_argument("--with-lidar", action="store_true",
                        help=f"Append {NUM_RAY_BINS} normalised ray bins to each observation.")
    return parser.parse_args(argv)


def _merge_with_existing(
    save_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    episode_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Append new arrays onto an existing .npz produced by this script."""
    if not save_path.is_file():
        return states, actions, next_states, episode_starts
    existing = np.load(save_path)
    if "episode_starts" not in existing.files:
        raise RuntimeError(
            f"--append target {save_path} is missing 'episode_starts' "
            f"(legacy iid dataset). Write to a new path instead.")
    old_dim = int(existing["states"].shape[-1])
    new_dim = int(states.shape[-1])
    if old_dim != new_dim:
        raise RuntimeError(
            f"--append target has obs_dim={old_dim} but new episodes have obs_dim={new_dim}. "
            "Use a fresh --save-path or delete the old file.")
    old_rows = existing["states"].shape[0]
    return (
        np.concatenate([existing["states"], states], axis=0),
        np.concatenate([existing["actions"], actions], axis=0),
        np.concatenate([existing["next_states"], next_states], axis=0),
        np.concatenate([existing["episode_starts"], episode_starts + old_rows], axis=0).astype(np.int64),
    )


def main() -> None:
    args = _parse_args()

    reader = StateReader()
    lidar: LidarReader | None = LidarReader() if args.with_lidar else None
    progress: dict = {}

    def _flush_and_exit(sig, frame):  # type: ignore[no-untyped-def]
        print("\n[interrupt] saving completed episodes…")
        try:
            s_list = progress.get("states") or []
            if not s_list:
                print("[interrupt] no completed episodes yet — nothing to save.")
            else:
                s = np.stack(s_list, dtype=np.float32)
                a = np.stack(progress["actions"], dtype=np.float32)
                ns = np.stack(progress["next_states"], dtype=np.float32)
                starts = np.asarray(progress["episode_starts"], dtype=np.int64)
                if args.append and args.save_path.is_file():
                    s, a, ns, starts = _merge_with_existing(args.save_path, s, a, ns, starts)
                    print(f"[interrupt] merged with existing {args.save_path}")
                save_trajectories(args.save_path, s, a, ns, starts)
        finally:
            reader.stop()
            if lidar is not None:
                lidar.stop()
            sys.exit(0)

    signal.signal(signal.SIGINT, _flush_and_exit)

    reader.start()
    if not reader.wait_ready(timeout=60.0):
        print("ERROR: reader never received a pose message.")
        reader.stop()
        sys.exit(1)

    if lidar is not None:
        lidar.start()
        print(f"lidar    : waiting up to {LIDAR_WAIT_TIMEOUT:.0f}s for first scan…")
        if not lidar.wait_ready(timeout=LIDAR_WAIT_TIMEOUT):
            print("ERROR: no scan messages received.  Is the Sensors plugin loaded in the world SDF?")
            reader.stop()
            lidar.stop()
            sys.exit(1)
        stats = lidar.sensor_status()
        print("lidar    : " + "  ".join(
            f"{k}={'ok' if v < 2.0 else f'stale ({v:.1f}s)'}" for k, v in stats.items()))

    unpause_sim()

    s, a, ns, starts = collect_trajectories(
        reader,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        persist_prob=args.persist_prob,
        persist_max=args.persist_max,
        forward_bias=args.forward_bias,
        min_episode_length=args.min_episode_length,
        mode_weights=tuple(args.mode_weights),  # type: ignore[arg-type]
        seed=args.seed,
        progress=progress,
        lidar=lidar,
    )

    if args.append and args.save_path.is_file():
        s, a, ns, starts = _merge_with_existing(args.save_path, s, a, ns, starts)
        print(f"[append] merged with existing {args.save_path}")

    save_trajectories(args.save_path, s, a, ns, starts)
    reader.stop()
    if lidar is not None:
        lidar.stop()


if __name__ == "__main__":
    main()
