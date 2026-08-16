#!/usr/bin/env python3
"""Compare open-loop imagination accuracy across rollout modes.

Rolls one shared batch of trajectories through three rollout variants and
reports MSE at several horizons, so the effect of grounding part of the
dynamics in exact physics can be measured directly:

    pure        fully learned RSSM prior rollout
    hybrid-full overwrite all 5 pose dims from the *commanded* action and an
                analytically integrated yaw  (assumes instant actuator
                response -- kept as the A/B baseline)
    hybrid-pos  integrate only x, y exactly, driven by the RSSM's own
                *decoded* yaw and v; yaw/v/w are left untouched so the
                learned command-to-realised mapping is preserved

Every variant starts from the same teacher-forced belief, so differences
come only from the splice. Metrics are reported per slice of the
observation vector:

    total  all dims          pos   dims 0..1 (x, y)
    pose   dims 0..4         yaw   dim 2
    lidar  dims 5..          vel   dims 3..4 (v, w)

The question this answers: does exact position integration beat the
learned pose estimate at long horizons, once it is fed the model's own
realised velocity instead of the raw command?
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_SD_DIR = Path(__file__).resolve().parent
if str(_SD_DIR) not in sys.path:
    sys.path.insert(0, str(_SD_DIR))

import train_tugbot_world_model as wm


POSE_DIM = 5
SLICES = ("total", "pose", "pos", "yaw", "vel", "lidar")


@torch.no_grad()
def _teacher_force(rssm, encoder, obs, act, teacher_steps, device):
    """Teacher-force ``teacher_steps`` steps; return the belief ``(z, d)``."""
    bsz = obs.shape[0]
    prior, deterministic = rssm.recurrent_model_input_init(bsz)
    latent = prior.to(device)
    deterministic = deterministic.to(device)
    embedded = encoder(obs)
    for t in range(1, teacher_steps + 1):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        _, _prior = rssm.transition_model(deterministic)
        _, posterior = rssm.representation_model(embedded[:, t], deterministic)
        latent = posterior
    return latent, deterministic


@torch.no_grad()
def rollout(
    rssm,
    encoder,
    decoder,
    obs: torch.Tensor,
    act: torch.Tensor,
    teacher_steps: int,
    open_steps: int,
    device: torch.device,
    mode: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    action_dt: float,
    yaw_mode: str = "mid",
    v_mode: str = "mid",
) -> torch.Tensor:
    """Open-loop rollout in ``mode``; returns normalised ``(B, open_steps, D)``."""
    latent, deterministic = _teacher_force(rssm, encoder, obs, act, teacher_steps, device)

    mean_v = mean.view(1, -1)
    std_v = std.view(1, -1)
    mean_pose = mean_v[:, :POSE_DIM]
    std_pose = std_v[:, :POSE_DIM]

    start_world = obs[:, teacher_steps] * std_v + mean_v
    x_prev = start_world[:, 0].clone()
    y_prev = start_world[:, 1].clone()
    yaw_prev = start_world[:, 2].clone()
    v_prev = start_world[:, 3].clone()

    preds: list[torch.Tensor] = []
    for t in range(teacher_steps + 1, teacher_steps + open_steps + 1):
        a = act[:, t - 1]
        deterministic = rssm.recurrent_model(latent, a, deterministic)
        prior_dist, _ = rssm.transition_model(deterministic)
        z_prior = prior_dist.mean
        dec = decoder(z_prior.unsqueeze(1), deterministic.unsqueeze(1))
        pred_norm = dec.mean.squeeze(1)

        if mode == "pure":
            latent = z_prior
            preds.append(pred_norm)
            continue

        corrected = pred_norm.clone()
        if mode == "hybrid-full":
            v_cmd = a[:, 0]
            w_cmd = a[:, 1]
            x_new = x_prev + v_cmd * action_dt * torch.cos(yaw_prev)
            y_new = y_prev + v_cmd * action_dt * torch.sin(yaw_prev)
            yaw_new = wm.wrap_pi(yaw_prev + w_cmd * action_dt)
            pose_new = torch.stack([x_new, y_new, yaw_new, v_cmd, w_cmd], dim=-1)
            corrected[:, :POSE_DIM] = (pose_new - mean_pose) / std_pose
            yaw_prev = yaw_new
            v_prev = v_cmd
        elif mode == "hybrid-pos":
            pred_world = pred_norm * std_v + mean_v
            yaw_next = pred_world[:, 2]
            v_next = pred_world[:, 3]
            if yaw_mode == "mid":
                yaw_int = yaw_prev + 0.5 * wm.wrap_pi(yaw_next - yaw_prev)
            elif yaw_mode == "prev":
                yaw_int = yaw_prev
            else:
                yaw_int = yaw_next
            v_int = 0.5 * (v_prev + v_next) if v_mode == "mid" else v_next
            x_new = x_prev + v_int * action_dt * torch.cos(yaw_int)
            y_new = y_prev + v_int * action_dt * torch.sin(yaw_int)
            corrected[:, 0] = (x_new - mean_v[0, 0]) / std_v[0, 0]
            corrected[:, 1] = (y_new - mean_v[0, 1]) / std_v[0, 1]
            yaw_prev = yaw_next
            v_prev = v_next
        else:
            raise ValueError(f"unknown mode {mode!r}")

        emb_corr = encoder(corrected.unsqueeze(1)).squeeze(1)
        post_dist, _ = rssm.representation_model(emb_corr, deterministic)
        latent = post_dist.mean

        preds.append(corrected)
        x_prev = x_new
        y_prev = y_new

    return torch.stack(preds, dim=1)


def split_mse(preds, targets, probe_ks) -> dict[str, float]:
    """Per-slice MSE at each probe horizon, averaged over the batch."""
    err2 = (preds - targets) ** 2
    obs_dim = err2.shape[-1]
    out: dict[str, float] = {}
    for k in probe_ks:
        i = k - 1
        out[f"total_k{k}"] = float(err2[:, i].mean().cpu())
        out[f"pose_k{k}"] = float(err2[:, i, :POSE_DIM].mean().cpu())
        out[f"pos_k{k}"] = float(err2[:, i, 0:2].mean().cpu())
        out[f"yaw_k{k}"] = float(err2[:, i, 2:3].mean().cpu())
        out[f"vel_k{k}"] = float(err2[:, i, 3:5].mean().cpu())
        out[f"lidar_k{k}"] = (
            float(err2[:, i, POSE_DIM:].mean().cpu()) if obs_dim > POSE_DIM else float("nan")
        )
    return out


def _fmt(x: float) -> str:
    return f"{'n/a':>10s}" if x != x else f"{x:>10.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path,
                        default=_SD_DIR.parent / "trajectories_lidar.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--teacher-steps", type=int, default=1)
    parser.add_argument("--open-steps", type=int, default=12)
    parser.add_argument("--action-dt", type=float, default=0.3,
                        help="Control period used during data collection.")
    parser.add_argument("--yaw-mode", choices=("mid", "prev", "next"), default="mid")
    parser.add_argument("--v-mode", choices=("mid", "next"), default="mid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = wm.pick_device(args.device)
    rssm, encoder, decoder, mean, std, _cfg = wm.load_world_model_bundle(args.checkpoint, device)
    obs_dim = int(mean.shape[-1])

    print(f"device        : {device}")
    print(f"checkpoint    : {args.checkpoint.name}")
    print(f"obs_dim       : {obs_dim}  (pose 0..{POSE_DIM - 1}, lidar {POSE_DIM}..{obs_dim - 1})")
    print(f"action_dt     : {args.action_dt}   yaw_mode={args.yaw_mode}  v_mode={args.v_mode}")

    states, actions, next_states, episode_starts = wm.load_transition_dataset(args.dataset)
    if episode_starts is None:
        raise SystemExit("dataset has no episode_starts -- need trajectory-mode data")

    window = args.teacher_steps + args.open_steps + 1
    obs_arr, act_arr, _ = wm.build_windows_from_episodes(
        states, actions, next_states, episode_starts, window
    )

    m_np = mean.cpu().numpy().reshape(1, 1, -1)
    s_np = std.cpu().numpy().reshape(1, 1, -1)
    obs_arr = ((obs_arr - m_np) / s_np).astype(np.float32)
    act_arr = act_arr.astype(np.float32)

    rng = np.random.default_rng(args.seed)
    bsz = min(args.batch_size, obs_arr.shape[0])
    idx = rng.choice(obs_arr.shape[0], size=bsz, replace=False)
    obs_t = torch.from_numpy(obs_arr[idx]).to(device)
    act_t = torch.from_numpy(act_arr[idx]).to(device)
    targets = obs_t[:, args.teacher_steps + 1 : args.teacher_steps + args.open_steps + 1]

    print(f"windows       : {obs_arr.shape[0]} total, {bsz} sampled (window={window})")
    print(f"rollout       : teacher_steps={args.teacher_steps}, open_steps={args.open_steps}")

    modes = ("pure", "hybrid-full", "hybrid-pos")
    probe_ks = tuple(sorted({1, 4, args.open_steps}))
    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        preds = rollout(
            rssm, encoder, decoder, obs_t, act_t,
            args.teacher_steps, args.open_steps, device, mode,
            mean=mean, std=std, action_dt=args.action_dt,
            yaw_mode=args.yaw_mode, v_mode=args.v_mode,
        )
        results[mode] = split_mse(preds, targets, probe_ks)

    khdr = " ".join(f"{f'k={k}':>10s}" for k in probe_ks)
    print("\nOpen-loop MSE, normalised units (lower is better)")
    for mode in modes:
        print(f"\n{mode.upper()}")
        print(f"  {'slice':<7s}{khdr}")
        for s in SLICES:
            print(f"  {s:<7s}" + " ".join(_fmt(results[mode][f'{s}_k{k}']) for k in probe_ks))

    base = results["pure"]
    print("\nDelta vs PURE (negative = hybrid wins)")
    for mode in ("hybrid-full", "hybrid-pos"):
        print(f"\n{mode.upper()} - PURE")
        print(f"  {'slice':<7s}{khdr}")
        for s in SLICES:
            print(f"  {s:<7s}" + " ".join(
                _fmt(results[mode][f'{s}_k{k}'] - base[f'{s}_k{k}']) for k in probe_ks))

    kmax = args.open_steps
    print(f"\n{'=' * 62}")
    print(f"VERDICT at k={kmax} (the horizon the planner actually uses)")
    print(f"{'=' * 62}")
    for s in ("pos", "pose", "lidar", "total"):
        pv = base[f"{s}_k{kmax}"]
        hv = results["hybrid-pos"][f"{s}_k{kmax}"]
        if pv != pv or hv != hv:
            continue
        rel = (hv - pv) / pv * 100.0 if pv else float("nan")
        tag = "WIN " if hv < pv else "LOSS"
        print(f"  {s:<6s} pure={pv:8.5f}  hybrid-pos={hv:8.5f}  "
              f"{rel:+7.1f}%  {tag}")


if __name__ == "__main__":
    main()
