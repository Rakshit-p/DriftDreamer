#!/usr/bin/env python3
"""Check whether the world model beats trivial baselines, per dimension.

A low reconstruction MSE is meaningless on its own: a model can score well
by predicting each dimension's marginal mean and never tracking anything.
This harness compares the model against three baselines that require no
learning at all, separately for every slice of the observation vector:

    predict-mean     always output the dataset mean for that dim
    best-constant    the optimal constant on this batch (= its variance);
                     a lower bound on what any constant predictor achieves
    frozen           persistence -- repeat the last observed value

Any slice where the model does not beat all three carries no learned
signal and is effectively broken, however good the aggregate loss looks.

Two probes are reported:

    recon(0-step)  encode the true observation, take the posterior, decode
                   it straight back. Isolates whether the encoder->decoder
                   path can represent the dimension at all, independent of
                   the transition model.
    open-loop k    roll the prior forward k steps from a teacher-forced
                   posterior. Adds the transition model on top.

If a dimension fails at 0-step, the problem is representation (encoding,
normalisation or decoder objective) and no amount of dynamics work will
fix it. If 0-step is fine but open-loop is not, the transition model is
the culprit.

Use ``--interior-only`` to restrict to windows whose yaw never approaches
the +-pi wrap, which tests whether the wrap discontinuity is what stops a
raw-angle representation from being learnable.
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
from obs_layout import ObsLayout


@torch.no_grad()
def zero_step_recon(rssm, encoder, decoder, obs, act, device) -> torch.Tensor:
    """Posterior reconstruction of each observed step; returns ``(B, T-1, D)``."""
    bsz, T, _ = obs.shape
    latent, deterministic = rssm.recurrent_model_input_init(bsz)
    latent = latent.to(device)
    deterministic = deterministic.to(device)
    embedded = encoder(obs)
    outs: list[torch.Tensor] = []
    for t in range(1, T):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        _, posterior = rssm.representation_model(embedded[:, t], deterministic)
        latent = posterior
        x_hat = decoder(latent.unsqueeze(1), deterministic.unsqueeze(1)).mean.squeeze(1)
        outs.append(x_hat)
    return torch.stack(outs, dim=1)


@torch.no_grad()
def open_loop(rssm, encoder, decoder, obs, act, teacher_steps, open_steps, device) -> torch.Tensor:
    """Prior rollout after teacher forcing; returns ``(B, open_steps, D)``."""
    bsz = obs.shape[0]
    latent, deterministic = rssm.recurrent_model_input_init(bsz)
    latent = latent.to(device)
    deterministic = deterministic.to(device)
    embedded = encoder(obs)
    for t in range(1, teacher_steps + 1):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        _, posterior = rssm.representation_model(embedded[:, t], deterministic)
        latent = posterior
    preds: list[torch.Tensor] = []
    for t in range(teacher_steps + 1, teacher_steps + open_steps + 1):
        deterministic = rssm.recurrent_model(latent, act[:, t - 1], deterministic)
        prior_dist, _ = rssm.transition_model(deterministic)
        latent = prior_dist.mean
        preds.append(decoder(latent.unsqueeze(1), deterministic.unsqueeze(1)).mean.squeeze(1))
    return torch.stack(preds, dim=1)


def _mse(pred: torch.Tensor, tgt: torch.Tensor, cols: list[int]) -> float:
    return float(((pred[..., cols] - tgt[..., cols]) ** 2).mean().cpu())


def report_table(
    title: str,
    slices: dict[str, list[int]],
    model_pred: torch.Tensor,
    targets: torch.Tensor,
    frozen: torch.Tensor,
) -> dict[str, bool]:
    """Print a model-vs-baselines table; return {slice: model_wins}."""
    print(f"\n{title}")
    print(f"  {'slice':<12s}{'model':>10s}{'pred-mean':>11s}{'best-const':>12s}"
          f"{'frozen':>10s}   verdict")
    verdicts: dict[str, bool] = {}
    zeros = torch.zeros_like(targets)
    for name, cols in slices.items():
        m = _mse(model_pred, targets, cols)
        pm = _mse(zeros, targets, cols)
        bc = float(targets[..., cols].var(dim=(0, 1), unbiased=False).mean().cpu())
        fr = _mse(frozen, targets, cols)
        best_baseline = min(pm, bc, fr)
        wins = m < best_baseline
        verdicts[name] = wins
        tag = "ok" if wins else "*** NO SIGNAL ***"
        print(f"  {name:<12s}{m:>10.4f}{pm:>11.4f}{bc:>12.4f}{fr:>10.4f}   {tag}")
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=_SD_DIR.parent / "trajectories_lidar.npz")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--teacher-steps", type=int, default=1)
    ap.add_argument("--open-steps", type=int, default=12)
    ap.add_argument("--probe-ks", type=int, nargs="+", default=[1, 4, 12])
    ap.add_argument("--interior-only", action="store_true",
                    help="Keep only windows whose |yaw| stays below --interior-thresh "
                         "for the whole window (tests the wrap-discontinuity hypothesis).")
    ap.add_argument("--interior-thresh", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = wm.pick_device(args.device)
    rssm, encoder, decoder, mean, std, _cfg, layout = wm.load_world_model_bundle(
        args.checkpoint, device)
    obs_dim = int(mean.view(-1).shape[0])
    yaw_dim = None if layout.is_cossin else 2

    print(f"checkpoint    : {args.checkpoint.name}")
    print(f"device        : {device}    obs_dim: {obs_dim}    layout: {layout.yaw_encoding}")

    states, actions, next_states, eps = wm.load_transition_dataset(args.dataset)
    if eps is None:
        raise SystemExit("dataset has no episode_starts -- need trajectory-mode data")
    window = args.teacher_steps + args.open_steps + 1
    obs_w, act_w, _ = wm.build_windows_from_episodes(states, actions, next_states, eps, window)
    total_windows = obs_w.shape[0]

    if args.interior_only:
        keep = np.all(np.abs(obs_w[:, :, 2]) < args.interior_thresh, axis=1)
        obs_w, act_w = obs_w[keep], act_w[keep]
        print(f"windows       : {obs_w.shape[0]} of {total_windows} kept "
              f"(|yaw| < {args.interior_thresh} throughout)")
        if obs_w.shape[0] < 32:
            raise SystemExit("too few interior windows to evaluate")
    else:
        print(f"windows       : {total_windows} (no filtering)")

    obs_w = layout.encode(obs_w)
    m_np = mean.cpu().numpy().reshape(1, 1, -1)
    s_np = std.cpu().numpy().reshape(1, 1, -1)
    obs_n = ((obs_w - m_np) / s_np).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    bsz = min(args.batch_size, obs_n.shape[0])
    idx = rng.choice(obs_n.shape[0], size=bsz, replace=False)
    obs_t = torch.from_numpy(obs_n[idx]).to(device)
    act_t = torch.from_numpy(act_w[idx].astype(np.float32)).to(device)
    print(f"sampled       : {bsz}")

    slices = layout.slice_map(obs_dim)

    recon = zero_step_recon(rssm, encoder, decoder, obs_t, act_t, device)
    recon_tgt = obs_t[:, 1:]
    recon_frozen = obs_t[:, :-1]
    print("\n" + "=" * 78)
    print("PROBE 1: zero-step reconstruction (posterior saw the true obs)")
    print("  Failure here => the encoder->decoder path cannot represent the dim.")
    print("=" * 78)
    v_recon = report_table("recon MSE (normalised)", slices, recon, recon_tgt, recon_frozen)

    ol = open_loop(rssm, encoder, decoder, obs_t, act_t,
                   args.teacher_steps, args.open_steps, device)
    print("\n" + "=" * 78)
    print("PROBE 2: open-loop prior rollout")
    print("=" * 78)
    ks = [k for k in args.probe_ks if 1 <= k <= args.open_steps]
    v_ol: dict[int, dict[str, bool]] = {}
    for k in ks:
        tgt_k = obs_t[:, args.teacher_steps + k].unsqueeze(1)
        pred_k = ol[:, k - 1].unsqueeze(1)
        frozen_k = obs_t[:, args.teacher_steps].unsqueeze(1)
        v_ol[k] = report_table(f"open-loop k={k} MSE (normalised)",
                               slices, pred_k, tgt_k, frozen_k)

    mean_v = mean.view(1, -1)
    std_v = std.view(1, -1)

    def ang_err_deg(pred_n: torch.Tensor, tgt_n: torch.Tensor) -> float:
        """RMS heading error in degrees, comparing whole observation vectors.

        Denormalises first and recovers the angle through the layout, so the
        number is directly comparable between the raw and (cos, sin)
        encodings even though they occupy different columns.
        """
        shp = pred_n.shape[:-1]
        pw = pred_n.reshape(-1, pred_n.shape[-1]) * std_v + mean_v
        tw = tgt_n.reshape(-1, tgt_n.shape[-1]) * std_v + mean_v
        err = wm.wrap_pi(layout.get_yaw(pw) - layout.get_yaw(tw)).reshape(shp)
        return float(torch.sqrt((err ** 2).mean()).cpu()) * 180.0 / math.pi

    print("\n" + "=" * 78)
    print("HEADING, wrap-aware (RMS angular error in degrees)")
    print("  Comparable across yaw encodings; 'pred-mean' emits the dataset mean pose.")
    print("=" * 78)
    print(f"  {'probe':<16s}{'model':>10s}{'pred-mean':>11s}{'frozen':>10s}")
    zeros_r = torch.zeros_like(recon_tgt)
    print(f"  {'recon(0-step)':<16s}"
          f"{ang_err_deg(recon, recon_tgt):>10.1f}"
          f"{ang_err_deg(zeros_r, recon_tgt):>11.1f}"
          f"{ang_err_deg(recon_frozen, recon_tgt):>10.1f}")
    for k in ks:
        t_k = obs_t[:, args.teacher_steps + k].unsqueeze(1)
        p_k = ol[:, k - 1].unsqueeze(1)
        f_k = obs_t[:, args.teacher_steps].unsqueeze(1)
        print(f"  {'open-loop k=' + str(k):<16s}"
              f"{ang_err_deg(p_k, t_k):>10.1f}"
              f"{ang_err_deg(torch.zeros_like(t_k), t_k):>11.1f}"
              f"{ang_err_deg(f_k, t_k):>10.1f}")

    print("\n" + "=" * 78)
    print("SUMMARY: slices with no learned signal (lose to a trivial baseline)")
    print("=" * 78)
    failed_recon = [s for s, ok in v_recon.items() if not ok]
    print(f"  zero-step recon : {failed_recon or 'none -- all slices beat baselines'}")
    for k in ks:
        failed = [s for s, ok in v_ol[k].items() if not ok]
        print(f"  open-loop k={k:<4d}: {failed or 'none'}")


if __name__ == "__main__":
    main()
