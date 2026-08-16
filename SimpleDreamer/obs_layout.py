"""Observation layout for the tugbot, including the yaw encoding.

The robot's heading is an angle on a circle, but the dataset stores it as a
raw value in ``[-pi, pi)``. That representation has a discontinuity at the
seam: +179 deg and -179 deg are two degrees apart physically yet maximally
far apart numerically. A network trained under MSE cannot fit that, and
measurements on this project showed the decoder responding by parking yaw
near its marginal mean -- roughly 71 deg RMS error even when reconstructing
an observation it had just encoded, worse than simply repeating the previous
value.

Encoding the heading as ``(cos yaw, sin yaw)`` removes the seam: nearby
headings are always nearby in the representation, and the angle comes back
out with ``atan2``. This module owns both layouts so the trainer, planner,
reward and evaluation code never index pose columns by hand:

    raw     [x, y, yaw,      v, w, rays...]   pose_dim = 5
    cossin  [x, y, cos, sin, v, w, rays...]   pose_dim = 6

Datasets stay on disk in ``raw`` form; :meth:`ObsLayout.encode` converts a
batch on the way in, so existing recordings and checkpoints remain valid.

Normalisation note: ``cos`` and ``sin`` are deliberately *not* z-scored
per dimension. Because ``cos^2 + sin^2 == 1`` the pair has a fixed combined
scale, so both share one constant ``std = 1/sqrt(2)`` and zero mean. A
shared scale maps the unit circle to a circle (an independent per-dimension
scale would distort it into an ellipse and bias ``atan2`` recovery), while
still giving each dimension unit variance so it carries the same weight in
the summed reconstruction loss as every other dimension.
"""

from __future__ import annotations

import math

import numpy as np
import torch

RAW = "raw"
COSSIN = "cossin"

COSSIN_STD = 1.0 / math.sqrt(2.0)

ArrayLike = torch.Tensor | np.ndarray


class ObsLayout:
    """Describes where pose quantities live in an observation vector."""

    def __init__(self, yaw_encoding: str = RAW) -> None:
        if yaw_encoding not in (RAW, COSSIN):
            raise ValueError(f"yaw_encoding must be {RAW!r} or {COSSIN!r}, got {yaw_encoding!r}")
        self.yaw_encoding = yaw_encoding

    def __repr__(self) -> str:
        return f"ObsLayout({self.yaw_encoding!r}, pose_dim={self.pose_dim})"

    @property
    def is_cossin(self) -> bool:
        return self.yaw_encoding == COSSIN

    @property
    def pose_dim(self) -> int:
        """Number of leading columns describing the pose."""
        return 6 if self.is_cossin else 5

    @property
    def rel_goal_dim(self) -> int:
        """Width of the relative-goal vector fed to the reward head."""
        return 4 if self.is_cossin else 3

    @property
    def v_index(self) -> int:
        return 4 if self.is_cossin else 3

    @property
    def w_index(self) -> int:
        return 5 if self.is_cossin else 4

    def obs_dim(self, n_rays: int) -> int:
        return self.pose_dim + n_rays

    def n_rays(self, obs_dim: int) -> int:
        return max(0, obs_dim - self.pose_dim)

    def get_yaw(self, obs: ArrayLike) -> ArrayLike:
        """Heading in radians, recovered from whichever encoding is in use."""
        if not self.is_cossin:
            return obs[..., 2]
        if isinstance(obs, torch.Tensor):
            return torch.atan2(obs[..., 3], obs[..., 2])
        return np.arctan2(obs[..., 3], obs[..., 2])

    def get_v(self, obs: ArrayLike) -> ArrayLike:
        return obs[..., self.v_index]

    def get_w(self, obs: ArrayLike) -> ArrayLike:
        return obs[..., self.w_index]

    def encode(self, obs_raw: ArrayLike) -> ArrayLike:
        """Convert a ``raw``-layout observation to this layout."""
        if not self.is_cossin:
            return obs_raw
        yaw = obs_raw[..., 2]
        if isinstance(obs_raw, torch.Tensor):
            return torch.cat(
                [obs_raw[..., :2], torch.cos(yaw).unsqueeze(-1),
                 torch.sin(yaw).unsqueeze(-1), obs_raw[..., 3:]],
                dim=-1,
            )
        return np.concatenate(
            [obs_raw[..., :2], np.cos(yaw)[..., None],
             np.sin(yaw)[..., None], obs_raw[..., 3:]],
            axis=-1,
        )

    def decode(self, obs: ArrayLike) -> ArrayLike:
        """Convert an observation in this layout back to ``raw`` layout."""
        if not self.is_cossin:
            return obs
        yaw = self.get_yaw(obs)
        if isinstance(obs, torch.Tensor):
            return torch.cat([obs[..., :2], yaw.unsqueeze(-1), obs[..., 4:]], dim=-1)
        return np.concatenate([obs[..., :2], yaw[..., None], obs[..., 4:]], axis=-1)

    def pose_to_xyyaw(self, obs: ArrayLike) -> ArrayLike:
        """Extract ``[x, y, yaw]`` regardless of encoding."""
        yaw = self.get_yaw(obs)
        if isinstance(obs, torch.Tensor):
            return torch.stack([obs[..., 0], obs[..., 1], yaw], dim=-1)
        return np.stack([obs[..., 0], obs[..., 1], yaw], axis=-1)

    def build_pose(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble a pose block in this layout from scalar components."""
        if self.is_cossin:
            return torch.stack([x, y, torch.cos(yaw), torch.sin(yaw), v, w], dim=-1)
        return torch.stack([x, y, yaw, v, w], dim=-1)

    def norm_stats(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-dimension ``(mean, std)``, with the heading pair handled specially.

        Every dimension is z-scored from the data except ``cos``/``sin``,
        which get mean 0 and a shared constant ``std`` so the unit circle
        survives normalisation undistorted.
        """
        flat = obs.reshape(-1, obs.shape[-1])
        mean = flat.mean(0, keepdims=True)
        std = flat.std(0, keepdims=True) + 1e-6
        if self.is_cossin:
            mean[0, 2:4] = 0.0
            std[0, 2:4] = COSSIN_STD
        return mean.astype(np.float32), std.astype(np.float32)

    def slice_map(self, obs_dim: int) -> dict[str, list[int]]:
        """Named column groups, for per-dimension diagnostics."""
        sl: dict[str, list[int]] = {"x": [0], "y": [1]}
        if self.is_cossin:
            sl["cos_yaw"] = [2]
            sl["sin_yaw"] = [3]
            sl["heading"] = [2, 3]
        else:
            sl["yaw"] = [2]
        sl["v"] = [self.v_index]
        sl["w"] = [self.w_index]
        sl["pose(all)"] = list(range(self.pose_dim))
        if obs_dim > self.pose_dim:
            sl["lidar(all)"] = list(range(self.pose_dim, obs_dim))
        sl["ALL"] = list(range(obs_dim))
        return sl


def layout_from_checkpoint(ckpt: dict) -> ObsLayout:
    """Recover the layout a checkpoint was trained with.

    Checkpoints written before the heading encoding existed have no
    ``yaw_encoding`` key and are ``raw`` by definition.
    """
    return ObsLayout(str(ckpt.get("yaw_encoding", RAW)))


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi)``."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "RAW",
    "COSSIN",
    "COSSIN_STD",
    "ObsLayout",
    "layout_from_checkpoint",
    "wrap_pi",
]
