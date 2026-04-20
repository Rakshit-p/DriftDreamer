#!/usr/bin/env python3
"""Data collection + I/O primitives for the Gazebo tugbot.

This module exposes the low-level helpers used by the episodic collector
and by the live MPC runner: ``StateReader`` (polls the robot pose via
``gz topic -e``), ``LidarReader`` (streams both planar scanners and
min-pools them into 16 bearing bins), ``send_cmd_vel`` (publishes Twist
in-process when possible, otherwise falls back to a ``gz topic``
subprocess), and ``reset_robot`` / ``unpause_sim`` service calls.

Running this file directly also performs a simple i.i.d. transition
collection into ``transitions.npz`` (the first-iteration dataset format).
"""

import os
import sys

_gz_lib = "/opt/homebrew/lib"
_existing_dyld = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
if _gz_lib not in _existing_dyld.split(":"):
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
        _gz_lib + (":" + _existing_dyld if _existing_dyld else "")
    )

_GZ_SITE = "/opt/homebrew/lib/python3.12/site-packages"
if _GZ_SITE not in sys.path:
    sys.path.insert(0, _GZ_SITE)

os.environ.setdefault("GZ_IP", "127.0.0.1")
os.environ.setdefault("GZ_DISCOVERY_LOCALHOST_ENABLED", "1")

import math
import re as _re
import signal
import subprocess
import threading
import time

import numpy as np

_GZ_TRANSPORT_OK = False
try:
    import gz.transport as _gzt  # type: ignore # noqa: F401
    _GZ_TRANSPORT_OK = True
except Exception:
    pass

_GZ_ENV = {
    **os.environ,
    "GZ_IP": "127.0.0.1",
    "GZ_DISCOVERY_LOCALHOST_ENABLED": "1",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


NUM_TRANSITIONS = _env_int("NUM_TRANSITIONS", 3_000)
ACTION_DURATION = _env_float("ACTION_DURATION", 0.3)
SAVE_PATH = "/Users/rakshitpradhan/Desktop/factory/transitions.npz"

WORLD_NAME = "world_demo"
ROBOT_NAME = "tugbot"

LINEAR_VEL_RANGE = (-0.4, 0.4)
ANGULAR_VEL_RANGE = (-1.2, 1.2)

X_BOUNDS = (-5.0, 20.0)
Y_BOUNDS = (-5.5, 8.0)
RESET_POSE = {"x": 2.75, "y": 1.0, "z": 0.132, "yaw": 0.0}
FOCUS_BOUNDS: dict[str, tuple[float, float]] | None = None

POSE_TOPIC = f"/world/{WORLD_NAME}/pose/info"
CMD_VEL_TOPIC = f"/model/{ROBOT_NAME}/cmd_vel"
SET_POSE_SVC = f"/world/{WORLD_NAME}/set_pose"

NUM_RAY_BINS = 16
RAY_MAX_RANGE = 5.0

SCAN_FRONT_TOPIC = (
    f"/world/{WORLD_NAME}/model/{ROBOT_NAME}/link/scan_front/sensor/scan_front/scan"
)
SCAN_BACK_TOPIC = (
    f"/world/{WORLD_NAME}/model/{ROBOT_NAME}/link/scan_back/sensor/scan_back/scan"
)
SCAN_FRONT_LINK_YAW = 0.0
SCAN_BACK_LINK_YAW = math.pi


_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def _parse_pose_block(block: str) -> dict | None:
    """Return the ``{x, y, qx..qw}`` dict for ``ROBOT_NAME`` in a Pose_V text block."""
    name_pat = _re.compile(r'name:\s*"([^"]*)"')
    chunks = _re.split(r'(?=\bpose\s*\{)', block)
    for chunk in chunks:
        m = name_pat.search(chunk)
        if not m or m.group(1) != ROBOT_NAME:
            continue
        pos_m = _re.search(r'position\s*\{([^}]*)\}', chunk, _re.DOTALL)
        ori_m = _re.search(r'orientation\s*\{([^}]*)\}', chunk, _re.DOTALL)
        if not pos_m or not ori_m:
            continue

        def _field(text: str, name: str) -> float:
            fm = _re.search(r'\b' + name + r'\s*:\s*(' + _NUM + r')', text)
            return float(fm.group(1)) if fm else 0.0

        return {
            'x':  _field(pos_m.group(1), 'x'),
            'y':  _field(pos_m.group(1), 'y'),
            'qx': _field(ori_m.group(1), 'x'),
            'qy': _field(ori_m.group(1), 'y'),
            'qz': _field(ori_m.group(1), 'z'),
            'qw': _field(ori_m.group(1), 'w') or 1.0,
        }
    return None


class StateReader:
    """Polls the robot pose via ``gz topic -e -n 1`` and derives (x, y, yaw, v_lin, v_ang)."""

    _TOPICS = [POSE_TOPIC, f"/world/{WORLD_NAME}/dynamic_pose/info"]

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_state: np.ndarray | None = None
        self._prev_pose: tuple | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._active_topic = POSE_TOPIC

    def start(self) -> None:
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True

    def wait_ready(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_state is not None:
                    return True
            time.sleep(0.2)
        return False

    def get_state(self) -> np.ndarray | None:
        with self._lock:
            s = self._latest_state
            return s.copy() if s is not None else None

    def topic_list(self) -> list[str]:
        try:
            r = subprocess.run(
                ["gz", "topic", "-l"],
                capture_output=True, text=True, timeout=5, env=_GZ_ENV,
            )
            return [t.strip() for t in r.stdout.splitlines() if t.strip()]
        except Exception:
            return []

    def _reader_loop(self) -> None:
        topic_idx = 0
        while not self._stopped:
            topic = self._TOPICS[topic_idx % len(self._TOPICS)]
            text = self._fetch_one(topic)
            if text:
                self._active_topic = topic
                self._process_block(text)
            else:
                topic_idx += 1
            time.sleep(0.05)

    def _fetch_one(self, topic: str) -> str:
        for cmd in (
            ["gz", "topic", "-e", "-n", "1", "-t", topic],
            ["gz", "topic", "-e", "-t", topic],
        ):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_GZ_ENV,
            )
            out: bytes = b""
            err: bytes = b""
            try:
                out, err = proc.communicate(timeout=4.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    out, err = proc.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    out, err = b"", b""
            text = out.decode("utf-8", errors="replace")
            err_text = err.decode("utf-8", errors="replace") if proc.returncode else ""
            if "Unknown flag" in err_text or "unrecognized" in err_text.lower():
                continue
            return text
        return ""

    def _process_block(self, block: str) -> None:
        p = _parse_pose_block(block)
        if p is None:
            return
        x, y = p['x'], p['y']
        qx, qy, qz, qw = p['qx'], p['qy'], p['qz'], p['qw']
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        now = time.monotonic()
        with self._lock:
            v_lin = v_ang = 0.0
            if self._prev_pose is not None:
                px0, py0, yaw0, t0 = self._prev_pose
                dt = now - t0
                if 0.005 < dt < 1.0:
                    dx, dy = x - px0, y - py0
                    dist = math.sqrt(dx * dx + dy * dy)
                    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
                    v_lin = math.copysign(dist / dt, fwd)
                    dyaw = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
                    v_ang = dyaw / dt
            self._prev_pose = (x, y, yaw, now)
            self._latest_state = np.array([x, y, yaw, v_lin, v_ang], dtype=np.float32)


class LidarReader:
    """Streams scan_front + scan_back and exposes a 16-bin ``[0, 1]`` ray vector.

    Index 0 covers bearing ≈ −π in the robot's base frame; bin width = 2π/16.
    1.0 means "nothing within ``RAY_MAX_RANGE``", 0.0 means "touching".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rays = np.ones((NUM_RAY_BINS,), dtype=np.float32)
        self._per_sensor_bins: dict[str, np.ndarray] = {
            "scan_front": np.full(NUM_RAY_BINS, RAY_MAX_RANGE, dtype=np.float32),
            "scan_back":  np.full(NUM_RAY_BINS, RAY_MAX_RANGE, dtype=np.float32),
        }
        self._last_seen: dict[str, float] = {"scan_front": 0.0, "scan_back": 0.0}
        self._threads: list[threading.Thread] = []
        self._stopped = False

    def start(self) -> None:
        specs = [
            (SCAN_FRONT_TOPIC, "scan_front", SCAN_FRONT_LINK_YAW),
            (SCAN_BACK_TOPIC,  "scan_back",  SCAN_BACK_LINK_YAW),
        ]
        for topic, name, link_yaw in specs:
            t = threading.Thread(
                target=self._reader_loop,
                args=(topic, name, link_yaw),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stopped = True

    def wait_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if any(v > 0.0 for v in self._last_seen.values()):
                    return True
            time.sleep(0.1)
        return False

    def get_rays(self) -> np.ndarray:
        with self._lock:
            return self._rays.copy()

    def sensor_status(self) -> dict[str, float]:
        """Seconds since last message per sensor (``inf`` = never)."""
        now = time.monotonic()
        with self._lock:
            return {k: (now - v if v > 0.0 else float("inf"))
                    for k, v in self._last_seen.items()}

    def _reader_loop(self, topic: str, name: str, link_yaw: float) -> None:
        while not self._stopped:
            proc = subprocess.Popen(
                ["gz", "topic", "-e", "-t", topic],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=_GZ_ENV, bufsize=1, text=True,
            )
            try:
                self._parse_stream(proc, name, link_yaw)
            except Exception:
                pass
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except Exception:
                pass
            if not self._stopped:
                time.sleep(0.5)

    def _parse_stream(self, proc, name: str, link_yaw: float) -> None:
        """Parse one LaserScan at a time from ``proc.stdout``.

        ``gz topic -e`` does not emit a consistent separator across gz versions
        on macOS, so we detect message boundaries via ``header {`` at column 0
        (after we've seen some ranges) and via a ``count: N`` line matching
        ``len(ranges)``. A literal ``---`` line is still accepted as a fallback.
        """
        angle_min: float | None = None
        angle_step: float | None = None
        expected_count: int | None = None
        ranges: list[float] = []

        def _flush() -> None:
            nonlocal angle_min, angle_step, expected_count, ranges
            if angle_min is not None and angle_step is not None and len(ranges) > 0:
                self._process_scan(
                    name, link_yaw,
                    float(angle_min), float(angle_step), ranges,
                )
            angle_min = None
            angle_step = None
            expected_count = None
            ranges = []

        for line in proc.stdout:
            if self._stopped:
                return
            line = line.rstrip()
            if not line:
                continue

            if line == "---":
                _flush()
                continue
            if line.startswith("header {") and ranges:
                _flush()
                continue

            if line.startswith("ranges:"):
                try:
                    ranges.append(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
                if expected_count is not None and len(ranges) >= expected_count:
                    _flush()
            elif line.startswith("angle_min:"):
                try:
                    angle_min = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("angle_step:"):
                try:
                    angle_step = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("count:"):
                try:
                    expected_count = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                if (expected_count is not None
                        and expected_count > 0
                        and len(ranges) >= expected_count):
                    _flush()

        _flush()

    def _process_scan(
        self,
        name: str,
        link_yaw: float,
        angle_min: float,
        angle_step: float,
        ranges: list[float],
    ) -> None:
        arr = np.asarray(ranges, dtype=np.float32)
        if arr.size == 0:
            return
        bad = ~np.isfinite(arr) | (arr <= 0.0)
        arr = np.where(bad, RAY_MAX_RANGE, arr)
        arr = np.minimum(arr, RAY_MAX_RANGE)

        bearings = angle_min + angle_step * np.arange(arr.size, dtype=np.float32) + link_yaw
        bearings = (bearings + math.pi) % (2.0 * math.pi) - math.pi

        bin_width = 2.0 * math.pi / NUM_RAY_BINS
        bins = np.floor((bearings + math.pi) / bin_width).astype(np.int32)
        bins = np.clip(bins, 0, NUM_RAY_BINS - 1)

        per_bin = np.full(NUM_RAY_BINS, RAY_MAX_RANGE, dtype=np.float32)
        np.minimum.at(per_bin, bins, arr)

        now = time.monotonic()
        with self._lock:
            self._per_sensor_bins[name] = per_bin
            self._last_seen[name] = now
            combined = np.minimum(
                self._per_sensor_bins["scan_front"],
                self._per_sensor_bins["scan_back"],
            )
            self._rays = (combined / RAY_MAX_RANGE).astype(np.float32)


_cmd_vel_pub = None


def _try_init_cmd_vel_transport() -> None:
    """Try to set up an in-process ``gz.transport`` Twist publisher; else use a subprocess fallback."""
    global _cmd_vel_pub
    if _cmd_vel_pub is not None:
        return
    if os.environ.get("GZ_CMD_VEL_USE_SUBPROCESS", "").lower() in ("1", "true", "yes"):
        _cmd_vel_pub = False
        return
    try:
        from gz.transport import Node
        from gz.msgs.twist_pb2 import Twist

        node = Node()
        pub = node.advertise(CMD_VEL_TOPIC, Twist)
        _cmd_vel_pub = (node, pub, Twist)
    except Exception:
        _cmd_vel_pub = False


def _send_cmd_vel_subprocess(v: float, omega: float) -> None:
    data = f"linear {{ x: {v:.4f} }} angular {{ z: {omega:.4f} }}"
    subprocess.run(
        ["gz", "topic", "-t", CMD_VEL_TOPIC, "-m", "gz.msgs.Twist", "-p", data],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_GZ_ENV,
    )


def send_cmd_vel(v: float, omega: float) -> None:
    """Publish a Twist(linear.x=v, angular.z=omega) on ``CMD_VEL_TOPIC``."""
    if _cmd_vel_pub is None:
        _try_init_cmd_vel_transport()
    if _cmd_vel_pub is not False:
        _node, pub, Twist = _cmd_vel_pub
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = omega
        pub.publish(msg)
        return
    _send_cmd_vel_subprocess(v, omega)


def reset_robot() -> None:
    """Teleport the robot to ``RESET_POSE`` via the Gazebo ``set_pose`` service."""
    yaw = float(RESET_POSE.get("yaw", 0.0))
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    req = (
        f"name: '{ROBOT_NAME}' "
        f"position {{ x: {RESET_POSE['x']} y: {RESET_POSE['y']} z: {RESET_POSE['z']} }} "
        f"orientation {{ z: {qz} w: {qw} }}"
    )
    subprocess.run(
        ["gz", "service", "-s", SET_POSE_SVC,
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "1000", "-r", req],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_GZ_ENV,
    )
    time.sleep(0.4)


def unpause_sim() -> None:
    """Send ``pause: false`` to the world-control service."""
    result = subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/control",
         "--reqtype", "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "-r", "pause: false"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_GZ_ENV,
    )
    if result.returncode != 0:
        print(f"  [WARN] unpause service returned {result.returncode} — "
              "click Play (▶) in Gazebo; cmd_vel does nothing while paused.", flush=True)
        if result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()[:200]}", flush=True)
    else:
        print("  [exec] Sent unpause (pause: false) via gz service.", flush=True)


def in_bounds(state: np.ndarray) -> bool:
    return (X_BOUNDS[0] < state[0] < X_BOUNDS[1]
            and Y_BOUNDS[0] < state[1] < Y_BOUNDS[1])


def in_focus_bounds(state: np.ndarray) -> bool:
    if FOCUS_BOUNDS is None:
        return True
    xb = FOCUS_BOUNDS["x"]
    yb = FOCUS_BOUNDS["y"]
    return xb[0] < state[0] < xb[1] and yb[0] < state[1] < yb[1]


def run_diagnostic(reader: StateReader, seconds: int = 5) -> None:
    """Stream the live robot state for ``seconds`` so a human can sanity-check it."""
    print(f"\n[Diagnostic] Streaming raw state for {seconds}s — verify values look correct.")
    print("  Format: [x, y, yaw, v_lin, v_ang]\n")
    t0 = time.time()
    last = None
    while time.time() - t0 < seconds:
        s = reader.get_state()
        if s is not None and (last is None or not np.allclose(s, last, atol=1e-4)):
            print(f"  state = [{s[0]:+7.3f}, {s[1]:+7.3f}, {s[2]:+6.3f}, "
                  f"{s[3]:+5.2f}, {s[4]:+5.2f}]")
            last = s
        time.sleep(0.05)
    print()


def collect(reader: StateReader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect ``NUM_TRANSITIONS`` i.i.d. (state, action, next_state) triples."""
    rng = np.random.default_rng(seed=42)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    resets = 0
    skipped = 0
    t_start = time.time()

    _try_init_cmd_vel_transport()
    if _cmd_vel_pub is not False:
        print("cmd_vel : in-process gz.transport (no per-step subprocess)")
    else:
        print("cmd_vel : subprocess `gz topic` (slower; use /opt/homebrew/bin/python3.12 for fast path)")

    _t_cmd = []
    for _ in range(3):
        t0 = time.perf_counter()
        send_cmd_vel(0.0, 0.0)
        _t_cmd.append(time.perf_counter() - t0)
    avg_cmd = sum(_t_cmd) / len(_t_cmd)
    sec_per_transition = avg_cmd + ACTION_DURATION
    eta_min = NUM_TRANSITIONS * sec_per_transition / 60.0

    print(f"Target  : {NUM_TRANSITIONS:,} transitions")
    print(f"Action  : {ACTION_DURATION}s hold, uniform random ∈ "
          f"[{LINEAR_VEL_RANGE[0]},{LINEAR_VEL_RANGE[1]}] m/s  ×  "
          f"[{ANGULAR_VEL_RANGE[0]},{ANGULAR_VEL_RANGE[1]}] rad/s")
    print(f"Reset   : x={RESET_POSE['x']:.2f}, y={RESET_POSE['y']:.2f}, "
          f"z={RESET_POSE['z']:.3f}, yaw={RESET_POSE.get('yaw', 0.0):.3f}")
    if FOCUS_BOUNDS is not None:
        print(f"Focus   : x in [{FOCUS_BOUNDS['x'][0]},{FOCUS_BOUNDS['x'][1]}], "
              f"y in [{FOCUS_BOUNDS['y'][0]},{FOCUS_BOUNDS['y'][1]}]")
    print(f"Estimate: ~{eta_min:.0f} min  (~{sec_per_transition:.2f}s per transition "
          f"= cmd {avg_cmd*1000:.0f}ms + hold {ACTION_DURATION*1000:.0f}ms)\n")

    collected = 0
    while collected < NUM_TRANSITIONS:
        s0 = reader.get_state()
        if s0 is None:
            time.sleep(0.05)
            continue

        if not in_bounds(s0) or not in_focus_bounds(s0):
            send_cmd_vel(0.0, 0.0)
            reset_robot()
            resets += 1
            continue

        v = float(rng.uniform(*LINEAR_VEL_RANGE))
        omega = float(rng.uniform(*ANGULAR_VEL_RANGE))

        state_before = reader.get_state()
        send_cmd_vel(v, omega)
        time.sleep(ACTION_DURATION)
        state_after = reader.get_state()

        if state_before is None or state_after is None:
            skipped += 1
            continue

        states.append(state_before)
        actions.append(np.array([v, omega], dtype=np.float32))
        next_states.append(state_after)
        collected += 1

        if collected % 200 == 0 or collected == NUM_TRANSITIONS:
            elapsed = time.time() - t_start
            rate = collected / elapsed
            eta = (NUM_TRANSITIONS - collected) / rate if rate > 0 else 0.0
            print(f"  [{collected:>6}/{NUM_TRANSITIONS}]  "
                  f"{elapsed/60:5.1f} min  |  ETA {eta/60:5.1f} min  |  "
                  f"pos=({state_after[0]:.2f},{state_after[1]:.2f})  "
                  f"resets={resets}  skipped={skipped}")

    send_cmd_vel(0.0, 0.0)
    return (np.stack(states, dtype=np.float32),
            np.stack(actions, dtype=np.float32),
            np.stack(next_states, dtype=np.float32))


def save_and_report(states, actions, next_states) -> None:
    """Save arrays to ``SAVE_PATH`` and print a small summary table."""
    np.savez_compressed(SAVE_PATH, states=states, actions=actions, next_states=next_states)
    n = len(states)
    sep = "─" * 57
    print(f"\n{sep}")
    print(f"  Saved {n:,} transitions → {SAVE_PATH}")
    print(sep)
    print(f"  {'Array':<14} {'Shape':<18} {'min':>8} {'max':>8} {'mean':>8}")
    print(f"  {'─' * 54}")
    for name, arr in [("states", states), ("actions", actions), ("next_states", next_states)]:
        print(f"  {name:<14} {str(arr.shape):<18} "
              f"{arr.min():>8.3f} {arr.max():>8.3f} {arr.mean():>8.3f}")
    print(f"\n  State  columns : x, y, yaw, v_linear, v_angular")
    print(f"  Action columns : cmd_linear, cmd_angular")


def main() -> None:
    reader = StateReader()
    partial: dict = {}

    def _on_interrupt(sig, frame):
        print("\n\nInterrupted — saving partial dataset …")
        send_cmd_vel(0.0, 0.0)
        reader.stop()
        if partial.get("states"):
            s = np.stack(partial["states"], dtype=np.float32)
            a = np.stack(partial["actions"], dtype=np.float32)
            ns = np.stack(partial["next_states"], dtype=np.float32)
            save_and_report(s, a, ns)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)

    print("Subscribing to pose topic (gz topic subprocess) …", end=" ", flush=True)
    reader.start()
    print("ok")

    print("Waiting for pose data (up to 60 s — includes gz-transport discovery) …", flush=True)
    print("  Make sure gz-sim is running in Terminal 1 with GZ_IP=127.0.0.1")
    if not reader.wait_ready(timeout=60.0):
        print("\n[ERROR] No pose data after 60 s.")
        topics = reader.topic_list()
        print(f"  In-process topic list ({len(topics)} topics):")
        for t in sorted(topics)[:30]:
            print(f"    {t}")
        if not topics:
            print("  No topics at all → gz-sim is NOT running or GZ_IP/discovery mismatch.")
        else:
            raw = reader._fetch_one(POSE_TOPIC) or reader._fetch_one(
                f"/world/{WORLD_NAME}/dynamic_pose/info")
            if raw:
                print("  --- raw output (first 800 chars) ---")
                print(raw[:800])
                print("  ---")
                print(f"  ROBOT_NAME={ROBOT_NAME!r} — verify this name appears above.")
        sys.exit(1)

    topics = reader.topic_list()
    print(f"connected.  ({len(topics)} topics discovered)\n")

    print("Unpausing simulation …", end=" ", flush=True)
    unpause_sim()
    time.sleep(0.5)
    print("done\n")

    run_diagnostic(reader, seconds=5)

    confirm = input("Values look correct? [y/n]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        reader.stop()
        sys.exit(0)

    states, actions, next_states = collect(reader)
    partial["states"] = list(states)
    partial["actions"] = list(actions)
    partial["next_states"] = list(next_states)

    reader.stop()
    save_and_report(states, actions, next_states)


def _parse_cli_and_apply() -> None:
    """Override module-level globals from command-line arguments."""
    import argparse
    global NUM_TRANSITIONS, ACTION_DURATION, SAVE_PATH
    global LINEAR_VEL_RANGE, ANGULAR_VEL_RANGE
    global RESET_POSE, FOCUS_BOUNDS
    p = argparse.ArgumentParser(
        description="Collect (state, action, next_state) from the tugbot for world-model training.")
    p.add_argument("-n", "--num-transitions", type=int, default=None, metavar="N")
    p.add_argument("--full", action="store_true", help="Collect 10 000 transitions.")
    p.add_argument("-d", "--action-duration", type=float, default=None, metavar="SEC")
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--reset-pose", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW"), default=None)
    p.add_argument("--lin-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    p.add_argument("--ang-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    p.add_argument("--focus-bounds", type=float, nargs=4,
                   metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"), default=None)
    args = p.parse_args()
    if args.full:
        NUM_TRANSITIONS = 10_000
    if args.num_transitions is not None:
        NUM_TRANSITIONS = args.num_transitions
    if args.action_duration is not None:
        ACTION_DURATION = args.action_duration
    if args.save_path is not None:
        SAVE_PATH = args.save_path
    if args.reset_pose is not None:
        RESET_POSE = {
            "x": float(args.reset_pose[0]),
            "y": float(args.reset_pose[1]),
            "z": float(args.reset_pose[2]),
            "yaw": float(args.reset_pose[3]),
        }
    if args.lin_range is not None:
        LINEAR_VEL_RANGE = (float(args.lin_range[0]), float(args.lin_range[1]))
    if args.ang_range is not None:
        ANGULAR_VEL_RANGE = (float(args.ang_range[0]), float(args.ang_range[1]))
    if args.focus_bounds is not None:
        FOCUS_BOUNDS = {
            "x": (float(args.focus_bounds[0]), float(args.focus_bounds[1])),
            "y": (float(args.focus_bounds[2]), float(args.focus_bounds[3])),
        }


if __name__ == "__main__":
    _parse_cli_and_apply()
    main()
