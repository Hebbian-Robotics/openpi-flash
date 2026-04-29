"""Real-time + offline rerun integration for the bimanual sim.

Three sink modes, picked at construction time:

* `RerunStreamer.serve_grpc(port)` — open a gRPC server on the local
  host so a rerun viewer (running anywhere with TCP access) can
  connect and stream live. The intended remote-debugging pattern is
  EC2 serves on `localhost:PORT`, the laptop SSH-tunnels with
  `-L PORT:localhost:PORT`, the laptop's rerun viewer connects to
  `rerun+http://localhost:PORT`.

* `RerunStreamer.connect_grpc(url)` — push events to an already-running
  rerun viewer. Useful when the viewer host runs `rerun --serve` and
  this process is the producer.

* `RerunStreamer.save_rrd(path)` — write to a `.rrd` file for later
  offline replay. This is what `tools/mj.py contracts --rerun-rrd ...`
  uses.

The streamer wraps a single `RecordingStream` so the rest of the
codebase doesn't have to thread `recording=` through every `rr.log`
call. It exposes typed helpers for the four event flavours we care
about:

* `set_sim_time(t)` — set the `sim_time` timeline cursor.
* `log_phase_event(...)` — text log of phase boundary results.
* `log_joint_scalars(side_prefix, joint_names, qpos)` — per-joint
  scalar plots so you can see the planned trajectories.
* `log_body_transform(name, xpos, xquat)` — 3D transform of one body
  for the rerun 3D viewer.
* `log_camera_frame(name, image)` — an RGB frame from a named camera.

Importing rerun lazily (only on construction) keeps the rest of the
codebase callable when rerun-sdk isn't installed — handy for unit
tests or minimal CI where the streamer is unused.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — type-only import
    from rerun import RecordingStream


def _import_rerun() -> Any:
    """Import rerun-sdk lazily, with a clear error if missing.

    The PyPI package is `rerun-sdk` (the rerun.io visualizer); the
    bare `rerun` package is an unrelated file-change-detector
    library. Both expose the `rerun` import namespace so the import
    statement looks identical — the install command differs.
    """
    try:
        import rerun as rr
    except ImportError as err:  # pragma: no cover — exercised when sdk missing
        raise RuntimeError(
            "rerun streaming requires `rerun-sdk` (NOT the bare `rerun` package). "
            "Install with `uv add rerun-sdk` or `pip install rerun-sdk`."
        ) from err
    return rr


@dataclass
class RerunStreamer:
    """One rerun recording stream with typed log helpers.

    Construct via the class methods (`serve_grpc`, `connect_grpc`,
    `save_rrd`); direct instantiation is reserved for tests that
    inject a stub `rr` module.
    """

    # Justified type exception: `rr` is the lazily-imported rerun-sdk
    # module. Its archetype constructors (`Scalars`, `TextLog`,
    # `Transform3D`, etc.) live as module attributes that ty can't
    # resolve from a TYPE_CHECKING-only import without bringing the
    # SDK in at type-check time. Carrying the module reference as
    # `Any` is the minimum-cost loosening — every call site that uses
    # `self.rr.Foo(...)` is wrapped in a typed helper method on this
    # class, so the looseness doesn't leak.
    rr: Any
    recording: RecordingStream
    application_id: str
    # Known body names for which a transform was logged at least once. Lets
    # callers avoid double-logging static bodies on every tick.
    _logged_static: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Lifecycle constructors
    # ------------------------------------------------------------------

    @classmethod
    def serve_grpc(cls, *, scene_name: str, grpc_port: int) -> RerunStreamer:
        """Serve a gRPC endpoint on `localhost:grpc_port` for a viewer to connect to."""
        rr = _import_rerun()
        application_id = f"bimanual_sim_{scene_name}"
        recording = rr.RecordingStream(application_id=application_id)
        endpoint = rr.serve_grpc(grpc_port=grpc_port, recording=recording)
        print(f"rerun: serving on {endpoint}")
        print(
            "  remote viewer pattern: "
            "ssh -L "
            f"{grpc_port}:localhost:{grpc_port} <user>@<host> ; "
            f"rerun connect rerun+http://localhost:{grpc_port}"
        )
        return cls(rr=rr, recording=recording, application_id=application_id)

    @classmethod
    def connect_grpc(cls, *, scene_name: str, url: str) -> RerunStreamer:
        """Push events to an already-running viewer at `url`."""
        rr = _import_rerun()
        application_id = f"bimanual_sim_{scene_name}"
        recording = rr.RecordingStream(application_id=application_id)
        rr.connect_grpc(url=url, recording=recording)
        print(f"rerun: connected to {url}")
        return cls(rr=rr, recording=recording, application_id=application_id)

    @classmethod
    def save_rrd(cls, *, scene_name: str, rrd_path: Path) -> RerunStreamer:
        """Write events to a `.rrd` file for offline replay."""
        rr = _import_rerun()
        application_id = f"bimanual_sim_{scene_name}"
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        recording = rr.RecordingStream(application_id=application_id)
        recording.save(str(rrd_path))
        print(f"rerun: writing rrd → {rrd_path}")
        return cls(rr=rr, recording=recording, application_id=application_id)

    # ------------------------------------------------------------------
    # Time + log helpers
    # ------------------------------------------------------------------

    def set_sim_time(self, sim_t: float) -> None:
        """Move the `sim_time` timeline cursor to `sim_t` seconds.

        All subsequent `log_*` calls are tagged with this timestamp
        until the next `set_sim_time`.
        """
        self.rr.set_time("sim_time", duration=float(sim_t), recording=self.recording)

    def log_phase_event(
        self,
        *,
        phase: str,
        boundary: str,
        contract_ok: bool,
        message: str,
    ) -> None:
        """Append a phase-boundary text log entry."""
        level = self.rr.TextLogLevel.INFO if contract_ok else self.rr.TextLogLevel.ERROR
        full_message = f"{phase} {boundary}: {message}"
        self.rr.log(
            "log/phases",
            self.rr.TextLog(full_message, level=level),
            recording=self.recording,
        )

    def log_joint_scalars(
        self,
        *,
        side_prefix: str,
        joint_names: Iterable[str],
        qpos: np.ndarray,
    ) -> None:
        """Log per-joint scalar plots.

        Each joint becomes its own scalar entity at
        `arms/<side>/joints/<joint_name>` so the rerun viewer plots
        them on separate tracks; the user can group / overlay them
        via the viewer's UI.
        """
        side = side_prefix.rstrip("/")
        joints = list(joint_names)
        values = np.asarray(qpos, dtype=float).reshape(-1)
        if len(joints) != len(values):
            raise ValueError(
                f"log_joint_scalars: {len(joints)} joint names but qpos has length {len(values)}"
            )
        for joint_name, value in zip(joints, values, strict=True):
            self.rr.log(
                f"arms/{side}/joints/{joint_name}",
                self.rr.Scalars([float(value)]),
                recording=self.recording,
            )

    def log_body_transform(
        self,
        *,
        name: str,
        xpos: np.ndarray,
        xquat: np.ndarray,
    ) -> None:
        """Log one body's world pose as a 3D transform.

        MuJoCo's quaternion is `(w, x, y, z)`; rerun expects
        `(x, y, z, w)`. The conversion happens here so callers can
        pass `data.xquat[body_id]` directly.
        """
        translation = np.asarray(xpos, dtype=float).reshape(3)
        wxyz = np.asarray(xquat, dtype=float).reshape(4)
        xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)
        self.rr.log(
            f"bodies/{name}",
            self.rr.Transform3D(
                translation=translation,
                rotation=self.rr.Quaternion(xyzw=xyzw),
            ),
            recording=self.recording,
        )

    def log_camera_frame(self, *, name: str, image: np.ndarray) -> None:
        """Log one RGB camera frame.

        Camera frames are large; callers should sample sparingly (every
        N render ticks) to keep gRPC bandwidth manageable over an SSH
        tunnel. A 640x480 RGB frame is ~0.9 MB before rerun's internal
        compression.
        """
        self.rr.log(
            f"cameras/{name}",
            self.rr.Image(image),
            recording=self.recording,
        )
