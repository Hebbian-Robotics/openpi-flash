"""Agilex Piper loader — Menagerie-adapter for the bimanual arm attachments.

Menagerie's `agilex_piper/piper.xml` stays untouched on disk; our scene
calls `attach_piper(scene_spec, prefix=..., frame=..., config=...)` which
reads upstream, attaches with the requested prefix, then overrides a
fixed set of actuator parameters via native `MjsActuator` properties
(`gainprm`, `biasprm`, `forcerange`).

Why we override: upstream ships `kp=80 N·m/rad kv=5 forcerange=±100` on
joints 1–3, tuned for an empty wrist. Our scene carries a wrist camera
payload and plans far-reach cable poses where gravity torque on joint 2
reaches ~8 N·m; at the default PD that lands ~0.55 rad of steady-state
droop (~28 cm of TCP offset). The override stays additive — we don't
delete or rename anything from upstream — so Menagerie bumps carry over
cleanly.

Boundary assertions (`_assert_menagerie_shape`) check that the actuator
names we plan to touch actually exist in the upstream XML. If Menagerie
renames `joint1` → `shoulder_pitch`, load fails here with a clear
message instead of silently no-op'ing and leaving the arm at stock
gains (which would reintroduce the 28 cm droop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

from paths import PIPER_XML


@dataclass(frozen=True)
class PiperJointGain:
    """kp / kv / forcerange override for a single Piper position actuator.

    `joint_suffix` is the name upstream uses for both the joint and the
    actuator that drives it (Menagerie's convention). We concatenate the
    scene-supplied `prefix` at attach time to resolve the final name.
    """

    joint_suffix: str
    kp: float
    kv: float
    forcerange: tuple[float, float]


@dataclass(frozen=True)
class PiperConfig:
    """Declarative customization of Menagerie's `agilex_piper/piper.xml`.

    Only actuator parameter overrides live here — not joint damping, not
    masses, not geom tweaks. That's deliberate: the more we change, the
    more drift risk on upstream bumps. If a future scene needs extra
    knobs (e.g. per-joint damping), add them as new fields here so all
    customizations stay in one typed place.
    """

    gains: tuple[PiperJointGain, ...] = field(default_factory=lambda: _DEFAULT_DATA_CENTER_GAINS)


# Default gain profile used by the data-center scene. Calibrated to
# eliminate gravity droop on loaded cable-reach poses:
#   * kp bumped ~60× over upstream's 80 N·m/rad so kp·err doesn't saturate
#     at modest tracking error.
#   * kv ≈ 1.2× critical damping (2·√(kp·I)) for each joint's effective
#     inertia — same damping ratio upstream uses, just at a stiffer kp.
#     Over-damping (an earlier kv=150 try) asymptoted slowly; under-damping
#     introduces audible oscillation at the PD step-response.
#   * forcerange widened ±1000 N·m so the PD has headroom to converge
#     before saturating (upstream ±100 clamped error at 0.02 rad, leaving
#     ~3 cm of droop on shoulder-heavy poses).
_DEFAULT_DATA_CENTER_GAINS: tuple[PiperJointGain, ...] = (
    PiperJointGain("joint1", kp=5000.0, kv=60.0, forcerange=(-1000.0, 1000.0)),
    PiperJointGain("joint2", kp=5000.0, kv=60.0, forcerange=(-1000.0, 1000.0)),
    PiperJointGain("joint3", kp=3000.0, kv=40.0, forcerange=(-1000.0, 1000.0)),
    PiperJointGain("joint4", kp=1500.0, kv=25.0, forcerange=(-1000.0, 1000.0)),
    PiperJointGain("joint5", kp=800.0, kv=15.0, forcerange=(-1000.0, 1000.0)),
    PiperJointGain("joint6", kp=800.0, kv=15.0, forcerange=(-1000.0, 1000.0)),
)


_PIPER_REQUIRED_ACTUATORS: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
)


def _assert_menagerie_shape(piper_spec: mujoco.MjSpec, config: PiperConfig) -> None:
    """Fail fast if the upstream actuator names we plan to override are
    missing or the config references one that isn't upstream."""
    upstream_actuators = {a.name for a in piper_spec.actuators if a.name}
    for required in _PIPER_REQUIRED_ACTUATORS:
        if required not in upstream_actuators:
            raise RuntimeError(
                f"Piper upstream XML missing expected actuator {required!r}. "
                "Menagerie's agilex_piper/piper.xml may have changed shape — "
                "update robots/piper.py if the new naming is intentional."
            )
    for gain in config.gains:
        if gain.joint_suffix not in upstream_actuators:
            raise RuntimeError(
                f"PiperConfig.gains references actuator {gain.joint_suffix!r} "
                f"not in Piper upstream (actuators: {sorted(upstream_actuators)!r})."
            )


def attach_piper(
    scene_spec: mujoco.MjSpec,
    *,
    prefix: str,
    frame: mujoco.MjsFrame,
    config: PiperConfig = PiperConfig(),  # noqa: B008 — frozen dataclass, no shared state
) -> None:
    """Attach a prefixed Piper to `scene_spec` at `frame`, then apply
    gain/forcerange overrides per `config`.

    `prefix` is prepended to every Piper body/joint/actuator/geom name on
    attach (MjSpec convention). The post-attach override uses the native
    `spec.actuator(name).gainprm = [...]` form — same as writing the
    values directly into `<position>` elements, just expressed in Python.

    Mutates `scene_spec` in place; no return value.
    """
    piper_spec = mujoco.MjSpec.from_file(str(PIPER_XML))
    _assert_menagerie_shape(piper_spec, config)
    scene_spec.attach(piper_spec, prefix=prefix, frame=frame)

    for g in config.gains:
        name = f"{prefix}{g.joint_suffix}"
        act = scene_spec.actuator(name)
        if act is None:
            raise RuntimeError(
                f"attach_piper: actuator {name!r} not found on scene spec after "
                f"attach. Check prefix={prefix!r} / PiperConfig.gains."
            )
        act.gainprm = [g.kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        act.biasprm = [0.0, -g.kp, -g.kv, 0, 0, 0, 0, 0, 0, 0]
        act.forcerange = list(g.forcerange)
