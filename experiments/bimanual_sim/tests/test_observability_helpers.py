"""Property tests for observability helpers that have non-trivial input space.

Hypothesis earns its keep here: the quaternion → pitch/roll decomposition has
edge cases near gimbal lock that hand-written cases would likely miss, and
`_phase_windows` operates over per-arm `Step` lists with combinatorial
structure that's tedious to enumerate manually.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from arm_handles import ArmSide
from scene_base import Step, TaskPhase
from tools.mj import _phase_windows
from tools.observability import _quat_pitch_roll_rad

# ---------------------------------------------------------------------------
# _quat_pitch_roll_rad — numerical stability across the unit quaternion sphere
# ---------------------------------------------------------------------------


@st.composite
def unit_quaternions(draw: st.DrawFn) -> np.ndarray:
    """Uniformly-ish distributed unit quaternions via 4 floats normalised."""
    components = draw(
        st.tuples(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    quat = np.asarray(components, dtype=float)
    norm = float(np.linalg.norm(quat))
    # Discard degenerate near-zero quats; hypothesis will retry.
    assume(norm > 1e-3)
    return quat / norm


@given(quat=unit_quaternions())
def test_pitch_roll_finite_and_in_range(quat: np.ndarray) -> None:
    """Pitch ∈ [-π/2, π/2], roll ∈ [-π, π], both finite — the contract any
    ZYX Tait-Bryan decomposition has to satisfy. Catches NaN production from
    the `asin` near gimbal lock if `clamp` ever drifts."""
    pitch, roll = _quat_pitch_roll_rad(quat)
    assert math.isfinite(pitch)
    assert math.isfinite(roll)
    # asin clamps to [-π/2, π/2] by definition.
    assert -math.pi / 2 - 1e-9 <= pitch <= math.pi / 2 + 1e-9
    assert -math.pi - 1e-9 <= roll <= math.pi + 1e-9


def test_pitch_roll_identity_quat() -> None:
    pitch, roll = _quat_pitch_roll_rad(np.array([1.0, 0.0, 0.0, 0.0]))
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)


def test_pitch_roll_pure_x_rotation() -> None:
    """Quat for 30° rotation about world +x: roll should be ≈ 30°, pitch ≈ 0."""
    angle = math.radians(30.0)
    quat = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
    pitch, roll = _quat_pitch_roll_rad(quat)
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert roll == pytest.approx(angle, abs=1e-9)


def test_pitch_roll_pure_y_rotation() -> None:
    """Quat for 30° rotation about world +y: pitch should be ≈ 30°, roll ≈ 0."""
    angle = math.radians(30.0)
    quat = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])
    pitch, roll = _quat_pitch_roll_rad(quat)
    assert pitch == pytest.approx(angle, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)


def test_pitch_roll_at_gimbal_lock() -> None:
    """At pitch = ±π/2 the ZYX decomposition is degenerate (roll/yaw share an
    axis). The implementation clamps `sin_pitch` to [-1, 1] — verify that the
    near-gimbal case returns finite values and not NaN from a `asin(1+ε)`."""
    quat = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])  # 90° about +y
    pitch, roll = _quat_pitch_roll_rad(quat)
    assert math.isfinite(pitch)
    assert math.isfinite(roll)
    assert pitch == pytest.approx(math.pi / 2, abs=1e-9)


# ---------------------------------------------------------------------------
# _phase_windows — combinatorial grouping over per-arm Step lists
# ---------------------------------------------------------------------------


_NON_UNPHASED = [p for p in TaskPhase if p is not TaskPhase.UNPHASED]


def _make_step(phase: TaskPhase, duration: float) -> Step:
    """Minimal Step with a fixed (6,) arm_q. We don't care about the joint
    config for these tests — we're testing the timeline grouping."""
    return Step(
        label="t",
        arm_q=np.zeros(6, dtype=float),
        gripper="open",
        duration=duration,
        phase=phase,
    )


step_strategy = st.builds(
    _make_step,
    phase=st.sampled_from([*_NON_UNPHASED, TaskPhase.UNPHASED]),
    duration=st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
)


@given(
    left=st.lists(step_strategy, min_size=0, max_size=12),
    right=st.lists(step_strategy, min_size=0, max_size=12),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_phase_windows_have_nonnegative_duration(left: list[Step], right: list[Step]) -> None:
    """Every emitted window has start_s ≤ end_s. Catches sign/order bugs in
    the min/max accumulation in `_phase_windows`."""
    plan = {ArmSide.LEFT: left, ArmSide.RIGHT: right}
    for window in _phase_windows(plan).values():
        assert window.start_s <= window.end_s


@given(
    left=st.lists(step_strategy, min_size=0, max_size=12),
    right=st.lists(step_strategy, min_size=0, max_size=12),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_unphased_steps_never_appear_as_windows(left: list[Step], right: list[Step]) -> None:
    """`UNPHASED` is the "no contract here" sentinel — it must not get a
    window of its own (which would mean the runner thinks we entered an
    UNPHASED contract)."""
    plan = {ArmSide.LEFT: left, ArmSide.RIGHT: right}
    windows = _phase_windows(plan)
    assert TaskPhase.UNPHASED not in windows


@given(steps=st.lists(step_strategy, min_size=1, max_size=10))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_phase_windows_cover_only_real_phases(steps: list[Step]) -> None:
    """Every key in the result must be one of the phases that actually
    appeared in the input (post-UNPHASED filter)."""
    plan = {ArmSide.LEFT: steps, ArmSide.RIGHT: []}
    seen_phases = {s.phase for s in steps if s.phase is not TaskPhase.UNPHASED}
    windows = _phase_windows(plan)
    assert set(windows.keys()) == seen_phases
