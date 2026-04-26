"""Structured artifacts and checks for scripted simulation demos.

The demo runner is intentionally visual, but debugging needs a colder
interface: named phase boundaries, machine-readable events, and exact
snapshots of MuJoCo state. This module keeps those concerns out of the
scene definition and CLI command bodies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from scene_base import (
    GrippablePoseExpectation,
    PhaseContract,
    PhaseInvariant,
    PhaseState,
    QaccSentinel,
    TaskPhase,
    WeldHoldInvariant,
)

PhaseBoundary = Literal["start", "end"]


def make_run_id() -> str:
    """Filesystem-safe UTC run identifier."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def make_run_directory(root: Path, *, scene_name: str, run_id: str | None = None) -> Path:
    """Create the canonical artifact directory for one demo/debug run."""
    actual_run_id = run_id or make_run_id()
    run_dir = root / scene_name / actual_run_id
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass(frozen=True)
class ContractFailure:
    """One concrete mismatch at a phase boundary."""

    kind: str
    name: str
    expected: str | float
    observed: str | float
    message: str


@dataclass(frozen=True)
class ContractCheckReport:
    """Result of evaluating a declared `PhaseState` against MuJoCo data."""

    ok: bool
    failures: tuple[ContractFailure, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": [asdict(failure) for failure in self.failures],
        }


class RunArtifactWriter:
    """Append-only writer for one simulation run's artifacts."""

    def __init__(self, run_dir: Path, *, scene_name: str) -> None:
        self.run_dir = run_dir
        self.scene_name = scene_name
        self.events_path = run_dir / "events.jsonl"

    def write_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "scene": self.scene_name,
            "event_type": event_type,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(event, sort_keys=True) + "\n")

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_phase_contracts(self, phase_contracts: tuple[PhaseContract, ...]) -> Path:
        return self.write_json(
            "phase_contracts.json",
            [phase_contract_to_json_dict(contract) for contract in phase_contracts],
        )

    def write_summary(self, payload: dict[str, Any]) -> Path:
        return self.write_json("summary.json", payload)

    def write_snapshot(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        phase: TaskPhase,
        boundary: PhaseBoundary,
        phase_state: PhaseState,
        contract_report: ContractCheckReport,
    ) -> Path:
        stem = f"{phase.value}_{boundary}"
        snapshot_path = self.run_dir / "snapshots" / f"{stem}.npz"
        np.savez_compressed(
            snapshot_path,
            qpos=np.asarray(data.qpos, dtype=float).copy(),
            qvel=np.asarray(data.qvel, dtype=float).copy(),
            ctrl=np.asarray(data.ctrl, dtype=float).copy(),
            eq_active=np.asarray(data.eq_active, dtype=np.uint8).copy(),
            eq_data=np.asarray(model.eq_data, dtype=float).copy(),
            time=np.asarray([float(data.time)], dtype=float),
        )
        metadata = {
            "scene": self.scene_name,
            "phase": phase.value,
            "boundary": boundary,
            "time_s": float(data.time),
            "description": phase_state.description,
            "expected_state": phase_state_to_json_dict(phase_state),
            "contract": contract_report.to_json_dict(),
            "snapshot": str(snapshot_path),
        }
        self.write_json(f"snapshots/{stem}.json", metadata)
        self.write_event(
            "phase_snapshot",
            phase=phase.value,
            boundary=boundary,
            time_s=float(data.time),
            snapshot=str(snapshot_path),
            contract_ok=contract_report.ok,
            failures=[asdict(failure) for failure in contract_report.failures],
        )
        return snapshot_path


def phase_contract_to_json_dict(contract: PhaseContract) -> dict[str, Any]:
    return {
        "phase": contract.phase.value,
        "starts": phase_state_to_json_dict(contract.starts),
        "ends": phase_state_to_json_dict(contract.ends),
        "invariants": [phase_invariant_to_json_dict(inv) for inv in contract.invariants],
    }


def phase_state_to_json_dict(state: PhaseState) -> dict[str, Any]:
    return {
        "description": state.description,
        "active_attachments": [str(name) for name in state.active_attachments],
        "inactive_attachments": [str(name) for name in state.inactive_attachments],
        "base_aux": [{"name": str(name), "value": value} for name, value in state.base_aux],
        "expected_grippable_poses": [
            {
                "name": pose.name,
                "position": list(pose.position),
                "tolerance_m": pose.tolerance_m,
            }
            for pose in state.expected_grippable_poses
        ],
    }


def phase_invariant_to_json_dict(invariant: PhaseInvariant) -> dict[str, Any]:
    match invariant:
        case QaccSentinel():
            return {"kind": "qacc_sentinel", "max_increase": invariant.max_increase}
        case WeldHoldInvariant():
            return {
                "kind": "weld_hold",
                "name": invariant.name,
                "must_be_active": invariant.must_be_active,
            }


def check_phase_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase_state: PhaseState,
    *,
    base_tolerance: float = 1e-3,
) -> ContractCheckReport:
    """Evaluate a declarative phase boundary against current MuJoCo state."""
    failures: list[ContractFailure] = []
    failures.extend(
        _check_attachment_expectations(
            model,
            data,
            phase_state.active_attachments,
            expected_active=True,
        )
    )
    failures.extend(
        _check_attachment_expectations(
            model,
            data,
            phase_state.inactive_attachments,
            expected_active=False,
        )
    )
    failures.extend(_check_base_aux_expectations(model, data, phase_state, base_tolerance))
    failures.extend(
        _check_grippable_pose_expectations(model, data, phase_state.expected_grippable_poses)
    )
    return ContractCheckReport(ok=not failures, failures=tuple(failures))


@dataclass(frozen=True)
class _InvariantBaseline:
    """Phase-start state needed to evaluate cumulative invariants at end.

    QACC warnings are cumulative across the run, so the only meaningful
    metric is *delta from phase start*. This snapshot captures both the
    cumulative counters and the per-tick sample state used by tick-rate
    invariants (current weld status).
    """

    qacc_warning_count_at_start: int


def capture_invariant_baseline(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> _InvariantBaseline:
    _ = model  # unused now; kept for symmetry with checker signatures
    """Snapshot the baseline state needed to evaluate phase invariants.

    Call once at phase start; pass the result back into
    `check_phase_invariants` at phase end (or every tick for live
    sampling).
    """
    return _InvariantBaseline(
        qacc_warning_count_at_start=int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number),
    )


def check_phase_invariants(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    invariants: tuple[PhaseInvariant, ...],
    baseline: _InvariantBaseline,
) -> ContractCheckReport:
    """Evaluate every invariant against current MuJoCo state vs. baseline.

    Cheap enough to call every tick — the contained checks are O(N
    invariants) with small N. The runner samples per-tick when in
    --strict mode; offline tools call once at phase end.
    """
    failures: list[ContractFailure] = []
    for invariant in invariants:
        match invariant:
            case QaccSentinel():
                failures.extend(_check_qacc_sentinel(data, invariant, baseline))
            case WeldHoldInvariant():
                failures.extend(_check_weld_hold(model, data, invariant))
    return ContractCheckReport(ok=not failures, failures=tuple(failures))


def restore_snapshot(model: mujoco.MjModel, data: mujoco.MjData, snapshot_path: Path) -> None:
    """Restore a snapshot saved by `RunArtifactWriter.write_snapshot`."""
    with np.load(snapshot_path) as snapshot:
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.ctrl[:] = snapshot["ctrl"]
        data.eq_active[:] = snapshot["eq_active"]
        model.eq_data[:] = snapshot["eq_data"]
        data.time = float(snapshot["time"][0])
    mujoco.mj_forward(model, data)


def _check_attachment_expectations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    attachment_names: tuple[str, ...],
    *,
    expected_active: bool,
) -> tuple[ContractFailure, ...]:
    failures: list[ContractFailure] = []
    for attachment_name in attachment_names:
        eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, str(attachment_name)))
        if eq_id < 0:
            failures.append(
                ContractFailure(
                    kind="attachment",
                    name=str(attachment_name),
                    expected="present",
                    observed="missing",
                    message=f"attachment equality {attachment_name!s} does not exist",
                )
            )
            continue
        observed_active = bool(data.eq_active[eq_id])
        if observed_active != expected_active:
            failures.append(
                ContractFailure(
                    kind="attachment",
                    name=str(attachment_name),
                    expected="active" if expected_active else "inactive",
                    observed="active" if observed_active else "inactive",
                    message=(
                        f"attachment {attachment_name!s} expected "
                        f"{'active' if expected_active else 'inactive'}"
                    ),
                )
            )
    return tuple(failures)


def _check_base_aux_expectations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase_state: PhaseState,
    base_tolerance: float,
) -> tuple[ContractFailure, ...]:
    failures: list[ContractFailure] = []
    for joint_or_actuator_name, expected_value in phase_state.base_aux:
        qpos_address = _resolve_joint_qpos_address(model, str(joint_or_actuator_name))
        if qpos_address is None:
            failures.append(
                ContractFailure(
                    kind="base_aux",
                    name=str(joint_or_actuator_name),
                    expected=float(expected_value),
                    observed="missing",
                    message=f"base joint/actuator {joint_or_actuator_name!s} does not exist",
                )
            )
            continue
        observed_value = float(data.qpos[qpos_address])
        if abs(observed_value - float(expected_value)) > base_tolerance:
            failures.append(
                ContractFailure(
                    kind="base_aux",
                    name=str(joint_or_actuator_name),
                    expected=float(expected_value),
                    observed=observed_value,
                    message=(
                        f"base value {joint_or_actuator_name!s} expected "
                        f"{float(expected_value):+.4f}, observed {observed_value:+.4f}"
                    ),
                )
            )
    return tuple(failures)


def _resolve_joint_qpos_address(model: mujoco.MjModel, name: str) -> int | None:
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if joint_id < 0:
        actuator_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        if actuator_id < 0:
            return None
        joint_id = int(model.actuator_trnid[actuator_id][0])
    return int(model.jnt_qposadr[joint_id])


def _check_grippable_pose_expectations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    expectations: tuple[GrippablePoseExpectation, ...],
) -> tuple[ContractFailure, ...]:
    failures: list[ContractFailure] = []
    for expected in expectations:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, expected.name))
        if body_id < 0:
            failures.append(
                ContractFailure(
                    kind="grippable_pose",
                    name=expected.name,
                    expected="present",
                    observed="missing",
                    message=f"grippable body {expected.name!r} does not exist in model",
                )
            )
            continue
        observed_xyz = np.asarray(data.xpos[body_id], dtype=float)
        expected_xyz = np.asarray(expected.position, dtype=float)
        delta = float(np.linalg.norm(observed_xyz - expected_xyz))
        if delta > expected.tolerance_m:
            failures.append(
                ContractFailure(
                    kind="grippable_pose",
                    name=expected.name,
                    expected=f"{tuple(round(v, 4) for v in expected_xyz)}",
                    observed=f"{tuple(round(v, 4) for v in observed_xyz)}",
                    message=(
                        f"grippable {expected.name!r} drifted {delta * 1000:.1f} mm "
                        f"from expected pose (tol {expected.tolerance_m * 1000:.0f} mm)"
                    ),
                )
            )
    return tuple(failures)


def _check_qacc_sentinel(
    data: mujoco.MjData,
    sentinel: QaccSentinel,
    baseline: _InvariantBaseline,
) -> tuple[ContractFailure, ...]:
    """QACC sentinel: phase introduced no more than `max_increase` BADQACC warnings.

    `mjData.warning[mjWARN_BADQACC].number` is cumulative across the
    sim run, so the only meaningful comparison is current minus
    baseline. A non-zero increase typically means the integrator
    blew up — the demo will visibly twitch/explode soon after.
    """
    current = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    delta = current - baseline.qacc_warning_count_at_start
    if delta <= sentinel.max_increase:
        return ()
    return (
        ContractFailure(
            kind="qacc_sentinel",
            name="mjWARN_BADQACC",
            expected=float(sentinel.max_increase),
            observed=float(delta),
            message=(
                f"QACC warning count grew by {delta} during phase "
                f"(allowed {sentinel.max_increase}); physics likely unstable"
            ),
        ),
    )


def _check_weld_hold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    invariant: WeldHoldInvariant,
) -> tuple[ContractFailure, ...]:
    eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, invariant.name))
    if eq_id < 0:
        return (
            ContractFailure(
                kind="weld_hold",
                name=invariant.name,
                expected="present",
                observed="missing",
                message=f"weld {invariant.name!r} does not exist",
            ),
        )
    observed_active = bool(data.eq_active[eq_id])
    if observed_active == invariant.must_be_active:
        return ()
    return (
        ContractFailure(
            kind="weld_hold",
            name=invariant.name,
            expected="active" if invariant.must_be_active else "inactive",
            observed="active" if observed_active else "inactive",
            message=(
                f"weld {invariant.name!r} flickered to "
                f"{'active' if observed_active else 'inactive'} "
                f"mid-phase (must remain {'active' if invariant.must_be_active else 'inactive'})"
            ),
        ),
    )
