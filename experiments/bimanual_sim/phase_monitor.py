"""Runtime enforcement of `PhaseContract`s declared by scenes.

`tools/mj.py contracts` evaluates phase boundaries in batch mode. The guard
here is the *online* counterpart: it watches a live runner's task-plan
execution, fires `check_phase_state` at each phase transition, and samples
per-tick invariants in between. `--strict` raises on the first failure;
otherwise failures are collected so the user can keep watching and inspect
the report after.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco

from scene_base import PhaseContract, TaskPhase
from tools.observability import (
    ContractCheckReport,
    ContractFailure,
    capture_invariant_baseline,
    check_phase_invariants,
    check_phase_state,
)
from tools.observability import (
    _InvariantBaseline as InvariantBaseline,
)

# Physics ticks between invariant samples. 10 is plenty for catching QACC
# blow-ups (a typical divergence inflates the warning counter within a few
# ticks); 5 keeps headroom without measurable overhead.
DEFAULT_INVARIANT_SAMPLE_EVERY: int = 5


@dataclass(frozen=True)
class GuardEvent:
    """One contract-evaluation event recorded by the guard.

    Field shape mirrors `RunArtifactWriter.write_event` so the runner can
    stream events to disk directly when an artifact directory is configured.
    """

    kind: str  # "phase_start" | "phase_end" | "invariant_sample"
    phase: TaskPhase
    sim_time_s: float
    report: ContractCheckReport


class PhaseRuntimeMonitor:
    """Tracks the active task phase and evaluates contracts in-line.

    Constructing with an empty `contracts` tuple short-circuits every method,
    so scenes with no contracts pay zero per-tick cost.
    """

    def __init__(
        self,
        contracts: tuple[PhaseContract, ...],
        *,
        strict: bool = False,
        invariant_sample_every: int = DEFAULT_INVARIANT_SAMPLE_EVERY,
        on_failure: Callable[[ContractFailure], None] | None = None,
    ) -> None:
        self._contracts_by_phase: dict[TaskPhase, PhaseContract] = {
            contract.phase: contract for contract in contracts
        }
        self._strict = strict
        self._sample_every = max(1, invariant_sample_every)
        self._on_failure = on_failure
        self._current_phase: TaskPhase | None = None
        self._invariant_baseline: InvariantBaseline | None = None
        self._tick_counter: int = 0
        self.events: list[GuardEvent] = []
        self.failures: list[ContractFailure] = []

    @property
    def enabled(self) -> bool:
        return bool(self._contracts_by_phase)

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def current_phase(self) -> TaskPhase | None:
        return self._current_phase

    def on_phase_observed(
        self,
        phase: TaskPhase | None,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        """Called by the runner when the scene has a single active phase.

        `None` and `TaskPhase.UNPHASED` continue whatever phase was last
        active. The runner derives this from the full bimanual timeline so a
        fast arm cannot advance the scene-wide contract boundary before the
        slower arm has crossed into the same phase.
        """
        if not self.enabled:
            return
        if phase is None or phase is TaskPhase.UNPHASED:
            return
        new_phase = phase
        if new_phase == self._current_phase:
            return
        self._check_predecessor(new_phase, data)
        if self._current_phase is not None:
            self._evaluate_end(self._current_phase, model, data)
        self._evaluate_start(new_phase, model, data)
        self._current_phase = new_phase
        self._tick_counter = 0

    def _check_predecessor(self, new_phase: TaskPhase, data: mujoco.MjData) -> None:
        """Fail-fast check that `_current_phase → new_phase` is a transition the
        scene's contract declares. No constraint is recorded if the incoming
        phase has `legal_predecessors=()`, so legacy scenes are unaffected."""
        contract = self._contracts_by_phase.get(new_phase)
        if contract is None or not contract.legal_predecessors:
            return
        actual = self._current_phase
        if actual is not None and actual in contract.legal_predecessors:
            return
        actual_label = actual.value if actual is not None else "(initial)"
        expected_label = ", ".join(p.value for p in contract.legal_predecessors)
        failure = ContractFailure(
            kind="phase_predecessor",
            name=f"{actual_label}→{new_phase.value}",
            expected=expected_label,
            observed=actual_label,
            message=(
                f"phase {new_phase.value!r} entered from {actual_label!r}; "
                f"declared legal predecessors: [{expected_label}]"
            ),
        )
        self._record(
            GuardEvent(
                kind="phase_predecessor",
                phase=new_phase,
                sim_time_s=float(data.time),
                report=ContractCheckReport(ok=False, failures=(failure,)),
            )
        )

    def on_tick(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        """Sample invariants periodically while a phase is active."""
        if not self.enabled or self._current_phase is None:
            return
        self._tick_counter += 1
        if self._tick_counter % self._sample_every != 0:
            return
        contract = self._contracts_by_phase.get(self._current_phase)
        if contract is None or not contract.invariants:
            return
        assert self._invariant_baseline is not None
        report = check_phase_invariants(model, data, contract.invariants, self._invariant_baseline)
        if not report.ok:
            self._record(
                GuardEvent(
                    kind="invariant_sample",
                    phase=self._current_phase,
                    sim_time_s=float(data.time),
                    report=report,
                )
            )

    def on_plan_finished(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        """Evaluate the end contract for whatever phase is still open."""
        if not self.enabled or self._current_phase is None:
            return
        self._evaluate_end(self._current_phase, model, data)
        self._current_phase = None
        self._invariant_baseline = None

    def reset(self) -> None:
        """Forget tracking state. Keep `events` / `failures` so post-run
        inspection has the full history — the caller clears them manually
        if a fresh log is wanted."""
        self._current_phase = None
        self._invariant_baseline = None
        self._tick_counter = 0

    def _evaluate_start(
        self,
        phase: TaskPhase,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        contract = self._contracts_by_phase.get(phase)
        if contract is None:
            self._invariant_baseline = capture_invariant_baseline(model, data)
            return
        report = check_phase_state(model, data, contract.starts)
        self._record(
            GuardEvent(
                kind="phase_start",
                phase=phase,
                sim_time_s=float(data.time),
                report=report,
            )
        )
        self._invariant_baseline = capture_invariant_baseline(model, data)

    def _evaluate_end(
        self,
        phase: TaskPhase,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        contract = self._contracts_by_phase.get(phase)
        if contract is None:
            return
        boundary_report = check_phase_state(model, data, contract.ends)
        self._record(
            GuardEvent(
                kind="phase_end",
                phase=phase,
                sim_time_s=float(data.time),
                report=boundary_report,
            )
        )
        if contract.invariants and self._invariant_baseline is not None:
            invariant_report = check_phase_invariants(
                model, data, contract.invariants, self._invariant_baseline
            )
            if not invariant_report.ok:
                self._record(
                    GuardEvent(
                        kind="invariant_sample",
                        phase=phase,
                        sim_time_s=float(data.time),
                        report=invariant_report,
                    )
                )

    def _record(self, event: GuardEvent) -> None:
        self.events.append(event)
        if event.report.ok:
            return
        for failure in event.report.failures:
            self.failures.append(failure)
            if self._on_failure is not None:
                self._on_failure(failure)
        if self._strict:
            failures_summary = "; ".join(failure.message for failure in event.report.failures)
            raise PhaseContractViolation(
                f"phase {event.phase.value} {event.kind} at "
                f"sim_t={event.sim_time_s:.3f}s: {failures_summary}"
            )


class PhaseContractViolation(RuntimeError):
    """Raised by `PhaseRuntimeMonitor` in strict mode when a contract fails."""
