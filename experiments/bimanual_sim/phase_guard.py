"""Runtime enforcement of `PhaseContract`s declared by scenes.

`tools/mj.py contracts` already evaluates phase boundaries in batch
mode. The guard here is the *online* counterpart: it watches a live
runner's task-plan execution, fires `check_phase_state` at every
phase transition, and samples per-tick invariants in between. In
`--strict` mode the first failure raises; in non-strict mode failures
are collected so the user can keep watching the demo and inspect the
report after.

Wired in by `runner.py`:
  guard = PhaseRuntimeGuard(scene.PHASE_CONTRACTS, strict=args.strict)
  ...
  for each step:
      guard.on_step_started(step, model, data)
  for each physics tick:
      guard.on_tick(model, data)
  on plan completion:
      guard.on_plan_finished(model, data)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco

from scene_base import PhaseContract, Step, TaskPhase
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

# Number of physics ticks between invariant samples. 1 (every tick) is
# the most paranoid; 10 is plenty for catching QACC blow-ups (a typical
# divergence inflates the warning counter within a few ticks). Higher
# values reduce overhead in tight runtime loops.
DEFAULT_INVARIANT_SAMPLE_EVERY: int = 5


@dataclass(frozen=True)
class GuardEvent:
    """One contract-evaluation event recorded by the guard.

    These are appended to the guard's internal log so callers can read
    a full audit trail after the run. The fields mirror what
    `RunArtifactWriter.write_event` accepts so the runner can stream
    events to disk directly when an artifact directory is configured.
    """

    kind: str  # "phase_start" | "phase_end" | "invariant_sample"
    phase: TaskPhase
    sim_time_s: float
    report: ContractCheckReport


class PhaseRuntimeGuard:
    """Tracks the active task phase and evaluates contracts in-line.

    The runner calls into the guard at three event types:

    * `on_step_started(step, model, data)` — every time a new
      `Step` becomes the active waypoint. Detects phase transitions
      and triggers boundary checks.
    * `on_tick(model, data)` — every physics tick. Samples invariants
      according to `invariant_sample_every`.
    * `on_plan_finished(model, data)` — once the full plan completes
      so the *end* boundary of the final phase is evaluated.

    The guard is intentionally cheap when no contracts are declared —
    constructing with an empty tuple short-circuits every method.
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

    def on_step_started(
        self,
        step: Step,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        """Called by the runner when a new `Step` becomes active.

        Phase transitions detected here trigger the *end* check of
        the outgoing phase and the *start* check of the incoming
        phase, plus an invariant baseline capture for the new phase.
        Steps tagged `TaskPhase.UNPHASED` are treated as continuing
        whatever phase was last active — the contracts only assert
        what scenes have actually declared.
        """
        if not self.enabled:
            return
        if step.phase is TaskPhase.UNPHASED:
            return
        new_phase = step.phase
        if new_phase == self._current_phase:
            return
        # Outgoing phase: evaluate the end contract + final invariant pass.
        if self._current_phase is not None:
            self._evaluate_end(self._current_phase, model, data)
        # Incoming phase: evaluate start contract, capture baseline.
        self._evaluate_start(new_phase, model, data)
        self._current_phase = new_phase
        self._tick_counter = 0

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
        """Forget all tracking state — call when the user hits Reset."""
        self._current_phase = None
        self._invariant_baseline = None
        self._tick_counter = 0
        # Keep self.events / self.failures so post-run inspection has
        # the full history; the caller can clear them manually if a
        # fresh log is wanted.

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
    """Raised by `PhaseRuntimeGuard` in strict mode when a contract fails."""
