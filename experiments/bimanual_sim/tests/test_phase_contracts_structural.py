"""Structural assertions over the data_center scene's phase contracts.

These are pytest cases (not hypothesis) because the input space is fixed: there
is exactly one `PHASE_CONTRACTS` tuple in the scene, and we want to assert
properties of *that* tuple. Generated inputs add nothing here.

Skips gracefully when `mujoco_menagerie` isn't installed locally — the
data_center scene's `paths.py` validates menagerie XMLs at import.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Probe the menagerie path before importing `scenes.data_center`. The scene
# accesses lazy paths constants at import (UR10E_XML, ROBOTIQ_2F85_XML,
# D435I_XML, D405_MESH_STL via robots.ur10e and the direct paths import),
# any of which would raise FileNotFoundError if menagerie is missing. We
# probe the UR10e XML specifically because it's the asset most distinctive
# to the live scene (Piper/TIAGo are legacy and aren't on this code path).
# Skipping at the path-probe level keeps ty's flow analysis happy; a
# `try: from ... except FileNotFoundError: skip()` pattern leaves the
# imported name "possibly unbound" to the type checker.
_MENAGERIE = Path(os.environ.get("MENAGERIE_PATH") or (Path.home() / "mujoco_menagerie"))
if not (_MENAGERIE / "universal_robots_ur10e" / "ur10e.xml").is_file():
    pytest.skip(f"menagerie not found at {_MENAGERIE}", allow_module_level=True)

from scene_base import PhaseContract, TaskPhase  # noqa: E402  (path-probe gate above)
from scenes import data_center  # noqa: E402

PHASE_CONTRACTS: tuple[PhaseContract, ...] = data_center.PHASE_CONTRACTS


def test_every_phase_has_at_most_one_contract() -> None:
    phases = [c.phase for c in PHASE_CONTRACTS]
    assert len(phases) == len(set(phases)), (
        f"duplicate phase contracts: {[p for p in phases if phases.count(p) > 1]}"
    )


def test_legal_predecessors_reference_known_phases() -> None:
    declared_phases = {c.phase for c in PHASE_CONTRACTS}
    for contract in PHASE_CONTRACTS:
        for prev in contract.legal_predecessors:
            assert prev in declared_phases, (
                f"{contract.phase.value} declares predecessor {prev.value!r} "
                f"but no contract for that phase exists"
            )


def test_legal_predecessors_form_a_dag() -> None:
    """Topological sort of the predecessor graph must succeed (no cycles)."""
    contracts_by_phase = {c.phase: c for c in PHASE_CONTRACTS}
    visited: set[TaskPhase] = set()
    visiting: set[TaskPhase] = set()

    def visit(phase: TaskPhase) -> None:
        if phase in visited:
            return
        assert phase not in visiting, f"cycle through {phase.value}"
        visiting.add(phase)
        contract = contracts_by_phase.get(phase)
        if contract is not None:
            for prev in contract.legal_predecessors:
                visit(prev)
        visiting.remove(phase)
        visited.add(phase)

    for contract in PHASE_CONTRACTS:
        visit(contract.phase)


def test_exactly_one_initial_phase() -> None:
    """An "initial phase" is one with no declared predecessors. Multiple
    initial phases would mean the choreography has multiple legal entry
    points, which the current scene doesn't support."""
    initial = [c.phase for c in PHASE_CONTRACTS if not c.legal_predecessors]
    assert len(initial) == 1, f"expected exactly one initial phase, got {initial}"
    assert initial[0] is TaskPhase.SETUP


def test_active_inactive_attachments_disjoint_at_starts() -> None:
    for contract in PHASE_CONTRACTS:
        active = set(contract.starts.active_attachments)
        inactive = set(contract.starts.inactive_attachments)
        overlap = active & inactive
        assert not overlap, (
            f"{contract.phase.value}.starts: attachment(s) declared both "
            f"active and inactive: {overlap}"
        )


def test_active_inactive_attachments_disjoint_at_ends() -> None:
    for contract in PHASE_CONTRACTS:
        active = set(contract.ends.active_attachments)
        inactive = set(contract.ends.inactive_attachments)
        overlap = active & inactive
        assert not overlap, (
            f"{contract.phase.value}.ends: attachment(s) declared both "
            f"active and inactive: {overlap}"
        )


def test_predecessor_chain_is_total_for_data_center() -> None:
    """The data_center scene declares a strict total order: each non-initial
    phase has exactly one predecessor. This isn't required by the contract
    type, but it's what the current choreography uses; if the test starts
    failing it means someone introduced a branch and should update this test
    to reflect the new graph shape."""
    for contract in PHASE_CONTRACTS:
        if contract.phase is TaskPhase.SETUP:
            assert contract.legal_predecessors == ()
            continue
        assert len(contract.legal_predecessors) == 1, (
            f"{contract.phase.value} has {len(contract.legal_predecessors)} predecessors; "
            f"data_center expects exactly one"
        )


def test_predecessor_chain_reaches_setup_from_every_phase() -> None:
    """Walk backward from each phase through `legal_predecessors`; must reach
    SETUP. Catches phases that are unreachable from the initial state."""
    contracts_by_phase = {c.phase: c for c in PHASE_CONTRACTS}
    for contract in PHASE_CONTRACTS:
        seen: set[TaskPhase] = set()
        current: TaskPhase | None = contract.phase
        while current is not None and current not in seen:
            seen.add(current)
            preds = contracts_by_phase[current].legal_predecessors
            current = preds[0] if preds else None
        assert TaskPhase.SETUP in seen, (
            f"{contract.phase.value} not reachable from SETUP via predecessor chain"
        )
