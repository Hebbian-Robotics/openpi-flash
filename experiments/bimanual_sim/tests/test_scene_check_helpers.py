"""Property tests for scene_check geometry helpers.

`_aabb_overlap` is one line but it's the kind of one-liner that asymmetry
bugs hide in (mismatched `<` vs `<=`, eps applied to one side only).
Hypothesis sweeps random pairs and confirms the algebraic property.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scene_check import _aabb_overlap, _GeomBox, _point_in_aabb


def _box(min_xyz: tuple[float, float, float], max_xyz: tuple[float, float, float]) -> _GeomBox:
    """Construct a `_GeomBox` from two corners. The geom_id / body_id /
    name / body_name fields are unused by the predicates we test."""
    return _GeomBox(
        geom_id=0,
        name="t",
        body_id=0,
        body_name="t",
        min_xyz=np.asarray(min_xyz, dtype=float),
        max_xyz=np.asarray(max_xyz, dtype=float),
    )


@st.composite
def aabbs(draw: st.DrawFn) -> _GeomBox:
    """Random axis-aligned box with non-degenerate extent. Half-extent is
    bounded so floats stay well-conditioned."""
    cx = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    cy = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    cz = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    hx = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    hy = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    hz = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    return _box((cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))


@given(a=aabbs(), b=aabbs())
def test_aabb_overlap_is_symmetric(a: _GeomBox, b: _GeomBox) -> None:
    """Catches asymmetric eps handling — if the predicate ever uses `<` on
    one side and `<=` on the other, two boxes that just touch at a face
    would disagree depending on argument order."""
    assert _aabb_overlap(a, b) == _aabb_overlap(b, a)


def test_aabb_overlap_disjoint_boxes_return_false() -> None:
    a = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = _box((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
    assert not _aabb_overlap(a, b)


def test_aabb_overlap_self_returns_true() -> None:
    a = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert _aabb_overlap(a, a)


def test_aabb_overlap_touching_faces_does_not_count() -> None:
    """Faces that touch at exactly a plane shouldn't register as overlap —
    the predicate uses strict inequality with eps. Important for the static-
    overlap allow-list: bodies sharing edge planes (rack panels at seams)
    aren't flagged as colliding."""
    a = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    assert not _aabb_overlap(a, b)


@given(box=aabbs())
def test_box_centre_is_inside_self(box: _GeomBox) -> None:
    """The centre of any AABB with positive extent must be inside it."""
    centre = (box.min_xyz + box.max_xyz) * 0.5
    assert _point_in_aabb(centre, box)


def test_point_outside_aabb() -> None:
    box = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert not _point_in_aabb(np.array([2.0, 0.5, 0.5]), box)


@pytest.mark.parametrize(
    "point",
    [
        (1.0, 0.5, 0.5),  # on +x face
        (0.0, 0.5, 0.5),  # on -x face
        (0.5, 1.0, 0.5),  # on +y face
    ],
)
def test_point_on_face_is_not_strictly_inside(point: tuple[float, float, float]) -> None:
    """Strict containment: faces / edges aren't "inside" — the TCP-clip
    check would otherwise false-fire on grippers parked at a static-geom
    surface."""
    box = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert not _point_in_aabb(np.asarray(point, dtype=float), box)
