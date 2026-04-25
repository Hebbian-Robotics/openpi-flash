"""Universal Robots UR10e + Robotiq 2F-85 loader.

Composes Menagerie's `universal_robots_ur10e/ur10e.xml` with a
`robotiq_2f85/2f85.xml` mounted on the wrist's `attachment_site`.
Returns a `mjcf.RootElement` ready for the data-center scene to
attach at a torso mount site.

Conventions:

* The UR10e has 6 named revolute joints
  (`shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`,
  `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`) and one
  `<general>` actuator per joint.
* The 2F-85 is tendon-coupled with one driving actuator
  `fingers_actuator` (ctrlrange 0..255, where 0 = open and 255 =
  closed under the upstream tendon convention).
* After attaching with `model="left"` (or "right"), every UR/2F85
  joint, body, actuator, etc. gets the `left/`-prefixed compiled
  name (e.g. `left/shoulder_pan_joint`, `left/fingers_actuator`).

The earlier Piper loader's role — `load_piper(side)` — is preserved
in spirit by `load_ur10e_with_gripper(side)`: caller mutates the
returned root if needed (TCP site, wrist camera) then attaches via
`mount_site.attach(root)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from dm_control import mjcf

from arm_handles import ArmSide
from paths import ROBOTIQ_2F85_XML, UR10E_XML


@dataclass(frozen=True)
class UR10eConfig:
    """Customizations for the UR10e + 2F-85 composite.

    Kept minimal — Menagerie ships sane gain values for both pieces.
    Add fields here if a future scene needs to bump kp / forcerange /
    etc. (the data-center scene under puppet mode just needs the
    arm in the spec; runtime drives qpos directly).
    """

    # Default `<general>` joint actuator gain on the UR's `size4`,
    # `size3_limited`, and `size2` defaults are already set in
    # ur10e.xml; we don't override.
    pass


_UR10E_REQUIRED_BODIES: tuple[str, ...] = (
    "base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)
"""Bodies the scene's IK / weld code dereferences after attach. Loading
fails fast if Menagerie renames any of them."""

UR10E_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
"""6 arm joints in canonical chain order. `arm_handles.get_arm_handles`
uses this list to assemble the per-side joint id arrays."""

UR10E_ACTUATOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
"""Position actuators driving the 6 arm joints. Same order as
`UR10E_JOINT_NAMES`. Note: actuator names DROP the `_joint` suffix
(Menagerie convention)."""

GRIPPER_ACTUATOR_NAME = "fingers_actuator"
"""Single driving actuator on the 2F-85. ctrlrange 0..255 in the
upstream MJCF; 0 = fully open, 255 = fully closed."""

GRIPPER_DRIVER_JOINT_NAME = "right_driver_joint"
"""One of the 8 joints in the 2F-85 finger linkage. We read this
joint's `qpos` to inspect gripper state; the others are coupled via
tendons / equality constraints."""

# UR10e wrist-3 has an `attachment_site` already declared in the
# upstream MJCF — this is where the 2F-85 mounts.
_UR10E_TOOL_SITE_NAME = "attachment_site"


def _assert_menagerie_shape(root: mjcf.RootElement) -> None:
    """Fail at load time if upstream renamed something we depend on.

    Same defensive pattern as the Piper / TIAGo loaders: catch a
    Menagerie shape change here with an actionable message rather
    than a cryptic compile error 400 lines later.
    """
    for name in _UR10E_REQUIRED_BODIES:
        if root.find("body", name) is None:
            raise RuntimeError(
                f"UR10e upstream XML missing expected body {name!r}. "
                "Menagerie's universal_robots_ur10e/ur10e.xml may have "
                "changed shape — update robots/ur10e.py if so."
            )
    for jname in UR10E_JOINT_NAMES:
        if root.find("joint", jname) is None:
            raise RuntimeError(
                f"UR10e upstream XML missing expected joint {jname!r}."
            )
    if root.find("site", _UR10E_TOOL_SITE_NAME) is None:
        raise RuntimeError(
            f"UR10e upstream XML missing tool-mount site "
            f"{_UR10E_TOOL_SITE_NAME!r}; cannot attach 2F-85."
        )


def load_ur10e_with_gripper(
    side: ArmSide,
    config: UR10eConfig = UR10eConfig(),  # noqa: B008 — frozen dataclass
) -> mjcf.RootElement:
    """Load UR10e, attach a Robotiq 2F-85 to its wrist flange, and
    namespace the whole assembly with `side`'s value (e.g. `left/`).

    Returns the UR10e `mjcf.RootElement` so the caller can mutate it
    further (add a TCP site, wrist camera, etc.) BEFORE calling
    `parent_site.attach(root)`. This mirrors `load_piper`'s contract —
    any element added inside the subtree before attach inherits the
    namespace prefix.
    """
    del config  # currently unused; kept for parity with PiperConfig

    arm = mjcf.from_path(str(UR10E_XML))
    arm.model = side.rstrip("/")
    _assert_menagerie_shape(arm)

    # Attach the 2F-85 at the UR's `attachment_site`. dm_control's
    # site.attach() returns the attachment frame body — we don't need
    # the handle here; the gripper's own internal joints/tendons run
    # the parallel-jaw motion.
    gripper = mjcf.from_path(str(ROBOTIQ_2F85_XML))
    # Empty model so gripper bodies sit directly under `<side>/` rather
    # than `<side>/<gripper-model>/`. Matches a flat arm-with-tool
    # naming convention; if a future scene wants to address the
    # gripper subtree distinctly, set this to e.g. `gripper`.
    gripper.model = ""
    tool_site = arm.find("site", _UR10E_TOOL_SITE_NAME)
    tool_site.attach(gripper)

    return arm
