"""Mobile ALOHA base loader — Stanford CAD + UR10e at ARM #3 and #4 positions.

Single-purpose loader: builds a `mjcf.RootElement` carrying

* The Stanford Mobile ALOHA body silhouette (chassis + central column +
  4 mount-post columns + dual top platforms), stripped of the original
  four ViperX arms. See `tools/strip_aloha_arms.py`.
* Three planar-mobility joints on `base_link` (slide x, slide y, hinge
  yaw) so the runner can puppet the base around the floor.
* Two attachment SITES at the *exact* xy coordinates where the original
  CAD's ARM #3 and ARM #4 (the rear pair, identified by the user via
  `tools/label_aloha_arms.py`) lived. The scene module attaches a
  UR10e + Robotiq-2F85 sub-MJCF at each site.

Coordinate convention
---------------------
The Stanford CAD's "front" — the side the arms POINT toward — is
along -y_cad. Our scene convention has the rack at world +x. We
rotate the mesh visual +π/2 about z, mapping CAD's -y direction to
world +x:

    CAD (x, y, z) → body (-y, x, z)

ARM #3 and #4 cluster centroids (from `tools/label_aloha_arms.py`):

    ARM #3 CAD (-294, -539, +1338) → body (+0.539, -0.294, +1.338)
    ARM #4 CAD (+295, -538, +1338) → body (+0.538, +0.295, +1.338)

These are the centroids of the *whole arm assembly* (mid-arm,
including the gripper). The shoulder yaw motor mounts at the same xy
but at z ≈ 1.0 m (top of the body's mount-post columns at z_cad =
996 mm). The UR10e base flange sits there.
"""

from __future__ import annotations

from dm_control import mjcf

from paths import MOBILE_ALOHA_STANFORD_BODY_STL

# --- Mesh -------------------------------------------------------------
# Stanford CAD bounds post-strip (mm): x [-435, +400], y [-516, +926.7],
# z [0, +1076]. Mesh is grounded at z=0 so base_link sits on the floor.
_MESH_SCALE = (0.001, 0.001, 0.001)  # mm → m

# Mesh quat: +π/2 about z. CAD's -y_cad (where ARMS #3 & #4 cantilever
# to) maps onto world +x (the rack direction).
# Quat (wxyz): cos(+π/4) at w, sin(+π/4) at z.
_BODY_ROTATION_QUAT = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]

# --- Arm-mount sites = ARM #3 / #4 spots, pulled back onto the body --
# Originally I tried body x = +0.538 (the exact ARM #3, #4 CENTROID
# x in body frame). That puts the bases ~312 mm forward of the body's
# last visible structure (the back rail at body x ≈ +0.226), so they
# floated over the front edge of the platform. Pulled back to body
# x = +0.226 — sitting ON the back rail's top surface — so the bases
# rest on visible body geometry while keeping the y separation that
# matches the original arms.
#
# Y_ABS = +0.295 still matches |x_cad| of the ARM #3, #4 centroids
# so the arms read as "in #3 / #4's lateral spot".
# Z = +1.000 is just above the post tops (z_cad = 996 mm).
_ARM_MOUNT_X = 0.226
_ARM_MOUNT_Y_ABS = 0.295
_ARM_MOUNT_Z = 1.000

# --- Top-camera mount + stand ----------------------------------------
# Original Stanford CAD's central pole tops out at z = 1.076 m (post-
# strip). Camera at z = 1.150 had no visible support — appeared to
# float. Raise to z = 1.60 m for a wider field-of-view over the rack
# and add a vertical pole geom from the body top (z = 1.0) up to the
# camera mount, so the camera reads as "mounted on a stand".
# CAD central pole at x ≈ 0, y ≈ -76 → post +π/2 rotation:
# body x ≈ +0.076, y ≈ 0.
_TOP_CAM_X = 0.076
_TOP_CAM_Y = 0.0
_TOP_CAM_Z = 1.600

# Camera-stand pole geometry: thin cylinder rising from the body's top
# (z = 1.0) up to just below the camera mount (z = 1.58 — leaving a
# 20 mm gap so the D435i body is visible above it).
_STAND_BASE_Z = 1.000
_STAND_TOP_Z = _TOP_CAM_Z - 0.020
_STAND_RADIUS = 0.020


def load_mobile_aloha() -> mjcf.RootElement:
    """Return the Mobile ALOHA mobile-base `mjcf.RootElement`.

    The scene module attaches UR10e + 2F85 sub-MJCFs at
    `left_arm_mount` / `right_arm_mount` and a D435i at `top_cam_mount`.
    Three planar joints on `base_link` provide puppet-mode floor
    mobility (slide-x, slide-y, hinge-yaw).
    """
    root = mjcf.RootElement(model="aloha_base")

    # --- Mesh asset -------------------------------------------------
    root.asset.add(
        "mesh",
        name="aloha_body",
        file=str(MOBILE_ALOHA_STANFORD_BODY_STL),
        scale=list(_MESH_SCALE),
    )

    # --- Base body with planar mobility joints ----------------------
    base = root.worldbody.add("body", name="base_link", pos=[0.0, 0.0, 0.0])
    base.add(
        "joint",
        name="base_x",
        type="slide",
        axis=[1.0, 0.0, 0.0],
        damping=50.0,
        limited="false",
    )
    base.add(
        "joint",
        name="base_y",
        type="slide",
        axis=[0.0, 1.0, 0.0],
        damping=50.0,
        limited="false",
    )
    base.add(
        "joint",
        name="base_yaw",
        type="hinge",
        axis=[0.0, 0.0, 1.0],
        damping=20.0,
        limited="false",
    )

    # --- Body silhouette --------------------------------------------
    base.add(
        "geom",
        name="aloha_body",
        type="mesh",
        mesh="aloha_body",
        quat=_BODY_ROTATION_QUAT,
        rgba=[0.18, 0.20, 0.24, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.0,
    )

    # --- Arm mount sites (exact ARM #3 / #4 positions) --------------
    # left_arm_mount maps to ARM #4's body-frame y = +0.295 (CAD x =
    # +295 mm); right_arm_mount maps to ARM #3's y = -0.295 (CAD x =
    # -294 mm). Identity site quat: arm bases stand vertical with
    # shoulder pan around world +z.
    base.add(
        "site",
        name="left_arm_mount",
        pos=[_ARM_MOUNT_X, +_ARM_MOUNT_Y_ABS, _ARM_MOUNT_Z],
        quat=[1.0, 0.0, 0.0, 0.0],
        size=[0.001, 0.001, 0.001],
    )
    base.add(
        "site",
        name="right_arm_mount",
        pos=[_ARM_MOUNT_X, -_ARM_MOUNT_Y_ABS, _ARM_MOUNT_Z],
        quat=[1.0, 0.0, 0.0, 0.0],
        size=[0.001, 0.001, 0.001],
    )

    # --- Top-camera stand (visible vertical pole) -------------------
    # Cylinder geom from the body's top up to just below the camera
    # mount. Same dark-grey rgba as the body so it reads as part of
    # the robot's structure, not a foreign primitive.
    _stand_centre_z = (_STAND_BASE_Z + _STAND_TOP_Z) / 2.0
    _stand_half_height = (_STAND_TOP_Z - _STAND_BASE_Z) / 2.0
    base.add(
        "geom",
        name="top_cam_stand",
        type="cylinder",
        size=[_STAND_RADIUS, _stand_half_height],
        pos=[_TOP_CAM_X, _TOP_CAM_Y, _stand_centre_z],
        rgba=[0.18, 0.20, 0.24, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.0,
    )

    # --- Top-camera mount -------------------------------------------
    # 90°-x quat rotates the D435i mesh's local +z onto -y so the body
    # lies horizontal with its lens pointing +x (toward the rack).
    base.add(
        "site",
        name="top_cam_mount",
        pos=[_TOP_CAM_X, _TOP_CAM_Y, _TOP_CAM_Z],
        quat=[0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
        size=[0.001, 0.001, 0.001],
    )

    return root


# --- Aux-actuator names exposed for the scene module's DataCenterAux --
BASE_X_JOINT_NAME = "base_x"
BASE_Y_JOINT_NAME = "base_y"
BASE_YAW_JOINT_NAME = "base_yaw"

LEFT_ARM_MOUNT_SITE = "left_arm_mount"
RIGHT_ARM_MOUNT_SITE = "right_arm_mount"
TOP_CAM_MOUNT_SITE = "top_cam_mount"
