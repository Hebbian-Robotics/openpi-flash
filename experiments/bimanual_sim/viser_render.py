"""Bridge MuJoCo geometry to Viser scene handles.

`build_viser_scene` walks every geom in the compiled model once, creates a
matching Viser mesh handle at its current world pose, and returns handle
pairs for the subset that can *move* in later frames. `update_viser` then
pushes the current world pose of each moving geom — this is what animates
the browser-side render.

Static-vs-moving classification uses `model.body_weldid`: a body whose weld
root is the worldbody (id 0) is rigidly welded into the world frame and will
never change pose, so we skip it in the per-frame update and save the
websocket write plus the quaternion math. In the live server-swap scene this
drops ~⅔ of the per-frame update volume (chassis, rack, tower, pedestal
walls, bin walls, cable anchors are all static).

Matrix → quaternion conversion uses `mujoco.mju_mat2Quat` (C kernel) rather
than `viser.transforms.SO3.from_matrix`. The latter runs multiple
`np.allclose` scans per call to pick between four quaternion branches and,
with ~60 geoms x 125 Hz, dominated the runtime profile at ~45 % of wallclock.

Per-frame pose push bypasses viser's `MeshHandle.position` / `.wxyz`
property setters. The property setters each run `np.allclose(new, current)`
to suppress redundant websocket messages — useful for user-driven GUI
updates, expensive when we *know* every moving geom changed pose this
frame. We poke `handle._impl.{position,wxyz}[:]` in-place and enqueue the
`SetPosition` / `SetOrientation` messages ourselves, skipping the diff.
All enqueued messages are then sent as a single atomic websocket frame via
`server.atomic()`.

Shapes supported: BOX, SPHERE, CYLINDER, CAPSULE, ELLIPSOID, MESH. Planes
are skipped (we add a Viser grid instead since a flat plane looks awkward).
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser
from viser._messages import SetOrientationMessage, SetPositionMessage


def _geom_to_trimesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    gtype = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]

    match gtype:
        case mujoco.mjtGeom.mjGEOM_BOX:
            return trimesh.creation.box(extents=2 * size)
        case mujoco.mjtGeom.mjGEOM_SPHERE:
            return trimesh.creation.icosphere(radius=float(size[0]))
        case mujoco.mjtGeom.mjGEOM_CYLINDER:
            return trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
        case mujoco.mjtGeom.mjGEOM_CAPSULE:
            return trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
        case mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            m = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
            m.apply_scale(size)
            return m
        case mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[geom_id])
            vs = int(model.mesh_vertadr[mid])
            vn = int(model.mesh_vertnum[mid])
            fs = int(model.mesh_faceadr[mid])
            fn = int(model.mesh_facenum[mid])
            verts = np.asarray(model.mesh_vert[vs : vs + vn], dtype=np.float32).copy()
            faces = np.asarray(model.mesh_face[fs : fs + fn], dtype=np.int32).copy()
            return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return None


def _is_static_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    """A geom is static iff its body's weld-equivalence root is the worldbody.

    `model.body_weldid[body]` gives the id of the nearest ancestor body with
    at least one DoF, or 0 (worldbody) if the body is rigidly welded to the
    world frame. Bodies attached via a frame offset but no joint (typical of
    pedestals, rack structure, bin walls) inherit weldid == 0 and are
    correctly classified as static.
    """
    body_id = int(model.geom_bodyid[geom_id])
    return int(model.body_weldid[body_id]) == 0


def _scene_grid_extents(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    minimum_extent: float,
) -> tuple[float, float, tuple[float, float, float]]:
    """Compute (width, height, centre_xyz) for a viser floor grid that covers
    every non-plane geom in the model with `minimum_extent` as a floor and
    2 m margin past the outermost geom. Mesh geoms are treated as their
    `geom_rbound` enclosing sphere (loose but cheap and never wrong)."""
    min_xy = np.array([np.inf, np.inf], dtype=float)
    max_xy = np.array([-np.inf, -np.inf], dtype=float)
    for g in range(model.ngeom):
        gtype = int(model.geom_type[g])
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = np.asarray(data.geom_xpos[g], dtype=float)
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            r = float(model.geom_rbound[g])
            half_xy = np.array([r, r], dtype=float)
        else:
            size = np.asarray(model.geom_size[g], dtype=float)
            half_xy = np.array([float(np.max(size)), float(np.max(size))], dtype=float)
        min_xy = np.minimum(min_xy, pos[:2] - half_xy)
        max_xy = np.maximum(max_xy, pos[:2] + half_xy)
    if not np.all(np.isfinite(min_xy)):
        return minimum_extent, minimum_extent, (0.0, 0.0, 0.0)
    width = max(float(max_xy[0] - min_xy[0]) + 2.0, minimum_extent)
    height = max(float(max_xy[1] - min_xy[1]) + 2.0, minimum_extent)
    centre = (float((max_xy[0] + min_xy[0]) * 0.5), float((max_xy[1] + min_xy[1]) * 0.5), 0.0)
    return width, height, centre


def _viser_mesh_path(model: mujoco.MjModel, geom_id: int) -> str:
    """Stable scene-graph path for a geom's viser handle. Used by both
    initial publish and the recolour helper so the path matches across
    remove + re-add."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"g{geom_id}"
    return f"/geoms/{geom_id:04d}_{name.replace('/', '_')}"


def build_viser_scene(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ground_width: float = 3.0,
    ground_cell: float = 0.1,
) -> tuple[list[tuple[int, viser.MeshHandle]], dict[int, viser.MeshHandle]]:
    """Publish every geom in `model` to the Viser scene.

    Returns `(moving_handles, handle_by_geom_id)`:
      * `moving_handles` — `(geom_id, handle)` pairs for the geoms whose
        world pose can change across frames; consumed by `update_viser`.
        Static geoms are created with their current world pose baked in
        and never touched by the per-frame update.
      * `handle_by_geom_id` — full registry covering *every* geom (moving
        and static), keyed by geom id. Lets `update_geom_rgba` look up
        and rebuild a static geom's handle when its colour needs to flip
        at runtime (the per-frame moving list alone wouldn't know about
        static-geom handles).

    The caller must have initialised `data` to the scene's start state
    (typically via `scene.apply_initial_state`) before calling — the
    static geom poses read here are frozen at handle-creation time.
    """
    server.scene.set_up_direction("+z")

    mujoco.mj_forward(model, data)

    # Auto-fit the floor grid to the scene's geom XY footprint so multi-rack
    # scenes (e.g. the indicator-check aisle spanning X ≈ 0..6) don't render
    # with the grid cropped to a 3 x 3 m square at origin. `ground_width` is
    # now the *minimum* — if the scene's geom XY extent exceeds it we expand,
    # plus 2 m margin so the grid breathes past the last geom.
    grid_width, grid_height, grid_centre = _scene_grid_extents(model, data, ground_width)
    server.scene.add_grid(
        "/ground",
        width=grid_width,
        height=grid_height,
        cell_size=ground_cell,
        position=grid_centre,
        plane="xy",
    )

    quat_buf = np.empty(4, dtype=np.float64)
    moving_handles: list[tuple[int, viser.MeshHandle]] = []
    handle_by_geom_id: dict[int, viser.MeshHandle] = {}

    for g in range(model.ngeom):
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        mesh = _geom_to_trimesh(model, g)
        if mesh is None:
            continue
        rgba = model.geom_rgba[g]
        color: tuple[float, float, float] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

        pos = data.geom_xpos[g]
        # geom_xmat rows are length-9 flat, which is what mju_mat2Quat wants.
        mujoco.mju_mat2Quat(quat_buf, data.geom_xmat[g])

        h = server.scene.add_mesh_simple(
            _viser_mesh_path(model, g),
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            color=color,
            opacity=float(rgba[3]),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            wxyz=(
                float(quat_buf[0]),
                float(quat_buf[1]),
                float(quat_buf[2]),
                float(quat_buf[3]),
            ),
        )
        handle_by_geom_id[g] = h
        if not _is_static_geom(model, g):
            moving_handles.append((g, h))
    return moving_handles, handle_by_geom_id


def update_geom_rgba(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    moving_handles: list[tuple[int, viser.MeshHandle]],
    handle_by_geom_id: dict[int, viser.MeshHandle],
    geom_name: str,
    rgba: tuple[float, float, float, float],
) -> None:
    """Recolour one geom at runtime. Writes `model.geom_rgba` and rebuilds
    the corresponding viser handle (viser bakes colour into the mesh at
    creation, so we remove the old handle and re-add with the new RGBA).

    No-ops cleanly if the geom doesn't exist or wasn't published — the
    runtime path will hit a missing-name long before viser would, since
    the scene's `Step.set_geom_rgba` references geoms declared in its
    layout module by stable name.
    """
    geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    if geom_id < 0:
        return
    old_handle = handle_by_geom_id.get(geom_id)
    if old_handle is None:
        return
    model.geom_rgba[geom_id] = rgba
    mesh = _geom_to_trimesh(model, geom_id)
    if mesh is None:
        return
    pos = data.geom_xpos[geom_id]
    quat_buf = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_buf, data.geom_xmat[geom_id])
    old_handle.remove()
    new_handle = server.scene.add_mesh_simple(
        _viser_mesh_path(model, geom_id),
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        color=(float(rgba[0]), float(rgba[1]), float(rgba[2])),
        opacity=float(rgba[3]),
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        wxyz=(
            float(quat_buf[0]),
            float(quat_buf[1]),
            float(quat_buf[2]),
            float(quat_buf[3]),
        ),
    )
    handle_by_geom_id[geom_id] = new_handle
    # If the geom was also in the moving list, swap it in place so
    # update_viser keeps pushing pose to the live handle, not the removed one.
    for i, (gid, _) in enumerate(moving_handles):
        if gid == geom_id:
            moving_handles[i] = (geom_id, new_handle)
            break


def update_viser(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: list[tuple[int, viser.MeshHandle]],
) -> None:
    """Push world pose of every moving geom to its Viser handle.

    Three layers of batching vs. the naive `h.position = ...` /
    `h.wxyz = ...`:

    1. `server.atomic()` groups every enqueued message into one
       client-side atomic update, halving the websocket framing overhead
       and guaranteeing the browser sees all geoms move synchronously.
       It also suppresses the per-`push` `call_soon_threadsafe` signal —
       the atomic-end issues a single cross-thread wake instead of one
       per message.
    2. Bypass the per-handle `position` / `wxyz` property setters. The
       setters each run `np.allclose(new, current)` to suppress redundant
       sends — right default for GUI-driven updates, pure overhead for
       physics-driven updates that change every tick (cost ~44 % of
       wallclock before this change). We update `_impl.{position,wxyz}[:]`
       in-place so viser's cached state stays coherent with the wire.
    3. Bypass `queue_message`: cache the broadcast buffer's `push` once
       and call it directly. `queue_message` layers a record-handle loop
       (we don't record) and a `get_message_buffer()` call on top of
       `buffer.push`; it showed ~14 % inclusive overhead over `push`
       alone in profiling. All handles share the same `WebsockServer`,
       so the cached `buffer_push` is valid for every geom.
    """
    if not handles:
        return

    # All handles from the same `ViserServer` share the broadcast buffer.
    # `wsi.get_message_buffer()` returns `WebsockServer._broadcast_buffer`;
    # `server.atomic()` starts/ends an atomic block on that same buffer.
    wsi = handles[0][1]._impl.api._websock_interface
    buffer_push = wsi.get_message_buffer().push

    # Local-bind hot-path symbols. Inside a ~60-iteration loop at 60 Hz this
    # turns LOAD_GLOBAL/LOAD_METHOD into LOAD_FAST — measurable on profiled
    # workloads, free to apply.
    set_pos_msg = SetPositionMessage
    set_ori_msg = SetOrientationMessage
    mat2quat = mujoco.mju_mat2Quat
    geom_xpos = data.geom_xpos
    geom_xmat = data.geom_xmat

    quat_buf = np.empty(4, dtype=np.float64)
    with server.atomic():
        for g, h in handles:
            pos = geom_xpos[g]
            mat2quat(quat_buf, geom_xmat[g])
            pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
            wxyz_tuple = (
                float(quat_buf[0]),
                float(quat_buf[1]),
                float(quat_buf[2]),
                float(quat_buf[3]),
            )
            impl = h._impl
            impl.position[:] = pos_tuple
            impl.wxyz[:] = wxyz_tuple
            name = impl.name
            buffer_push(set_pos_msg(name, pos_tuple))
            buffer_push(set_ori_msg(name, wxyz_tuple))
