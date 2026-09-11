#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)


XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

IK_CONFIG = (
    ROOT
    / "general_motion_retargeting/ik_configs"
    / "smplx_to_linglong20_rootlock_v1.json"
)

SMPLX_FILE = (
    ROOT
    / "output/openloong_v2_kneeguide"
    / "stream_demo"
    / "gmr_smplx_results.npz"
)

BODY_MODEL_DIR = (
    ROOT
    / "assets/body_models"
)

OUT_DIR = (
    ROOT
    / "output/linglong20_body_v2_rootlock"
)

OUT_CSV = (
    OUT_DIR
    / "linglong20_body_v2_rootlock_qpos37.csv"
)

EXPECTED_FRAMES = 661
OUTPUT_FPS = 30.0


def get_body_id(model, name):
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(
            f"Body not found: {name}"
        )

    return bid


def body_geom_ids(model, body_name):
    bid = get_body_id(
        model,
        body_name,
    )

    ids = np.where(
        np.asarray(model.geom_bodyid) == bid
    )[0]

    if len(ids) == 0:
        raise RuntimeError(
            f"No geoms attached to {body_name}"
        )

    return ids


def geom_bottom_z(model, data, gid):
    gtype = int(model.geom_type[gid])

    pos = np.asarray(
        data.geom_xpos[gid],
        dtype=np.float64,
    )

    rot = np.asarray(
        data.geom_xmat[gid],
        dtype=np.float64,
    ).reshape(3, 3)

    size = np.asarray(
        model.geom_size[gid],
        dtype=np.float64,
    )

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(
            model.geom_dataid[gid]
        )

        if mesh_id < 0:
            return float(pos[2])

        adr = int(
            model.mesh_vertadr[mesh_id]
        )
        num = int(
            model.mesh_vertnum[mesh_id]
        )

        verts = np.asarray(
            model.mesh_vert[
                adr:adr + num
            ],
            dtype=np.float64,
        )

        # world_z = R[2,:] @ local_vertex + pos_z
        z = (
            verts @ rot[2, :]
            + pos[2]
        )

        return float(
            np.min(z)
        )

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        vertical_extent = float(
            np.sum(
                np.abs(rot[2, :])
                * size
            )
        )
        return float(
            pos[2] - vertical_extent
        )

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(
            pos[2] - size[0]
        )

    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = float(
            np.sqrt(
                np.sum(
                    (
                        rot[2, :]
                        * size
                    ) ** 2
                )
            )
        )
        return float(
            pos[2] - extent
        )

    if gtype in (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    ):
        radius = float(size[0])
        half_length = float(size[1])

        axis_vertical = abs(
            float(rot[2, 2])
        )

        radial_vertical = float(
            np.sqrt(
                rot[2, 0] ** 2
                + rot[2, 1] ** 2
            )
        )

        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = (
                axis_vertical
                * half_length
                + radius
            )
        else:
            extent = (
                axis_vertical
                * half_length
                + radial_vertical
                * radius
            )

        return float(
            pos[2] - extent
        )

    return float(pos[2])


def foot_bottoms(
    model,
    data,
    qpos,
    left_geoms,
    right_geoms,
):
    data.qpos[:] = qpos

    mujoco.mj_forward(
        model,
        data,
    )

    left = min(
        geom_bottom_z(
            model,
            data,
            int(gid),
        )
        for gid in left_geoms
    )

    right = min(
        geom_bottom_z(
            model,
            data,
            int(gid),
        )
        for gid in right_geoms
    )

    return (
        float(left),
        float(right),
    )


def validate_joint_limits(
    model,
    qpos,
):
    violations = []

    for jid in range(1, model.njnt):
        if not bool(
            model.jnt_limited[jid]
        ):
            continue

        qadr = int(
            model.jnt_qposadr[jid]
        )

        lo, hi = model.jnt_range[jid]

        vals = qpos[:, qadr]

        if (
            np.min(vals) < lo - 1e-6
            or np.max(vals) > hi + 1e-6
        ):
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )

            violations.append(
                (
                    name,
                    float(np.min(vals)),
                    float(np.max(vals)),
                    float(lo),
                    float(hi),
                )
            )

    return violations


def main():
    for p in (
        XML,
        IK_CONFIG,
        SMPLX_FILE,
        BODY_MODEL_DIR,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("LINGLONG 2.0 BODY V2 ROOTLOCK RETARGET")
    print("=" * 90)

    print("robot :", XML)
    print("config:", IK_CONFIG)
    print("smplx :", SMPLX_FILE)
    print("output:", OUT_CSV)

    # Temporary registration.
    IK_CONFIG_DICT["smplx"]["linglong2"] = (
        IK_CONFIG
    )

    (
        smplx_data,
        body_model,
        smplx_output,
        human_height,
    ) = load_smplx_file(
        str(SMPLX_FILE),
        str(BODY_MODEL_DIR),
        coord_fix="auto",
    )

    (
        human_frames,
        source_metadata_fps,
    ) = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    print()
    print(
        "estimated human height:",
        float(human_height),
    )
    print(
        "human frames          :",
        len(human_frames),
    )
    print(
        "NPZ metadata fps      :",
        source_metadata_fps,
    )
    print(
        "chosen output fps     :",
        OUTPUT_FPS,
    )

    if len(human_frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} frames, "
            f"got {len(human_frames)}"
        )

    gmr = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="linglong2",
        robot_path=str(XML),
        actual_human_height=float(
            human_height
        ),
        verbose=False,
    )

    raw_qpos = []
    errors = []

    print()
    print("=" * 90)
    print("RETARGET")
    print("=" * 90)

    for i, frame in enumerate(
        human_frames
    ):
        q, err = gmr.retarget(
            frame,
            offset_to_ground=False,
        )

        q = np.asarray(
            q,
            dtype=np.float64,
        )

        if q.shape != (37,):
            raise RuntimeError(
                f"frame {i}: "
                f"qpos shape={q.shape}"
            )

        if not np.all(
            np.isfinite(q)
        ):
            raise RuntimeError(
                f"frame {i}: "
                "non-finite qpos"
            )

        raw_qpos.append(q)
        errors.append(err)

        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == EXPECTED_FRAMES - 1
        ):
            print(
                f"frame {i:03d} | "
                f"root_z={q[2]: .4f} | "
                f"err1={err[0]:.4f} | "
                f"err2={err[1]:.4f}"
            )

    raw_qpos = np.stack(
        raw_qpos,
        axis=0,
    )

    # ----------------------------------------------------------
    # CONSTANT ground normalization.
    #
    # We estimate the floor from the REAL foot meshes during the
    # first 30 frames, then add ONE constant root-z offset to the
    # entire motion.
    #
    # This preserves all relative vertical motion.
    # ----------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )
    data = mujoco.MjData(model)

    left_geoms = body_geom_ids(
        model,
        "left_ankle_roll_link",
    )

    right_geoms = body_geom_ids(
        model,
        "right_ankle_roll_link",
    )

    print()
    print("=" * 90)
    print("FOOT GEOMETRY")
    print("=" * 90)

    print(
        "left foot geom ids :",
        left_geoms.tolist(),
    )
    print(
        "right foot geom ids:",
        right_geoms.tolist(),
    )

    reference_count = min(
        30,
        len(raw_qpos),
    )

    reference_bottoms = []

    for i in range(
        reference_count
    ):
        left_z, right_z = (
            foot_bottoms(
                model,
                data,
                raw_qpos[i],
                left_geoms,
                right_geoms,
            )
        )

        reference_bottoms.append(
            min(left_z, right_z)
        )

    reference_bottoms = np.asarray(
        reference_bottoms,
        dtype=np.float64,
    )

    estimated_floor = float(
        np.median(
            reference_bottoms
        )
    )

    ground_z_offset = (
        -estimated_floor
    )

    qpos = raw_qpos.copy()

    # ONE constant shift only.
    qpos[:, 2] += ground_z_offset

    print()
    print("=" * 90)
    print("GROUND NORMALIZATION")
    print("=" * 90)

    print(
        "reference frames    :",
        reference_count,
    )
    print(
        "pre-ground median z :",
        estimated_floor,
    )
    print(
        "constant root offset:",
        ground_z_offset,
    )

    # Full-sequence foot check after the constant shift.
    full_lowest = []
    left_all = []
    right_all = []

    for q in qpos:
        left_z, right_z = (
            foot_bottoms(
                model,
                data,
                q,
                left_geoms,
                right_geoms,
            )
        )

        left_all.append(left_z)
        right_all.append(right_z)

        full_lowest.append(
            min(left_z, right_z)
        )

    left_all = np.asarray(left_all)
    right_all = np.asarray(right_all)
    full_lowest = np.asarray(full_lowest)

    quat_norm = np.linalg.norm(
        qpos[:, 3:7],
        axis=1,
    )

    max_joint_step = float(
        np.max(
            np.abs(
                np.diff(
                    qpos[:, 7:],
                    axis=0,
                )
            )
        )
    )

    violations = (
        validate_joint_limits(
            model,
            qpos,
        )
    )

    np.savetxt(
        OUT_CSV,
        qpos,
        delimiter=",",
        fmt="%.9f",
    )

    print()
    print("=" * 90)
    print("FINAL VALIDATION")
    print("=" * 90)

    print(
        "shape             :",
        qpos.shape,
    )

    print(
        "finite            :",
        bool(
            np.all(
                np.isfinite(qpos)
            )
        ),
    )

    print(
        "quat norm min/max :",
        float(quat_norm.min()),
        float(quat_norm.max()),
    )

    print(
        "root z min/max    :",
        float(qpos[:, 2].min()),
        float(qpos[:, 2].max()),
    )

    print(
        "left foot z range :",
        float(left_all.min()),
        float(left_all.max()),
    )

    print(
        "right foot z range:",
        float(right_all.min()),
        float(right_all.max()),
    )

    print(
        "lowest foot range :",
        float(full_lowest.min()),
        float(full_lowest.max()),
    )

    print(
        "first30 median    :",
        float(
            np.median(
                full_lowest[
                    :reference_count
                ]
            )
        ),
    )

    print(
        "max joint step    :",
        max_joint_step,
        "rad/frame",
    )

    print(
        "joint violations  :",
        len(violations),
    )

    for item in violations:
        print(
            "  ",
            item,
        )

    print()
    print(
        "saved:",
        OUT_CSV,
    )

    print(
        "duration @30 FPS:",
        qpos.shape[0]
        / OUTPUT_FPS,
        "sec",
    )

    assert qpos.shape == (
        EXPECTED_FRAMES,
        37,
    )

    assert np.all(
        np.isfinite(qpos)
    )

    assert np.max(
        np.abs(
            quat_norm - 1.0
        )
    ) < 1e-5

    assert len(violations) == 0

    print()
    print(
        "LINGLONG20 BODY V1 FULL: PASS"
    )


if __name__ == "__main__":
    main()
