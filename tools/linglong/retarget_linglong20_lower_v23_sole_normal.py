#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

V21_CSV = (
    ROOT
    / "output/linglong20_lower_v21"
    / "linglong20_lower_v21_qpos37.csv"
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
    / "output/linglong20_lower_v23"
)

OUTPUT_CSV = (
    OUT_DIR
    / "linglong20_lower_v23_qpos37.csv"
)


EXPECTED_FRAMES = 661


LEFT_ANKLE = [
    11,  # pitch
    12,  # roll
]

RIGHT_ANKLE = [
    17,  # pitch
    18,  # roll
]


NORMAL_WEIGHT = 1.0

# Small preference for temporal continuity.
# This is not a hard rate clamp.
CONTINUITY_WEIGHT = 0.01


def normalize(v):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(v)
    )

    if norm < 1e-10:
        return None

    return v / norm


def angle_deg(a, b):

    a = normalize(a)
    b = normalize(b)

    dot = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(dot)
        )
    )


def quat_wxyz_matrix(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()


def anatomical_basis(frame):

    pelvis = np.asarray(
        frame["pelvis"][0],
        dtype=np.float64,
    )

    spine = np.asarray(
        frame["spine3"][0],
        dtype=np.float64,
    )

    lh = np.asarray(
        frame["left_hip"][0],
        dtype=np.float64,
    )

    rh = np.asarray(
        frame["right_hip"][0],
        dtype=np.float64,
    )

    # +Y = human anatomical left.
    y_axis = normalize(
        lh - rh
    )

    z0 = normalize(
        spine - pelvis
    )

    z_axis = normalize(
        z0
        - np.dot(
            z0,
            y_axis,
        )
        * y_axis
    )

    # Y x Z = X forward
    x_axis = normalize(
        np.cross(
            y_axis,
            z_axis,
        )
    )

    y_axis = normalize(
        np.cross(
            z_axis,
            x_axis,
        )
    )

    basis = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    return (
        pelvis,
        basis,
    )


def raw_foot_normal(
    frame,
    side,
):
    """
    SMPL-X diagnosis showed:

        foot local +Y

    is the most reliable foot-up / sole-normal axis.

    Return it in human anatomical coordinates:
        X forward
        Y left
        Z up
    """

    _, basis = (
        anatomical_basis(frame)
    )

    rot_world = (
        quat_wxyz_matrix(
            frame[
                f"{side}_foot"
            ][1]
        )
    )

    normal_world = (
        rot_world
        @ np.array(
            [
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )
    )

    normal_local = (
        basis.T
        @ normal_world
    )

    return normalize(
        normal_local
    )


def build_normal_sequence(
    frames,
    side,
):
    """
    Build a continuous *directed* sole-normal signal.

    Important:
    do not independently force +Z every frame.

    Doing that can create an artificial 180-degree flip near
    horizontal orientations.

    Instead:
      1. orient frame 0 upward
      2. each later frame chooses the hemisphere closest to
         the previous frame
    """

    normals = []

    previous = None

    for i, frame in enumerate(
        frames
    ):

        n = raw_foot_normal(
            frame,
            side,
        )

        if n is None:
            if previous is None:
                n = np.array(
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                )
            else:
                n = previous.copy()

        if previous is None:

            # Initial sole normal should face upward.
            if n[2] < 0.0:
                n = -n

        else:

            # Hemisphere continuity.
            if np.dot(
                n,
                previous,
            ) < 0.0:
                n = -n

        normals.append(
            n
        )

        previous = (
            n.copy()
        )

    return np.asarray(
        normals,
        dtype=np.float64,
    )


def body_id(
    model,
    name,
):

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


def limits_for_qpos(
    model,
    indices,
):

    lower = []
    upper = []

    for qidx in indices:

        found = False

        for jid in range(
            model.njnt
        ):

            if int(
                model.jnt_qposadr[jid]
            ) != qidx:
                continue

            found = True

            lo, hi = (
                model.jnt_range[jid]
            )

            lower.append(
                float(lo)
            )

            upper.append(
                float(hi)
            )

            break

        if not found:
            raise RuntimeError(
                f"No joint at qpos {qidx}"
            )

    return (
        np.asarray(
            lower,
            dtype=np.float64,
        ),
        np.asarray(
            upper,
            dtype=np.float64,
        ),
    )


def make_fk(
    model,
    foot_body_name,
    ankle_indices,
):

    data = mujoco.MjData(
        model
    )

    base_id = body_id(
        model,
        "base_link",
    )

    foot_id = body_id(
        model,
        foot_body_name,
    )

    def fk(
        q_base,
        ankle_values,
    ):

        data.qpos[:] = q_base

        for idx, value in zip(
            ankle_indices,
            ankle_values,
        ):
            data.qpos[idx] = value

        mujoco.mj_forward(
            model,
            data,
        )

        base_rot = np.asarray(
            data.xmat[
                base_id
            ],
            dtype=np.float64,
        ).reshape(
            3,
            3,
        )

        foot_rot = np.asarray(
            data.xmat[
                foot_id
            ],
            dtype=np.float64,
        ).reshape(
            3,
            3,
        )

        # LingLong foot link +Z = foot-up normal.
        foot_up = (
            base_rot.T
            @ foot_rot[:, 2]
        )

        # Diagnostic only:
        # LingLong foot link +X = forward.
        foot_forward = (
            base_rot.T
            @ foot_rot[:, 0]
        )

        return (
            normalize(
                foot_up
            ),
            normalize(
                foot_forward
            ),
        )

    return fk


def solve_ankle(
    fk,
    q_base,
    target_normal,
    previous,
    lower,
    upper,
):

    def residual(x):

        robot_normal, _ = fk(
            q_base,
            x,
        )

        return np.concatenate(
            (
                NORMAL_WEIGHT
                * (
                    robot_normal
                    - target_normal
                ),

                CONTINUITY_WEIGHT
                * (
                    x
                    - previous
                ),
            )
        )

    seed = np.clip(
        previous,
        lower + 1e-8,
        upper - 1e-8,
    )

    result = least_squares(
        residual,
        seed,

        bounds=(
            lower,
            upper,
        ),

        method="trf",

        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,

        max_nfev=80,
    )

    robot_normal, robot_forward = (
        fk(
            q_base,
            result.x,
        )
    )

    normal_error = angle_deg(
        robot_normal,
        target_normal,
    )

    return (
        result.x,
        normal_error,
        robot_normal,
        robot_forward,
    )


def target_toe_direction(
    frame,
    side,
):
    """
    Diagnostic only.

    Do NOT use this to drive ankle pitch/roll.
    """

    _, basis = (
        anatomical_basis(frame)
    )

    ankle = np.asarray(
        frame[
            f"{side}_ankle"
        ][0],
        dtype=np.float64,
    )

    foot = np.asarray(
        frame[
            f"{side}_foot"
        ][0],
        dtype=np.float64,
    )

    d_world = (
        foot - ankle
    )

    d = (
        basis.T
        @ d_world
    )

    return normalize(
        d
    )


def geom_bottom_z(
    model,
    data,
    gid,
):

    gtype = int(
        model.geom_type[gid]
    )

    pos = np.asarray(
        data.geom_xpos[gid],
        dtype=np.float64,
    )

    rot = np.asarray(
        data.geom_xmat[gid],
        dtype=np.float64,
    ).reshape(
        3,
        3,
    )

    size = np.asarray(
        model.geom_size[gid],
        dtype=np.float64,
    )

    if (
        gtype
        == mujoco.mjtGeom.mjGEOM_MESH
    ):

        mesh_id = int(
            model.geom_dataid[gid]
        )

        adr = int(
            model.mesh_vertadr[
                mesh_id
            ]
        )

        num = int(
            model.mesh_vertnum[
                mesh_id
            ]
        )

        verts = np.asarray(
            model.mesh_vert[
                adr:adr + num
            ],
            dtype=np.float64,
        )

        z = (
            verts
            @ rot[2, :]
            + pos[2]
        )

        return float(
            np.min(z)
        )

    if (
        gtype
        == mujoco.mjtGeom.mjGEOM_BOX
    ):

        extent = float(
            np.sum(
                np.abs(
                    rot[2, :]
                )
                * size
            )
        )

        return float(
            pos[2]
            - extent
        )

    if (
        gtype
        == mujoco.mjtGeom.mjGEOM_SPHERE
    ):

        return float(
            pos[2]
            - size[0]
        )

    return float(
        pos[2]
    )


def foot_geom_ids(
    model,
    body_name,
):

    bid = body_id(
        model,
        body_name,
    )

    ids = np.where(
        np.asarray(
            model.geom_bodyid
        )
        == bid
    )[0]

    if len(ids) == 0:
        raise RuntimeError(
            f"No geoms on {body_name}"
        )

    return ids


def lowest_z(
    model,
    data,
    q,
    geom_ids,
):

    data.qpos[:] = q

    mujoco.mj_forward(
        model,
        data,
    )

    return min(
        geom_bottom_z(
            model,
            data,
            int(gid),
        )
        for gid in geom_ids
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    q21 = np.loadtxt(
        V21_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert q21.shape == (
        EXPECTED_FRAMES,
        37,
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

    frames, metadata_fps = (
        get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=30,
        )
    )

    assert len(frames) == (
        EXPECTED_FRAMES
    )


    left_target_normal = (
        build_normal_sequence(
            frames,
            "left",
        )
    )

    right_target_normal = (
        build_normal_sequence(
            frames,
            "right",
        )
    )


    # --------------------------------------------------------
    # Target continuity diagnostic BEFORE solving robot.
    # --------------------------------------------------------

    left_target_step = [
        angle_deg(
            left_target_normal[i - 1],
            left_target_normal[i],
        )
        for i in range(
            1,
            EXPECTED_FRAMES,
        )
    ]

    right_target_step = [
        angle_deg(
            right_target_normal[i - 1],
            right_target_normal[i],
        )
        for i in range(
            1,
            EXPECTED_FRAMES,
        )
    ]


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )


    left_fk = make_fk(
        model,
        "left_ankle_roll_link",
        LEFT_ANKLE,
    )

    right_fk = make_fk(
        model,
        "right_ankle_roll_link",
        RIGHT_ANKLE,
    )


    left_lo, left_hi = (
        limits_for_qpos(
            model,
            LEFT_ANKLE,
        )
    )

    right_lo, right_hi = (
        limits_for_qpos(
            model,
            RIGHT_ANKLE,
        )
    )


    refined = (
        q21.copy()
    )


    # Explicitly do not reuse V1 ankle seed.
    left_prev = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )

    right_prev = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )


    left_normal_error = []
    right_normal_error = []

    left_forward_error = []
    right_forward_error = []


    print("=" * 100)
    print("LINGLONG LOWER RETARGET V2.3")
    print("SOLE-NORMAL -> ANKLE PITCH / ROLL")
    print("=" * 100)

    print(
        "frames:",
        EXPECTED_FRAMES,
    )

    print(
        "metadata fps:",
        metadata_fps,
    )

    print()

    print(
        "left target-normal max step:",
        np.max(
            left_target_step
        ),
        "deg",
    )

    print(
        "right target-normal max step:",
        np.max(
            right_target_step
        ),
        "deg",
    )


    for i, frame in enumerate(
        frames
    ):

        (
            left_solution,
            left_err,
            _,
            left_forward,
        ) = solve_ankle(
            left_fk,

            refined[i],

            left_target_normal[i],

            left_prev,

            left_lo,
            left_hi,
        )


        (
            right_solution,
            right_err,
            _,
            right_forward,
        ) = solve_ankle(
            right_fk,

            refined[i],

            right_target_normal[i],

            right_prev,

            right_lo,
            right_hi,
        )


        refined[
            i,
            LEFT_ANKLE,
        ] = left_solution

        refined[
            i,
            RIGHT_ANKLE,
        ] = right_solution


        left_prev = (
            left_solution.copy()
        )

        right_prev = (
            right_solution.copy()
        )


        left_normal_error.append(
            left_err
        )

        right_normal_error.append(
            right_err
        )


        # Toe direction is now diagnostic only.
        left_toe = (
            target_toe_direction(
                frame,
                "left",
            )
        )

        right_toe = (
            target_toe_direction(
                frame,
                "right",
            )
        )

        left_forward_error.append(
            angle_deg(
                left_forward,
                left_toe,
            )
        )

        right_forward_error.append(
            angle_deg(
                right_forward,
                right_toe,
            )
        )


        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == 660
        ):

            print(
                f"frame {i:03d} | "
                f"L normal={left_err:6.2f}° "
                f"R normal={right_err:6.2f}°"
            )


    # --------------------------------------------------------
    # Recompute ONE constant ground offset.
    # --------------------------------------------------------

    data = mujoco.MjData(
        model
    )

    left_geoms = (
        foot_geom_ids(
            model,
            "left_ankle_roll_link",
        )
    )

    right_geoms = (
        foot_geom_ids(
            model,
            "right_ankle_roll_link",
        )
    )

    bottoms = []

    for i in range(30):

        lz = lowest_z(
            model,
            data,
            refined[i],
            left_geoms,
        )

        rz = lowest_z(
            model,
            data,
            refined[i],
            right_geoms,
        )

        bottoms.append(
            min(
                lz,
                rz,
            )
        )

    ground_shift = (
        -float(
            np.median(
                bottoms
            )
        )
    )

    refined[:, 2] += (
        ground_shift
    )


    left_normal_error = np.asarray(
        left_normal_error
    )

    right_normal_error = np.asarray(
        right_normal_error
    )

    left_forward_error = np.asarray(
        left_forward_error
    )

    right_forward_error = np.asarray(
        right_forward_error
    )


    np.savetxt(
        OUTPUT_CSV,
        refined,
        delimiter=",",
        fmt="%.9f",
    )


    print()
    print("=" * 100)
    print("SOLE NORMAL FIT")
    print("=" * 100)

    print(
        "left mean/median/p95/max:",
        left_normal_error.mean(),
        np.median(
            left_normal_error
        ),
        np.percentile(
            left_normal_error,
            95,
        ),
        left_normal_error.max(),
    )

    print(
        "right mean/median/p95/max:",
        right_normal_error.mean(),
        np.median(
            right_normal_error
        ),
        np.percentile(
            right_normal_error,
            95,
        ),
        right_normal_error.max(),
    )


    print()
    print("=" * 100)
    print("TOE DIRECTION DIAGNOSTIC ONLY")
    print("=" * 100)

    print(
        "left toe error "
        "mean/median/p95:",
        left_forward_error.mean(),
        np.median(
            left_forward_error
        ),
        np.percentile(
            left_forward_error,
            95,
        ),
    )

    print(
        "right toe error "
        "mean/median/p95:",
        right_forward_error.mean(),
        np.median(
            right_forward_error
        ),
        np.percentile(
            right_forward_error,
            95,
        ),
    )


    print()
    print("=" * 100)
    print("ANKLE RANGE / CONTINUITY")
    print("=" * 100)

    checks = [
        (
            "left ankle pitch",
            11,
        ),
        (
            "left ankle roll",
            12,
        ),
        (
            "right ankle pitch",
            17,
        ),
        (
            "right ankle roll",
            18,
        ),
    ]

    for name, idx in checks:

        x = np.degrees(
            refined[:, idx]
        )

        dx = np.degrees(
            np.diff(
                refined[:, idx]
            )
        )

        print(
            f"{name:20s}: "
            f"{x.min():8.2f} .. "
            f"{x.max():8.2f} deg | "
            f"max step="
            f"{np.max(np.abs(dx)):6.2f} deg"
        )


    print()
    print("LIMIT COUNTS")

    print(
        "left pitch lower:",
        np.sum(
            np.degrees(
                refined[:, 11]
            ) < -45.9
        ),
    )

    print(
        "left pitch upper:",
        np.sum(
            np.degrees(
                refined[:, 11]
            ) > 34.4
        ),
    )

    print(
        "left roll lower:",
        np.sum(
            np.degrees(
                refined[:, 12]
            ) < -29.9
        ),
    )

    print(
        "left roll upper:",
        np.sum(
            np.degrees(
                refined[:, 12]
            ) > 29.9
        ),
    )

    print(
        "right pitch lower:",
        np.sum(
            np.degrees(
                refined[:, 17]
            ) < -45.9
        ),
    )

    print(
        "right pitch upper:",
        np.sum(
            np.degrees(
                refined[:, 17]
            ) > 34.4
        ),
    )

    print(
        "right roll lower:",
        np.sum(
            np.degrees(
                refined[:, 18]
            ) < -29.9
        ),
    )

    print(
        "right roll upper:",
        np.sum(
            np.degrees(
                refined[:, 18]
            ) > 29.9
        ),
    )


    print()

    print(
        "constant root-z shift:",
        ground_shift,
    )

    print(
        "shape:",
        refined.shape,
    )

    print(
        "finite:",
        np.all(
            np.isfinite(
                refined
            )
        ),
    )

    print(
        "saved:",
        OUTPUT_CSV,
    )


    assert refined.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(
            refined
        )
    )


    print()

    print(
        "LINGLONG LOWER RETARGET V2.3: PASS"
    )


if __name__ == "__main__":
    main()
