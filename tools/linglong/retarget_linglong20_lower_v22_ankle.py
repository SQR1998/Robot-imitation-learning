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
    / "output/linglong20_lower_v22"
)

OUTPUT_CSV = (
    OUT_DIR
    / "linglong20_lower_v22_qpos37.csv"
)


EXPECTED_FRAMES = 661

LATERAL_GAIN = 0.57


LEFT_ANKLE = [
    11,  # ankle pitch
    12,  # ankle roll
]

RIGHT_ANKLE = [
    17,  # ankle pitch
    18,  # ankle roll
]


# Foot forward is more important than exact foot-normal fit.
FORWARD_WEIGHT = 1.00

UP_WEIGHT = 0.75

# Small temporal preference only.
CONTINUITY_WEIGHT = 0.015


def normalize(v):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-9:
        return None

    return v / n


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

    # +Y = anatomical left
    y_axis = normalize(
        lh - rh
    )

    z0 = normalize(
        spine - pelvis
    )

    # Orthogonalize body up against left axis.
    z_axis = normalize(
        z0
        - np.dot(
            z0,
            y_axis,
        )
        * y_axis
    )

    # Y x Z = X
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

    return pelvis, basis


def quat_wxyz_matrix(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()


def mapped_forward(v):
    """
    Same lateral morphology mapping as Lower V2.
    """

    d = normalize(v)

    d = np.array(
        [
            d[0],
            LATERAL_GAIN * d[1],
            d[2],
        ],
        dtype=np.float64,
    )

    return normalize(d)


def build_foot_target(
    frame,
    side,
):
    """
    Construct desired robot-style foot frame:

      +X = toe / forward
      +Z = foot up normal

    Toe direction:
      human ankle -> foot positional vector

    Up:
      SMPL-X foot quaternion local +Y,
      verified empirically from the 661-frame sequence.
    """

    pelvis, basis = (
        anatomical_basis(frame)
    )

    ankle_world = np.asarray(
        frame[
            f"{side}_ankle"
        ][0],
        dtype=np.float64,
    )

    foot_world = np.asarray(
        frame[
            f"{side}_foot"
        ][0],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Target +X: geometric toe direction.
    # --------------------------------------------------------

    forward_world = (
        foot_world
        - ankle_world
    )

    forward_human = (
        basis.T
        @ forward_world
    )

    forward_target = (
        mapped_forward(
            forward_human
        )
    )


    # --------------------------------------------------------
    # Target +Z:
    #
    # Diagnosis showed SMPL-X foot local +Y is strongly aligned
    # with anatomical UP:
    #
    # left  median up ~= +0.942
    # right median up ~= +0.978
    # --------------------------------------------------------

    foot_rot_world = (
        quat_wxyz_matrix(
            frame[
                f"{side}_foot"
            ][1]
        )
    )

    smpl_up_world = (
        foot_rot_world
        @ np.array(
            [
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )
    )

    smpl_up_human = (
        basis.T
        @ smpl_up_world
    )

    # Remove the component parallel to the toe direction.
    # Robot +X and +Z must be orthogonal.
    up_target = (
        smpl_up_human
        - np.dot(
            smpl_up_human,
            forward_target,
        )
        * forward_target
    )

    up_target = normalize(
        up_target
    )

    if up_target is None:
        up_target = np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

    # Human is never expected to have the sole upside-down here.
    # This is only a sign consistency safeguard, not a clamp.
    if up_target[2] < 0.0:
        up_target = -up_target

    return (
        forward_target,
        up_target,
    )


def body_id(model, name):

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Body not found: {name}"
        )

    return idx


def limits_for_qpos(
    model,
    qpos_indices,
):

    lower = []
    upper = []

    for qidx in qpos_indices:

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
                f"No joint for qpos {qidx}"
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


def make_foot_fk(
    model,
    body_name,
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
        body_name,
    )

    def fk(
        base_qpos,
        ankle_values,
    ):

        # Keep the full V2.1 lower-body solution.
        data.qpos[:] = base_qpos

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

        # Robot foot link:
        #
        # +X = forward
        # +Z = up
        foot_forward = (
            base_rot.T
            @ foot_rot[:, 0]
        )

        foot_up = (
            base_rot.T
            @ foot_rot[:, 2]
        )

        return (
            normalize(
                foot_forward
            ),
            normalize(
                foot_up
            ),
        )

    return fk


def angle_deg(a, b):

    a = normalize(a)
    b = normalize(b)

    dot = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(dot)
        )
    )


def solve_ankle(
    fk,
    base_qpos,
    target_forward,
    target_up,
    previous,
    lower,
    upper,
):

    def residual(x):

        forward, up = fk(
            base_qpos,
            x,
        )

        return np.concatenate(
            (
                FORWARD_WEIGHT
                * (
                    forward
                    - target_forward
                ),

                UP_WEIGHT
                * (
                    up
                    - target_up
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

    forward, up = fk(
        base_qpos,
        result.x,
    )

    return (
        result.x,
        angle_deg(
            forward,
            target_forward,
        ),
        angle_deg(
            up,
            target_up,
        ),
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

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:

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

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:

        extent = float(
            np.sum(
                np.abs(
                    rot[2, :]
                )
                * size
            )
        )

        return float(
            pos[2] - extent
        )

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:

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
        ) == bid
    )[0]

    if len(ids) == 0:
        raise RuntimeError(
            f"No geoms on {body_name}"
        )

    return ids


def lowest_foot_z(
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

    (
        frames,
        metadata_fps,
    ) = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    assert len(frames) == (
        EXPECTED_FRAMES
    )


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )


    left_fk = make_foot_fk(
        model,
        "left_ankle_roll_link",
        LEFT_ANKLE,
    )

    right_fk = make_foot_fk(
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


    print("=" * 100)
    print("LINGLONG LOWER RETARGET V2.2")
    print("ANKLE PITCH / ROLL FROM HUMAN FOOT ORIENTATION")
    print("=" * 100)

    print(
        "frames:",
        len(frames),
    )

    print(
        "lateral gain:",
        LATERAL_GAIN,
    )

    print(
        "human height:",
        human_height,
    )

    print(
        "metadata fps:",
        metadata_fps,
    )

    print()

    print(
        "left ankle limits deg:",
        np.degrees(
            np.column_stack(
                (
                    left_lo,
                    left_hi,
                )
            )
        ),
    )

    print(
        "right ankle limits deg:",
        np.degrees(
            np.column_stack(
                (
                    right_lo,
                    right_hi,
                )
            )
        ),
    )


    refined = (
        q21.copy()
    )

    # Do NOT seed from old V1 ankle values.
    # Start neutral, then previous V2.2 solution.
    left_prev = np.array(
        [
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    right_prev = np.array(
        [
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


    left_forward_error = []
    left_up_error = []

    right_forward_error = []
    right_up_error = []


    for i, frame in enumerate(
        frames
    ):

        (
            lfwd,
            lup,
        ) = build_foot_target(
            frame,
            "left",
        )

        (
            rfwd,
            rup,
        ) = build_foot_target(
            frame,
            "right",
        )


        (
            left_solution,
            lfe,
            lue,
        ) = solve_ankle(
            left_fk,

            refined[i],

            lfwd,
            lup,

            left_prev,

            left_lo,
            left_hi,
        )


        (
            right_solution,
            rfe,
            rue,
        ) = solve_ankle(
            right_fk,

            refined[i],

            rfwd,
            rup,

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


        left_forward_error.append(
            lfe
        )

        left_up_error.append(
            lue
        )

        right_forward_error.append(
            rfe
        )

        right_up_error.append(
            rue
        )


        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == EXPECTED_FRAMES - 1
        ):

            print(
                f"frame {i:03d} | "
                f"LF={lfe:6.2f}° "
                f"LU={lue:6.2f}° | "
                f"RF={rfe:6.2f}° "
                f"RU={rue:6.2f}°"
            )


    # ========================================================
    # Since changing foot orientation changes the actual mesh
    # bottom, recompute ONE constant root-Z ground correction.
    #
    # This is still coordinate normalization, not per-frame
    # collision/contact fixing.
    # ========================================================

    ground_data = mujoco.MjData(
        model
    )

    left_geoms = foot_geom_ids(
        model,
        "left_ankle_roll_link",
    )

    right_geoms = foot_geom_ids(
        model,
        "right_ankle_roll_link",
    )


    first30_bottom = []

    for i in range(30):

        lz = lowest_foot_z(
            model,
            ground_data,
            refined[i],
            left_geoms,
        )

        rz = lowest_foot_z(
            model,
            ground_data,
            refined[i],
            right_geoms,
        )

        first30_bottom.append(
            min(
                lz,
                rz,
            )
        )


    ground_shift = (
        -float(
            np.median(
                first30_bottom
            )
        )
    )

    refined[:, 2] += (
        ground_shift
    )


    # ========================================================
    # Diagnostics
    # ========================================================

    left_forward_error = np.asarray(
        left_forward_error
    )

    left_up_error = np.asarray(
        left_up_error
    )

    right_forward_error = np.asarray(
        right_forward_error
    )

    right_up_error = np.asarray(
        right_up_error
    )


    np.savetxt(
        OUTPUT_CSV,
        refined,
        delimiter=",",
        fmt="%.9f",
    )


    print()
    print("=" * 100)
    print("FOOT ORIENTATION FIT")
    print("=" * 100)

    print(
        "left forward mean/median/p95/max:",
        left_forward_error.mean(),
        np.median(
            left_forward_error
        ),
        np.percentile(
            left_forward_error,
            95,
        ),
        left_forward_error.max(),
    )

    print(
        "left up mean/median/p95/max     :",
        left_up_error.mean(),
        np.median(
            left_up_error
        ),
        np.percentile(
            left_up_error,
            95,
        ),
        left_up_error.max(),
    )

    print()

    print(
        "right forward mean/median/p95/max:",
        right_forward_error.mean(),
        np.median(
            right_forward_error
        ),
        np.percentile(
            right_forward_error,
            95,
        ),
        right_forward_error.max(),
    )

    print(
        "right up mean/median/p95/max     :",
        right_up_error.mean(),
        np.median(
            right_up_error
        ),
        np.percentile(
            right_up_error,
            95,
        ),
        right_up_error.max(),
    )


    print()
    print("=" * 100)
    print("ANKLE JOINT RANGE / CONTINUITY")
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

    print(
        "left roll at lower limit:",
        int(
            np.sum(
                np.degrees(
                    refined[:, 12]
                ) < -29.9
            )
        ),
    )

    print(
        "left roll at upper limit:",
        int(
            np.sum(
                np.degrees(
                    refined[:, 12]
                ) > 29.9
            )
        ),
    )

    print(
        "right roll at lower limit:",
        int(
            np.sum(
                np.degrees(
                    refined[:, 18]
                ) < -29.9
            )
        ),
    )

    print(
        "right roll at upper limit:",
        int(
            np.sum(
                np.degrees(
                    refined[:, 18]
                ) > 29.9
            )
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
        bool(
            np.all(
                np.isfinite(
                    refined
                )
            )
        ),
    )

    print(
        "saved:",
        OUTPUT_CSV,
    )


    assert refined.shape == (
        EXPECTED_FRAMES,
        37,
    )

    assert np.all(
        np.isfinite(
            refined
        )
    )


    print()

    print(
        "LINGLONG LOWER RETARGET V2.2: PASS"
    )


if __name__ == "__main__":
    main()
