#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares


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

BODY_V1_CSV = (
    ROOT
    / "output/linglong20_body_v1_full"
    / "linglong20_body_v1_full_qpos37.csv"
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
    / "output/linglong20_lower_v20"
)

OUTPUT_CSV = (
    OUT_DIR
    / "linglong20_lower_v20_qpos37.csv"
)


# ============================================================
# LingLong 2.0 measured geometry
# ============================================================

HIP_HALF_WIDTH = 0.1138
HIP_Z = -0.0925

THIGH_LENGTH = 0.3823
SHANK_LENGTH = 0.4200

# Calibrated from actual SMPL-X source + LingLong morphology.
LATERAL_GAIN = 0.57


# ============================================================
# 37-D qpos layout
# ============================================================

LEFT_LEG_QPOS = [
    7,   # hip pitch
    8,   # hip roll
    9,   # hip yaw
    10,  # knee
]

RIGHT_LEG_QPOS = [
    13,  # hip pitch
    14,  # hip roll
    15,  # hip yaw
    16,  # knee
]


EXPECTED_FRAMES = 661


def normalize(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(np.linalg.norm(v))

    if n < 1e-10:
        raise RuntimeError(
            "Cannot normalize near-zero vector"
        )

    return v / n


def anatomical_basis(frame):
    """
    Human anatomical coordinates:

        X = forward
        Y = left
        Z = up

    Derived from human geometry rather than relying on
    SMPL-X pelvis local quaternion axes.
    """

    pelvis = np.asarray(
        frame["pelvis"][0],
        dtype=np.float64,
    )

    spine = np.asarray(
        frame["spine3"][0],
        dtype=np.float64,
    )

    left_hip = np.asarray(
        frame["left_hip"][0],
        dtype=np.float64,
    )

    right_hip = np.asarray(
        frame["right_hip"][0],
        dtype=np.float64,
    )

    y_axis = normalize(
        left_hip - right_hip
    )

    z_raw = normalize(
        spine - pelvis
    )

    z_axis = normalize(
        z_raw
        - np.dot(
            z_raw,
            y_axis,
        )
        * y_axis
    )

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


def to_local(
    point,
    pelvis,
    basis,
):
    return (
        basis.T
        @ (
            np.asarray(
                point,
                dtype=np.float64,
            )
            - pelvis
        )
    )


def mapped_direction(v):
    """
    Preserve sagittal motion.

    X forward: 100 %
    Y lateral: 57 %
    Z vertical: 100 %

    Then re-normalize because this is a direction.
    """

    d = normalize(v)

    mapped = np.array(
        [
            d[0],
            LATERAL_GAIN * d[1],
            d[2],
        ],
        dtype=np.float64,
    )

    return normalize(mapped)


def build_targets(frame):
    pelvis, basis = (
        anatomical_basis(frame)
    )

    p = {}

    for name in (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ):
        p[name] = to_local(
            frame[name][0],
            pelvis,
            basis,
        )

    l_thigh = (
        p["left_knee"]
        - p["left_hip"]
    )

    r_thigh = (
        p["right_knee"]
        - p["right_hip"]
    )

    l_shank = (
        p["left_ankle"]
        - p["left_knee"]
    )

    r_shank = (
        p["right_ankle"]
        - p["right_knee"]
    )

    l_hip = np.array(
        [
            0.0,
            +HIP_HALF_WIDTH,
            HIP_Z,
        ],
        dtype=np.float64,
    )

    r_hip = np.array(
        [
            0.0,
            -HIP_HALF_WIDTH,
            HIP_Z,
        ],
        dtype=np.float64,
    )

    l_knee = (
        l_hip
        + THIGH_LENGTH
        * mapped_direction(
            l_thigh
        )
    )

    r_knee = (
        r_hip
        + THIGH_LENGTH
        * mapped_direction(
            r_thigh
        )
    )

    l_ankle = (
        l_knee
        + SHANK_LENGTH
        * mapped_direction(
            l_shank
        )
    )

    r_ankle = (
        r_knee
        + SHANK_LENGTH
        * mapped_direction(
            r_shank
        )
    )

    return {
        "left": (
            l_knee,
            l_ankle,
        ),
        "right": (
            r_knee,
            r_ankle,
        ),
    }


def body_id(model, name):
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


def joint_limits_for_qpos(
    model,
    qpos_indices,
):
    lower = []
    upper = []

    for qidx in qpos_indices:

        found = False

        for jid in range(model.njnt):

            if int(
                model.jnt_qposadr[jid]
            ) != qidx:
                continue

            found = True

            if bool(
                model.jnt_limited[jid]
            ):
                lo, hi = (
                    model.jnt_range[jid]
                )
            else:
                lo = -np.inf
                hi = +np.inf

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


def make_fk_context(
    model,
    knee_body,
    ankle_body,
    qpos_indices,
):
    data = mujoco.MjData(model)

    base_id = body_id(
        model,
        "base_link",
    )

    knee_id = body_id(
        model,
        knee_body,
    )

    ankle_id = body_id(
        model,
        ankle_body,
    )

    def fk(x):
        # Local kinematic model.
        # Root fixed to identity.
        data.qpos[:] = model.qpos0

        data.qpos[:7] = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        for idx, value in zip(
            qpos_indices,
            x,
        ):
            data.qpos[idx] = value

        mujoco.mj_forward(
            model,
            data,
        )

        base_pos = np.asarray(
            data.xpos[base_id],
            dtype=np.float64,
        )

        base_rot = np.asarray(
            data.xmat[base_id],
            dtype=np.float64,
        ).reshape(3, 3)

        knee = (
            base_rot.T
            @ (
                np.asarray(
                    data.xpos[knee_id],
                    dtype=np.float64,
                )
                - base_pos
            )
        )

        ankle = (
            base_rot.T
            @ (
                np.asarray(
                    data.xpos[ankle_id],
                    dtype=np.float64,
                )
                - base_pos
            )
        )

        return (
            knee,
            ankle,
        )

    return fk


def solve_leg(
    fk,
    target_knee,
    target_ankle,
    seed,
    lower,
    upper,
):

    def residual(x):
        knee, ankle = fk(x)

        return np.concatenate(
            (
                knee
                - target_knee,

                ankle
                - target_ankle,
            )
        )

    seed = np.clip(
        seed,
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

        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,

        max_nfev=100,
    )

    knee, ankle = fk(
        result.x
    )

    knee_error = float(
        np.linalg.norm(
            knee - target_knee
        )
    )

    ankle_error = float(
        np.linalg.norm(
            ankle - target_ankle
        )
    )

    return (
        result.x,
        knee_error,
        ankle_error,
    )


def body_geom_ids(
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

    return ids


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
    ).reshape(3, 3)

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


def foot_bottom(
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

    base_qpos = np.loadtxt(
        BODY_V1_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    if base_qpos.shape != (
        EXPECTED_FRAMES,
        37,
    ):
        raise RuntimeError(
            f"Unexpected Body V1 shape: "
            f"{base_qpos.shape}"
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
        metadata_fps,
    ) = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    if len(human_frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected 661 frames, "
            f"got {len(human_frames)}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    left_fk = make_fk_context(
        model,
        "left_knee_link",
        "left_ankle_roll_link",
        LEFT_LEG_QPOS,
    )

    right_fk = make_fk_context(
        model,
        "right_knee_link",
        "right_ankle_roll_link",
        RIGHT_LEG_QPOS,
    )

    left_lower, left_upper = (
        joint_limits_for_qpos(
            model,
            LEFT_LEG_QPOS,
        )
    )

    right_lower, right_upper = (
        joint_limits_for_qpos(
            model,
            RIGHT_LEG_QPOS,
        )
    )

    print("=" * 95)
    print("LINGLONG LOWER RETARGET V2.0")
    print("=" * 95)

    print(
        "frames:",
        len(human_frames),
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
        "left limits:",
        np.degrees(
            np.column_stack(
                (
                    left_lower,
                    left_upper,
                )
            )
        ),
    )

    print(
        "right limits:",
        np.degrees(
            np.column_stack(
                (
                    right_lower,
                    right_upper,
                )
            )
        ),
    )

    refined = base_qpos.copy()

    left_knee_error = []
    left_ankle_error = []

    right_knee_error = []
    right_ankle_error = []

    target_knee_width = []
    target_ankle_width = []

    # --------------------------------------------------------
    # First frame:
    #
    # Do not seed roll/yaw from old V1 because V1 mapping is
    # exactly what we are replacing.
    # --------------------------------------------------------

    left_seed = np.array(
        [
            base_qpos[
                0,
                7,
            ],
            0.0,
            0.0,
            base_qpos[
                0,
                10,
            ],
        ],
        dtype=np.float64,
    )

    right_seed = np.array(
        [
            base_qpos[
                0,
                13,
            ],
            0.0,
            0.0,
            base_qpos[
                0,
                16,
            ],
        ],
        dtype=np.float64,
    )

    for i, frame in enumerate(
        human_frames
    ):

        targets = build_targets(
            frame
        )

        (
            left_target_knee,
            left_target_ankle,
        ) = targets["left"]

        (
            right_target_knee,
            right_target_ankle,
        ) = targets["right"]

        target_knee_width.append(
            left_target_knee[1]
            - right_target_knee[1]
        )

        target_ankle_width.append(
            left_target_ankle[1]
            - right_target_ankle[1]
        )

        (
            left_solution,
            lke,
            lae,
        ) = solve_leg(
            left_fk,

            left_target_knee,
            left_target_ankle,

            left_seed,

            left_lower,
            left_upper,
        )

        (
            right_solution,
            rke,
            rae,
        ) = solve_leg(
            right_fk,

            right_target_knee,
            right_target_ankle,

            right_seed,

            right_lower,
            right_upper,
        )

        refined[
            i,
            LEFT_LEG_QPOS,
        ] = left_solution

        refined[
            i,
            RIGHT_LEG_QPOS,
        ] = right_solution

        # Current frame becomes next frame's seed.
        left_seed = (
            left_solution.copy()
        )

        right_seed = (
            right_solution.copy()
        )

        left_knee_error.append(
            lke
        )

        left_ankle_error.append(
            lae
        )

        right_knee_error.append(
            rke
        )

        right_ankle_error.append(
            rae
        )

        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == EXPECTED_FRAMES - 1
        ):
            print(
                f"frame {i:03d} | "
                f"LK={lke*1000:6.2f}mm "
                f"LA={lae*1000:6.2f}mm | "
                f"RK={rke*1000:6.2f}mm "
                f"RA={rae*1000:6.2f}mm"
            )

    # --------------------------------------------------------
    # Re-normalize ground using ONE constant root-z shift.
    #
    # This is coordinate normalization, NOT collision repair.
    # --------------------------------------------------------

    ground_data = mujoco.MjData(
        model
    )

    left_geoms = body_geom_ids(
        model,
        "left_ankle_roll_link",
    )

    right_geoms = body_geom_ids(
        model,
        "right_ankle_roll_link",
    )

    first30_lowest = []

    for i in range(30):

        lz = foot_bottom(
            model,
            ground_data,
            refined[i],
            left_geoms,
        )

        rz = foot_bottom(
            model,
            ground_data,
            refined[i],
            right_geoms,
        )

        first30_lowest.append(
            min(
                lz,
                rz,
            )
        )

    ground_offset = -float(
        np.median(
            first30_lowest
        )
    )

    refined[:, 2] += (
        ground_offset
    )

    # --------------------------------------------------------
    # Diagnostics only.
    # No knee-distance constraint is applied.
    # --------------------------------------------------------

    diag_data = mujoco.MjData(
        model
    )

    left_knee_id = body_id(
        model,
        "left_knee_link",
    )

    right_knee_id = body_id(
        model,
        "right_knee_link",
    )

    base_id = body_id(
        model,
        "base_link",
    )

    actual_knee_width = []

    for q in refined:

        diag_data.qpos[:] = q

        mujoco.mj_forward(
            model,
            diag_data,
        )

        base_pos = np.asarray(
            diag_data.xpos[
                base_id
            ]
        )

        base_rot = np.asarray(
            diag_data.xmat[
                base_id
            ]
        ).reshape(3, 3)

        lk = (
            base_rot.T
            @ (
                np.asarray(
                    diag_data.xpos[
                        left_knee_id
                    ]
                )
                - base_pos
            )
        )

        rk = (
            base_rot.T
            @ (
                np.asarray(
                    diag_data.xpos[
                        right_knee_id
                    ]
                )
                - base_pos
            )
        )

        actual_knee_width.append(
            lk[1] - rk[1]
        )

    target_knee_width = np.asarray(
        target_knee_width
    )

    target_ankle_width = np.asarray(
        target_ankle_width
    )

    actual_knee_width = np.asarray(
        actual_knee_width
    )

    left_knee_error = np.asarray(
        left_knee_error
    )

    left_ankle_error = np.asarray(
        left_ankle_error
    )

    right_knee_error = np.asarray(
        right_knee_error
    )

    right_ankle_error = np.asarray(
        right_ankle_error
    )

    np.savetxt(
        OUTPUT_CSV,
        refined,
        delimiter=",",
        fmt="%.9f",
    )

    print()
    print("=" * 95)
    print("TARGET FIT")
    print("=" * 95)

    print(
        "left knee mean/max mm :",
        left_knee_error.mean()
        * 1000,
        left_knee_error.max()
        * 1000,
    )

    print(
        "left ankle mean/max mm:",
        left_ankle_error.mean()
        * 1000,
        left_ankle_error.max()
        * 1000,
    )

    print(
        "right knee mean/max mm:",
        right_knee_error.mean()
        * 1000,
        right_knee_error.max()
        * 1000,
    )

    print(
        "right ankle mean/max mm:",
        right_ankle_error.mean()
        * 1000,
        right_ankle_error.max()
        * 1000,
    )

    print()
    print("=" * 95)
    print("MAPPING WIDTH")
    print("=" * 95)

    print(
        "target knee min/median/max:",
        target_knee_width.min(),
        np.median(
            target_knee_width
        ),
        target_knee_width.max(),
    )

    print(
        "target ankle min/median/max:",
        target_ankle_width.min(),
        np.median(
            target_ankle_width
        ),
        target_ankle_width.max(),
    )

    print(
        "actual knee min/median/max:",
        actual_knee_width.min(),
        np.median(
            actual_knee_width
        ),
        actual_knee_width.max(),
    )

    print()
    print("=" * 95)
    print("JOINT RANGES")
    print("=" * 95)

    names = [
        (
            "left hip pitch",
            7,
        ),
        (
            "left hip roll",
            8,
        ),
        (
            "left hip yaw",
            9,
        ),
        (
            "left knee",
            10,
        ),
        (
            "right hip pitch",
            13,
        ),
        (
            "right hip roll",
            14,
        ),
        (
            "right hip yaw",
            15,
        ),
        (
            "right knee",
            16,
        ),
    ]

    for name, idx in names:

        vals = np.degrees(
            refined[:, idx]
        )

        print(
            f"{name:20s}: "
            f"{vals.min():8.2f} .. "
            f"{vals.max():8.2f} deg"
        )

    max_step = float(
        np.max(
            np.abs(
                np.diff(
                    refined[:, 7:],
                    axis=0,
                )
            )
        )
    )

    print()
    print(
        "constant ground offset:",
        ground_offset,
    )

    print(
        "max joint step:",
        max_step,
        "rad/frame",
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
        "LINGLONG LOWER RETARGET V2.0: PASS"
    )


if __name__ == "__main__":
    main()
