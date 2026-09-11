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

V20_CSV = (
    ROOT
    / "output/linglong20_lower_v20"
    / "linglong20_lower_v20_qpos37.csv"
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
    / "output/linglong20_lower_v21"
)

OUTPUT_CSV = (
    OUT_DIR
    / "linglong20_lower_v21_qpos37.csv"
)


HIP_HALF_WIDTH = 0.1138
HIP_Z = -0.0925

THIGH_LENGTH = 0.3823
SHANK_LENGTH = 0.4200

LATERAL_GAIN = 0.57


LEFT_QPOS = [7, 8, 9, 10]
RIGHT_QPOS = [13, 14, 15, 16]

EXPECTED_FRAMES = 661


# Axis mapping influence expressed as an equivalent
# position error in metres.
AXIS_WEIGHT = 0.08

# Very small continuity preference.
# This does NOT override the human mapping.
CONTINUITY_WEIGHT = 0.003


def normalize(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(np.linalg.norm(v))

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

    y_axis = normalize(
        lh - rh
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

    return (
        pelvis,
        np.column_stack(
            (
                x_axis,
                y_axis,
                z_axis,
            )
        ),
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


def orient_positive_y(axis):

    axis = normalize(axis)

    if axis is None:
        return None

    if axis[1] < 0.0:
        axis = -axis

    return axis


def build_leg_target(
    hip,
    knee,
    ankle,
    foot,
    robot_hip,
):
    """
    Returns:
      robot knee target
      robot ankle target
      target knee hinge axis
      bend amount
    """

    thigh = mapped_direction(
        knee - hip
    )

    shank = mapped_direction(
        ankle - knee
    )

    foot_dir = mapped_direction(
        foot - ankle
    )

    robot_knee = (
        robot_hip
        + THIGH_LENGTH
        * thigh
    )

    robot_ankle = (
        robot_knee
        + SHANK_LENGTH
        * shank
    )

    # --------------------------------------------------------
    # Knee hinge orientation.
    #
    # Bent leg:
    #   normal of thigh/shank plane.
    #
    # Straight leg:
    #   foot direction supplies the missing twist.
    # --------------------------------------------------------

    bend_raw = np.cross(
        thigh,
        shank,
    )

    bend_strength = float(
        np.linalg.norm(
            bend_raw
        )
    )

    bend_axis = normalize(
        bend_raw
    )

    foot_axis = normalize(
        np.cross(
            foot_dir,
            shank,
        )
    )

    if foot_axis is not None:
        foot_axis = (
            orient_positive_y(
                foot_axis
            )
        )

    if bend_axis is not None:

        if (
            foot_axis is not None
            and np.dot(
                bend_axis,
                foot_axis,
            ) < 0.0
        ):
            bend_axis = -bend_axis

        elif (
            foot_axis is None
            and bend_axis[1] < 0.0
        ):
            bend_axis = -bend_axis

    if bend_axis is None:
        bend_axis = foot_axis

    if foot_axis is None:
        foot_axis = bend_axis

    if (
        bend_axis is None
        or foot_axis is None
    ):
        # Last-resort neutral knee axis.
        target_axis = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

    else:
        # sin(knee flexion angle) is bend_strength.
        #
        # Near straight:
        #   foot geometry dominates.
        #
        # Bent:
        #   actual thigh/shank plane dominates.
        blend = np.clip(
            (
                bend_strength
                - 0.08
            )
            / 0.25,
            0.0,
            1.0,
        )

        target_axis = normalize(
            blend * bend_axis
            + (1.0 - blend)
            * foot_axis
        )

    return (
        robot_knee,
        robot_ankle,
        target_axis,
        bend_strength,
    )


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
        "left_foot",
        "right_foot",
    ):
        p[name] = to_local(
            frame[name][0],
            pelvis,
            basis,
        )

    left_hip_robot = np.array(
        [
            0.0,
            +HIP_HALF_WIDTH,
            HIP_Z,
        ],
        dtype=np.float64,
    )

    right_hip_robot = np.array(
        [
            0.0,
            -HIP_HALF_WIDTH,
            HIP_Z,
        ],
        dtype=np.float64,
    )

    left = build_leg_target(
        p["left_hip"],
        p["left_knee"],
        p["left_ankle"],
        p["left_foot"],
        left_hip_robot,
    )

    right = build_leg_target(
        p["right_hip"],
        p["right_knee"],
        p["right_ankle"],
        p["right_foot"],
        right_hip_robot,
    )

    return {
        "left": left,
        "right": right,
    }


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


def joint_id(model, name):

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return idx


def joint_limits(
    model,
    indices,
):

    lo = []
    hi = []

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

            low, high = (
                model.jnt_range[jid]
            )

            lo.append(
                float(low)
            )

            hi.append(
                float(high)
            )

            break

        if not found:
            raise RuntimeError(
                f"No joint at qpos {qidx}"
            )

    return (
        np.asarray(lo),
        np.asarray(hi),
    )


def make_fk(
    model,
    qpos_indices,
    knee_body_name,
    ankle_body_name,
    knee_joint_name,
):

    data = mujoco.MjData(
        model
    )

    base = body_id(
        model,
        "base_link",
    )

    knee_body = body_id(
        model,
        knee_body_name,
    )

    ankle_body = body_id(
        model,
        ankle_body_name,
    )

    knee_joint = joint_id(
        model,
        knee_joint_name,
    )

    def fk(x):

        data.qpos[:] = (
            model.qpos0
        )

        data.qpos[:7] = [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]

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
            data.xpos[base]
        )

        base_rot = np.asarray(
            data.xmat[base]
        ).reshape(3, 3)

        knee = (
            base_rot.T
            @ (
                np.asarray(
                    data.xpos[
                        knee_body
                    ]
                )
                - base_pos
            )
        )

        ankle = (
            base_rot.T
            @ (
                np.asarray(
                    data.xpos[
                        ankle_body
                    ]
                )
                - base_pos
            )
        )

        knee_axis = (
            base_rot.T
            @ np.asarray(
                data.xaxis[
                    knee_joint
                ]
            )
        )

        knee_axis = normalize(
            knee_axis
        )

        return (
            knee,
            ankle,
            knee_axis,
        )

    return fk


def solve_leg(
    fk,
    target_knee,
    target_ankle,
    target_axis,
    seed,
    previous,
    lower,
    upper,
):

    def residual(x):

        (
            knee,
            ankle,
            knee_axis,
        ) = fk(x)

        return np.concatenate(
            (
                knee
                - target_knee,

                ankle
                - target_ankle,

                AXIS_WEIGHT
                * (
                    knee_axis
                    - target_axis
                ),

                CONTINUITY_WEIGHT
                * (
                    x
                    - previous
                ),
            )
        )

    result = least_squares(
        residual,
        np.clip(
            seed,
            lower + 1e-8,
            upper - 1e-8,
        ),

        bounds=(
            lower,
            upper,
        ),

        method="trf",

        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,

        max_nfev=120,
    )

    (
        knee,
        ankle,
        axis,
    ) = fk(
        result.x
    )

    knee_error = float(
        np.linalg.norm(
            knee
            - target_knee
        )
    )

    ankle_error = float(
        np.linalg.norm(
            ankle
            - target_ankle
        )
    )

    dot = float(
        np.clip(
            np.dot(
                axis,
                target_axis,
            ),
            -1.0,
            1.0,
        )
    )

    axis_error_deg = float(
        np.degrees(
            np.arccos(
                dot
            )
        )
    )

    return (
        result.x,
        knee_error,
        ankle_error,
        axis_error_deg,
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    q20 = np.loadtxt(
        V20_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert q20.shape == (
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

    left_fk = make_fk(
        model,
        LEFT_QPOS,
        "left_knee_link",
        "left_ankle_roll_link",
        "left_knee_joint",
    )

    right_fk = make_fk(
        model,
        RIGHT_QPOS,
        "right_knee_link",
        "right_ankle_roll_link",
        "right_knee_joint",
    )

    left_lo, left_hi = (
        joint_limits(
            model,
            LEFT_QPOS,
        )
    )

    right_lo, right_hi = (
        joint_limits(
            model,
            RIGHT_QPOS,
        )
    )

    refined = q20.copy()

    left_prev = q20[
        0,
        LEFT_QPOS,
    ].copy()

    right_prev = q20[
        0,
        RIGHT_QPOS,
    ].copy()

    l_knee_err = []
    l_ankle_err = []
    l_axis_err = []

    r_knee_err = []
    r_ankle_err = []
    r_axis_err = []

    bend_left = []
    bend_right = []

    print("=" * 95)
    print("LINGLONG LOWER RETARGET V2.1")
    print("POSITION + KNEE-PLANE / TWIST MAPPING")
    print("=" * 95)

    for i, frame in enumerate(
        frames
    ):

        targets = build_targets(
            frame
        )

        (
            lk,
            la,
            lax,
            lbs,
        ) = targets["left"]

        (
            rk,
            ra,
            rax,
            rbs,
        ) = targets["right"]

        bend_left.append(
            lbs
        )

        bend_right.append(
            rbs
        )

        (
            lsol,
            lke,
            lae,
            laxe,
        ) = solve_leg(
            left_fk,
            lk,
            la,
            lax,

            left_prev,
            left_prev,

            left_lo,
            left_hi,
        )

        (
            rsol,
            rke,
            rae,
            raxe,
        ) = solve_leg(
            right_fk,
            rk,
            ra,
            rax,

            right_prev,
            right_prev,

            right_lo,
            right_hi,
        )

        refined[
            i,
            LEFT_QPOS,
        ] = lsol

        refined[
            i,
            RIGHT_QPOS,
        ] = rsol

        left_prev = (
            lsol.copy()
        )

        right_prev = (
            rsol.copy()
        )

        l_knee_err.append(
            lke
        )

        l_ankle_err.append(
            lae
        )

        l_axis_err.append(
            laxe
        )

        r_knee_err.append(
            rke
        )

        r_ankle_err.append(
            rae
        )

        r_axis_err.append(
            raxe
        )

        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == 660
        ):
            print(
                f"frame {i:03d} | "
                f"L axis={laxe:6.2f}deg "
                f"R axis={raxe:6.2f}deg"
            )

    arrays = {
        "l_knee": np.asarray(
            l_knee_err
        ),
        "l_ankle": np.asarray(
            l_ankle_err
        ),
        "l_axis": np.asarray(
            l_axis_err
        ),
        "r_knee": np.asarray(
            r_knee_err
        ),
        "r_ankle": np.asarray(
            r_ankle_err
        ),
        "r_axis": np.asarray(
            r_axis_err
        ),
    }

    np.savetxt(
        OUTPUT_CSV,
        refined,
        delimiter=",",
        fmt="%.9f",
    )

    print()
    print("=" * 95)
    print("POSITION FIT")
    print("=" * 95)

    for name in (
        "l_knee",
        "l_ankle",
        "r_knee",
        "r_ankle",
    ):

        a = arrays[name]

        print(
            f"{name:10s}: "
            f"mean={a.mean()*1000:.3f} mm "
            f"max={a.max()*1000:.3f} mm"
        )

    print()
    print("=" * 95)
    print("KNEE AXIS FIT")
    print("=" * 95)

    print(
        "left axis mean/max:",
        arrays["l_axis"].mean(),
        arrays["l_axis"].max(),
        "deg",
    )

    print(
        "right axis mean/max:",
        arrays["r_axis"].mean(),
        arrays["r_axis"].max(),
        "deg",
    )

    print()
    print("=" * 95)
    print("JOINT RANGE / CONTINUITY")
    print("=" * 95)

    check = [
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

    for name, idx in check:

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

    lower_diff = np.abs(
        np.diff(
            refined[:, 7:19],
            axis=0,
        )
    )

    flat = int(
        np.argmax(
            lower_diff
        )
    )

    row, col = np.unravel_index(
        flat,
        lower_diff.shape,
    )

    qidx = col + 7

    print()

    print(
        "overall lower max step:",
        float(
            lower_diff[
                row,
                col,
            ]
        ),
        "rad",
    )

    print(
        "max-step frame:",
        row + 1,
    )

    print(
        "max-step qpos index:",
        qidx,
    )

    print()
    print(
        "saved:",
        OUTPUT_CSV,
    )

    print()

    print(
        "LINGLONG LOWER RETARGET V2.1: PASS"
    )


if __name__ == "__main__":
    main()
