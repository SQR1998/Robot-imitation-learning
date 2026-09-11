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


BASE_CSV = (
    ROOT
    / "output/linglong20_torso_head_v2"
    / "linglong20_torso_head_v2_qpos37.csv"
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

XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

OUT_DIR = (
    ROOT
    / "output/linglong20_arm_v1"
)

OUT_CSV = (
    OUT_DIR
    / "linglong20_arm_v1_qpos37.csv"
)


# ============================================================
# LingLong qpos
# ============================================================

LEFT_ARM = np.array(
    [23, 24, 25, 26, 27, 28, 29],
    dtype=int,
)

RIGHT_ARM = np.array(
    [30, 31, 32, 33, 34, 35, 36],
    dtype=int,
)


# Position is the primary objective.
ELBOW_POS_WEIGHT = 35.0
WRIST_POS_WEIGHT = 45.0

# Wrist orientation is intentionally coarse in Arm V1.
WRIST_ORI_WEIGHT = 0.20

# Only prevents branch jumping.
CONTINUITY_WEIGHT = 0.035

CALIBRATION_FRAMES = 30


def normalize(v):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-10:
        raise RuntimeError(
            "Degenerate vector"
        )

    return v / n


def quat_wxyz_matrix(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()


def chest_frame(frame):
    """
    Human anatomical chest frame:

      +X forward
      +Y left
      +Z up
    """

    ls = np.asarray(
        frame["left_shoulder"][0],
        dtype=np.float64,
    )

    rs = np.asarray(
        frame["right_shoulder"][0],
        dtype=np.float64,
    )

    spine = np.asarray(
        frame["spine3"][0],
        dtype=np.float64,
    )

    neck = np.asarray(
        frame["neck"][0],
        dtype=np.float64,
    )


    y = normalize(
        ls - rs
    )

    z0 = normalize(
        neck - spine
    )

    z = normalize(
        z0
        - np.dot(
            z0,
            y,
        ) * y
    )

    x = normalize(
        np.cross(
            y,
            z,
        )
    )

    y = normalize(
        np.cross(
            z,
            x,
        )
    )


    return np.column_stack(
        (
            x,
            y,
            z,
        )
    )


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


def joint_bounds(
    model,
    qpos_indices,
):

    low = []
    high = []


    for qidx in qpos_indices:

        found = False

        for jid in range(
            model.njnt
        ):

            if (
                int(
                    model.jnt_qposadr[jid]
                )
                == int(qidx)
            ):

                lo, hi = (
                    model.jnt_range[jid]
                )

                low.append(
                    float(lo)
                )

                high.append(
                    float(hi)
                )

                found = True
                break


        if not found:

            raise RuntimeError(
                f"No joint for qpos index {qidx}"
            )


    return (
        np.asarray(
            low,
            dtype=np.float64,
        ),
        np.asarray(
            high,
            dtype=np.float64,
        ),
    )


def joint_name_from_qpos(
    model,
    qidx,
):

    for jid in range(
        model.njnt
    ):

        if (
            int(
                model.jnt_qposadr[jid]
            )
            == int(qidx)
        ):

            return mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )

    return "UNKNOWN"


def orientation_error(
    target_R,
    actual_R,
):

    err_R = (
        target_R.T
        @ actual_R
    )

    return R.from_matrix(
        err_R
    ).as_rotvec()


def solve_side(
    side,
    model,
    data,
    qpos,
    frames,
    chest_id,
    shoulder_id,
    elbow_id,
    wrist_id,
    end_id,
    arm_indices,
):

    low, high = joint_bounds(
        model,
        arm_indices,
    )


    # ========================================================
    # Robot own morphology
    # ========================================================

    neutral = qpos[0].copy()

    neutral[
        arm_indices
    ] = 0.0

    # Make sure zero is actually inside all limits.
    neutral[
        arm_indices
    ] = np.minimum(
        np.maximum(
            neutral[
                arm_indices
            ],
            low,
        ),
        high,
    )

    data.qpos[:] = neutral

    mujoco.mj_forward(
        model,
        data,
    )


    shoulder_zero = (
        data.xpos[
            shoulder_id
        ].copy()
    )

    elbow_zero = (
        data.xpos[
            elbow_id
        ].copy()
    )

    wrist_zero = (
        data.xpos[
            wrist_id
        ].copy()
    )


    upper_len = float(
        np.linalg.norm(
            elbow_zero
            - shoulder_zero
        )
    )

    fore_len = float(
        np.linalg.norm(
            wrist_zero
            - elbow_zero
        )
    )


    # ========================================================
    # Coarse wrist-frame calibration
    #
    # Human wrist orientation is useful dynamically, but its
    # local coordinate system does not equal LingLong's.
    #
    # Align the mean first-30-frame human wrist frame to the
    # robot neutral wrist frame.
    # ========================================================

    human_rel_rots = []


    for i in range(
        CALIBRATION_FRAMES
    ):

        Ch = chest_frame(
            frames[i]
        )

        Rh = quat_wxyz_matrix(
            frames[i][
                f"{side}_wrist"
            ][1]
        )

        human_rel_rots.append(
            Ch.T
            @ Rh
        )


    human_rel_mean = (
        R.from_matrix(
            np.asarray(
                human_rel_rots
            )
        )
        .mean()
        .as_matrix()
    )


    # Robot neutral wrist frame relative to robot chest.
    neutral_robot = qpos[0].copy()

    neutral_robot[
        arm_indices
    ] = 0.0

    neutral_robot[
        arm_indices
    ] = np.minimum(
        np.maximum(
            neutral_robot[
                arm_indices
            ],
            low,
        ),
        high,
    )

    data.qpos[:] = neutral_robot

    mujoco.mj_forward(
        model,
        data,
    )


    Cr0 = (
        data.xmat[
            chest_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    Rw0 = (
        data.xmat[
            end_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    robot_rel_neutral = (
        Cr0.T
        @ Rw0
    )


    wrist_frame_correction = (
        human_rel_mean.T
        @ robot_rel_neutral
    )


    print()
    print("=" * 100)
    print(
        side.upper(),
        "ARM MORPHOLOGY / WRIST CALIBRATION",
    )
    print("=" * 100)

    print(
        "robot upper-arm length:",
        upper_len,
    )

    print(
        "robot forearm length:",
        fore_len,
    )

    print(
        "wrist-frame correction ZYX deg:",
        R.from_matrix(
            wrist_frame_correction
        ).as_euler(
            "ZYX",
            degrees=True,
        ),
    )


    solved = qpos.copy()

    elbow_errors = []
    wrist_errors = []
    ori_errors = []


    # Initial solution:
    # keep current arm if valid.
    prev = np.minimum(
        np.maximum(
            solved[
                0,
                arm_indices,
            ],
            low,
        ),
        high,
    )


    for i, frame in enumerate(
        frames
    ):

        # ====================================================
        # Human arm directions in HUMAN CHEST coordinates.
        # ====================================================

        Ch = chest_frame(
            frame
        )


        h_shoulder = np.asarray(
            frame[
                f"{side}_shoulder"
            ][0],
            dtype=np.float64,
        )

        h_elbow = np.asarray(
            frame[
                f"{side}_elbow"
            ][0],
            dtype=np.float64,
        )

        h_wrist = np.asarray(
            frame[
                f"{side}_wrist"
            ][0],
            dtype=np.float64,
        )


        upper_world = normalize(
            h_elbow
            - h_shoulder
        )

        fore_world = normalize(
            h_wrist
            - h_elbow
        )


        upper_local = (
            Ch.T
            @ upper_world
        )

        fore_local = (
            Ch.T
            @ fore_world
        )


        # ====================================================
        # Robot chest + shoulder anchor for this frame.
        # Torso/Head V2 stays untouched.
        # ====================================================

        frame_q = solved[i].copy()

        frame_q[
            arm_indices
        ] = prev


        data.qpos[:] = frame_q

        mujoco.mj_forward(
            model,
            data,
        )


        Cr = (
            data.xmat[
                chest_id
            ]
            .reshape(
                3,
                3,
            )
            .copy()
        )

        shoulder_pos = (
            data.xpos[
                shoulder_id
            ].copy()
        )


        # Use ROBOT'S OWN bone lengths.
        target_elbow = (
            shoulder_pos
            + Cr
            @ upper_local
            * upper_len
        )

        target_wrist = (
            target_elbow
            + Cr
            @ fore_local
            * fore_len
        )


        # ====================================================
        # Coarse wrist orientation target
        # ====================================================

        human_wrist_R = quat_wxyz_matrix(
            frame[
                f"{side}_wrist"
            ][1]
        )

        human_wrist_rel = (
            Ch.T
            @ human_wrist_R
        )

        target_wrist_rel = (
            human_wrist_rel
            @ wrist_frame_correction
        )

        target_wrist_R = (
            Cr
            @ target_wrist_rel
        )


        def residual(x):

            test_q = solved[i].copy()

            test_q[
                arm_indices
            ] = x


            data.qpos[:] = test_q

            mujoco.mj_forward(
                model,
                data,
            )


            actual_elbow = (
                data.xpos[
                    elbow_id
                ]
            )

            actual_wrist = (
                data.xpos[
                    wrist_id
                ]
            )

            actual_end_R = (
                data.xmat[
                    end_id
                ]
                .reshape(
                    3,
                    3,
                )
            )


            r_elbow = (
                ELBOW_POS_WEIGHT
                * (
                    actual_elbow
                    - target_elbow
                )
            )

            r_wrist = (
                WRIST_POS_WEIGHT
                * (
                    actual_wrist
                    - target_wrist
                )
            )

            r_ori = (
                WRIST_ORI_WEIGHT
                * orientation_error(
                    target_wrist_R,
                    actual_end_R,
                )
            )

            r_cont = (
                CONTINUITY_WEIGHT
                * (
                    x - prev
                )
            )


            return np.concatenate(
                (
                    r_elbow,
                    r_wrist,
                    r_ori,
                    r_cont,
                )
            )


        result = least_squares(
            residual,
            prev,
            bounds=(
                low,
                high,
            ),
            method="trf",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=120,
        )


        x = result.x

        solved[
            i,
            arm_indices,
        ] = x


        # ====================================================
        # Diagnostics
        # ====================================================

        data.qpos[:] = solved[i]

        mujoco.mj_forward(
            model,
            data,
        )


        elbow_err = float(
            np.linalg.norm(
                data.xpos[
                    elbow_id
                ]
                - target_elbow
            )
        )

        wrist_err = float(
            np.linalg.norm(
                data.xpos[
                    wrist_id
                ]
                - target_wrist
            )
        )

        actual_end_R = (
            data.xmat[
                end_id
            ]
            .reshape(
                3,
                3,
            )
        )

        ori_err = float(
            np.linalg.norm(
                orientation_error(
                    target_wrist_R,
                    actual_end_R,
                )
            )
        )


        elbow_errors.append(
            elbow_err
        )

        wrist_errors.append(
            wrist_err
        )

        ori_errors.append(
            ori_err
        )


        prev = x.copy()


        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == len(frames) - 1
        ):

            print(
                f"{side:5s} "
                f"frame {i:03d}/660 | "
                f"elbow={elbow_err*1000:7.2f} mm | "
                f"wrist={wrist_err*1000:7.2f} mm | "
                f"ori={np.degrees(ori_err):7.2f} deg"
            )


    return (
        solved,
        np.asarray(
            elbow_errors
        ),
        np.asarray(
            wrist_errors
        ),
        np.asarray(
            ori_errors
        ),
        low,
        high,
    )


def report_errors(
    side,
    elbow_error,
    wrist_error,
    ori_error,
):

    print()
    print("=" * 100)
    print(
        side.upper(),
        "ARM FIT",
    )
    print("=" * 100)

    for name, values in [
        (
            "elbow position mm",
            elbow_error
            * 1000.0,
        ),
        (
            "wrist position mm",
            wrist_error
            * 1000.0,
        ),
        (
            "coarse wrist ori deg",
            np.degrees(
                ori_error
            ),
        ),
    ]:

        print(
            f"{name:24s}"
            f"mean={np.mean(values):8.3f}  "
            f"median={np.median(values):8.3f}  "
            f"p95={np.percentile(values,95):8.3f}  "
            f"max={np.max(values):8.3f}"
        )


def report_joints(
    side,
    model,
    qpos,
    indices,
    low,
    high,
):

    print()
    print("=" * 100)
    print(
        side.upper(),
        "ARM JOINT RANGE / CONTINUITY",
    )
    print("=" * 100)


    for j, qidx in enumerate(
        indices
    ):

        x = np.degrees(
            qpos[
                :,
                qidx,
            ]
        )

        dx = np.diff(
            x
        )

        lower_count = int(
            np.sum(
                x
                < np.degrees(
                    low[j]
                )
                + 0.05
            )
        )

        upper_count = int(
            np.sum(
                x
                > np.degrees(
                    high[j]
                )
                - 0.05
            )
        )


        print(
            f"{joint_name_from_qpos(model,qidx):28s} "
            f"{x.min():8.2f} .. "
            f"{x.max():8.2f} deg | "
            f"max step={np.max(np.abs(dx)):7.2f} | "
            f"limits L/U={lower_count}/{upper_count}"
        )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    base_qpos = np.loadtxt(
        BASE_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert base_qpos.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(
            base_qpos
        )
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

    assert len(frames) == 661


    required_human = [
        "spine3",
        "neck",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ]


    for name in required_human:

        if name not in frames[0]:

            raise RuntimeError(
                f"Missing SMPL-X joint: {name}"
            )


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(
        model
    )


    chest_id = body_id(
        model,
        "waist_pitch_link",
    )


    left_ids = {
        "shoulder": body_id(
            model,
            "left_shoulder_pitch_link",
        ),
        "elbow": body_id(
            model,
            "left_elbow_link",
        ),
        "wrist": body_id(
            model,
            "left_wrist_roll_link",
        ),
        "end": body_id(
            model,
            "left_wrist_yaw_link",
        ),
    }


    right_ids = {
        "shoulder": body_id(
            model,
            "right_shoulder_pitch_link",
        ),
        "elbow": body_id(
            model,
            "right_elbow_link",
        ),
        "wrist": body_id(
            model,
            "right_wrist_roll_link",
        ),
        "end": body_id(
            model,
            "right_wrist_yaw_link",
        ),
    }


    print("=" * 100)
    print("LINGLONG FULL ARM RETARGET V1")
    print("=" * 100)

    print(
        "frames:",
        len(frames),
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
        "Arm V1 policy:"
    )

    print(
        "  shoulder/elbow/wrist position = PRIMARY"
    )

    print(
        "  wrist orientation = COARSE"
    )

    print(
        "  fine wrist mapping = later stage"
    )


    # ========================================================
    # LEFT
    # ========================================================

    (
        qpos_left,
        l_elbow_err,
        l_wrist_err,
        l_ori_err,
        l_low,
        l_high,
    ) = solve_side(
        "left",
        model,
        data,
        base_qpos.copy(),
        frames,
        chest_id,
        left_ids["shoulder"],
        left_ids["elbow"],
        left_ids["wrist"],
        left_ids["end"],
        LEFT_ARM,
    )


    # ========================================================
    # RIGHT
    #
    # Start from the LEFT-solved motion so both arms coexist.
    # ========================================================

    (
        qpos_final,
        r_elbow_err,
        r_wrist_err,
        r_ori_err,
        r_low,
        r_high,
    ) = solve_side(
        "right",
        model,
        data,
        qpos_left.copy(),
        frames,
        chest_id,
        right_ids["shoulder"],
        right_ids["elbow"],
        right_ids["wrist"],
        right_ids["end"],
        RIGHT_ARM,
    )


    # ========================================================
    # Reports
    # ========================================================

    report_errors(
        "left",
        l_elbow_err,
        l_wrist_err,
        l_ori_err,
    )

    report_errors(
        "right",
        r_elbow_err,
        r_wrist_err,
        r_ori_err,
    )


    report_joints(
        "left",
        model,
        qpos_final,
        LEFT_ARM,
        l_low,
        l_high,
    )

    report_joints(
        "right",
        model,
        qpos_final,
        RIGHT_ARM,
        r_low,
        r_high,
    )


    # ========================================================
    # Preservation:
    #
    # only qpos 23:37 may change.
    # Lower + torso + head MUST remain identical.
    # ========================================================

    unchanged_diff = float(
        np.max(
            np.abs(
                qpos_final[
                    :,
                    :23,
                ]
                - base_qpos[
                    :,
                    :23,
                ]
            )
        )
    )


    print()
    print("=" * 100)
    print("BASE MOTION PRESERVATION")
    print("=" * 100)

    print(
        "qpos[0:23] max diff:",
        unchanged_diff,
    )


    assert unchanged_diff < 1e-12

    assert qpos_final.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(
            qpos_final
        )
    )


    # Overall arm continuity.
    arm_all = np.concatenate(
        (
            LEFT_ARM,
            RIGHT_ARM,
        )
    )

    arm_steps = np.abs(
        np.diff(
            qpos_final[
                :,
                arm_all,
            ],
            axis=0,
        )
    )

    worst = np.unravel_index(
        np.argmax(
            arm_steps
        ),
        arm_steps.shape,
    )

    worst_frame = int(
        worst[0] + 1
    )

    worst_joint_qidx = int(
        arm_all[
            worst[1]
        ]
    )

    worst_step = float(
        arm_steps[
            worst
        ]
    )


    print()
    print("=" * 100)
    print("OVERALL ARM CONTINUITY")
    print("=" * 100)

    print(
        "max arm step:",
        np.degrees(
            worst_step
        ),
        "deg",
    )

    print(
        "frame:",
        worst_frame,
    )

    print(
        "joint:",
        joint_name_from_qpos(
            model,
            worst_joint_qidx,
        ),
    )


    np.savetxt(
        OUT_CSV,
        qpos_final,
        delimiter=",",
        fmt="%.12f",
    )


    print()
    print("=" * 100)
    print("OUTPUT")
    print("=" * 100)

    print(
        "shape:",
        qpos_final.shape,
    )

    print(
        "finite:",
        bool(
            np.all(
                np.isfinite(
                    qpos_final
                )
            )
        ),
    )

    print(
        "saved:",
        OUT_CSV,
    )

    print()

    print(
        "LINGLONG FULL ARM "
        "RETARGET V1: PASS"
    )


if __name__ == "__main__":
    main()
