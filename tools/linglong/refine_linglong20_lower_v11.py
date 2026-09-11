#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np


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

INPUT_CSV = (
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
    / "output/linglong20_lower_v11"
)

OUTPUT_CSV = (
    OUT_DIR
    / "linglong20_lower_v11_qpos37.csv"
)


# ------------------------------------------------------------
# LingLong 2.0 validated qpos layout
# ------------------------------------------------------------

LEFT_HIP_PITCH = 7
LEFT_KNEE = 10
LEFT_ANKLE_PITCH = 11

RIGHT_HIP_PITCH = 13
RIGHT_KNEE = 16
RIGHT_ANKLE_PITCH = 17


# ------------------------------------------------------------
# Conservative V1.1 parameters
# ------------------------------------------------------------

# Human knee flex -> robot knee flex.
KNEE_GAIN = 0.90

# How strongly we pull IK knee toward human knee.
BLEND = 0.55

# Human knee signal smoothing.
FLEX_SMOOTH_ALPHA = 0.30

# Avoid perfectly locked knee.
STRAIGHT_KNEE = 0.03

# Conservative working range.
KNEE_MIN = 0.00
KNEE_MAX = 1.65

# Do not change the original IK knee by more than this
# in a single frame during guidance.
MAX_CORRECTION = 0.30


def human_knee_flexion(frame, side):
    hip = np.asarray(
        frame[f"{side}_hip"][0],
        dtype=np.float64,
    )

    knee = np.asarray(
        frame[f"{side}_knee"][0],
        dtype=np.float64,
    )

    foot = np.asarray(
        frame[f"{side}_foot"][0],
        dtype=np.float64,
    )

    thigh = hip - knee
    shank = foot - knee

    n1 = np.linalg.norm(thigh)
    n2 = np.linalg.norm(shank)

    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    cosine = np.dot(
        thigh,
        shank,
    ) / (n1 * n2)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    inner_angle = np.arccos(
        cosine
    )

    # Straight leg:
    # inner angle ~= pi
    # flexion ~= 0
    return float(
        max(
            0.0,
            np.pi - inner_angle,
        )
    )


def smooth_signal(values, alpha):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    result = values.copy()

    for i in range(1, len(result)):
        result[i] = (
            alpha * values[i]
            + (1.0 - alpha)
            * result[i - 1]
        )

    return result


def joint_limits(model, qpos_index):
    for jid in range(model.njnt):
        if int(
            model.jnt_qposadr[jid]
        ) == qpos_index:

            if not model.jnt_limited[jid]:
                return -np.inf, np.inf

            return tuple(
                map(
                    float,
                    model.jnt_range[jid],
                )
            )

    raise RuntimeError(
        f"No joint for qpos {qpos_index}"
    )


def apply_side(
    q,
    target_flex,
    hip_idx,
    knee_idx,
    ankle_idx,
    limits,
):
    old_hip = float(
        q[hip_idx]
    )

    old_knee = float(
        q[knee_idx]
    )

    old_ankle = float(
        q[ankle_idx]
    )

    target_knee = (
        STRAIGHT_KNEE
        + KNEE_GAIN
        * target_flex
    )

    target_knee = float(
        np.clip(
            target_knee,
            KNEE_MIN,
            KNEE_MAX,
        )
    )

    guided_knee = (
        old_knee
        + BLEND
        * (
            target_knee
            - old_knee
        )
    )

    correction = np.clip(
        guided_knee - old_knee,
        -MAX_CORRECTION,
        MAX_CORRECTION,
    )

    new_knee = (
        old_knee
        + correction
    )

    # Preserve approximate sagittal chain orientation.
    new_hip = (
        old_hip
        - 0.5 * correction
    )

    new_ankle = (
        old_ankle
        - 0.5 * correction
    )

    hip_lo, hip_hi = limits[
        hip_idx
    ]

    knee_lo, knee_hi = limits[
        knee_idx
    ]

    ankle_lo, ankle_hi = limits[
        ankle_idx
    ]

    q[hip_idx] = np.clip(
        new_hip,
        hip_lo,
        hip_hi,
    )

    q[knee_idx] = np.clip(
        new_knee,
        max(knee_lo, KNEE_MIN),
        min(knee_hi, KNEE_MAX),
    )

    q[ankle_idx] = np.clip(
        new_ankle,
        ankle_lo,
        ankle_hi,
    )


def correlation(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if (
        np.std(a) < 1e-8
        or np.std(b) < 1e-8
    ):
        return 0.0

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qpos = np.loadtxt(
        INPUT_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    if qpos.shape != (
        661,
        37,
    ):
        raise RuntimeError(
            f"Unexpected qpos shape: {qpos.shape}"
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
        _,
    ) = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    if len(human_frames) != 661:
        raise RuntimeError(
            f"Expected 661 human frames, "
            f"got {len(human_frames)}"
        )

    human_left = [
        human_knee_flexion(
            frame,
            "left",
        )
        for frame in human_frames
    ]

    human_right = [
        human_knee_flexion(
            frame,
            "right",
        )
        for frame in human_frames
    ]

    human_left = smooth_signal(
        human_left,
        FLEX_SMOOTH_ALPHA,
    )

    human_right = smooth_signal(
        human_right,
        FLEX_SMOOTH_ALPHA,
    )

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    indices = [
        LEFT_HIP_PITCH,
        LEFT_KNEE,
        LEFT_ANKLE_PITCH,
        RIGHT_HIP_PITCH,
        RIGHT_KNEE,
        RIGHT_ANKLE_PITCH,
    ]

    limits = {
        idx: joint_limits(
            model,
            idx,
        )
        for idx in indices
    }

    refined = qpos.copy()

    before_left = qpos[
        :,
        LEFT_KNEE,
    ].copy()

    before_right = qpos[
        :,
        RIGHT_KNEE,
    ].copy()

    for i in range(661):

        apply_side(
            refined[i],
            human_left[i],
            LEFT_HIP_PITCH,
            LEFT_KNEE,
            LEFT_ANKLE_PITCH,
            limits,
        )

        apply_side(
            refined[i],
            human_right[i],
            RIGHT_HIP_PITCH,
            RIGHT_KNEE,
            RIGHT_ANKLE_PITCH,
            limits,
        )

    after_left = refined[
        :,
        LEFT_KNEE,
    ]

    after_right = refined[
        :,
        RIGHT_KNEE,
    ]

    # --------------------------------------------------------
    # Since leg geometry changed slightly, recompute ONE
    # constant root-z offset from first 30 frames.
    # Use ankle-roll body origins here only for relative
    # correction; V1 already contains the mesh-based ground
    # normalization.
    # --------------------------------------------------------

    data = mujoco.MjData(
        model
    )

    left_ankle_body = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_ankle_roll_link",
        )
    )

    right_ankle_body = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_ankle_roll_link",
        )
    )

    old_bottom = []
    new_bottom = []

    for i in range(30):

        data.qpos[:] = qpos[i]
        mujoco.mj_forward(
            model,
            data,
        )

        old_bottom.append(
            min(
                data.xpos[
                    left_ankle_body,
                    2,
                ],
                data.xpos[
                    right_ankle_body,
                    2,
                ],
            )
        )

        data.qpos[:] = refined[i]
        mujoco.mj_forward(
            model,
            data,
        )

        new_bottom.append(
            min(
                data.xpos[
                    left_ankle_body,
                    2,
                ],
                data.xpos[
                    right_ankle_body,
                    2,
                ],
            )
        )

    z_correction = (
        np.median(old_bottom)
        - np.median(new_bottom)
    )

    refined[:, 2] += (
        z_correction
    )

    np.savetxt(
        OUTPUT_CSV,
        refined,
        delimiter=",",
        fmt="%.9f",
    )

    print("=" * 90)
    print("LINGLONG LOWER V1.1")
    print("=" * 90)

    print(
        "human left knee deg range :",
        np.degrees(
            [
                np.min(human_left),
                np.max(human_left),
            ]
        ),
    )

    print(
        "human right knee deg range:",
        np.degrees(
            [
                np.min(human_right),
                np.max(human_right),
            ]
        ),
    )

    print()

    print(
        "robot left before deg:",
        np.degrees(
            [
                np.min(before_left),
                np.max(before_left),
            ]
        ),
    )

    print(
        "robot left after deg :",
        np.degrees(
            [
                np.min(after_left),
                np.max(after_left),
            ]
        ),
    )

    print(
        "robot right before deg:",
        np.degrees(
            [
                np.min(before_right),
                np.max(before_right),
            ]
        ),
    )

    print(
        "robot right after deg :",
        np.degrees(
            [
                np.min(after_right),
                np.max(after_right),
            ]
        ),
    )

    print()

    print(
        "left correlation before:",
        correlation(
            human_left,
            before_left,
        ),
    )

    print(
        "left correlation after :",
        correlation(
            human_left,
            after_left,
        ),
    )

    print(
        "right correlation before:",
        correlation(
            human_right,
            before_right,
        ),
    )

    print(
        "right correlation after :",
        correlation(
            human_right,
            after_right,
        ),
    )

    print()

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

    print(
        "constant z correction:",
        z_correction,
    )

    print(
        "max joint step:",
        max_step,
        "rad/frame",
    )

    print(
        "output:",
        OUTPUT_CSV,
    )

    print()

    print(
        "LINGLONG LOWER V1.1: PASS"
    )


if __name__ == "__main__":
    main()
