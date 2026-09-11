#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
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


ROBOT_HIP_WIDTH = 0.2276
ROBOT_THIGH = 0.3823
ROBOT_SHANK = 0.4200

CURRENT_SCALE = 0.9


def quat_wxyz_to_rot(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()


def local_point(
    point,
    pelvis_pos,
    pelvis_rot,
):
    return (
        pelvis_rot.T
        @ (
            point
            - pelvis_pos
        )
    )


def stats(name, x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    print(
        f"{name:32s} "
        f"min={x.min(): .4f}  "
        f"p05={np.percentile(x,5): .4f}  "
        f"median={np.median(x): .4f}  "
        f"p95={np.percentile(x,95): .4f}  "
        f"max={x.max(): .4f}"
    )


def main():
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

    print("=" * 95)
    print("SMPL-X -> LINGLONG LOWER-BODY MAPPING DIAGNOSIS")
    print("=" * 95)

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

    hip_width = []
    knee_width = []
    foot_width = []

    left_thigh_len = []
    right_thigh_len = []

    left_shank_len = []
    right_shank_len = []

    left_knee_inward = []
    right_knee_inward = []

    left_foot_inward = []
    right_foot_inward = []

    for frame in frames:

        pelvis_pos = np.asarray(
            frame["pelvis"][0],
            dtype=np.float64,
        )

        pelvis_rot = quat_wxyz_to_rot(
            frame["pelvis"][1]
        )

        points = {}

        for name in (
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_foot",
            "right_foot",
        ):
            points[name] = local_point(
                np.asarray(
                    frame[name][0],
                    dtype=np.float64,
                ),
                pelvis_pos,
                pelvis_rot,
            )

        lh = points["left_hip"]
        rh = points["right_hip"]

        lk = points["left_knee"]
        rk = points["right_knee"]

        lf = points["left_foot"]
        rf = points["right_foot"]

        # +Y = left side in pelvis-local frame.
        hip_width.append(
            lh[1] - rh[1]
        )

        knee_width.append(
            lk[1] - rk[1]
        )

        foot_width.append(
            lf[1] - rf[1]
        )

        left_thigh_len.append(
            np.linalg.norm(
                lk - lh
            )
        )

        right_thigh_len.append(
            np.linalg.norm(
                rk - rh
            )
        )

        left_shank_len.append(
            np.linalg.norm(
                lf - lk
            )
        )

        right_shank_len.append(
            np.linalg.norm(
                rf - rk
            )
        )

        # Positive value = knee moves inward relative to hip.
        left_knee_inward.append(
            lh[1] - lk[1]
        )

        right_knee_inward.append(
            rk[1] - rh[1]
        )

        left_foot_inward.append(
            lh[1] - lf[1]
        )

        right_foot_inward.append(
            rf[1] - rh[1]
        )

    print()
    print("=" * 95)
    print("HUMAN RAW GEOMETRY")
    print("=" * 95)

    stats(
        "human hip width",
        hip_width,
    )

    stats(
        "human knee width",
        knee_width,
    )

    stats(
        "human foot width",
        foot_width,
    )

    print()

    stats(
        "human left thigh",
        left_thigh_len,
    )

    stats(
        "human right thigh",
        right_thigh_len,
    )

    stats(
        "human left shank",
        left_shank_len,
    )

    stats(
        "human right shank",
        right_shank_len,
    )

    print()
    print("=" * 95)
    print("CURRENT V1 TARGET AFTER ×0.9")
    print("=" * 95)

    stats(
        "target hip width",
        np.asarray(
            hip_width
        ) * CURRENT_SCALE,
    )

    stats(
        "target knee width",
        np.asarray(
            knee_width
        ) * CURRENT_SCALE,
    )

    stats(
        "target foot width",
        np.asarray(
            foot_width
        ) * CURRENT_SCALE,
    )

    print()
    print("=" * 95)
    print("LINGLONG GEOMETRY")
    print("=" * 95)

    print(
        "robot hip width :",
        ROBOT_HIP_WIDTH,
        "m",
    )

    print(
        "robot thigh     :",
        ROBOT_THIGH,
        "m",
    )

    print(
        "robot shank     :",
        ROBOT_SHANK,
        "m",
    )

    print()
    print("=" * 95)
    print("INWARD LEG MOTION IN HUMAN SOURCE")
    print("=" * 95)

    stats(
        "left knee inward",
        left_knee_inward,
    )

    stats(
        "right knee inward",
        right_knee_inward,
    )

    stats(
        "left foot inward",
        left_foot_inward,
    )

    stats(
        "right foot inward",
        right_foot_inward,
    )

    print()
    print("=" * 95)
    print("KEY RATIOS")
    print("=" * 95)

    median_human_hip = float(
        np.median(
            hip_width
        )
    )

    print(
        "human median hip width:",
        median_human_hip,
    )

    print(
        "current ×0.9 hip target:",
        median_human_hip
        * CURRENT_SCALE,
    )

    print(
        "LingLong hip width     :",
        ROBOT_HIP_WIDTH,
    )

    print(
        "required lateral ratio :",
        ROBOT_HIP_WIDTH
        / median_human_hip,
    )

    print()

    print(
        "LINGLONG LOWER MAPPING "
        "DIAGNOSIS: PASS"
    )


if __name__ == "__main__":
    main()
