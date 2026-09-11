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
    Anatomical chest frame:

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


def stats(name, x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    dx = np.diff(x)

    print(
        f"{name:30s}"
        f"min={x.min():8.2f}  "
        f"max={x.max():8.2f}  "
        f"median={np.median(x):8.2f}  "
        f"std={np.std(x):7.2f}  "
        f"maxstep={np.max(np.abs(dx)):7.2f}"
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

    frames, fps = (
        get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=30,
        )
    )

    assert len(frames) == 661


    chest_rot = []

    head_raw_rot = []


    for frame in frames:

        chest_rot.append(
            chest_frame(
                frame
            )
        )

        head_raw_rot.append(
            quat_wxyz_matrix(
                frame["head"][1]
            )
        )


    chest_rot = np.asarray(
        chest_rot,
        dtype=np.float64,
    )

    head_raw_rot = np.asarray(
        head_raw_rot,
        dtype=np.float64,
    )


    # ============================================================
    # CALIBRATION
    #
    # We do NOT assume that the SMPL-X head local coordinate
    # system is an anatomical XYZ frame.
    #
    # Find a CONSTANT rotation C so that:
    #
    #     R_head_raw @ C
    #
    # approximately aligns with the anatomical chest frame
    # during the first CALIBRATION_FRAMES.
    #
    # Each frame gives:
    #
    #     C_i = R_head_raw^T @ R_chest
    #
    # Average these constant rotations on SO(3).
    # ============================================================

    corrections = []

    for i in range(
        CALIBRATION_FRAMES
    ):

        C_i = (
            head_raw_rot[i].T
            @ chest_rot[i]
        )

        corrections.append(
            C_i
        )


    correction_rot = (
        R.from_matrix(
            np.asarray(
                corrections
            )
        )
        .mean()
    )

    correction = (
        correction_rot
        .as_matrix()
    )


    correction_euler = (
        correction_rot
        .as_euler(
            "ZYX",
            degrees=True,
        )
    )


    # ============================================================
    # Apply constant frame correction.
    # ============================================================

    head_yaw = []
    head_pitch = []
    head_roll = []


    for i in range(
        len(frames)
    ):

        corrected_head = (
            head_raw_rot[i]
            @ correction
        )


        # Head relative to geometrically constructed chest.
        rel = (
            chest_rot[i].T
            @ corrected_head
        )


        angles = (
            R.from_matrix(
                rel
            )
            .as_euler(
                "ZYX",
                degrees=True,
            )
        )


        head_yaw.append(
            angles[0]
        )

        head_pitch.append(
            angles[1]
        )

        head_roll.append(
            angles[2]
        )


    head_yaw = np.asarray(
        head_yaw
    )

    head_pitch = np.asarray(
        head_pitch
    )

    head_roll = np.asarray(
        head_roll
    )


    print("=" * 100)
    print("LINGLONG HEAD CALIBRATED DIAGNOSIS V3")
    print("=" * 100)

    print(
        "frames:",
        len(frames),
    )

    print(
        "fps metadata:",
        fps,
    )

    print(
        "calibration frames:",
        CALIBRATION_FRAMES,
    )


    print()
    print("=" * 100)
    print("1. CONSTANT SMPL-X HEAD FRAME CORRECTION")
    print("=" * 100)

    print(
        "correction ZYX deg:",
        correction_euler,
    )


    print()
    print("=" * 100)
    print("2. CALIBRATED HEAD RELATIVE TO GEOMETRIC CHEST")
    print("=" * 100)

    stats(
        "calibrated head yaw",
        head_yaw,
    )

    stats(
        "calibrated head pitch",
        head_pitch,
    )

    stats(
        "calibrated head roll",
        head_roll,
    )


    print()
    print("=" * 100)
    print("3. CALIBRATION WINDOW")
    print("=" * 100)

    stats(
        "first30 head yaw",
        head_yaw[
            :CALIBRATION_FRAMES
        ],
    )

    stats(
        "first30 head pitch",
        head_pitch[
            :CALIBRATION_FRAMES
        ],
    )

    stats(
        "first30 head roll",
        head_roll[
            :CALIBRATION_FRAMES
        ],
    )


    print()
    print("=" * 100)
    print("4. SAMPLE FRAMES")
    print("=" * 100)

    for i in [
        0,
        30,
        60,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        660,
    ]:

        print(
            f"{i:03d} | "
            f"yaw={head_yaw[i]:8.2f}°  "
            f"pitch={head_pitch[i]:8.2f}°  "
            f"roll={head_roll[i]:8.2f}°"
        )


    print()
    print("=" * 100)

    # LingLong:
    #
    # head yaw   +/- 90 deg
    # head pitch +/- 45 deg

    yaw_violation = np.sum(
        np.abs(
            head_yaw
        ) > 90.0
    )

    pitch_violation = np.sum(
        np.abs(
            head_pitch
        ) > 45.0
    )


    print(
        "head yaw > +/-90:",
        int(
            yaw_violation
        )
    )

    print(
        "head pitch > +/-45:",
        int(
            pitch_violation
        )
    )


    print()
    print(
        "LINGLONG HEAD CALIBRATED "
        "DIAGNOSIS V3: PASS"
    )


if __name__ == "__main__":
    main()
