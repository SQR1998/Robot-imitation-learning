#!/usr/bin/env python3

from pathlib import Path
import sys
import csv

import numpy as np
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)


V23_CSV = (
    ROOT
    / "output/linglong20_lower_v23"
    / "linglong20_lower_v23_qpos37.csv"
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
    / "output/linglong20_torso_head_v1"
)

OUT_CSV = (
    OUT_DIR
    / "torso_head_diagnostics.csv"
)


WAIST_YAW = 19
WAIST_PITCH = 20

HEAD_YAW = 21
HEAD_PITCH = 22


def quat_wxyz_to_rot(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    )


def relative_zyx_deg(
    parent_q,
    child_q,
):
    """
    Relative orientation:

        parent^-1 * child

    Return:
        yaw(Z), pitch(Y), roll(X)

    This corresponds naturally to LingLong's
    yaw(Z) + pitch(Y) joint structure.
    """

    parent = quat_wxyz_to_rot(
        parent_q
    )

    child = quat_wxyz_to_rot(
        child_q
    )

    rel = (
        parent.inv()
        * child
    )

    zyx = rel.as_euler(
        "ZYX",
        degrees=True,
    )

    return zyx


def stats(
    name,
    x,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    dx = np.diff(x)

    print(
        f"{name:24s}: "
        f"min={x.min():8.2f}  "
        f"max={x.max():8.2f}  "
        f"median={np.median(x):8.2f}  "
        f"std={np.std(x):7.2f}  "
        f"max_step={np.max(np.abs(dx)):7.2f}"
    )


def compare(
    title,
    human,
    robot,
):

    human = np.asarray(
        human,
        dtype=np.float64,
    )

    robot = np.asarray(
        robot,
        dtype=np.float64,
    )

    hc = (
        human
        - np.median(human)
    )

    rc = (
        robot
        - np.median(robot)
    )


    if (
        np.std(hc) < 1e-9
        or np.std(rc) < 1e-9
    ):
        corr = np.nan
    else:
        corr = float(
            np.corrcoef(
                hc,
                rc,
            )[0, 1]
        )


    denom = float(
        np.dot(
            hc,
            hc,
        )
    )

    if denom > 1e-12:

        gain = float(
            np.dot(
                hc,
                rc,
            )
            / denom
        )

    else:

        gain = np.nan


    if (
        np.std(hc) < 1e-9
        or np.std(-rc) < 1e-9
    ):
        mirror_corr = np.nan
    else:
        mirror_corr = float(
            np.corrcoef(
                hc,
                -rc,
            )[0, 1]
        )


    print()
    print(title)

    print(
        "  correlation:",
        corr,
    )

    print(
        "  gain:",
        gain,
    )

    print(
        "  mirrored correlation:",
        mirror_corr,
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    qpos = np.loadtxt(
        V23_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert qpos.shape == (
        661,
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


    assert len(frames) == 661


    required = [
        "pelvis",
        "spine3",
        "neck",
        "head",
    ]


    print("=" * 100)
    print("LINGLONG TORSO / HEAD DIAGNOSIS V1")
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
        "required SMPL-X joints:"
    )

    for name in required:

        print(
            f"  {name:8s}:",
            name in frames[0],
        )

        if name not in frames[0]:
            raise RuntimeError(
                f"Missing SMPL-X joint: {name}"
            )


    torso_yaw = []
    torso_pitch = []
    torso_roll = []

    neck_yaw = []
    neck_pitch = []
    neck_roll = []

    head_yaw = []
    head_pitch = []
    head_roll = []


    for frame in frames:

        # ====================================================
        # Chest relative to pelvis
        # ====================================================

        t = relative_zyx_deg(
            frame["pelvis"][1],
            frame["spine3"][1],
        )

        torso_yaw.append(
            t[0]
        )

        torso_pitch.append(
            t[1]
        )

        torso_roll.append(
            t[2]
        )


        # ====================================================
        # Neck relative to chest
        # ====================================================

        n = relative_zyx_deg(
            frame["spine3"][1],
            frame["neck"][1],
        )

        neck_yaw.append(
            n[0]
        )

        neck_pitch.append(
            n[1]
        )

        neck_roll.append(
            n[2]
        )


        # ====================================================
        # Full head relative to chest
        # ====================================================

        h = relative_zyx_deg(
            frame["spine3"][1],
            frame["head"][1],
        )

        head_yaw.append(
            h[0]
        )

        head_pitch.append(
            h[1]
        )

        head_roll.append(
            h[2]
        )


    torso_yaw = np.asarray(
        torso_yaw
    )

    torso_pitch = np.asarray(
        torso_pitch
    )

    torso_roll = np.asarray(
        torso_roll
    )


    neck_yaw = np.asarray(
        neck_yaw
    )

    neck_pitch = np.asarray(
        neck_pitch
    )

    neck_roll = np.asarray(
        neck_roll
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


    robot_waist_yaw = np.degrees(
        qpos[
            :,
            WAIST_YAW,
        ]
    )

    robot_waist_pitch = np.degrees(
        qpos[
            :,
            WAIST_PITCH,
        ]
    )

    robot_head_yaw = np.degrees(
        qpos[
            :,
            HEAD_YAW,
        ]
    )

    robot_head_pitch = np.degrees(
        qpos[
            :,
            HEAD_PITCH,
        ]
    )


    print()
    print("=" * 100)
    print("1. HUMAN TORSO: pelvis -> spine3")
    print("=" * 100)

    stats(
        "human torso yaw",
        torso_yaw,
    )

    stats(
        "human torso pitch",
        torso_pitch,
    )

    stats(
        "human torso roll",
        torso_roll,
    )


    print()
    print("=" * 100)
    print("2. CURRENT ROBOT WAIST")
    print("=" * 100)

    stats(
        "robot waist yaw",
        robot_waist_yaw,
    )

    stats(
        "robot waist pitch",
        robot_waist_pitch,
    )


    print()

    print(
        "waist pitch at +30deg:",
        int(
            np.sum(
                robot_waist_pitch
                > 29.9
            )
        ),
        "frames",
    )

    print(
        "waist pitch at -30deg:",
        int(
            np.sum(
                robot_waist_pitch
                < -29.9
            )
        ),
        "frames",
    )


    print()
    print("=" * 100)
    print("3. HUMAN TORSO -> ROBOT WAIST")
    print("=" * 100)

    compare(
        "TORSO YAW -> WAIST YAW",
        torso_yaw,
        robot_waist_yaw,
    )

    compare(
        "TORSO PITCH -> WAIST PITCH",
        torso_pitch,
        robot_waist_pitch,
    )


    print()
    print("=" * 100)
    print("4. HUMAN NECK: spine3 -> neck")
    print("=" * 100)

    stats(
        "human neck yaw",
        neck_yaw,
    )

    stats(
        "human neck pitch",
        neck_pitch,
    )

    stats(
        "human neck roll",
        neck_roll,
    )


    print()
    print("=" * 100)
    print("5. HUMAN HEAD: spine3 -> head")
    print("=" * 100)

    stats(
        "human head yaw",
        head_yaw,
    )

    stats(
        "human head pitch",
        head_pitch,
    )

    stats(
        "human head roll",
        head_roll,
    )


    print()
    print("=" * 100)
    print("6. CURRENT ROBOT HEAD")
    print("=" * 100)

    stats(
        "robot head yaw",
        robot_head_yaw,
    )

    stats(
        "robot head pitch",
        robot_head_pitch,
    )


    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "frame",

                "human_torso_yaw_deg",
                "human_torso_pitch_deg",
                "human_torso_roll_deg",

                "robot_waist_yaw_deg",
                "robot_waist_pitch_deg",

                "human_neck_yaw_deg",
                "human_neck_pitch_deg",
                "human_neck_roll_deg",

                "human_head_yaw_deg",
                "human_head_pitch_deg",
                "human_head_roll_deg",

                "robot_head_yaw_deg",
                "robot_head_pitch_deg",
            ]
        )


        for i in range(661):

            writer.writerow(
                [
                    i,

                    torso_yaw[i],
                    torso_pitch[i],
                    torso_roll[i],

                    robot_waist_yaw[i],
                    robot_waist_pitch[i],

                    neck_yaw[i],
                    neck_pitch[i],
                    neck_roll[i],

                    head_yaw[i],
                    head_pitch[i],
                    head_roll[i],

                    robot_head_yaw[i],
                    robot_head_pitch[i],
                ]
            )


    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        "saved:",
        OUT_CSV,
    )

    print()

    print(
        "LINGLONG TORSO / HEAD "
        "DIAGNOSIS V1: PASS"
    )


if __name__ == "__main__":
    main()
