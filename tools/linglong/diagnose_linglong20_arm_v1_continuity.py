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


CSV = (
    ROOT
    / "output/linglong20_arm_v1"
    / "linglong20_arm_v1_qpos37.csv"
)

XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
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


LEFT_ARM = np.arange(23, 30)
RIGHT_ARM = np.arange(30, 37)


def normalize(v):

    v = np.asarray(v, dtype=np.float64)

    n = np.linalg.norm(v)

    if n < 1e-10:
        return None

    return v / n


def angle_deg(a, b):

    a = normalize(a)
    b = normalize(b)

    if a is None or b is None:
        return np.nan

    c = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


def body_id(model, name):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(name)

    return bid


def joint_name(model, qidx):

    for jid in range(model.njnt):

        if int(
            model.jnt_qposadr[jid]
        ) == int(qidx):

            return mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )

    return "UNKNOWN"


def human_segment_steps(frames, side):

    upper = []
    fore = []


    for frame in frames:

        shoulder = np.asarray(
            frame[
                f"{side}_shoulder"
            ][0]
        )

        elbow = np.asarray(
            frame[
                f"{side}_elbow"
            ][0]
        )

        wrist = np.asarray(
            frame[
                f"{side}_wrist"
            ][0]
        )


        upper.append(
            normalize(
                elbow - shoulder
            )
        )

        fore.append(
            normalize(
                wrist - elbow
            )
        )


    upper_step = np.array(
        [
            angle_deg(
                upper[i - 1],
                upper[i],
            )
            for i in range(1, len(upper))
        ]
    )


    fore_step = np.array(
        [
            angle_deg(
                fore[i - 1],
                fore[i],
            )
            for i in range(1, len(fore))
        ]
    )


    return (
        upper_step,
        fore_step,
    )


def robot_segment_steps(
    model,
    data,
    qpos,
    side,
):

    shoulder_id = body_id(
        model,
        f"{side}_shoulder_pitch_link",
    )

    elbow_id = body_id(
        model,
        f"{side}_elbow_link",
    )

    wrist_id = body_id(
        model,
        f"{side}_wrist_roll_link",
    )


    upper = []
    fore = []


    for q in qpos:

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data,
        )


        shoulder = (
            data.xpos[
                shoulder_id
            ].copy()
        )

        elbow = (
            data.xpos[
                elbow_id
            ].copy()
        )

        wrist = (
            data.xpos[
                wrist_id
            ].copy()
        )


        upper.append(
            normalize(
                elbow - shoulder
            )
        )

        fore.append(
            normalize(
                wrist - elbow
            )
        )


    upper_step = np.array(
        [
            angle_deg(
                upper[i - 1],
                upper[i],
            )
            for i in range(1, len(upper))
        ]
    )


    fore_step = np.array(
        [
            angle_deg(
                fore[i - 1],
                fore[i],
            )
            for i in range(1, len(fore))
        ]
    )


    return (
        upper_step,
        fore_step,
    )


def report_side(
    side,
    model,
    qpos,
    indices,
    human_upper,
    human_fore,
    robot_upper,
    robot_fore,
):

    joint_steps = np.degrees(
        np.abs(
            np.diff(
                qpos[:, indices],
                axis=0,
            )
        )
    )


    print()
    print("=" * 105)
    print(
        side.upper(),
        "ARM CONTINUITY",
    )
    print("=" * 105)


    print(
        "human upper-arm max step :",
        np.max(human_upper),
        "deg",
    )

    print(
        "robot upper-arm max step :",
        np.max(robot_upper),
        "deg",
    )

    print(
        "human forearm max step   :",
        np.max(human_fore),
        "deg",
    )

    print(
        "robot forearm max step   :",
        np.max(robot_fore),
        "deg",
    )


    # --------------------------------------------------------
    # Largest individual robot joint jumps
    # --------------------------------------------------------

    flat_order = np.argsort(
        joint_steps.ravel()
    )[::-1]


    print()
    print(
        "TOP 15 ROBOT JOINT STEPS"
    )


    shown = 0


    for flat_idx in flat_order:

        row, col = np.unravel_index(
            flat_idx,
            joint_steps.shape,
        )

        value = joint_steps[
            row,
            col,
        ]

        frame = row + 1
        qidx = int(
            indices[col]
        )


        print(
            f"frame {frame:03d} | "
            f"{joint_name(model,qidx):28s} "
            f"{value:7.2f}° | "
            f"H upper={human_upper[row]:6.2f}° "
            f"H fore={human_fore[row]:6.2f}° | "
            f"R upper={robot_upper[row]:6.2f}° "
            f"R fore={robot_fore[row]:6.2f}°"
        )


        shown += 1

        if shown >= 15:
            break


    # --------------------------------------------------------
    # Actual Cartesian segment jump frames
    # --------------------------------------------------------

    metric = np.maximum(
        robot_upper,
        robot_fore,
    )

    order = np.argsort(
        metric
    )[::-1][:12]


    print()
    print(
        "TOP ACTUAL ARM-SEGMENT MOTION FRAMES"
    )


    for row in order:

        frame = int(
            row + 1
        )

        print(
            f"frame {frame:03d} | "
            f"H upper={human_upper[row]:6.2f}° "
            f"H fore={human_fore[row]:6.2f}° | "
            f"R upper={robot_upper[row]:6.2f}° "
            f"R fore={robot_fore[row]:6.2f}°"
        )


def main():

    qpos = np.loadtxt(
        CSV,
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


    frames, fps = (
        get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=30,
        )
    )

    assert len(frames) == 661


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(
        model
    )


    print("=" * 105)
    print("LINGLONG ARM V1 CONTINUITY DIAGNOSIS")
    print("=" * 105)

    print(
        "frames:",
        len(frames),
    )

    print(
        "human height:",
        human_height,
    )

    print(
        "fps metadata:",
        fps,
    )


    for side, indices in [
        (
            "left",
            LEFT_ARM,
        ),
        (
            "right",
            RIGHT_ARM,
        ),
    ]:

        (
            human_upper,
            human_fore,
        ) = human_segment_steps(
            frames,
            side,
        )


        (
            robot_upper,
            robot_fore,
        ) = robot_segment_steps(
            model,
            data,
            qpos,
            side,
        )


        report_side(
            side,
            model,
            qpos,
            indices,
            human_upper,
            human_fore,
            robot_upper,
            robot_fore,
        )


    print()
    print("=" * 105)
    print(
        "LINGLONG ARM V1 "
        "CONTINUITY DIAGNOSIS: PASS"
    )
    print("=" * 105)


if __name__ == "__main__":
    main()
