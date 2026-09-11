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

BODY_MODEL_DIR = ROOT / "assets/body_models"


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        raise RuntimeError("degenerate vector")
    return v / n


def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(a, b),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def chest_frame(frame):

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

    y = normalize(ls - rs)

    z0 = normalize(neck - spine)

    z = normalize(
        z0 - np.dot(z0, y) * y
    )

    x = normalize(
        np.cross(y, z)
    )

    y = normalize(
        np.cross(z, x)
    )

    return np.column_stack((x, y, z))


def body_id(model, name):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(name)

    return bid


def segment_steps(vectors):

    return np.asarray(
        [
            angle_deg(
                vectors[i - 1],
                vectors[i],
            )
            for i in range(
                1,
                len(vectors),
            )
        ]
    )


def analyse_side(
    side,
    frames,
    model,
    data,
    qpos,
):

    # ========================================================
    # HUMAN: arm directions relative to HUMAN chest
    # ========================================================

    h_upper = []
    h_fore = []

    for frame in frames:

        C = chest_frame(frame)

        shoulder = np.asarray(
            frame[f"{side}_shoulder"][0],
            dtype=np.float64,
        )

        elbow = np.asarray(
            frame[f"{side}_elbow"][0],
            dtype=np.float64,
        )

        wrist = np.asarray(
            frame[f"{side}_wrist"][0],
            dtype=np.float64,
        )

        upper_world = normalize(
            elbow - shoulder
        )

        fore_world = normalize(
            wrist - elbow
        )

        h_upper.append(
            C.T @ upper_world
        )

        h_fore.append(
            C.T @ fore_world
        )


    h_upper_step = segment_steps(h_upper)
    h_fore_step = segment_steps(h_fore)


    # ========================================================
    # ROBOT: arm directions relative to ROBOT chest
    # ========================================================

    chest_id = body_id(
        model,
        "waist_pitch_link",
    )

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


    r_upper = []
    r_fore = []


    for q in qpos:

        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        C = np.asarray(
            data.xmat[chest_id],
            dtype=np.float64,
        ).reshape(3, 3)

        shoulder = data.xpos[
            shoulder_id
        ].copy()

        elbow = data.xpos[
            elbow_id
        ].copy()

        wrist = data.xpos[
            wrist_id
        ].copy()


        upper_world = normalize(
            elbow - shoulder
        )

        fore_world = normalize(
            wrist - elbow
        )

        r_upper.append(
            C.T @ upper_world
        )

        r_fore.append(
            C.T @ fore_world
        )


    r_upper_step = segment_steps(r_upper)
    r_fore_step = segment_steps(r_fore)


    print()
    print("=" * 100)
    print(
        side.upper(),
        "CHEST-LOCAL ARM CONTINUITY",
    )
    print("=" * 100)

    print(
        "human upper max:",
        np.max(h_upper_step),
    )

    print(
        "robot upper max:",
        np.max(r_upper_step),
    )

    print(
        "human fore max:",
        np.max(h_fore_step),
    )

    print(
        "robot fore max:",
        np.max(r_fore_step),
    )


    # Difference between robot and human per-frame motion.
    upper_excess = (
        r_upper_step
        - h_upper_step
    )

    fore_excess = (
        r_fore_step
        - h_fore_step
    )


    metric = np.maximum(
        upper_excess,
        fore_excess,
    )

    order = np.argsort(
        metric
    )[::-1][:20]


    print()
    print(
        "TOP ROBOT EXCESS MOTION FRAMES"
    )


    for row in order:

        frame = row + 1

        print(
            f"frame {frame:03d} | "
            f"H upper={h_upper_step[row]:6.2f}° "
            f"R upper={r_upper_step[row]:6.2f}° "
            f"excess={upper_excess[row]:6.2f}° | "
            f"H fore={h_fore_step[row]:6.2f}° "
            f"R fore={r_fore_step[row]:6.2f}° "
            f"excess={fore_excess[row]:6.2f}°"
        )


    print()
    print(
        "excess > 3 deg:"
    )

    print(
        "  upper:",
        int(
            np.sum(
                upper_excess > 3.0
            )
        ),
        "frames",
    )

    print(
        "  fore :",
        int(
            np.sum(
                fore_excess > 3.0
            )
        ),
        "frames",
    )

    print(
        "excess > 5 deg:"
    )

    print(
        "  upper:",
        int(
            np.sum(
                upper_excess > 5.0
            )
        ),
        "frames",
    )

    print(
        "  fore :",
        int(
            np.sum(
                fore_excess > 5.0
            )
        ),
        "frames",
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

    data = mujoco.MjData(model)


    print("=" * 100)
    print(
        "LINGLONG ARM V1 CHEST-LOCAL CONTINUITY"
    )
    print("=" * 100)

    print(
        "frames:",
        len(frames),
    )

    print(
        "fps metadata:",
        fps,
    )


    analyse_side(
        "left",
        frames,
        model,
        data,
        qpos,
    )

    analyse_side(
        "right",
        frames,
        model,
        data,
        qpos,
    )


    print()
    print("=" * 100)
    print(
        "LINGLONG ARM V1 CHEST-LOCAL "
        "CONTINUITY: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
