#!/usr/bin/env python3

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

TARGET_BODIES = [
    "base_link",

    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",

    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",

    "waist_yaw_link",
    "waist_pitch_link",

    "head_yaw_link",
    "head_pitch_link",

    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_wrist_yaw_link",

    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def body_id(model, name):
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )
    if bid < 0:
        raise RuntimeError(f"Body not found: {name}")
    return bid


def joint_id(model, name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")
    return jid


def main():
    if not XML.exists():
        raise FileNotFoundError(XML)

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )
    data = mujoco.MjData(model)

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    print("=" * 110)
    print("LINGLONG 2.0 ZERO-POSE BODY FRAMES")
    print("=" * 110)

    print(
        "model:",
        f"nq={model.nq}",
        f"nv={model.nv}",
        f"njnt={model.njnt}",
        f"nbody={model.nbody}",
    )

    print()
    print(
        "BODY                                     "
        "POSITION xyz [m]                  "
        "QUATERNION wxyz"
    )
    print("-" * 110)

    for name in TARGET_BODIES:
        bid = body_id(model, name)

        p = np.asarray(
            data.xpos[bid],
            dtype=np.float64,
        )
        q = np.asarray(
            data.xquat[bid],
            dtype=np.float64,
        )

        print(
            f"{name:38s} "
            f"[{p[0]: .6f} {p[1]: .6f} {p[2]: .6f}]  "
            f"[{q[0]: .6f} {q[1]: .6f} {q[2]: .6f} {q[3]: .6f}]"
        )

    print()
    print("=" * 110)
    print("JOINT AXES AT ZERO POSE")
    print("=" * 110)

    for jid in range(1, model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        qadr = int(model.jnt_qposadr[jid])

        axis = np.asarray(
            data.xaxis[jid],
            dtype=np.float64,
        )

        anchor = np.asarray(
            data.xanchor[jid],
            dtype=np.float64,
        )

        print(
            f"{qadr:02d} {name:34s} "
            f"axis=[{axis[0]: .6f} {axis[1]: .6f} {axis[2]: .6f}] "
            f"anchor=[{anchor[0]: .6f} {anchor[1]: .6f} {anchor[2]: .6f}]"
        )

    print()
    print("=" * 110)
    print("GEOMETRIC LENGTH CHECK")
    print("=" * 110)

    pairs = [
        (
            "left arm upper",
            "left_shoulder_pitch_joint",
            "left_elbow_joint",
        ),
        (
            "left arm forearm",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ),
        (
            "right arm upper",
            "right_shoulder_pitch_joint",
            "right_elbow_joint",
        ),
        (
            "right arm forearm",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ),
        (
            "left thigh",
            "left_hip_pitch_joint",
            "left_knee_joint",
        ),
        (
            "left shank",
            "left_knee_joint",
            "left_ankle_pitch_joint",
        ),
        (
            "right thigh",
            "right_hip_pitch_joint",
            "right_knee_joint",
        ),
        (
            "right shank",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ),
    ]

    for label, j1_name, j2_name in pairs:
        j1 = joint_id(model, j1_name)
        j2 = joint_id(model, j2_name)

        p1 = np.asarray(data.xanchor[j1])
        p2 = np.asarray(data.xanchor[j2])

        length = float(
            np.linalg.norm(p2 - p1)
        )

        print(
            f"{label:20s}: {length:.6f} m"
        )

    print()
    print("LINGLONG2 FRAME INSPECTION: PASS")


if __name__ == "__main__":
    main()
