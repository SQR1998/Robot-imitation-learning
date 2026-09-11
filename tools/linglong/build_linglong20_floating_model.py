#!/usr/bin/env python3

from pathlib import Path
import mujoco


ROOT = Path(__file__).resolve().parents[2]

URDF = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_mujoco.urdf"
)


EXPECTED_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_pitch_joint",

    "head_yaw_joint",
    "head_pitch_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def build_model():
    if not URDF.exists():
        raise FileNotFoundError(
            f"{URDF}\n"
            "Run tools/linglong/prepare_linglong20_mujoco.py first."
        )

    spec = mujoco.MjSpec.from_file(str(URDF))

    base = spec.body("base_link")
    if base is None:
        raise RuntimeError("base_link not found")

    base.add_freejoint(name="root_joint")

    model = spec.compile()

    return model


def main():
    model = build_model()

    print("=" * 90)
    print("LingLong 2.0 floating-base model")
    print("=" * 90)

    print("nq    :", model.nq)
    print("nv    :", model.nv)
    print("njnt  :", model.njnt)
    print("nbody :", model.nbody)

    assert model.nq == 37, model.nq
    assert model.nv == 36, model.nv
    assert model.njnt == 31, model.njnt
    assert model.nbody == 32, model.nbody

    root_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        0,
    )

    assert root_name == "root_joint", root_name
    assert int(model.jnt_qposadr[0]) == 0
    assert int(model.jnt_dofadr[0]) == 0

    print()
    print("=" * 90)
    print("37-D QPOS LAYOUT")
    print("=" * 90)

    print("00 root_x")
    print("01 root_y")
    print("02 root_z")
    print("03 root_qw")
    print("04 root_qx")
    print("05 root_qy")
    print("06 root_qz")

    actual_robot_joints = []

    for jid in range(1, model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        actual_robot_joints.append(name)

        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])

        print(
            f"{qadr:02d} "
            f"{name:34s} "
            f"dof={dadr:02d}"
        )

    assert actual_robot_joints == EXPECTED_JOINTS, (
        "Joint order mismatch"
    )

    assert int(model.jnt_qposadr[1]) == 7
    assert int(model.jnt_qposadr[-1]) == 36

    print()
    print("LINGLONG2 37-D FLOATING MODEL: PASS")


if __name__ == "__main__":
    main()
