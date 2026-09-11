#!/usr/bin/env python3

from pathlib import Path
import hashlib
import mujoco


ROOT = Path(__file__).resolve().parents[2]

ROBOT_DIR = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
)

SOURCE = ROBOT_DIR / "LingLong2.0_mujoco.urdf"
OUTPUT = ROBOT_DIR / "LingLong2.0_floating.xml"

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_model(model):
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

    actual = []

    for jid in range(1, model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )
        actual.append(name)

    assert actual == EXPECTED_JOINTS, (
        "LingLong2.0 joint order changed"
    )

    assert int(model.jnt_qposadr[0]) == 0
    assert int(model.jnt_qposadr[1]) == 7
    assert int(model.jnt_qposadr[-1]) == 36


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE}\n"
            "Run prepare_linglong20_mujoco.py first."
        )

    print("=" * 90)
    print("BUILD LINGLONG 2.0 FLOATING MJCF")
    print("=" * 90)
    print("source :", SOURCE)
    print("output :", OUTPUT)

    spec = mujoco.MjSpec.from_file(str(SOURCE))

    base = spec.body("base_link")
    if base is None:
        raise RuntimeError("base_link not found")

    base.add_freejoint(
        name="root_joint"
    )

    # First validate the in-memory model.
    model_memory = spec.compile()
    check_model(model_memory)

    print()
    print("In-memory model:")
    print("  nq    =", model_memory.nq)
    print("  nv    =", model_memory.nv)
    print("  njnt  =", model_memory.njnt)
    print("  nbody =", model_memory.nbody)

    # Serialize the MjSpec to real MJCF XML.
    xml_text = spec.to_xml()

    OUTPUT.write_text(
        xml_text,
        encoding="utf-8",
    )

    print()
    print("MJCF written:", OUTPUT)

    # Critical test:
    # reload from disk exactly as GMR will do later.
    model_disk = mujoco.MjModel.from_xml_path(
        str(OUTPUT)
    )

    check_model(model_disk)

    print()
    print("=" * 90)
    print("RELOADED FROM XML")
    print("=" * 90)

    print("nq    :", model_disk.nq)
    print("nv    :", model_disk.nv)
    print("njnt  :", model_disk.njnt)
    print("nbody :", model_disk.nbody)

    print()
    print("qpos0 root:")
    print(model_disk.qpos0[:7])

    print()
    print("37-D joint layout:")

    print("00-02 root xyz")
    print("03-06 root quaternion wxyz")

    for jid in range(1, model_disk.njnt):
        name = mujoco.mj_id2name(
            model_disk,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        qadr = int(
            model_disk.jnt_qposadr[jid]
        )

        print(
            f"{qadr:02d} {name}"
        )

    print()
    print("XML sha256:", sha256(OUTPUT))

    print()
    print(
        "LINGLONG2 FLOATING XML: PASS"
    )


if __name__ == "__main__":
    main()
