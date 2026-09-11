#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)


ROOT = Path(__file__).resolve().parents[2]

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

IK_CONFIG = (
    ROOT
    / "general_motion_retargeting/ik_configs"
    / "smplx_to_linglong20.json"
)

ROBOT_CSV = (
    ROOT
    / "output/linglong20_arm_v15_strong_extension"
    / "linglong20_arm_v15_strong_extension_qpos37.csv"
)


def quat_wxyz_to_rot(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)

    return R.from_quat(
        q[[1, 2, 3, 0]]
    )


def rot_from_robot_qpos(q):
    return quat_wxyz_to_rot(
        q[3:7]
    )


def rpy_deg(rot):
    yaw, pitch, roll = rot.as_euler(
        "ZYX",
        degrees=True,
    )

    return np.array(
        [roll, pitch, yaw],
        dtype=np.float64,
    )


def tilt_deg(rot):
    M = rot.as_matrix()

    up = M[:, 2]

    c = float(
        np.clip(
            up[2]
            / max(
                np.linalg.norm(up),
                1e-12,
            ),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


with IK_CONFIG.open(
    "r",
    encoding="utf-8",
) as f:
    cfg = json.load(f)


# Both stage-1 and stage-2 use the same pelvis rotation offset.
entry = cfg[
    "ik_match_table1"
][
    "base_link"
]

(
    human_body,
    position_weight,
    rotation_weight,
    position_offset,
    rotation_offset_wxyz,
) = entry


print("=" * 110)
print("LINGLONG ROOT TARGET DIAGNOSTIC V2")
print("=" * 110)

print("human body       :", human_body)
print("position weight  :", position_weight)
print("rotation weight  :", rotation_weight)
print(
    "rotation offset  :",
    rotation_offset_wxyz,
)


# Exactly the same conversion used by motion_retarget.py.
rot_offset = R.from_quat(
    np.asarray(
        rotation_offset_wxyz,
        dtype=np.float64,
    )[[1, 2, 3, 0]]
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


frames, fps = get_smplx_data_offline_fast(
    smplx_data,
    body_model,
    smplx_output,
    tgt_fps=30,
)


robot_q = np.loadtxt(
    ROBOT_CSV,
    delimiter=",",
)


if len(frames) != len(robot_q):
    raise RuntimeError(
        f"frame mismatch: "
        f"human={len(frames)} "
        f"robot={len(robot_q)}"
    )


human_raw_rpy = []
target_rpy = []
robot_rpy = []

human_raw_tilt = []
target_tilt = []
robot_tilt = []

target_robot_error = []


for i, frame in enumerate(frames):

    # ----------------------------------------------------------
    # Human pelvis BEFORE GMR rotation offset.
    # ----------------------------------------------------------

    human_rot = quat_wxyz_to_rot(
        frame["pelvis"][1]
    )

    # ----------------------------------------------------------
    # EXACT GMR orientation target:
    #
    # motion_retarget.py:
    #
    #   r_quat * rot_offsets[body_name]
    #
    # ----------------------------------------------------------

    target_rot = (
        human_rot
        * rot_offset
    )

    # ----------------------------------------------------------
    # Actual solved LingLong root.
    # ----------------------------------------------------------

    robot_rot = rot_from_robot_qpos(
        robot_q[i]
    )

    human_raw_rpy.append(
        rpy_deg(human_rot)
    )

    target_rpy.append(
        rpy_deg(target_rot)
    )

    robot_rpy.append(
        rpy_deg(robot_rot)
    )

    human_raw_tilt.append(
        tilt_deg(human_rot)
    )

    target_tilt.append(
        tilt_deg(target_rot)
    )

    robot_tilt.append(
        tilt_deg(robot_rot)
    )

    # Angular difference target -> actual robot root.
    delta = (
        target_rot.inv()
        * robot_rot
    )

    target_robot_error.append(
        np.degrees(
            np.linalg.norm(
                delta.as_rotvec()
            )
        )
    )


human_raw_rpy = np.asarray(
    human_raw_rpy
)

target_rpy = np.asarray(
    target_rpy
)

robot_rpy = np.asarray(
    robot_rpy
)

human_raw_tilt = np.asarray(
    human_raw_tilt
)

target_tilt = np.asarray(
    target_tilt
)

robot_tilt = np.asarray(
    robot_tilt
)

target_robot_error = np.asarray(
    target_robot_error
)


def stats(name, x):

    print(
        f"{name:30s} "
        f"mean={np.mean(x):8.3f} "
        f"median={np.median(x):8.3f} "
        f"min={np.min(x):8.3f} "
        f"max={np.max(x):8.3f} "
        f"std={np.std(x):8.3f}"
    )


print()
print("=" * 110)
print("TILT STATISTICS")
print("=" * 110)

stats(
    "human RAW pelvis tilt",
    human_raw_tilt,
)

stats(
    "GMR target pelvis tilt",
    target_tilt,
)

stats(
    "actual robot root tilt",
    robot_tilt,
)

stats(
    "target -> robot angle err",
    target_robot_error,
)


print()
print("=" * 110)
print("TARGET ROLL / PITCH")
print("=" * 110)

stats(
    "target roll",
    target_rpy[:, 0],
)

stats(
    "target pitch",
    target_rpy[:, 1],
)

stats(
    "robot roll",
    robot_rpy[:, 0],
)

stats(
    "robot pitch",
    robot_rpy[:, 1],
)


sample_frames = [
    0,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    407,
    422,
    450,
    500,
    550,
    600,
    650,
    660,
]


print()
print("=" * 110)
print("REPRESENTATIVE FRAMES")
print("=" * 110)


for i in sample_frames:

    print(
        f"frame={i:3d} | "
        f"target "
        f"roll={target_rpy[i,0]:7.2f} "
        f"pitch={target_rpy[i,1]:7.2f} "
        f"tilt={target_tilt[i]:7.2f} || "
        f"robot "
        f"roll={robot_rpy[i,0]:7.2f} "
        f"pitch={robot_rpy[i,1]:7.2f} "
        f"tilt={robot_tilt[i]:7.2f} || "
        f"err={target_robot_error[i]:7.2f}"
    )


print()
print("=" * 110)
print("WORST ROBOT ROOT EXCESS TILT")
print("=" * 110)

excess = (
    robot_tilt
    - target_tilt
)

order = np.argsort(
    excess
)[::-1]


for i in order[:30]:

    print(
        f"frame={i:3d} "
        f"target={target_tilt[i]:7.2f} "
        f"robot={robot_tilt[i]:7.2f} "
        f"excess={excess[i]:7.2f} "
        f"ori_err={target_robot_error[i]:7.2f}"
    )


print()
print("ROOT TARGET DIAGNOSTIC V2: PASS")
