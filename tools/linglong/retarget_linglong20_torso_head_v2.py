#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
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

XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

OUT_DIR = (
    ROOT
    / "output/linglong20_torso_head_v2"
)

OUT_CSV = (
    OUT_DIR
    / "linglong20_torso_head_v2_qpos37.csv"
)


WAIST_YAW = 19
WAIST_PITCH = 20
HEAD_YAW = 21
HEAD_PITCH = 22

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
            "Degenerate anatomical vector"
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


def pelvis_frame(frame):
    """
    +X forward
    +Y left
    +Z up
    """

    pelvis = np.asarray(
        frame["pelvis"][0],
        dtype=np.float64,
    )

    left_hip = np.asarray(
        frame["left_hip"][0],
        dtype=np.float64,
    )

    right_hip = np.asarray(
        frame["right_hip"][0],
        dtype=np.float64,
    )

    spine3 = np.asarray(
        frame["spine3"][0],
        dtype=np.float64,
    )


    y = normalize(
        left_hip - right_hip
    )

    z0 = normalize(
        spine3 - pelvis
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


def chest_frame(frame):
    """
    +X chest forward
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

    spine3 = np.asarray(
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
        neck - spine3
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


def print_stats(name, x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    dx = np.diff(x)

    print(
        f"{name:22s} "
        f"{x.min():8.2f} .. "
        f"{x.max():8.2f} deg | "
        f"median={np.median(x):7.2f} | "
        f"max step={np.max(np.abs(dx)):6.2f} deg"
    )


def joint_limit_deg(
    model,
    joint_name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    qadr = int(
        model.jnt_qposadr[jid]
    )

    limit = np.degrees(
        model.jnt_range[jid]
    )

    return (
        qadr,
        float(limit[0]),
        float(limit[1]),
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    base_qpos = np.loadtxt(
        V23_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert base_qpos.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(base_qpos)
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


    # ========================================================
    # 1. Build pelvis / chest frames
    # ========================================================

    pelvis_R = []
    chest_R = []
    raw_head_R = []


    for frame in frames:

        pelvis_R.append(
            pelvis_frame(
                frame
            )
        )

        chest_R.append(
            chest_frame(
                frame
            )
        )

        raw_head_R.append(
            quat_wxyz_matrix(
                frame["head"][1]
            )
        )


    pelvis_R = np.asarray(
        pelvis_R,
        dtype=np.float64,
    )

    chest_R = np.asarray(
        chest_R,
        dtype=np.float64,
    )

    raw_head_R = np.asarray(
        raw_head_R,
        dtype=np.float64,
    )


    # ========================================================
    # 2. Torso geometry
    #
    # pelvis^-1 * chest
    # ========================================================

    torso_angles = []


    for i in range(661):

        rel = (
            pelvis_R[i].T
            @ chest_R[i]
        )

        torso_angles.append(
            R.from_matrix(
                rel
            ).as_euler(
                "ZYX",
                degrees=False,
            )
        )


    torso_angles = np.asarray(
        torso_angles,
        dtype=np.float64,
    )


    # ========================================================
    # 3. Calibrate fixed SMPL-X head frame offset
    #
    # Find constant C:
    #
    # raw_head_R @ C ~= chest_R
    #
    # over neutral-ish first 30 frames.
    # ========================================================

    corrections = []


    for i in range(
        CALIBRATION_FRAMES
    ):

        corrections.append(
            raw_head_R[i].T
            @ chest_R[i]
        )


    correction = (
        R.from_matrix(
            np.asarray(
                corrections
            )
        )
        .mean()
        .as_matrix()
    )


    # ========================================================
    # 4. Calibrated head relative to geometric chest
    # ========================================================

    head_angles = []


    for i in range(661):

        corrected_head = (
            raw_head_R[i]
            @ correction
        )

        rel = (
            chest_R[i].T
            @ corrected_head
        )

        head_angles.append(
            R.from_matrix(
                rel
            ).as_euler(
                "ZYX",
                degrees=False,
            )
        )


    head_angles = np.asarray(
        head_angles,
        dtype=np.float64,
    )


    # ========================================================
    # 5. Start strictly from Lower V2.3
    # ========================================================

    qpos = base_qpos.copy()


    # LingLong joint axes:
    #
    # waist_yaw   +Z
    # waist_pitch +Y
    # head_yaw    +Z
    # head_pitch  +Y

    qpos[
        :,
        WAIST_YAW,
    ] = torso_angles[:, 0]

    qpos[
        :,
        WAIST_PITCH,
    ] = torso_angles[:, 1]

    qpos[
        :,
        HEAD_YAW,
    ] = head_angles[:, 0]

    qpos[
        :,
        HEAD_PITCH,
    ] = head_angles[:, 1]


    # ========================================================
    # 6. Validation
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )


    expected = {
        "waist_yaw_joint": WAIST_YAW,
        "waist_pitch_joint": WAIST_PITCH,
        "head_yaw_joint": HEAD_YAW,
        "head_pitch_joint": HEAD_PITCH,
    }


    print("=" * 100)
    print("LINGLONG TORSO / HEAD RETARGET V2")
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


    print("=" * 100)
    print("JOINT ADDRESS / LIMIT CHECK")
    print("=" * 100)


    limits = {}


    for name, expected_qadr in expected.items():

        (
            qadr,
            low,
            high,
        ) = joint_limit_deg(
            model,
            name,
        )

        print(
            f"{name:24s} "
            f"qpos={qadr:2d} "
            f"limit=[{low:7.2f}, {high:7.2f}] deg"
        )

        assert (
            qadr
            == expected_qadr
        )

        limits[
            expected_qadr
        ] = (
            low,
            high,
        )


    print()
    print("=" * 100)
    print("TORSO / HEAD OUTPUT")
    print("=" * 100)


    waist_yaw_deg = np.degrees(
        qpos[:, WAIST_YAW]
    )

    waist_pitch_deg = np.degrees(
        qpos[:, WAIST_PITCH]
    )

    head_yaw_deg = np.degrees(
        qpos[:, HEAD_YAW]
    )

    head_pitch_deg = np.degrees(
        qpos[:, HEAD_PITCH]
    )


    print_stats(
        "waist yaw",
        waist_yaw_deg,
    )

    print_stats(
        "waist pitch",
        waist_pitch_deg,
    )

    print_stats(
        "head yaw",
        head_yaw_deg,
    )

    print_stats(
        "head pitch",
        head_pitch_deg,
    )


    print()
    print("=" * 100)
    print("LIMIT COUNTS")
    print("=" * 100)


    for (
        name,
        idx,
        values,
    ) in [
        (
            "waist yaw",
            WAIST_YAW,
            waist_yaw_deg,
        ),
        (
            "waist pitch",
            WAIST_PITCH,
            waist_pitch_deg,
        ),
        (
            "head yaw",
            HEAD_YAW,
            head_yaw_deg,
        ),
        (
            "head pitch",
            HEAD_PITCH,
            head_pitch_deg,
        ),
    ]:

        low, high = limits[idx]

        lower_count = int(
            np.sum(
                values
                < low - 1e-6
            )
        )

        upper_count = int(
            np.sum(
                values
                > high + 1e-6
            )
        )

        print(
            f"{name:18s} "
            f"lower={lower_count:3d} "
            f"upper={upper_count:3d}"
        )

        assert lower_count == 0
        assert upper_count == 0


    # ========================================================
    # STRICT V2.3 PRESERVATION CHECK
    # ========================================================

    unchanged_mask = np.ones(
        37,
        dtype=bool,
    )

    unchanged_mask[
        19:23
    ] = False


    unchanged_diff = float(
        np.max(
            np.abs(
                qpos[
                    :,
                    unchanged_mask,
                ]
                - base_qpos[
                    :,
                    unchanged_mask,
                ]
            )
        )
    )


    changed_diff = np.max(
        np.abs(
            qpos[:, 19:23]
            - base_qpos[:, 19:23]
        ),
        axis=0,
    )


    print()
    print("=" * 100)
    print("V2.3 PRESERVATION")
    print("=" * 100)

    print(
        "all non torso/head qpos max diff:",
        unchanged_diff,
    )

    print(
        "waist_yaw change max:",
        np.degrees(
            changed_diff[0]
        ),
        "deg",
    )

    print(
        "waist_pitch change max:",
        np.degrees(
            changed_diff[1]
        ),
        "deg",
    )

    print(
        "head_yaw change max:",
        np.degrees(
            changed_diff[2]
        ),
        "deg",
    )

    print(
        "head_pitch change max:",
        np.degrees(
            changed_diff[3]
        ),
        "deg",
    )


    assert unchanged_diff < 1e-12


    assert qpos.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(qpos)
    )


    np.savetxt(
        OUT_CSV,
        qpos,
        delimiter=",",
        fmt="%.12f",
    )


    print()
    print("=" * 100)
    print("OUTPUT")
    print("=" * 100)

    print(
        "shape:",
        qpos.shape,
    )

    print(
        "finite:",
        bool(
            np.all(
                np.isfinite(qpos)
            )
        ),
    )

    print(
        "saved:",
        OUT_CSV,
    )

    print()

    print(
        "LINGLONG TORSO / HEAD "
        "RETARGET V2: PASS"
    )


if __name__ == "__main__":
    main()
