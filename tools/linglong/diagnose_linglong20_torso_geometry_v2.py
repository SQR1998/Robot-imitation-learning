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


def normalize(v):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = np.linalg.norm(v)

    if n < 1e-10:
        raise RuntimeError(
            "Degenerate vector"
        )

    return v / n


def quat_wxyz_to_rot(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    )


def pelvis_frame(frame):
    """
    Anatomical pelvis frame.

    +X = forward
    +Y = left
    +Z = up
    """

    pelvis = np.asarray(
        frame["pelvis"][0],
        dtype=np.float64,
    )

    lh = np.asarray(
        frame["left_hip"][0],
        dtype=np.float64,
    )

    rh = np.asarray(
        frame["right_hip"][0],
        dtype=np.float64,
    )

    spine = np.asarray(
        frame["spine3"][0],
        dtype=np.float64,
    )

    y = normalize(
        lh - rh
    )

    z0 = normalize(
        spine - pelvis
    )

    z = normalize(
        z0
        - np.dot(
            z0,
            y,
        )
        * y
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
    Anatomical chest frame from shoulders.

    +X = chest forward
    +Y = left shoulder direction
    +Z = chest up
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
        )
        * y
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


def geometry_torso_angles(frame):

    Rp = pelvis_frame(
        frame
    )

    Rc = chest_frame(
        frame
    )

    # Chest relative to pelvis.
    Rrel = (
        Rp.T
        @ Rc
    )

    return R.from_matrix(
        Rrel
    ).as_euler(
        "ZYX",
        degrees=True,
    )


def quaternion_torso_angles(frame):

    pelvis = quat_wxyz_to_rot(
        frame["pelvis"][1]
    )

    chest = quat_wxyz_to_rot(
        frame["spine3"][1]
    )

    rel = (
        pelvis.inv()
        * chest
    )

    return rel.as_euler(
        "ZYX",
        degrees=True,
    )


def stats(name, values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    dx = np.diff(
        x
    )

    print(
        f"{name:30s}"
        f"min={x.min():8.2f}  "
        f"max={x.max():8.2f}  "
        f"median={np.median(x):8.2f}  "
        f"std={np.std(x):7.2f}  "
        f"maxstep={np.max(np.abs(dx)):7.2f}"
    )


def corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    a = (
        a
        - np.median(a)
    )

    b = (
        b
        - np.median(b)
    )

    if (
        np.std(a) < 1e-10
        or np.std(b) < 1e-10
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
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


    print("=" * 100)
    print("LINGLONG TORSO GEOMETRY DIAGNOSIS V2")
    print("=" * 100)

    print(
        "frames:",
        len(frames),
    )

    print(
        "fps metadata:",
        fps,
    )

    print()

    print("AVAILABLE UPPER-BODY JOINTS")

    for key in [
        "pelvis",
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "head",
        "left_shoulder",
        "right_shoulder",
        "jaw",
        "left_eye",
        "right_eye",
    ]:

        print(
            f"{key:18s}:",
            key in frames[0],
        )


    geom = []
    quat = []


    for frame in frames:

        geom.append(
            geometry_torso_angles(
                frame
            )
        )

        quat.append(
            quaternion_torso_angles(
                frame
            )
        )


    geom = np.asarray(
        geom
    )

    quat = np.asarray(
        quat
    )


    print()
    print("=" * 100)
    print("1. GEOMETRY-BASED CHEST RELATIVE TO PELVIS")
    print("=" * 100)

    stats(
        "geometry torso yaw",
        geom[:, 0],
    )

    stats(
        "geometry torso pitch",
        geom[:, 1],
    )

    stats(
        "geometry torso roll",
        geom[:, 2],
    )


    print()
    print("=" * 100)
    print("2. QUATERNION-BASED SPINE3 RELATIVE TO PELVIS")
    print("=" * 100)

    stats(
        "quaternion torso yaw",
        quat[:, 0],
    )

    stats(
        "quaternion torso pitch",
        quat[:, 1],
    )

    stats(
        "quaternion torso roll",
        quat[:, 2],
    )


    print()
    print("=" * 100)
    print("3. GEOMETRY vs QUATERNION")
    print("=" * 100)

    print(
        "yaw correlation  :",
        corr(
            geom[:, 0],
            quat[:, 0],
        )
    )

    print(
        "pitch correlation:",
        corr(
            geom[:, 1],
            quat[:, 1],
        )
    )

    print(
        "roll correlation :",
        corr(
            geom[:, 2],
            quat[:, 2],
        )
    )


    print()
    print("=" * 100)
    print("4. FIRST 10 GEOMETRY FRAMES")
    print("=" * 100)

    for i in range(10):

        print(
            f"{i:03d} | "
            f"yaw={geom[i,0]:8.2f}° "
            f"pitch={geom[i,1]:8.2f}° "
            f"roll={geom[i,2]:8.2f}°"
        )


    print()
    print(
        "LINGLONG TORSO GEOMETRY "
        "DIAGNOSIS V2: PASS"
    )


if __name__ == "__main__":
    main()
