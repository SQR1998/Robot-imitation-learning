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

    n = float(
        np.linalg.norm(v)
    )

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


def chest_frame(frame):
    """
    Chest anatomical frame.

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


def head_frame(frame):
    """
    Head anatomical frame from:

      neck -> head : approximate UP
      head -> jaw  : forward/down direction

    Remove jaw vector's component along UP to
    obtain horizontal face-forward direction.

    Result:

      +X face forward
      +Y head left
      +Z head up
    """

    neck = np.asarray(
        frame["neck"][0],
        dtype=np.float64,
    )

    head = np.asarray(
        frame["head"][0],
        dtype=np.float64,
    )

    jaw = np.asarray(
        frame["jaw"][0],
        dtype=np.float64,
    )


    z = normalize(
        head - neck
    )


    forward0 = (
        jaw - head
    )

    forward = (
        forward0
        - np.dot(
            forward0,
            z,
        ) * z
    )

    x = normalize(
        forward
    )


    # +X forward, +Y left, +Z up
    y = normalize(
        np.cross(
            z,
            x,
        )
    )

    x = normalize(
        np.cross(
            y,
            z,
        )
    )


    return np.column_stack(
        (
            x,
            y,
            z,
        )
    )


def geometry_head_angles(frame):

    Rc = chest_frame(
        frame
    )

    Rh = head_frame(
        frame
    )


    # Head relative to chest.
    rel = (
        Rc.T
        @ Rh
    )


    return R.from_matrix(
        rel
    ).as_euler(
        "ZYX",
        degrees=True,
    )


def quaternion_head_angles(frame):

    chest = quat_wxyz_to_rot(
        frame["spine3"][1]
    )

    head = quat_wxyz_to_rot(
        frame["head"][1]
    )

    rel = (
        chest.inv()
        * head
    )

    return rel.as_euler(
        "ZYX",
        degrees=True,
    )


def stats(name, x):

    x = np.asarray(
        x,
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


def correlation(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    a = (
        a - np.median(a)
    )

    b = (
        b - np.median(b)
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


    required = [
        "spine3",
        "neck",
        "head",
        "jaw",
        "left_shoulder",
        "right_shoulder",
    ]


    print("=" * 100)
    print("LINGLONG HEAD GEOMETRY DIAGNOSIS V2")
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


    for key in required:

        print(
            f"{key:18s}:",
            key in frames[0],
        )

        if key not in frames[0]:

            raise RuntimeError(
                f"Missing joint: {key}"
            )


    geom = []
    quat = []


    for frame in frames:

        geom.append(
            geometry_head_angles(
                frame
            )
        )

        quat.append(
            quaternion_head_angles(
                frame
            )
        )


    geom = np.asarray(
        geom,
        dtype=np.float64,
    )

    quat = np.asarray(
        quat,
        dtype=np.float64,
    )


    print()
    print("=" * 100)
    print("1. GEOMETRY HEAD RELATIVE TO CHEST")
    print("=" * 100)

    stats(
        "geometry head yaw",
        geom[:, 0],
    )

    stats(
        "geometry head pitch",
        geom[:, 1],
    )

    stats(
        "geometry head roll",
        geom[:, 2],
    )


    print()
    print("=" * 100)
    print("2. QUATERNION HEAD RELATIVE TO SPINE3")
    print("=" * 100)

    stats(
        "quaternion head yaw",
        quat[:, 0],
    )

    stats(
        "quaternion head pitch",
        quat[:, 1],
    )

    stats(
        "quaternion head roll",
        quat[:, 2],
    )


    print()
    print("=" * 100)
    print("3. GEOMETRY vs QUATERNION")
    print("=" * 100)

    print(
        "yaw correlation:",
        correlation(
            geom[:, 0],
            quat[:, 0],
        )
    )

    print(
        "pitch correlation:",
        correlation(
            geom[:, 1],
            quat[:, 1],
        )
    )

    print(
        "roll correlation:",
        correlation(
            geom[:, 2],
            quat[:, 2],
        )
    )


    print()
    print("=" * 100)
    print("4. FIRST 15 GEOMETRY HEAD FRAMES")
    print("=" * 100)

    for i in range(15):

        print(
            f"{i:03d} | "
            f"yaw={geom[i,0]:8.2f}° "
            f"pitch={geom[i,1]:8.2f}° "
            f"roll={geom[i,2]:8.2f}°"
        )


    print()
    print(
        "LINGLONG HEAD GEOMETRY "
        "DIAGNOSIS V2: PASS"
    )


if __name__ == "__main__":
    main()
