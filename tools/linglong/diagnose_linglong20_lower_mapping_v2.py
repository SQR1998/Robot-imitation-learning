#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


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


# Validated LingLong geometry.
ROBOT_HIP_HALF_WIDTH = 0.1138
ROBOT_HIP_WIDTH = 0.2276

ROBOT_THIGH = 0.3823
ROBOT_SHANK = 0.4200

CURRENT_SCALE = 0.9


def normalize(v):
    v = np.asarray(v, dtype=np.float64)

    n = float(np.linalg.norm(v))

    if n < 1e-8:
        raise RuntimeError(
            "Cannot normalize near-zero vector"
        )

    return v / n


def stats(name, values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    print(
        f"{name:34s}"
        f"min={x.min(): .4f}  "
        f"p05={np.percentile(x,5): .4f}  "
        f"median={np.median(x): .4f}  "
        f"p95={np.percentile(x,95): .4f}  "
        f"max={x.max(): .4f}"
    )


def anatomical_basis(frame):
    """
    Return a robot-style human anatomical frame.

        X = forward
        Y = left
        Z = up

    This is determined from joint geometry, NOT from
    the SMPL pelvis quaternion axes.
    """

    pelvis = np.asarray(
        frame["pelvis"][0],
        dtype=np.float64,
    )

    spine = np.asarray(
        frame["spine3"][0],
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

    # Human left direction.
    left_axis = normalize(
        left_hip - right_hip
    )

    # Approximate human up.
    up_raw = normalize(
        spine - pelvis
    )

    # Remove any component parallel to left.
    up_axis = normalize(
        up_raw
        - np.dot(
            up_raw,
            left_axis,
        )
        * left_axis
    )

    # Robot convention:
    # Y(left) × Z(up) = X(forward)
    forward_axis = normalize(
        np.cross(
            left_axis,
            up_axis,
        )
    )

    # Re-orthogonalize left.
    left_axis = normalize(
        np.cross(
            up_axis,
            forward_axis,
        )
    )

    # Columns correspond to X/Y/Z axes.
    basis = np.column_stack(
        (
            forward_axis,
            left_axis,
            up_axis,
        )
    )

    return pelvis, basis


def to_anatomical(
    world_point,
    pelvis,
    basis,
):
    """
    World point -> robot-style human anatomical coordinates:
      x forward
      y left
      z up
    """

    return (
        basis.T
        @ (
            np.asarray(
                world_point,
                dtype=np.float64,
            )
            - pelvis
        )
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

    (
        frames,
        metadata_fps,
    ) = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    required_names = (
        "pelvis",
        "spine3",

        "left_hip",
        "right_hip",

        "left_knee",
        "right_knee",

        "left_ankle",
        "right_ankle",

        "left_foot",
        "right_foot",
    )

    missing = [
        name
        for name in required_names
        if name not in frames[0]
    ]

    if missing:
        raise RuntimeError(
            f"Missing SMPL-X joints: {missing}"
        )

    print("=" * 100)
    print("LINGLONG LOWER MAPPING DIAGNOSIS V2")
    print("ANATOMICAL FRAME + HIP-KNEE-ANKLE")
    print("=" * 100)

    print(
        "frames       :",
        len(frames),
    )

    print(
        "human height :",
        human_height,
    )

    print(
        "metadata fps :",
        metadata_fps,
    )


    hip_width = []
    knee_width = []
    ankle_width = []
    foot_width = []

    left_thigh = []
    right_thigh = []

    left_shank = []
    right_shank = []

    left_ankle_to_foot = []
    right_ankle_to_foot = []

    left_knee_inward = []
    right_knee_inward = []

    left_ankle_inward = []
    right_ankle_inward = []

    # What knee / ankle separation would be if we use
    # LingLong anchors + LingLong bone lengths but preserve
    # HUMAN thigh/shank directions?
    chain_knee_width = []
    chain_ankle_width = []

    basis_forward_world = []
    basis_left_world = []
    basis_up_world = []


    for frame in frames:

        pelvis, basis = (
            anatomical_basis(frame)
        )

        basis_forward_world.append(
            basis[:, 0]
        )

        basis_left_world.append(
            basis[:, 1]
        )

        basis_up_world.append(
            basis[:, 2]
        )

        p = {}

        for name in (
            "left_hip",
            "right_hip",

            "left_knee",
            "right_knee",

            "left_ankle",
            "right_ankle",

            "left_foot",
            "right_foot",
        ):
            p[name] = to_anatomical(
                frame[name][0],
                pelvis,
                basis,
            )

        lh = p["left_hip"]
        rh = p["right_hip"]

        lk = p["left_knee"]
        rk = p["right_knee"]

        la = p["left_ankle"]
        ra = p["right_ankle"]

        lf = p["left_foot"]
        rf = p["right_foot"]


        # ----------------------------------------------------
        # Real human geometry in anatomical frame.
        # ----------------------------------------------------

        hip_width.append(
            lh[1] - rh[1]
        )

        knee_width.append(
            lk[1] - rk[1]
        )

        ankle_width.append(
            la[1] - ra[1]
        )

        foot_width.append(
            lf[1] - rf[1]
        )


        l_thigh_vec = (
            lk - lh
        )

        r_thigh_vec = (
            rk - rh
        )

        l_shank_vec = (
            la - lk
        )

        r_shank_vec = (
            ra - rk
        )


        left_thigh.append(
            np.linalg.norm(
                l_thigh_vec
            )
        )

        right_thigh.append(
            np.linalg.norm(
                r_thigh_vec
            )
        )

        left_shank.append(
            np.linalg.norm(
                l_shank_vec
            )
        )

        right_shank.append(
            np.linalg.norm(
                r_shank_vec
            )
        )

        left_ankle_to_foot.append(
            np.linalg.norm(
                lf - la
            )
        )

        right_ankle_to_foot.append(
            np.linalg.norm(
                rf - ra
            )
        )


        # Positive = toward centerline.
        left_knee_inward.append(
            lh[1] - lk[1]
        )

        right_knee_inward.append(
            rk[1] - rh[1]
        )

        left_ankle_inward.append(
            lh[1] - la[1]
        )

        right_ankle_inward.append(
            ra[1] - rh[1]
        )


        # ----------------------------------------------------
        # Proposed bone-chain mapping.
        #
        # Start from LingLong's OWN hip anchors and only copy
        # human limb DIRECTION.
        # ----------------------------------------------------

        l_thigh_dir = normalize(
            l_thigh_vec
        )

        r_thigh_dir = normalize(
            r_thigh_vec
        )

        l_shank_dir = normalize(
            l_shank_vec
        )

        r_shank_dir = normalize(
            r_shank_vec
        )


        robot_l_hip = np.array(
            [
                0.0,
                +ROBOT_HIP_HALF_WIDTH,
                -0.0925,
            ],
            dtype=np.float64,
        )

        robot_r_hip = np.array(
            [
                0.0,
                -ROBOT_HIP_HALF_WIDTH,
                -0.0925,
            ],
            dtype=np.float64,
        )


        robot_l_knee = (
            robot_l_hip
            + ROBOT_THIGH
            * l_thigh_dir
        )

        robot_r_knee = (
            robot_r_hip
            + ROBOT_THIGH
            * r_thigh_dir
        )


        robot_l_ankle = (
            robot_l_knee
            + ROBOT_SHANK
            * l_shank_dir
        )

        robot_r_ankle = (
            robot_r_knee
            + ROBOT_SHANK
            * r_shank_dir
        )


        chain_knee_width.append(
            robot_l_knee[1]
            - robot_r_knee[1]
        )

        chain_ankle_width.append(
            robot_l_ankle[1]
            - robot_r_ankle[1]
        )


    print()
    print("=" * 100)
    print("TRUE HUMAN GEOMETRY IN ANATOMICAL FRAME")
    print("=" * 100)

    stats(
        "human hip width",
        hip_width,
    )

    stats(
        "human knee width",
        knee_width,
    )

    stats(
        "human ankle width",
        ankle_width,
    )

    stats(
        "human foot width",
        foot_width,
    )


    print()
    print("=" * 100)
    print("HUMAN BONE LENGTHS")
    print("=" * 100)

    stats(
        "left thigh hip->knee",
        left_thigh,
    )

    stats(
        "right thigh hip->knee",
        right_thigh,
    )

    stats(
        "left shank knee->ankle",
        left_shank,
    )

    stats(
        "right shank knee->ankle",
        right_shank,
    )

    stats(
        "left ankle->foot",
        left_ankle_to_foot,
    )

    stats(
        "right ankle->foot",
        right_ankle_to_foot,
    )


    print()
    print("=" * 100)
    print("CURRENT V1 UNIFORM ×0.9 TARGET WIDTH")
    print("=" * 100)

    stats(
        "V1 hip target width",
        np.asarray(
            hip_width
        )
        * CURRENT_SCALE,
    )

    stats(
        "V1 knee target width",
        np.asarray(
            knee_width
        )
        * CURRENT_SCALE,
    )

    stats(
        "V1 ankle target width",
        np.asarray(
            ankle_width
        )
        * CURRENT_SCALE,
    )


    print()
    print("=" * 100)
    print("SOURCE LEG INWARD MOTION")
    print("=" * 100)

    stats(
        "left knee inward",
        left_knee_inward,
    )

    stats(
        "right knee inward",
        right_knee_inward,
    )

    stats(
        "left ankle inward",
        left_ankle_inward,
    )

    stats(
        "right ankle inward",
        right_ankle_inward,
    )


    print()
    print("=" * 100)
    print("PROPOSED LINGLONG BONE-CHAIN MAPPING")
    print("=" * 100)

    print(
        "LingLong hip width:",
        ROBOT_HIP_WIDTH,
        "m",
    )

    print(
        "LingLong thigh:",
        ROBOT_THIGH,
        "m",
    )

    print(
        "LingLong shank:",
        ROBOT_SHANK,
        "m",
    )

    print()

    stats(
        "predicted robot knee width",
        chain_knee_width,
    )

    stats(
        "predicted robot ankle width",
        chain_ankle_width,
    )


    print()
    print("=" * 100)
    print("ANATOMICAL BASIS IN WORLD")
    print("=" * 100)

    print(
        "median forward axis:",
        np.median(
            np.asarray(
                basis_forward_world
            ),
            axis=0,
        ),
    )

    print(
        "median left axis   :",
        np.median(
            np.asarray(
                basis_left_world
            ),
            axis=0,
        ),
    )

    print(
        "median up axis     :",
        np.median(
            np.asarray(
                basis_up_world
            ),
            axis=0,
        ),
    )


    print()
    print(
        "LINGLONG LOWER MAPPING "
        "DIAGNOSIS V2: PASS"
    )


if __name__ == "__main__":
    main()
