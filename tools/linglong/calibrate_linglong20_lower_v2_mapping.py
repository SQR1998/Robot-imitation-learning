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


ROBOT_HIP_HALF_WIDTH = 0.1138
ROBOT_HIP_Z = -0.0925

ROBOT_THIGH = 0.3823
ROBOT_SHANK = 0.4200


def normalize(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-8:
        raise RuntimeError(
            "near-zero vector"
        )

    return v / n


def anatomical_basis(frame):

    pelvis = np.asarray(
        frame["pelvis"][0],
        dtype=np.float64,
    )

    spine = np.asarray(
        frame["spine3"][0],
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

    # robot-like convention:
    # X forward, Y left, Z up

    y_axis = normalize(
        lh - rh
    )

    z0 = normalize(
        spine - pelvis
    )

    z_axis = normalize(
        z0
        - np.dot(z0, y_axis)
        * y_axis
    )

    x_axis = normalize(
        np.cross(
            y_axis,
            z_axis,
        )
    )

    y_axis = normalize(
        np.cross(
            z_axis,
            x_axis,
        )
    )

    basis = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    return (
        pelvis,
        basis,
    )


def local_point(
    point,
    pelvis,
    basis,
):
    return (
        basis.T
        @ (
            np.asarray(
                point,
                dtype=np.float64,
            )
            - pelvis
        )
    )


def adapted_direction(
    vec,
    lateral_gain,
):
    """
    Keep sagittal X/Z motion,
    attenuate only lateral Y motion.
    """

    d = normalize(vec)

    d = np.array(
        [
            d[0],
            lateral_gain * d[1],
            d[2],
        ],
        dtype=np.float64,
    )

    return normalize(d)


def build_chain(
    l_thigh,
    r_thigh,
    l_shank,
    r_shank,
    gain,
):

    lhip = np.array(
        [
            0.0,
            +ROBOT_HIP_HALF_WIDTH,
            ROBOT_HIP_Z,
        ]
    )

    rhip = np.array(
        [
            0.0,
            -ROBOT_HIP_HALF_WIDTH,
            ROBOT_HIP_Z,
        ]
    )

    lknee = (
        lhip
        + ROBOT_THIGH
        * adapted_direction(
            l_thigh,
            gain,
        )
    )

    rknee = (
        rhip
        + ROBOT_THIGH
        * adapted_direction(
            r_thigh,
            gain,
        )
    )

    lankle = (
        lknee
        + ROBOT_SHANK
        * adapted_direction(
            l_shank,
            gain,
        )
    )

    rankle = (
        rknee
        + ROBOT_SHANK
        * adapted_direction(
            r_shank,
            gain,
        )
    )

    return (
        lknee,
        rknee,
        lankle,
        rankle,
    )


def describe(name, values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    print(
        f"{name:34s}"
        f"min={values.min():.4f} "
        f"p05={np.percentile(values,5):.4f} "
        f"median={np.median(values):.4f} "
        f"p95={np.percentile(values,95):.4f} "
        f"max={values.max():.4f}"
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

    frames, _ = (
        get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=30,
        )
    )

    human_knee_width = []
    human_ankle_width = []

    human_left_leg = []
    human_right_leg = []

    chain_data = []

    for frame in frames:

        pelvis, basis = (
            anatomical_basis(frame)
        )

        p = {}

        for name in (
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ):

            p[name] = local_point(
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

        lthigh = lk - lh
        rthigh = rk - rh

        lshank = la - lk
        rshank = ra - rk

        human_knee_width.append(
            lk[1] - rk[1]
        )

        human_ankle_width.append(
            la[1] - ra[1]
        )

        human_left_leg.append(
            np.linalg.norm(lthigh)
            + np.linalg.norm(lshank)
        )

        human_right_leg.append(
            np.linalg.norm(rthigh)
            + np.linalg.norm(rshank)
        )

        chain_data.append(
            (
                lthigh,
                rthigh,
                lshank,
                rshank,
            )
        )

    human_knee_width = np.asarray(
        human_knee_width
    )

    human_ankle_width = np.asarray(
        human_ankle_width
    )

    human_leg_length = 0.5 * (
        np.median(human_left_leg)
        + np.median(human_right_leg)
    )

    robot_leg_length = (
        ROBOT_THIGH
        + ROBOT_SHANK
    )

    length_ratio = (
        robot_leg_length
        / human_leg_length
    )

    # Morphology-normalized desired widths:
    # preserve the human motion scale relative to leg length.
    desired_knee_width = (
        np.median(
            human_knee_width
        )
        * length_ratio
    )

    desired_ankle_width = (
        np.median(
            human_ankle_width
        )
        * length_ratio
    )

    print("=" * 95)
    print("LINGLONG LOWER V2 MAPPING CALIBRATION")
    print("=" * 95)

    print(
        "human median leg length:",
        human_leg_length,
    )

    print(
        "robot leg length       :",
        robot_leg_length,
    )

    print(
        "length ratio           :",
        length_ratio,
    )

    print()

    print(
        "desired knee median:",
        desired_knee_width,
    )

    print(
        "desired ankle median:",
        desired_ankle_width,
    )


    results = []

    # Search only mapping parameter.
    # This is NOT a collision constraint.
    for gain in np.linspace(
        0.0,
        1.0,
        101,
    ):

        knee_widths = []
        ankle_widths = []

        for (
            lthigh,
            rthigh,
            lshank,
            rshank,
        ) in chain_data:

            (
                lk,
                rk,
                la,
                ra,
            ) = build_chain(
                lthigh,
                rthigh,
                lshank,
                rshank,
                gain,
            )

            knee_widths.append(
                lk[1] - rk[1]
            )

            ankle_widths.append(
                la[1] - ra[1]
            )

        knee_widths = np.asarray(
            knee_widths
        )

        ankle_widths = np.asarray(
            ankle_widths
        )

        knee_med = float(
            np.median(
                knee_widths
            )
        )

        ankle_med = float(
            np.median(
                ankle_widths
            )
        )

        # Dimensionless relative error.
        cost = (
            (
                (
                    knee_med
                    - desired_knee_width
                )
                / desired_knee_width
            ) ** 2
            +
            (
                (
                    ankle_med
                    - desired_ankle_width
                )
                / desired_ankle_width
            ) ** 2
        )

        results.append(
            (
                cost,
                gain,
                knee_widths,
                ankle_widths,
            )
        )

    (
        best_cost,
        best_gain,
        best_knee,
        best_ankle,
    ) = min(
        results,
        key=lambda x: x[0],
    )

    print()
    print("=" * 95)
    print("BEST MAPPING")
    print("=" * 95)

    print(
        "best lateral gain:",
        best_gain,
    )

    print(
        "mapping cost     :",
        best_cost,
    )

    print()

    describe(
        "V2 predicted knee width",
        best_knee,
    )

    describe(
        "V2 predicted ankle width",
        best_ankle,
    )

    print()
    print("=" * 95)
    print("REFERENCE GAINS")
    print("=" * 95)

    for gain in (
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
    ):

        kw = []
        aw = []

        for (
            lthigh,
            rthigh,
            lshank,
            rshank,
        ) in chain_data:

            lk, rk, la, ra = (
                build_chain(
                    lthigh,
                    rthigh,
                    lshank,
                    rshank,
                    gain,
                )
            )

            kw.append(
                lk[1] - rk[1]
            )

            aw.append(
                la[1] - ra[1]
            )

        print(
            f"gain={gain:.2f} | "
            f"knee median={np.median(kw):.4f} | "
            f"ankle median={np.median(aw):.4f}"
        )

    print()
    print(
        "LINGLONG LOWER V2 MAPPING "
        "CALIBRATION: PASS"
    )


if __name__ == "__main__":
    main()
