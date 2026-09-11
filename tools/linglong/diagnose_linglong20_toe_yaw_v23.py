#!/usr/bin/env python3

from pathlib import Path
import sys
import csv

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)


XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
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

OUT_DIR = (
    ROOT
    / "output/linglong20_lower_v23/diagnostics"
)

OUT_CSV = (
    OUT_DIR
    / "toe_yaw_v23.csv"
)


LEFT_HIP_YAW = 9
RIGHT_HIP_YAW = 15


def normalize(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-10:
        return None

    return v / n


def wrap_pi(x):
    return (
        x + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


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

    # +Y = anatomical left.
    y_axis = normalize(
        lh - rh
    )

    z0 = normalize(
        spine - pelvis
    )

    z_axis = normalize(
        z0
        - np.dot(
            z0,
            y_axis,
        )
        * y_axis
    )

    # Y x Z = X forward.
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

    return np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )


def body_id(model, name):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(
            f"Body not found: {name}"
        )

    return bid


def yaw_from_vector(v):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    horizontal_norm = float(
        np.hypot(
            v[0],
            v[1],
        )
    )

    if horizontal_norm < 1e-8:
        return np.nan, horizontal_norm

    return (
        float(
            np.arctan2(
                v[1],
                v[0],
            )
        ),
        horizontal_norm,
    )


def circular_offset(
    reference,
    measured,
):
    """
    measured ~= reference + offset
    """

    d = wrap_pi(
        measured
        - reference
    )

    return float(
        np.arctan2(
            np.mean(
                np.sin(d)
            ),
            np.mean(
                np.cos(d)
            ),
        )
    )


def circular_rmse(
    reference,
    measured,
    offset,
):

    error = wrap_pi(
        measured
        - reference
        - offset
    )

    return float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


def safe_corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
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


def analyse_side(
    name,
    human_yaw,
    robot_yaw,
    hip_yaw,
    valid,
):

    h = human_yaw[valid]
    r = robot_yaw[valid]
    hip = hip_yaw[valid]

    # --------------------------------------------------------
    # Absolute yaw:
    # find the best CONSTANT coordinate offset.
    # --------------------------------------------------------

    offset = circular_offset(
        h,
        r,
    )

    rmse = circular_rmse(
        h,
        r,
        offset,
    )

    err = wrap_pi(
        r
        - h
        - offset
    )


    # --------------------------------------------------------
    # Test whether the sign might be mirrored.
    # --------------------------------------------------------

    mirror_offset = circular_offset(
        -h,
        r,
    )

    mirror_rmse = circular_rmse(
        -h,
        r,
        mirror_offset,
    )


    # --------------------------------------------------------
    # Dynamics:
    #
    # unwrap first, then compare variations.
    # Absolute zero offsets no longer matter here.
    # --------------------------------------------------------

    hu = np.unwrap(h)
    ru = np.unwrap(r)
    hipu = np.unwrap(hip)

    hc = hu - np.median(hu)
    rc = ru - np.median(ru)
    hipc = hipu - np.median(hipu)

    correlation = safe_corr(
        hc,
        rc,
    )

    hip_correlation = safe_corr(
        hc,
        hipc,
    )

    # Linear motion gain:
    #
    # robot variation ~= gain * human variation
    denom = float(
        np.dot(
            hc,
            hc,
        )
    )

    if denom > 1e-12:
        gain = float(
            np.dot(
                hc,
                rc,
            )
            / denom
        )

        hip_gain = float(
            np.dot(
                hc,
                hipc,
            )
            / denom
        )

    else:
        gain = np.nan
        hip_gain = np.nan


    # Frame-to-frame direction agreement.
    dh = np.diff(hu)
    dr = np.diff(ru)

    motion_corr = safe_corr(
        dh,
        dr,
    )


    print()
    print("=" * 95)
    print(name)
    print("=" * 95)

    print(
        "valid frames:",
        len(h),
    )

    print()

    print("ABSOLUTE YAW")

    print(
        "human range deg:",
        np.degrees(
            [
                h.min(),
                h.max(),
            ]
        ),
    )

    print(
        "robot range deg:",
        np.degrees(
            [
                r.min(),
                r.max(),
            ]
        ),
    )

    print(
        "hip-yaw range deg:",
        np.degrees(
            [
                hip.min(),
                hip.max(),
            ]
        ),
    )

    print()

    print(
        "best constant offset:",
        np.degrees(offset),
        "deg",
    )

    print(
        "RMSE after offset:",
        np.degrees(rmse),
        "deg",
    )

    print(
        "p95 error after offset:",
        np.degrees(
            np.percentile(
                np.abs(err),
                95,
            )
        ),
        "deg",
    )

    print(
        "max error after offset:",
        np.degrees(
            np.max(
                np.abs(err)
            )
        ),
        "deg",
    )

    print()

    print("SIGN TEST")

    print(
        "normal-sign RMSE:",
        np.degrees(rmse),
        "deg",
    )

    print(
        "mirrored-sign RMSE:",
        np.degrees(
            mirror_rmse
        ),
        "deg",
    )

    print()

    print("MOTION TREND")

    print(
        "human->robot correlation:",
        correlation,
    )

    print(
        "human->robot motion gain:",
        gain,
    )

    print(
        "frame-delta correlation:",
        motion_corr,
    )

    print()

    print(
        "human toe -> hip_yaw correlation:",
        hip_correlation,
    )

    print(
        "human toe -> hip_yaw gain:",
        hip_gain,
    )

    print()

    # Largest errors.
    local_order = np.argsort(
        np.abs(err)
    )[::-1][:12]

    valid_indices = np.where(
        valid
    )[0]

    print(
        "largest residual frames:"
    )

    for j in local_order:

        frame_idx = int(
            valid_indices[j]
        )

        print(
            f"  frame {frame_idx:03d} | "
            f"human={np.degrees(h[j]):8.2f}° | "
            f"robot={np.degrees(r[j]):8.2f}° | "
            f"hip={np.degrees(hip[j]):8.2f}° | "
            f"residual={np.degrees(err[j]):7.2f}°"
        )

    return {
        "offset": offset,
        "rmse": rmse,
        "mirror_rmse": mirror_rmse,
        "correlation": correlation,
        "gain": gain,
        "motion_corr": motion_corr,
    }


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qpos = np.loadtxt(
        V23_CSV,
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

    frames, metadata_fps = (
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

    data = mujoco.MjData(
        model
    )

    base_id = body_id(
        model,
        "base_link",
    )

    left_foot_id = body_id(
        model,
        "left_ankle_roll_link",
    )

    right_foot_id = body_id(
        model,
        "right_ankle_roll_link",
    )


    human_left_yaw = []
    human_right_yaw = []

    robot_left_yaw = []
    robot_right_yaw = []

    human_left_horizontal = []
    human_right_horizontal = []

    robot_left_horizontal = []
    robot_right_horizontal = []


    for i, frame in enumerate(
        frames
    ):

        basis = anatomical_basis(
            frame
        )

        # ====================================================
        # Human toe direction
        # ====================================================

        for side, yaw_list, norm_list in (
            (
                "left",
                human_left_yaw,
                human_left_horizontal,
            ),
            (
                "right",
                human_right_yaw,
                human_right_horizontal,
            ),
        ):

            ankle = np.asarray(
                frame[
                    f"{side}_ankle"
                ][0],
                dtype=np.float64,
            )

            foot = np.asarray(
                frame[
                    f"{side}_foot"
                ][0],
                dtype=np.float64,
            )

            toe_world = (
                foot - ankle
            )

            toe_local = (
                basis.T
                @ toe_world
            )

            yaw, horizontal = (
                yaw_from_vector(
                    toe_local
                )
            )

            yaw_list.append(
                yaw
            )

            norm_list.append(
                horizontal
            )


        # ====================================================
        # Robot actual foot +X direction
        # ====================================================

        data.qpos[:] = qpos[i]

        mujoco.mj_forward(
            model,
            data,
        )

        base_rot = np.asarray(
            data.xmat[
                base_id
            ],
            dtype=np.float64,
        ).reshape(
            3,
            3,
        )

        for foot_id, yaw_list, norm_list in (
            (
                left_foot_id,
                robot_left_yaw,
                robot_left_horizontal,
            ),
            (
                right_foot_id,
                robot_right_yaw,
                robot_right_horizontal,
            ),
        ):

            foot_rot = np.asarray(
                data.xmat[
                    foot_id
                ],
                dtype=np.float64,
            ).reshape(
                3,
                3,
            )

            robot_forward = (
                base_rot.T
                @ foot_rot[:, 0]
            )

            yaw, horizontal = (
                yaw_from_vector(
                    robot_forward
                )
            )

            yaw_list.append(
                yaw
            )

            norm_list.append(
                horizontal
            )


    human_left_yaw = np.asarray(
        human_left_yaw
    )

    human_right_yaw = np.asarray(
        human_right_yaw
    )

    robot_left_yaw = np.asarray(
        robot_left_yaw
    )

    robot_right_yaw = np.asarray(
        robot_right_yaw
    )


    human_left_horizontal = np.asarray(
        human_left_horizontal
    )

    human_right_horizontal = np.asarray(
        human_right_horizontal
    )

    robot_left_horizontal = np.asarray(
        robot_left_horizontal
    )

    robot_right_horizontal = np.asarray(
        robot_right_horizontal
    )


    # Human ankle->foot length is ~0.14 m.
    # A horizontal projection > 3 cm is sufficient for a
    # stable yaw measurement.
    left_valid = (
        np.isfinite(
            human_left_yaw
        )
        & np.isfinite(
            robot_left_yaw
        )
        & (
            human_left_horizontal
            > 0.03
        )
        & (
            robot_left_horizontal
            > 0.20
        )
    )

    right_valid = (
        np.isfinite(
            human_right_yaw
        )
        & np.isfinite(
            robot_right_yaw
        )
        & (
            human_right_horizontal
            > 0.03
        )
        & (
            robot_right_horizontal
            > 0.20
        )
    )


    print("=" * 95)
    print("LINGLONG V2.3 TOE-YAW / HIP-YAW DIAGNOSIS")
    print("=" * 95)

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

    print(
        "left human horizontal "
        "min/median:",
        human_left_horizontal.min(),
        np.median(
            human_left_horizontal
        ),
    )

    print(
        "right human horizontal "
        "min/median:",
        human_right_horizontal.min(),
        np.median(
            human_right_horizontal
        ),
    )


    left_result = analyse_side(
        "LEFT",
        human_left_yaw,
        robot_left_yaw,
        qpos[:, LEFT_HIP_YAW],
        left_valid,
    )

    right_result = analyse_side(
        "RIGHT",
        human_right_yaw,
        robot_right_yaw,
        qpos[:, RIGHT_HIP_YAW],
        right_valid,
    )


    # ========================================================
    # Save diagnostics for later V2.4 if needed.
    # ========================================================

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "human_left_toe_yaw_deg",
                "robot_left_toe_yaw_deg",
                "left_hip_yaw_deg",
                "human_right_toe_yaw_deg",
                "robot_right_toe_yaw_deg",
                "right_hip_yaw_deg",
                "left_valid",
                "right_valid",
            ]
        )

        for i in range(661):

            writer.writerow(
                [
                    i,

                    np.degrees(
                        human_left_yaw[i]
                    ),

                    np.degrees(
                        robot_left_yaw[i]
                    ),

                    np.degrees(
                        qpos[
                            i,
                            LEFT_HIP_YAW,
                        ]
                    ),

                    np.degrees(
                        human_right_yaw[i]
                    ),

                    np.degrees(
                        robot_right_yaw[i]
                    ),

                    np.degrees(
                        qpos[
                            i,
                            RIGHT_HIP_YAW,
                        ]
                    ),

                    int(
                        left_valid[i]
                    ),

                    int(
                        right_valid[i]
                    ),
                ]
            )


    print()
    print("=" * 95)
    print("DECISION HINT")
    print("=" * 95)

    for side, result in (
        (
            "LEFT",
            left_result,
        ),
        (
            "RIGHT",
            right_result,
        ),
    ):

        print()
        print(side)

        print(
            "  offset deg:",
            np.degrees(
                result["offset"]
            ),
        )

        print(
            "  aligned RMSE deg:",
            np.degrees(
                result["rmse"]
            ),
        )

        print(
            "  correlation:",
            result["correlation"],
        )

        print(
            "  gain:",
            result["gain"],
        )

        print(
            "  delta correlation:",
            result[
                "motion_corr"
            ],
        )

        if (
            result["rmse"]
            < result["mirror_rmse"]
        ):
            print(
                "  sign:"
                " NORMAL is better"
            )
        else:
            print(
                "  sign:"
                " MIRRORED may be better"
            )


    print()

    print(
        "diagnostics saved:",
        OUT_CSV,
    )

    print()

    print(
        "LINGLONG V2.3 TOE-YAW "
        "DIAGNOSIS: PASS"
    )


if __name__ == "__main__":
    main()
