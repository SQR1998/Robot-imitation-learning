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


XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

BODY_V1 = (
    ROOT
    / "output/linglong20_body_v1_full"
    / "linglong20_body_v1_full_qpos37.csv"
)

V21 = (
    ROOT
    / "output/linglong20_lower_v21"
    / "linglong20_lower_v21_qpos37.csv"
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


LATERAL_GAIN = 0.57


ANKLE_INDICES = {
    "left_pitch": 11,
    "left_roll": 12,
    "right_pitch": 17,
    "right_roll": 18,
}


def normalize(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = np.linalg.norm(v)

    if n < 1e-10:
        return None

    return v / n


def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)

    if a is None or b is None:
        return np.nan

    dot = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(dot)
        )
    )


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

    return (
        pelvis,
        np.column_stack(
            (
                x_axis,
                y_axis,
                z_axis,
            )
        ),
    )


def mapped_direction(v):

    d = normalize(v)

    d = np.array(
        [
            d[0],
            LATERAL_GAIN * d[1],
            d[2],
        ],
        dtype=np.float64,
    )

    return normalize(d)


def quat_wxyz_matrix(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return R.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()


def body_id(model, name):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(name)

    return bid


def joint_id(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(name)

    return jid


def stats(name, values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    print(
        f"{name:34s}"
        f"mean={np.mean(x):7.2f}  "
        f"median={np.median(x):7.2f}  "
        f"p95={np.percentile(x,95):7.2f}  "
        f"max={np.max(x):7.2f}"
    )


def main():

    v1 = np.loadtxt(
        BODY_V1,
        delimiter=",",
    )

    v21 = np.loadtxt(
        V21,
        delimiter=",",
    )

    assert v1.shape == (661, 37)
    assert v21.shape == (661, 37)

    print("=" * 100)
    print("1. CHECK WHETHER V2.1 REALLY RETARGETED THE ANKLES")
    print("=" * 100)

    for name, idx in ANKLE_INDICES.items():

        diff = np.max(
            np.abs(
                v1[:, idx]
                - v21[:, idx]
            )
        )

        values = np.degrees(
            v21[:, idx]
        )

        step = np.degrees(
            np.diff(
                v21[:, idx]
            )
        )

        print()
        print(name)

        print(
            "  V1 -> V2.1 max diff:",
            diff,
        )

        print(
            "  range deg:",
            values.min(),
            values.max(),
        )

        print(
            "  max step deg:",
            np.max(
                np.abs(step)
            ),
        )

        print(
            "  max-step frame:",
            int(
                np.argmax(
                    np.abs(step)
                )
                + 1
            ),
        )

    print()

    print(
        "left roll at -30 deg:",
        int(
            np.sum(
                np.degrees(
                    v21[:, 12]
                ) < -29.9
            )
        ),
        "frames",
    )

    print(
        "right roll at +30 deg:",
        int(
            np.sum(
                np.degrees(
                    v21[:, 18]
                ) > 29.9
            )
        ),
        "frames",
    )

    print(
        "left pitch near +34.5:",
        int(
            np.sum(
                np.degrees(
                    v21[:, 11]
                ) > 34.4
            )
        ),
        "frames",
    )

    print(
        "right pitch near +34.5:",
        int(
            np.sum(
                np.degrees(
                    v21[:, 17]
                ) > 34.4
            )
        ),
        "frames",
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

    frames, _ = (
        get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=30,
        )
    )

    assert len(frames) == 661


    # ========================================================
    # Determine which SMPL foot local axis actually points
    # toward the toes.
    # ========================================================

    print()
    print("=" * 100)
    print("2. IDENTIFY SMPL-X FOOT LOCAL AXES")
    print("=" * 100)

    candidates = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }

    for side in (
        "left",
        "right",
    ):

        errors = {
            key: []
            for key in candidates
        }

        up_alignment = {
            key: []
            for key in candidates
        }

        for frame in frames:

            pelvis, basis = (
                anatomical_basis(frame)
            )

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

            toe_dir_world = normalize(
                foot - ankle
            )

            toe_dir_local = (
                basis.T
                @ toe_dir_world
            )

            foot_rot_world = (
                quat_wxyz_matrix(
                    frame[
                        f"{side}_foot"
                    ][1]
                )
            )

            for key, axis in (
                candidates.items()
            ):

                axis_world = (
                    foot_rot_world
                    @ axis
                )

                axis_human = (
                    basis.T
                    @ axis_world
                )

                errors[key].append(
                    angle_deg(
                        axis_human,
                        toe_dir_local,
                    )
                )

                up_alignment[
                    key
                ].append(
                    axis_human[2]
                )

        print()
        print(
            f"{side.upper()} FOOT"
        )

        ranking = sorted(
            candidates,
            key=lambda k:
                np.nanmedian(
                    errors[k]
                ),
        )

        print(
            "forward-axis candidates:"
        )

        for key in ranking:
            print(
                f"  {key:2s} "
                f"median toe error="
                f"{np.nanmedian(errors[key]):7.2f} deg | "
                f"median up="
                f"{np.nanmedian(up_alignment[key]): .3f}"
            )


    # ========================================================
    # Current robot foot direction vs mapped human foot
    # direction.
    # ========================================================

    print()
    print("=" * 100)
    print("3. CURRENT V2.1 ROBOT FOOT DIRECTION ERROR")
    print("=" * 100)

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

    lfoot_id = body_id(
        model,
        "left_ankle_roll_link",
    )

    rfoot_id = body_id(
        model,
        "right_ankle_roll_link",
    )

    left_error = []
    right_error = []

    left_up = []
    right_up = []

    for i, frame in enumerate(
        frames
    ):

        pelvis, basis = (
            anatomical_basis(frame)
        )

        human_l_forward = (
            mapped_direction(
                (
                    basis.T
                    @ (
                        np.asarray(
                            frame[
                                "left_foot"
                            ][0]
                        )
                        - np.asarray(
                            frame[
                                "left_ankle"
                            ][0]
                        )
                    )
                )
            )
        )

        human_r_forward = (
            mapped_direction(
                (
                    basis.T
                    @ (
                        np.asarray(
                            frame[
                                "right_foot"
                            ][0]
                        )
                        - np.asarray(
                            frame[
                                "right_ankle"
                            ][0]
                        )
                    )
                )
            )
        )

        data.qpos[:] = (
            v21[i]
        )

        mujoco.mj_forward(
            model,
            data,
        )

        base_rot = np.asarray(
            data.xmat[
                base_id
            ]
        ).reshape(3, 3)

        lrot = np.asarray(
            data.xmat[
                lfoot_id
            ]
        ).reshape(3, 3)

        rrot = np.asarray(
            data.xmat[
                rfoot_id
            ]
        ).reshape(3, 3)

        # LingLong zero-pose foot/link frame:
        # test +X as candidate forward axis.
        robot_l_forward = (
            base_rot.T
            @ lrot[:, 0]
        )

        robot_r_forward = (
            base_rot.T
            @ rrot[:, 0]
        )

        robot_l_up = (
            base_rot.T
            @ lrot[:, 2]
        )

        robot_r_up = (
            base_rot.T
            @ rrot[:, 2]
        )

        left_error.append(
            angle_deg(
                robot_l_forward,
                human_l_forward,
            )
        )

        right_error.append(
            angle_deg(
                robot_r_forward,
                human_r_forward,
            )
        )

        left_up.append(
            robot_l_up[2]
        )

        right_up.append(
            robot_r_up[2]
        )

    stats(
        "left foot forward error",
        left_error,
    )

    stats(
        "right foot forward error",
        right_error,
    )

    print()

    print(
        "left robot foot up-z "
        "min/median/max:",
        np.min(left_up),
        np.median(left_up),
        np.max(left_up),
    )

    print(
        "right robot foot up-z "
        "min/median/max:",
        np.min(right_up),
        np.median(right_up),
        np.max(right_up),
    )


    # ========================================================
    # Verify robot ankle axis/sign convention numerically.
    # ========================================================

    print()
    print("=" * 100)
    print("4. LINGLONG ANKLE AXIS / SIGN TEST")
    print("=" * 100)

    tests = [
        (
            "left ankle pitch",
            "left_ankle_pitch_joint",
            "left_ankle_roll_link",
        ),
        (
            "left ankle roll",
            "left_ankle_roll_joint",
            "left_ankle_roll_link",
        ),
        (
            "right ankle pitch",
            "right_ankle_pitch_joint",
            "right_ankle_roll_link",
        ),
        (
            "right ankle roll",
            "right_ankle_roll_joint",
            "right_ankle_roll_link",
        ),
    ]

    for label, jname, bname in tests:

        jid = joint_id(
            model,
            jname,
        )

        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )

        bid = body_id(
            model,
            bname,
        )

        print()
        print(label)

        print(
            "  joint axis:",
            model.jnt_axis[
                jid
            ],
        )

        for deg in (
            -10.0,
            0.0,
            +10.0,
        ):

            data.qpos[:] = (
                model.qpos0
            )

            data.qpos[:7] = [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
            ]

            data.qpos[
                qadr
            ] = np.deg2rad(
                deg
            )

            mujoco.mj_forward(
                model,
                data,
            )

            rot = np.asarray(
                data.xmat[
                    bid
                ]
            ).reshape(3, 3)

            print(
                f"  q={deg:+5.1f} deg | "
                f"foot +X={rot[:,0]} | "
                f"foot +Z={rot[:,2]}"
            )

    print()
    print(
        "LINGLONG ANKLE MAPPING "
        "DIAGNOSIS: PASS"
    )


if __name__ == "__main__":
    main()
