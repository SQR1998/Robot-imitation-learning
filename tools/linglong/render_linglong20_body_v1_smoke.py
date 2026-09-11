#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import general_motion_retargeting.robot_motion_viewer as rmv


XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

CSV = (
    ROOT
    / "output/linglong20_body_v1_smoke"
    / "linglong20_body_v1_qpos37.csv"
)

VIDEO = (
    ROOT
    / "output/linglong20_body_v1_smoke"
    / "linglong20_body_v1_preview.mp4"
)

FPS = 30.0


def main():
    if not XML.exists():
        raise FileNotFoundError(XML)

    if not CSV.exists():
        raise FileNotFoundError(CSV)

    qpos = np.loadtxt(
        CSV,
        delimiter=",",
        dtype=np.float64,
    )

    if qpos.ndim == 1:
        qpos = qpos[None, :]

    print("=" * 90)
    print("LINGLONG 2.0 BODY V1 VISUALIZATION")
    print("=" * 90)

    print("qpos shape:", qpos.shape)

    assert qpos.shape[1] == 37
    assert np.all(np.isfinite(qpos))

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )
    data = mujoco.MjData(model)

    left_foot = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "left_ankle_roll_link",
    )

    right_foot = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "right_ankle_roll_link",
    )

    assert left_foot >= 0
    assert right_foot >= 0

    # ----------------------------------------------------------
    # Visualization-only vertical offset.
    #
    # Important:
    #   We do NOT modify the saved qpos37 CSV.
    #
    # Use frame 0 as the common reference and apply one CONSTANT
    # z offset to every frame. This preserves the actual vertical
    # motion across the sequence.
    # ----------------------------------------------------------
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)

    first_left_z = float(
        data.xpos[left_foot, 2]
    )
    first_right_z = float(
        data.xpos[right_foot, 2]
    )

    first_lowest_ankle = min(
        first_left_z,
        first_right_z,
    )

    # Put ankle origin slightly above z=0 for visualization.
    target_ankle_z = 0.08

    display_z_offset = (
        target_ankle_z
        - first_lowest_ankle
    )

    print(
        "frame0 ankle z:",
        first_left_z,
        first_right_z,
    )

    print(
        "display z offset:",
        display_z_offset,
    )

    # Register LingLong only for viewer camera/default metadata.
    rmv.ROBOT_BASE_DICT["linglong20"] = "base_link"
    rmv.VIEWER_CAM_DISTANCE_DICT["linglong20"] = 2.2

    viewer = rmv.RobotMotionViewer(
        robot_type="linglong20",
        robot_path=str(XML),

        motion_fps=FPS,

        camera_follow=True,

        record_video=True,
        video_path=str(VIDEO),

        video_width=640,
        video_height=480,

        window_width=1100,
        window_height=850,

        camera_lookat_height_offset=0.05,
        camera_elevation=8.0,
        camera_distance_scale=1.15,
        camera_azimuth=140.0,
    )

    try:
        for i, q_raw in enumerate(qpos):
            q = q_raw.copy()

            # DISPLAY ONLY.
            q[2] += display_z_offset

            viewer.step(
                root_pos=q[:3],
                root_rot=q[3:7],
                dof_pos=q[7:],
                rate_limit=True,
                follow_camera=True,
            )

            if (
                i == 0
                or (i + 1) % 30 == 0
                or i == len(qpos) - 1
            ):
                print(
                    f"render frame "
                    f"{i:03d}/{len(qpos)-1:03d}"
                )

    finally:
        viewer.close()

    print()
    print("=" * 90)
    print("RESULT")
    print("=" * 90)

    print("video:", VIDEO)
    print("fps  :", FPS)
    print(
        "duration:",
        len(qpos) / FPS,
        "sec",
    )

    print()
    print(
        "LINGLONG20 BODY V1 VISUAL PREVIEW: PASS"
    )


if __name__ == "__main__":
    main()
