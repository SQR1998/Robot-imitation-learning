#!/usr/bin/env python3

from pathlib import Path
import sys

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
    / "output/linglong20_arm_v1"
    / "linglong20_arm_v1_qpos37.csv"
)

OUT_DIR = (
    ROOT
    / "output/linglong20_arm_v1"
)

FPS = 30.0


def render(
    qpos,
    filename,
    azimuth,
    elevation=5.0,
):

    video_path = (
        OUT_DIR
        / filename
    )

    print()
    print("=" * 100)
    print("RENDER")
    print("=" * 100)

    print("video:", video_path)
    print("azimuth:", azimuth)
    print("elevation:", elevation)


    rmv.ROBOT_BASE_DICT[
        "linglong20"
    ] = "base_link"

    rmv.VIEWER_CAM_DISTANCE_DICT[
        "linglong20"
    ] = 2.2


    viewer = rmv.RobotMotionViewer(
        robot_type="linglong20",

        robot_path=str(XML),

        motion_fps=FPS,

        camera_follow=True,

        record_video=True,

        video_path=str(video_path),

        video_width=640,
        video_height=480,

        window_width=1100,
        window_height=850,

        camera_lookat_height_offset=0.08,

        camera_elevation=elevation,

        camera_distance_scale=1.10,

        camera_azimuth=azimuth,
    )


    try:

        for i, q in enumerate(qpos):

            viewer.step(
                root_pos=q[:3],

                root_rot=q[3:7],

                dof_pos=q[7:],

                rate_limit=False,

                follow_camera=True,
            )


            if (
                i == 0
                or (i + 1) % 100 == 0
                or i == len(qpos) - 1
            ):

                print(
                    f"{filename}: "
                    f"frame {i:03d}/{len(qpos)-1:03d}"
                )

    finally:

        viewer.close()


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    qpos = np.loadtxt(
        CSV,
        delimiter=",",
        dtype=np.float64,
    )


    assert qpos.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(qpos)
    )


    print("=" * 100)
    print("LINGLONG FULL ARM V1 VISUAL CHECK")
    print("=" * 100)

    print(
        "input:",
        CSV,
    )

    print(
        "shape:",
        qpos.shape,
    )


    # Front:
    # shoulder roll / symmetry / arm spacing
    render(
        qpos,
        "linglong20_arm_v1_front.mp4",
        180.0,
    )


    # Side:
    # shoulder pitch / elbow flexion
    render(
        qpos,
        "linglong20_arm_v1_side.mp4",
        90.0,
    )


    # Three-quarter:
    # best overall inspection
    render(
        qpos,
        "linglong20_arm_v1_threequarter.mp4",
        135.0,
    )


    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    for name in [
        "linglong20_arm_v1_front.mp4",
        "linglong20_arm_v1_side.mp4",
        "linglong20_arm_v1_threequarter.mp4",
    ]:

        print(
            OUT_DIR / name
        )


    print()

    print(
        "LINGLONG FULL ARM V1 "
        "VISUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()
