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
    / "output/linglong20_lower_v21"
    / "linglong20_lower_v21_qpos37.csv"
)

VIDEO = (
    ROOT
    / "output/linglong20_lower_v21"
    / "linglong20_lower_v21_preview.mp4"
)

FPS = 30.0


def main():

    qpos = np.loadtxt(
        CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert qpos.shape == (661, 37)
    assert np.all(np.isfinite(qpos))

    print("=" * 90)
    print("LINGLONG LOWER V2.1 VISUAL CHECK")
    print("=" * 90)

    print("input :", CSV)
    print("shape :", qpos.shape)
    print("video :", VIDEO)
    print("fps   :", FPS)

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
        video_path=str(VIDEO),

        video_width=640,
        video_height=480,

        window_width=1100,
        window_height=850,

        camera_lookat_height_offset=0.05,

        camera_elevation=8.0,
        camera_distance_scale=1.15,

        # Keep same view as previous previews
        # so V1 and V2.1 are directly comparable.
        camera_azimuth=140.0,
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
                or i == 660
            ):
                print(
                    f"render frame "
                    f"{i:03d}/660"
                )

    finally:
        viewer.close()

    print()
    print("video saved:", VIDEO)
    print(
        "LINGLONG LOWER V2.1 "
        "VISUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()
