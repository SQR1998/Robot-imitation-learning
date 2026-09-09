from pathlib import Path
import csv
import subprocess

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# OpenLoong Hand Direction + Palm Roll Visualizer V1
#
# Purple arrow:
#   V6 final 2D finger direction (REAL V6 output)
#
# Green dial:
#   Palm Roll V2 scalar
#   0 deg   = palm-down calibration posture
#   +/-90   = side-roll
#   +/-180  = opposite / palm-up side
#
# IMPORTANT:
# The green dial is a ROLL GAUGE, not a literal image-plane palm-normal
# projection. This avoids giving a misleading 2D arrow for a 3D roll.
# ============================================================

VIDEO = Path(
    "dataset/openloong_master1/RGB_video_dataset.mp4"
)

POSE_MODEL = Path(
    "checkpoints/mediapipe/pose_landmarker_heavy.task"
)

V6_CSV = Path(
    "output/openloong_hand_full_filter_v6/"
    "hand_full_filtered_v6.csv"
)

PALM_CSV = Path(
    "output/openloong_palm_roll_v2/"
    "palm_roll_v2.csv"
)

OUT_DIR = Path(
    "output/openloong_hand_palm_visual_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TMP_VIDEO = OUT_DIR / "hand_palm_visual_v1_tmp.mp4"
OUT_VIDEO = OUT_DIR / "hand_palm_visual_v1_h264.mp4"


LEFT_WRIST = 15
RIGHT_WRIST = 16

FINGER_ARROW_LEN = 130

# Gauge size
GAUGE_RADIUS = 62
GAUGE_NEEDLE = 48


def wrap_deg(x):
    return (float(x) + 180.0) % 360.0 - 180.0


def unit2(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))

    if n < 1e-10:
        return np.array([1.0, 0.0], dtype=np.float64)

    return v / n


def palm_state(angle):
    a = abs(wrap_deg(angle))

    if a <= 30.0:
        return "DOWN"

    if a >= 150.0:
        return "UP/OPPOSITE"

    return "SIDE"


def draw_roll_gauge(
    image,
    center,
    angle_deg,
    title,
    color,
):
    cx, cy = center

    cv2.circle(
        image,
        (cx, cy),
        GAUGE_RADIUS,
        (210, 210, 210),
        2,
    )

    # Reference directions:
    # 0 deg = DOWN
    refs = [
        (0, "0 DOWN"),
        (90, "+90"),
        (-90, "-90"),
        (180, "180 UP"),
    ]

    for deg, label in refs:
        rad = np.deg2rad(deg)

        dx = int(
            np.sin(rad)
            * GAUGE_RADIUS
        )

        dy = int(
            np.cos(rad)
            * GAUGE_RADIUS
        )

        p1 = (
            cx + int(dx * 0.85),
            cy + int(dy * 0.85),
        )

        p2 = (
            cx + dx,
            cy + dy,
        )

        cv2.line(
            image,
            p1,
            p2,
            (160, 160, 160),
            2,
        )

    # Needle:
    # 0 = image down
    rad = np.deg2rad(
        wrap_deg(angle_deg)
    )

    end = (
        cx
        + int(
            np.sin(rad)
            * GAUGE_NEEDLE
        ),
        cy
        + int(
            np.cos(rad)
            * GAUGE_NEEDLE
        ),
    )

    cv2.arrowedLine(
        image,
        (cx, cy),
        end,
        color,
        5,
        tipLength=0.24,
    )

    cv2.putText(
        image,
        title,
        (
            cx - GAUGE_RADIUS,
            cy - GAUGE_RADIUS - 22,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
    )

    cv2.putText(
        image,
        (
            f"{wrap_deg(angle_deg):+.1f} deg "
            f"{palm_state(angle_deg)}"
        ),
        (
            cx - GAUGE_RADIUS,
            cy + GAUGE_RADIUS + 28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        color,
        2,
    )


# ============================================================
# Load V6
# ============================================================

with open(
    V6_CSV,
    "r",
    encoding="utf-8",
) as f:
    v6_rows = list(
        csv.DictReader(f)
    )

with open(
    PALM_CSV,
    "r",
    encoding="utf-8",
) as f:
    palm_rows = list(
        csv.DictReader(f)
    )

assert len(v6_rows) == len(palm_rows), (
    len(v6_rows),
    len(palm_rows),
)

N = len(v6_rows)


def col_float(rows, key):
    return np.array(
        [
            float(r[key])
            for r in rows
        ],
        dtype=np.float64,
    )


def col_int(rows, key):
    return np.array(
        [
            int(r[key])
            for r in rows
        ],
        dtype=int,
    )


v6_left_dir = np.stack(
    [
        col_float(
            v6_rows,
            "left_dir_x",
        ),
        col_float(
            v6_rows,
            "left_dir_y",
        ),
    ],
    axis=1,
)

v6_right_dir = np.stack(
    [
        col_float(
            v6_rows,
            "right_dir_x",
        ),
        col_float(
            v6_rows,
            "right_dir_y",
        ),
    ],
    axis=1,
)

v6_left_anchor = col_int(
    v6_rows,
    "left_anchor",
)

v6_right_anchor = col_int(
    v6_rows,
    "right_anchor",
)


left_roll = col_float(
    palm_rows,
    "left_palm_roll_v2_deg",
)

right_roll = col_float(
    palm_rows,
    "right_palm_roll_v2_deg",
)


print("=" * 72)
print("OpenLoong V6 + Palm Roll V2 Visualizer")
print("=" * 72)
print("frames:", N)
print(
    "V6 anchors:",
    "LEFT",
    int(
        v6_left_anchor.sum()
    ),
    "RIGHT",
    int(
        v6_right_anchor.sum()
    ),
)
print(
    "Palm source:",
    PALM_CSV,
)


# ============================================================
# Video / Pose
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO)
)

assert cap.isOpened(), VIDEO

fps = float(
    cap.get(
        cv2.CAP_PROP_FPS
    )
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


writer = cv2.VideoWriter(
    str(TMP_VIDEO),
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (
        width,
        height,
    ),
)

assert writer.isOpened()


BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


pose_options = PoseOptions(
    base_options=BaseOptions(
        model_asset_path=str(
            POSE_MODEL
        )
    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.35,
    min_pose_presence_confidence=0.35,
    min_tracking_confidence=0.35,
)


# Gauge positions
left_gauge_center = (
    95,
    145,
)

right_gauge_center = (
    width - 95,
    145,
)


with PoseLandmarker.create_from_options(
    pose_options
) as pose_detector:

    for i in range(N):
        ok, frame = cap.read()

        if not ok:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(
            round(
                i / fps * 1000.0
            )
        )

        result = (
            pose_detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        # Header
        cv2.putText(
            frame,
            (
                "V6 FINGER + PALM ROLL V2 "
                f"{i/fps:.2f}s"
            ),
            (
                20,
                height - 30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                0,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            frame,
            "PURPLE = V6 finger direction",
            (
                20,
                height - 62,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (
                220,
                80,
                220,
            ),
            2,
        )

        cv2.putText(
            frame,
            "GREEN DIAL = palm roll (0 = palm-down)",
            (
                20,
                height - 90,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (
                0,
                220,
                0,
            ),
            2,
        )

        # Palm roll gauges
        draw_roll_gauge(
            frame,
            left_gauge_center,
            left_roll[i],
            "LEFT PALM",
            (
                0,
                230,
                0,
            ),
        )

        draw_roll_gauge(
            frame,
            right_gauge_center,
            right_roll[i],
            "RIGHT PALM",
            (
                0,
                230,
                0,
            ),
        )

        # V6 directions anchored at pose wrist.
        if result.pose_landmarks:
            p = (
                result.pose_landmarks[0]
            )

            for (
                side,
                wrist_idx,
                direction,
                anchor,
            ) in [
                (
                    "LEFT",
                    LEFT_WRIST,
                    v6_left_dir[i],
                    v6_left_anchor[i],
                ),
                (
                    "RIGHT",
                    RIGHT_WRIST,
                    v6_right_dir[i],
                    v6_right_anchor[i],
                ),
            ]:

                lm = p[
                    wrist_idx
                ]

                wrist = np.array(
                    [
                        lm.x * width,
                        lm.y * height,
                    ],
                    dtype=np.float64,
                )

                d = unit2(
                    direction
                )

                end = (
                    wrist
                    + d
                    * FINGER_ARROW_LEN
                )

                color = (
                    255,
                    0,
                    255,
                )

                thickness = (
                    6
                    if anchor
                    else 3
                )

                cv2.arrowedLine(
                    frame,
                    tuple(
                        np.round(
                            wrist
                        ).astype(int)
                    ),
                    tuple(
                        np.round(
                            end
                        ).astype(int)
                    ),
                    color,
                    thickness,
                    tipLength=0.18,
                )

                cv2.putText(
                    frame,
                    (
                        f"{side} V6 "
                        f"A={int(anchor)}"
                    ),
                    (
                        int(wrist[0]) + 10,
                        int(wrist[1]) - 12,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

        writer.write(
            frame
        )

        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == N - 1
        ):
            print(
                f"rendered "
                f"{i + 1}/{N}"
            )


cap.release()
writer.release()


# ============================================================
# H264
# ============================================================

try:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(TMP_VIDEO),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(OUT_VIDEO),
        ],
        check=True,
    )

    try:
        TMP_VIDEO.unlink()
    except Exception:
        pass

except Exception:
    OUT_VIDEO = TMP_VIDEO


print()
print("=" * 72)
print("DONE")
print("=" * 72)
print("VIDEO:", OUT_VIDEO)
