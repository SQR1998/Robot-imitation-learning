from pathlib import Path
import csv

import cv2
import numpy as np


SRC_CSV = Path(
    "output/openloong_palm_roll_v1/palm_roll_v1.csv"
)

VIDEO = Path(
    "dataset/openloong_master1/RGB_video_dataset.mp4"
)

OUT_DIR = Path(
    "output/openloong_palm_roll_v11"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "palm_roll_v11.csv"
OUT_VIDEO = OUT_DIR / "palm_roll_v11.mp4"


CALIB_START_SEC = 15.0
CALIB_END_SEC = 18.0


def wrap_deg(x):
    return (
        np.asarray(x, dtype=np.float64)
        + 180.0
    ) % 360.0 - 180.0


# ============================================================
# Load V1
# ============================================================

with open(
    SRC_CSV,
    "r",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )

assert rows

frame = np.array(
    [int(r["frame"]) for r in rows],
    dtype=int,
)

time_sec = np.array(
    [float(r["time_sec"]) for r in rows],
    dtype=np.float64,
)

left_v1 = np.array(
    [
        float(
            r["left_palm_roll_deg"]
        )
        for r in rows
    ],
    dtype=np.float64,
)

right_v1 = np.array(
    [
        float(
            r["right_palm_roll_deg"]
        )
        for r in rows
    ],
    dtype=np.float64,
)


# ============================================================
# Calibration
#
# The user has visually verified:
#   during 15~18 s, BOTH palms face the ground.
#
# Therefore the physical palm-roll target in this interval is 0 deg.
#
# V1 resolved only the normal SIGN, but did not remove the remaining
# camera / landmark / local-frame zero offset.
#
# V1.1 removes that zero offset separately for left and right.
# ============================================================

calib = (
    (time_sec >= CALIB_START_SEC)
    & (time_sec <= CALIB_END_SEC)
)

assert np.count_nonzero(calib) >= 2


# Use median rather than mean because occasional hand detections are noisy.
left_zero = float(
    np.median(
        left_v1[calib]
    )
)

right_zero = float(
    np.median(
        right_v1[calib]
    )
)


left_unwrapped = (
    left_v1
    - left_zero
)

right_unwrapped = (
    right_v1
    - right_zero
)


# Physical orientation is periodic.
# Do NOT let "360 deg" become a new physical state.
left_roll = wrap_deg(
    left_unwrapped
)

right_roll = wrap_deg(
    right_unwrapped
)


# ============================================================
# Report
# ============================================================

print("=" * 72)
print("OpenLoong Palm Roll V1.1 - calibrated physical roll")
print("=" * 72)

print(
    "LEFT zero offset :",
    f"{left_zero:.2f} deg",
)

print(
    "RIGHT zero offset:",
    f"{right_zero:.2f} deg",
)


def report(
    name,
    t0,
    t1,
):
    sel = (
        (time_sec >= t0)
        & (time_sec <= t1)
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    for side, x in (
        ("LEFT", left_roll),
        ("RIGHT", right_roll),
    ):
        seg = x[sel]

        print(
            f"{side:5s} calibrated roll: "
            f"mean={np.mean(seg):7.2f} "
            f"median={np.median(seg):7.2f} "
            f"min={np.min(seg):7.2f} "
            f"max={np.max(seg):7.2f} deg"
        )


report(
    "7~9 sec",
    7.0,
    9.0,
)

report(
    "15~18 sec ZERO check",
    15.0,
    18.0,
)

print()
print(
    "full physical range:",
)

print(
    "LEFT :",
    f"{left_roll.min():.2f}",
    "~",
    f"{left_roll.max():.2f}",
    "deg",
)

print(
    "RIGHT:",
    f"{right_roll.min():.2f}",
    "~",
    f"{right_roll.max():.2f}",
    "deg",
)


# Detect suspicious single-frame changes on the physical wrapped circle.
left_step = np.abs(
    wrap_deg(
        np.diff(
            left_roll
        )
    )
)

right_step = np.abs(
    wrap_deg(
        np.diff(
            right_roll
        )
    )
)

print()
print(
    "max circular one-frame step:"
)

print(
    "LEFT :",
    f"{left_step.max():.2f} deg/frame"
)

print(
    "RIGHT:",
    f"{right_step.max():.2f} deg/frame"
)


# ============================================================
# Save CSV
# ============================================================

extra_cols = [
    "left_palm_roll_v11_deg",
    "right_palm_roll_v11_deg",
    "left_palm_roll_v11_unwrapped_deg",
    "right_palm_roll_v11_unwrapped_deg",
    "left_zero_offset_deg",
    "right_zero_offset_deg",
]


with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = (
        list(
            rows[0].keys()
        )
        + extra_cols
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for i, r in enumerate(rows):
        out = dict(r)

        out.update({
            "left_palm_roll_v11_deg":
                left_roll[i],

            "right_palm_roll_v11_deg":
                right_roll[i],

            "left_palm_roll_v11_unwrapped_deg":
                left_unwrapped[i],

            "right_palm_roll_v11_unwrapped_deg":
                right_unwrapped[i],

            "left_zero_offset_deg":
                left_zero,

            "right_zero_offset_deg":
                right_zero,
        })

        writer.writerow(
            out
        )


# ============================================================
# Simple video overlay
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO)
)

assert cap.isOpened()

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
    str(OUT_VIDEO),

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


for i in range(
    min(
        len(rows),
        len(frame),
    )
):
    ok, image = cap.read()

    if not ok:
        break

    cv2.putText(
        image,
        (
            "PALM ROLL V1.1  "
            f"{time_sec[i]:.2f}s"
        ),
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        image,
        (
            f"LEFT  roll={left_roll[i]:+7.1f} deg"
        ),
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 200, 0),
        2,
    )

    cv2.putText(
        image,
        (
            f"RIGHT roll={right_roll[i]:+7.1f} deg"
        ),
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 200, 255),
        2,
    )

    if (
        CALIB_START_SEC
        <= time_sec[i]
        <= CALIB_END_SEC
    ):
        cv2.putText(
            image,
            "KNOWN PALM-DOWN CALIBRATION",
            (20, 162),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
        )

    writer.write(
        image
    )


cap.release()
writer.release()


print()
print("=" * 72)
print("DONE")
print("=" * 72)
print("CSV  :", OUT_CSV)
print("VIDEO:", OUT_VIDEO)
