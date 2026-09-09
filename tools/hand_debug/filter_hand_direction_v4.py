from pathlib import Path
import csv
import subprocess

import cv2
import numpy as np


INPUT_CSV = Path(
    "output/openloong_hand_roi_debug_v2/hand_roi_v2.csv"
)

VIDEO = Path(
    "dataset/openloong_master1/RGB_video_dataset.mp4"
)

OUT_DIR = Path(
    "output/openloong_hand_filter_v4"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "hand_direction_filtered_v4.csv"
TMP_VIDEO = OUT_DIR / "hand_direction_filtered_v4_tmp.mp4"
OUT_VIDEO = OUT_DIR / "hand_direction_filtered_v4_h264.mp4"

FPS = 30.0

# ============================================================
# 关键参数
# ============================================================

# 连续帧通常不允许超过这个变化
BASE_MAX_STEP_DEG = 32.0

# 不管什么情况，超过这个角度视为“翻转误识别”
HARD_FLIP_DEG = 120.0

# 最终输出最大每帧角速度
FINAL_MAX_STEP_DEG = 14.0

# 低置信度点更容易被拒绝
MIN_SCORE = 0.20

# 对称平滑半径：7帧
SMOOTH_RADIUS = 3


def wrap_pi(x):
    return (
        x + np.pi
    ) % (2.0 * np.pi) - np.pi


def angle_diff(a, b):
    return wrap_pi(a - b)


def angle_from_vec(dx, dy):
    return np.arctan2(
        dy,
        dx,
    )


def smooth_angles_centered(
    angles,
    radius,
):
    n = len(angles)

    out = np.zeros(n)

    for i in range(n):

        lo = max(
            0,
            i - radius,
        )

        hi = min(
            n,
            i + radius + 1,
        )

        w = angles[lo:hi]

        cx = np.mean(
            np.cos(w)
        )

        cy = np.mean(
            np.sin(w)
        )

        out[i] = np.arctan2(
            cy,
            cx,
        )

    return out


def limit_step_forward(
    angles,
    max_step_rad,
):
    """
    最终保险：
    相邻帧最多转 max_step_rad。
    """

    out = angles.copy()

    for i in range(
        1,
        len(out),
    ):

        d = angle_diff(
            out[i],
            out[i - 1],
        )

        if abs(d) > max_step_rad:

            out[i] = wrap_pi(
                out[i - 1]
                + np.sign(d)
                * max_step_rad
            )

    return out


def limit_step_bidirectional(
    angles,
    max_step_rad,
):
    """
    前向 + 反向各限制一次。
    比只做单向限制更适合离线视频，
    不容易产生明显滞后。
    """

    out = limit_step_forward(
        angles,
        max_step_rad,
    )

    rev = out[::-1].copy()

    rev = limit_step_forward(
        rev,
        max_step_rad,
    )

    return rev[::-1]


# ============================================================
# 读取数据
# ============================================================

rows = []

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)


frames = sorted(
    set(
        int(r["frame"])
        for r in rows
    )
)

frame_to_idx = {
    f: i
    for i, f in enumerate(frames)
}

N = len(frames)

result = {}


print("=" * 70)
print("Hand Direction Filter V4")
print("=" * 70)
print(
    "frames:",
    N,
)
print(
    "frame range:",
    frames[0],
    "~",
    frames[-1],
)
print("=" * 70)


# ============================================================
# 左右手分别处理
# ============================================================

for side in [
    "LEFT",
    "RIGHT",
]:

    raw_angle = np.full(
        N,
        np.nan,
    )

    raw_score = np.zeros(
        N,
    )

    raw_wrist = np.full(
        (N, 2),
        np.nan,
    )

    raw_detected = np.zeros(
        N,
        dtype=bool,
    )


    for r in rows:

        if r["side"] != side:
            continue

        if int(
            r["detected"]
        ) != 1:
            continue

        i = frame_to_idx[
            int(r["frame"])
        ]

        try:

            dx = float(
                r["finger_dir_x"]
            )

            dy = float(
                r["finger_dir_y"]
            )

            wx = float(
                r["hand_wrist_x"]
            )

            wy = float(
                r["hand_wrist_y"]
            )

            score = float(
                r["score"]
            )

        except Exception:
            continue


        norm = np.hypot(
            dx,
            dy,
        )

        if norm < 1e-6:
            continue


        dx /= norm
        dy /= norm


        raw_angle[i] = angle_from_vec(
            dx,
            dy,
        )

        raw_score[i] = score

        raw_wrist[i] = [
            wx,
            wy,
        ]

        raw_detected[i] = True


    # ========================================================
    # 第一层：连续性门控
    # ========================================================

    accepted = np.zeros(
        N,
        dtype=bool,
    )

    rejected_flip = np.zeros(
        N,
        dtype=bool,
    )

    rejected_jump = np.zeros(
        N,
        dtype=bool,
    )

    rejected_score = np.zeros(
        N,
        dtype=bool,
    )


    valid_indices = np.where(
        raw_detected
    )[0]


    last_idx = None
    last_angle = None


    for i in valid_indices:

        a = raw_angle[i]


        # 置信度过低直接拒绝
        if raw_score[i] < MIN_SCORE:

            rejected_score[i] = True
            continue


        if last_idx is None:

            accepted[i] = True

            last_idx = i
            last_angle = a

            continue


        gap = i - last_idx

        delta = abs(
            np.rad2deg(
                angle_diff(
                    a,
                    last_angle,
                )
            )
        )


        # ----------------------------------------
        # 180度翻转硬拒绝
        # ----------------------------------------

        if delta >= HARD_FLIP_DEG:

            rejected_flip[i] = True

            # 注意：
            # 不更新 last_angle
            # 所以后续错误的180度数据也继续被拒绝

            continue


        # ----------------------------------------
        # 连续性阈值
        #
        # gap越大，允许变化稍大，
        # 但不能无限放宽
        # ----------------------------------------

        allowed = (
            BASE_MAX_STEP_DEG
            * max(
                1,
                min(
                    gap,
                    3,
                )
            )
        )


        if delta > allowed:

            rejected_jump[i] = True
            continue


        accepted[i] = True

        last_idx = i
        last_angle = a


    accepted_idx = np.where(
        accepted
    )[0]


    if len(
        accepted_idx
    ) < 2:

        raise RuntimeError(
            f"{side}: "
            "accepted points too few"
        )


    # ========================================================
    # 第二层：角度展开
    #
    # 这里不再使用cos/sin直接跨越180°插值，
    # 避免穿过原点导致方向突然翻转。
    # ========================================================

    accepted_angles = raw_angle[
        accepted_idx
    ]

    unwrapped = np.unwrap(
        accepted_angles
    )


    all_idx = np.arange(N)


    interp_unwrapped = np.interp(
        all_idx,
        accepted_idx,
        unwrapped,
    )


    interp_angle = np.array([
        wrap_pi(a)
        for a in interp_unwrapped
    ])


    # ========================================================
    # 第三层：中心平滑
    # ========================================================

    filtered = smooth_angles_centered(
        interp_angle,
        SMOOTH_RADIUS,
    )


    # ========================================================
    # 第四层：最终角速度硬限制
    # ========================================================

    filtered = limit_step_bidirectional(
        filtered,
        np.deg2rad(
            FINAL_MAX_STEP_DEG
        ),
    )


    # ========================================================
    # wrist只用于可视化
    # ========================================================

    wrist_valid = np.where(
        np.isfinite(
            raw_wrist[:, 0]
        )
    )[0]


    filtered_wrist = np.zeros(
        (N, 2)
    )


    for axis in range(2):

        filtered_wrist[
            :,
            axis
        ] = np.interp(
            all_idx,

            wrist_valid,

            raw_wrist[
                wrist_valid,
                axis
            ],
        )


    direction = np.stack(
        [
            np.cos(
                filtered
            ),

            np.sin(
                filtered
            ),
        ],
        axis=1,
    )


    result[side] = {
        "angle":
            filtered,

        "direction":
            direction,

        "wrist":
            filtered_wrist,

        "raw":
            raw_detected,

        "accepted":
            accepted,

        "flip":
            rejected_flip,

        "jump":
            rejected_jump,

        "score_reject":
            rejected_score,
    }


    print()
    print(
        f"[{side}]"
    )

    print(
        "raw detected:",
        int(
            raw_detected.sum()
        ),
    )

    print(
        "accepted:",
        int(
            accepted.sum()
        ),
    )

    print(
        "180deg flips rejected:",
        int(
            rejected_flip.sum()
        ),
    )

    print(
        "large jumps rejected:",
        int(
            rejected_jump.sum()
        ),
    )

    print(
        "low score rejected:",
        int(
            rejected_score.sum()
        ),
    )


# ============================================================
# 输出CSV
# ============================================================

with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "time_sec",

        "left_dir_x",
        "left_dir_y",

        "right_dir_x",
        "right_dir_y",

        "left_angle_deg",
        "right_angle_deg",

        "left_raw",
        "right_raw",

        "left_accepted",
        "right_accepted",

        "left_flip_rejected",
        "right_flip_rejected",

        "left_jump_rejected",
        "right_jump_rejected",
    ])


    for i, frame_id in enumerate(
        frames
    ):

        writer.writerow([
            frame_id,
            frame_id / FPS,

            result[
                "LEFT"
            ]["direction"][i, 0],

            result[
                "LEFT"
            ]["direction"][i, 1],

            result[
                "RIGHT"
            ]["direction"][i, 0],

            result[
                "RIGHT"
            ]["direction"][i, 1],

            np.rad2deg(
                result[
                    "LEFT"
                ]["angle"][i]
            ),

            np.rad2deg(
                result[
                    "RIGHT"
                ]["angle"][i]
            ),

            int(
                result[
                    "LEFT"
                ]["raw"][i]
            ),

            int(
                result[
                    "RIGHT"
                ]["raw"][i]
            ),

            int(
                result[
                    "LEFT"
                ]["accepted"][i]
            ),

            int(
                result[
                    "RIGHT"
                ]["accepted"][i]
            ),

            int(
                result[
                    "LEFT"
                ]["flip"][i]
            ),

            int(
                result[
                    "RIGHT"
                ]["flip"][i]
            ),

            int(
                result[
                    "LEFT"
                ]["jump"][i]
            ),

            int(
                result[
                    "RIGHT"
                ]["jump"][i]
            ),
        ])


print()
print(
    "filtered CSV:",
    OUT_CSV,
)


# ============================================================
# 可视化
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO)
)

assert cap.isOpened()


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

fps = float(
    cap.get(
        cv2.CAP_PROP_FPS
    )
)


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    frames[0],
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


for i, frame_id in enumerate(
    frames
):

    ok, frame = cap.read()

    if not ok:
        break


    cv2.putText(
        frame,

        (
            f"ANTI-FLIP FILTER "
            f"{frame_id/FPS:.2f}s"
        ),

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        (
            0,
            255,
            255,
        ),

        2,
    )


    for side in [
        "LEFT",
        "RIGHT",
    ]:

        start = result[
            side
        ]["wrist"][i]

        direction = result[
            side
        ]["direction"][i]

        end = (
            start
            + direction * 170
        )


        # 最终紫色方向
        cv2.arrowedLine(
            frame,

            tuple(
                start.astype(int)
            ),

            tuple(
                end.astype(int)
            ),

            (
                255,
                0,
                255,
            ),

            7,

            tipLength=0.25,
        )


        if result[
            side
        ]["accepted"][i]:

            circle_color = (
                0,
                255,
                0,
            )

        elif (
            result[
                side
            ]["flip"][i]
            or result[
                side
            ]["jump"][i]
        ):

            # 红色：
            # 本帧原来检测到了，
            # 但因为突变被拒绝
            circle_color = (
                0,
                0,
                255,
            )

        else:

            # 黄色：
            # 漏检/插值
            circle_color = (
                0,
                255,
                255,
            )


        cv2.circle(
            frame,

            tuple(
                start.astype(int)
            ),

            9,

            circle_color,

            2,
        )


        cv2.putText(
            frame,

            side,

            (
                int(start[0]) + 10,
                int(start[1]) - 10,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (
                255,
                255,
                0,
            ),

            2,
        )


    writer.write(
        frame
    )


writer.release()
cap.release()


subprocess.run(
    [
        "ffmpeg",
        "-y",

        "-i",
        str(TMP_VIDEO),

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(OUT_VIDEO),
    ],

    check=True,
)


print()
print(
    "video:",
    OUT_VIDEO,
)

print("=" * 70)
