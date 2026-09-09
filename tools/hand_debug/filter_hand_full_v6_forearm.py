from pathlib import Path
import csv
import subprocess

import cv2
import mediapipe as mp
import numpy as np


RAW_CSV = Path(
    "output/openloong_hand_full_v1/hand_full.csv"
)

V5_CSV = Path(
    "output/openloong_hand_full_filter_v5/"
    "hand_full_filtered_v5.csv"
)

VIDEO = Path(
    "dataset/openloong_master1/"
    "RGB_video_dataset.mp4"
)

POSE_MODEL = Path(
    "checkpoints/mediapipe/"
    "pose_landmarker_heavy.task"
)

OUT_DIR = Path(
    "output/openloong_hand_full_filter_v6"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "hand_full_filtered_v6.csv"
TMP_VIDEO = OUT_DIR / "hand_full_filtered_v6_tmp.mp4"
OUT_VIDEO = OUT_DIR / "hand_full_filtered_v6_h264.mp4"


FPS = 30.0

LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


# ============================================================
# V6关键参数
# ============================================================

# 手指相对前臂在图像平面内允许的最大弯折。
#
# 100°已经非常宽松，
# 但绝不允许产生360°。
MAX_RELATIVE_DEG = 100.0

# V5已经完成过第一轮误检剔除，
# V6继续使用V5 accepted作为初始锚点。

# 相邻有效锚点的相对腕角如果突然变化太多，
# 则认为当前锚点不可信。
#
# 每帧最多允许12°变化 + 12°容差。
ANCHOR_STEP_DEG_PER_FRAME = 12.0
ANCHOR_MARGIN_DEG = 12.0

# 即使隔了较长时间，
# 单次锚点跳变也不能超过80°。
MAX_ANCHOR_JUMP_DEG = 80.0

# 中心平滑半径：
# 4 -> 9帧 ≈ 0.30秒
SMOOTH_RADIUS = 4

# 最终相对腕角每帧最大变化
# 6°/frame @30FPS = 180°/s
FINAL_MAX_STEP_DEG = 6.0


def clamp(x, lo, hi):
    return np.minimum(
        np.maximum(x, lo),
        hi,
    )


def wrap_deg(x):
    return (
        x + 180.0
    ) % 360.0 - 180.0


def triangular_smooth(
    values,
    radius,
    passes=2,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if radius <= 0:
        return values.copy()

    ramp = np.arange(
        1,
        radius + 2,
        dtype=np.float64,
    )

    kernel = np.concatenate([
        ramp,
        ramp[-2::-1],
    ])

    kernel /= kernel.sum()

    out = values.copy()

    for _ in range(passes):

        padded = np.pad(
            out,
            (radius, radius),
            mode="edge",
        )

        out = np.convolve(
            padded,
            kernel,
            mode="valid",
        )

    return out


def speed_limit_scalar(
    values,
    max_step,
):
    """
    对相对腕角做前向+反向限速。

    注意：
    这里处理的是 [-100,100] 范围内的物理腕角，
    不再把它当成可以无限绕圈的角度。
    """

    out = values.copy()

    # forward
    for i in range(
        1,
        len(out),
    ):

        delta = (
            out[i]
            - out[i - 1]
        )

        delta = np.clip(
            delta,
            -max_step,
            max_step,
        )

        out[i] = (
            out[i - 1]
            + delta
        )

    # backward
    for i in range(
        len(out) - 2,
        -1,
        -1,
    ):

        delta = (
            out[i]
            - out[i + 1]
        )

        delta = np.clip(
            delta,
            -max_step,
            max_step,
        )

        out[i] = (
            out[i + 1]
            + delta
        )

    return out


# ============================================================
# 读取RAW CSV
# ============================================================

raw_rows = []

with open(
    RAW_CSV,
    "r",
    encoding="utf-8",
) as f:

    raw_rows = list(
        csv.DictReader(f)
    )


frames = sorted(
    set(
        int(r["frame"])
        for r in raw_rows
    )
)

N = len(frames)

assert N == 661, N

frame_to_idx = {
    frame: i
    for i, frame in enumerate(
        frames
    )
}


# ============================================================
# 读取V5 accepted结果
# ============================================================

v5_rows = []

with open(
    V5_CSV,
    "r",
    encoding="utf-8",
) as f:

    v5_rows = list(
        csv.DictReader(f)
    )


assert len(v5_rows) == N


v5_accept = {
    "LEFT": np.array(
        [
            int(
                r["left_accepted"]
            )
            for r in v5_rows
        ],
        dtype=bool,
    ),

    "RIGHT": np.array(
        [
            int(
                r["right_accepted"]
            )
            for r in v5_rows
        ],
        dtype=bool,
    ),
}


# ============================================================
# 使用Pose Heavy重新计算全视频 elbow→wrist
#
# 这里只跑Pose，不再重新跑Hand21。
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


pose_elbow = {
    "LEFT": np.full(
        (N, 2),
        np.nan,
    ),

    "RIGHT": np.full(
        (N, 2),
        np.nan,
    ),
}

pose_wrist = {
    "LEFT": np.full(
        (N, 2),
        np.nan,
    ),

    "RIGHT": np.full(
        (N, 2),
        np.nan,
    ),
}


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


def pix(lm):
    return np.array(
        [
            lm.x * width,
            lm.y * height,
        ],
        dtype=np.float64,
    )


pose_detected = 0


with PoseLandmarker.create_from_options(
    pose_options
) as detector:

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

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if not result.pose_landmarks:
            continue

        pose_detected += 1

        p = result.pose_landmarks[0]

        pose_elbow["LEFT"][i] = pix(
            p[LEFT_ELBOW]
        )

        pose_wrist["LEFT"][i] = pix(
            p[LEFT_WRIST]
        )

        pose_elbow["RIGHT"][i] = pix(
            p[RIGHT_ELBOW]
        )

        pose_wrist["RIGHT"][i] = pix(
            p[RIGHT_WRIST]
        )


cap.release()


print("=" * 72)
print("OpenLoong Hand Filter V6 - Forearm Relative")
print("=" * 72)

print(
    "pose detected:",
    pose_detected,
    "/",
    N,
)


# ============================================================
# 如果Pose偶尔漏一两帧，坐标做普通线性插值
# ============================================================

all_idx = np.arange(N)


for side in (
    "LEFT",
    "RIGHT",
):

    for array in (
        pose_elbow[side],
        pose_wrist[side],
    ):

        valid = np.where(
            np.isfinite(
                array[:, 0]
            )
        )[0]

        assert len(valid) >= 2

        for axis in range(2):

            array[:, axis] = np.interp(
                all_idx,
                valid,
                array[
                    valid,
                    axis
                ],
            )


# ============================================================
# 前臂方向
# ============================================================

forearm_angle = {}


for side in (
    "LEFT",
    "RIGHT",
):

    v = (
        pose_wrist[side]
        - pose_elbow[side]
    )

    forearm_angle[side] = np.rad2deg(
        np.arctan2(
            v[:, 1],
            v[:, 0],
        )
    )


# ============================================================
# 读取原始Hand21方向
# ============================================================

raw_hand_angle = {
    "LEFT": np.full(
        N,
        np.nan,
    ),

    "RIGHT": np.full(
        N,
        np.nan,
    ),
}


for r in raw_rows:

    side = r["side"]

    if side not in (
        "LEFT",
        "RIGHT",
    ):
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

    except Exception:
        continue


    if np.hypot(
        dx,
        dy,
    ) < 1e-8:
        continue


    raw_hand_angle[
        side
    ][i] = np.rad2deg(
        np.arctan2(
            dy,
            dx,
        )
    )


# ============================================================
# 正式V6过滤
# ============================================================

output = {}


for side in (
    "LEFT",
    "RIGHT",
):

    hand_angle = (
        raw_hand_angle[
            side
        ]
    )

    forearm = (
        forearm_angle[
            side
        ]
    )


    # --------------------------------------------------------
    # 手指相对前臂的角度
    # --------------------------------------------------------

    raw_relative = wrap_deg(
        hand_angle
        - forearm
    )


    anchors = (
        v5_accept[side]
        & np.isfinite(
            raw_relative
        )
    )


    # --------------------------------------------------------
    # 第一层：
    # 生理/几何范围硬门控
    #
    # 手不可能在2D里相对前臂连续绕360°
    # --------------------------------------------------------

    reject_geometry = (
        anchors
        & (
            np.abs(
                raw_relative
            )
            > MAX_RELATIVE_DEG
        )
    )

    anchors[
        reject_geometry
    ] = False


    # --------------------------------------------------------
    # 第二层：
    # 相邻可信锚点连续性判断
    # --------------------------------------------------------

    reject_jump = np.zeros(
        N,
        dtype=bool,
    )


    candidate_idx = np.where(
        anchors
    )[0]


    final_anchor = np.zeros(
        N,
        dtype=bool,
    )


    last_idx = None
    last_value = None


    for i in candidate_idx:

        value = raw_relative[i]

        if last_idx is None:

            final_anchor[i] = True

            last_idx = i
            last_value = value

            continue


        gap = (
            i - last_idx
        )


        allowed = min(
            MAX_ANCHOR_JUMP_DEG,

            (
                ANCHOR_STEP_DEG_PER_FRAME
                * gap
                + ANCHOR_MARGIN_DEG
            ),
        )


        delta = abs(
            value
            - last_value
        )


        if delta > allowed:

            reject_jump[i] = True

            # 注意：
            # 错误锚点不更新last_value
            continue


        final_anchor[i] = True

        last_idx = i
        last_value = value


    valid_idx = np.where(
        final_anchor
    )[0]


    assert len(valid_idx) >= 2


    # ========================================================
    # 核心变化：
    #
    # 这里绝对不再 np.unwrap()
    #
    # 相对腕角就是普通的 [-100,100] 标量，
    # 在有效锚点之间直接线性插值。
    #
    # 因此不可能凭空出现360°。
    # ========================================================

    relative = np.interp(
        all_idx,
        valid_idx,
        raw_relative[
            valid_idx
        ],
    )


    relative = clamp(
        relative,
        -MAX_RELATIVE_DEG,
        +MAX_RELATIVE_DEG,
    )


    # --------------------------------------------------------
    # 平滑
    # --------------------------------------------------------

    relative = triangular_smooth(
        relative,
        SMOOTH_RADIUS,
        passes=2,
    )


    # --------------------------------------------------------
    # 最终速度限制
    # --------------------------------------------------------

    relative = speed_limit_scalar(
        relative,
        FINAL_MAX_STEP_DEG,
    )


    relative = clamp(
        relative,
        -MAX_RELATIVE_DEG,
        +MAX_RELATIVE_DEG,
    )


    # --------------------------------------------------------
    # 最终绝对指尖方向
    #
    # = 前臂方向 + 稳定后的腕部相对角
    # --------------------------------------------------------

    final_angle = (
        forearm
        + relative
    )


    final_dir = np.stack(
        [
            np.cos(
                np.deg2rad(
                    final_angle
                )
            ),

            np.sin(
                np.deg2rad(
                    final_angle
                )
            ),
        ],

        axis=1,
    )


    output[side] = {
        "relative":
            relative,

        "final_angle":
            final_angle,

        "direction":
            final_dir,

        "raw_relative":
            raw_relative,

        "anchor":
            final_anchor,

        "reject_geometry":
            reject_geometry,

        "reject_jump":
            reject_jump,
    }


    print()
    print(
        f"[{side}]"
    )

    print(
        "V5 anchors       :",
        int(
            v5_accept[
                side
            ].sum()
        ),
    )

    print(
        "geometry rejected:",
        int(
            reject_geometry.sum()
        ),
    )

    print(
        "jump rejected    :",
        int(
            reject_jump.sum()
        ),
    )

    print(
        "V6 final anchors :",
        int(
            final_anchor.sum()
        ),
    )

    print(
        "relative range   :",
        f"{relative.min():.2f}",
        "~",
        f"{relative.max():.2f}",
        "deg",
    )


# ============================================================
# 专门检查7~9秒左手
# ============================================================

lo = int(
    7.0 * FPS
)

hi = int(
    9.0 * FPS
) + 1


left_seg = (
    output[
        "LEFT"
    ]["relative"][lo:hi]
)


print()
print("=" * 72)
print("LEFT 7~9 sec safety check")
print("=" * 72)

print(
    "relative min/max:",
    f"{left_seg.min():.2f}",
    "~",
    f"{left_seg.max():.2f}",
    "deg",
)

print(
    "total relative travel:",
    f"{np.sum(np.abs(np.diff(left_seg))):.2f}",
    "deg",
)

print(
    "maximum one-frame step:",
    f"{np.max(np.abs(np.diff(left_seg))):.2f}",
    "deg",
)


# ============================================================
# 保存CSV
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

        "left_relative_forearm_deg",
        "right_relative_forearm_deg",

        "left_forearm_deg",
        "right_forearm_deg",

        "left_anchor",
        "right_anchor",

        "left_geometry_rejected",
        "right_geometry_rejected",

        "left_jump_rejected",
        "right_jump_rejected",
    ])


    for i in range(N):

        writer.writerow([
            i,
            i / FPS,

            output[
                "LEFT"
            ]["direction"][i, 0],

            output[
                "LEFT"
            ]["direction"][i, 1],

            output[
                "RIGHT"
            ]["direction"][i, 0],

            output[
                "RIGHT"
            ]["direction"][i, 1],

            output[
                "LEFT"
            ]["relative"][i],

            output[
                "RIGHT"
            ]["relative"][i],

            forearm_angle[
                "LEFT"
            ][i],

            forearm_angle[
                "RIGHT"
            ][i],

            int(
                output[
                    "LEFT"
                ]["anchor"][i]
            ),

            int(
                output[
                    "RIGHT"
                ]["anchor"][i]
            ),

            int(
                output[
                    "LEFT"
                ]["reject_geometry"][i]
            ),

            int(
                output[
                    "RIGHT"
                ]["reject_geometry"][i]
            ),

            int(
                output[
                    "LEFT"
                ]["reject_jump"][i]
            ),

            int(
                output[
                    "RIGHT"
                ]["reject_jump"][i]
            ),
        ])


# ============================================================
# 可视化
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO)
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


for i in range(N):

    ok, frame = cap.read()

    if not ok:
        break


    cv2.putText(
        frame,

        (
            "V6 FOREARM-RELATIVE "
            f"{i/FPS:.2f}s"
        ),

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            0,
            255,
            255,
        ),

        2,
    )


    for side in (
        "LEFT",
        "RIGHT",
    ):

        elbow = pose_elbow[
            side
        ][i]

        wrist = pose_wrist[
            side
        ][i]

        direction = output[
            side
        ]["direction"][i]


        # 前臂方向：橙色
        cv2.line(
            frame,

            tuple(
                elbow.astype(int)
            ),

            tuple(
                wrist.astype(int)
            ),

            (
                0,
                165,
                255,
            ),

            4,
        )


        # 最终手指方向：紫色
        end = (
            wrist
            + direction * 170
        )


        cv2.arrowedLine(
            frame,

            tuple(
                wrist.astype(int)
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

            tipLength=0.22,
        )


        if output[
            side
        ]["anchor"][i]:

            color = (
                0,
                255,
                0,
            )

        elif (
            output[
                side
            ]["reject_geometry"][i]
            or output[
                side
            ]["reject_jump"][i]
        ):

            color = (
                0,
                0,
                255,
            )

        else:

            color = (
                0,
                255,
                255,
            )


        cv2.circle(
            frame,

            tuple(
                wrist.astype(int)
            ),

            9,

            color,

            2,
        )


        cv2.putText(
            frame,

            (
                f"{side} "
                f"rel="
                f"{output[side]['relative'][i]:.0f}"
            ),

            (
                int(wrist[0]) + 8,
                int(wrist[1]) - 8,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

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
print("=" * 72)
print("DONE")
print("CSV  :", OUT_CSV)
print("VIDEO:", OUT_VIDEO)
print("=" * 72)
