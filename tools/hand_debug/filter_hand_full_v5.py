from pathlib import Path
import csv
import subprocess

import cv2
import numpy as np


INPUT_CSV = Path(
    "output/openloong_hand_full_v1/hand_full.csv"
)

VIDEO = Path(
    "dataset/openloong_master1/RGB_video_dataset.mp4"
)

OUT_DIR = Path(
    "output/openloong_hand_full_filter_v5"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "hand_full_filtered_v5.csv"
TMP_VIDEO = OUT_DIR / "hand_full_filtered_v5_tmp.mp4"
OUT_VIDEO = OUT_DIR / "hand_full_filtered_v5_h264.mp4"


# ============================================================
# 参数
# ============================================================

FPS = 30.0

# 太低的置信度不要
MIN_SCORE = 0.60

# Hand wrist 和 Pose wrist 差太远不要
MAX_WRIST_ERROR_PX = 50.0

# 局部方向与邻域多数方向相差超过70°，认为误检
LOCAL_OUTLIER_DEG = 70.0

# 初步轨迹建立后，如果单点偏离平滑轨迹超过45°，继续剔除
TRAJECTORY_OUTLIER_DEG = 45.0

# 方向平滑窗口半径
# 5 -> 总窗口11帧，约0.37秒
SMOOTH_RADIUS = 5

# 最终每帧最大变化
# 10° / frame @ 30FPS = 300°/s
# 足够真人快速转腕，但不会出现瞬间90°/180°跳变
MAX_STEP_DEG = 10.0


def wrap_pi(x):
    return (
        x + np.pi
    ) % (2.0 * np.pi) - np.pi


def angle_diff(a, b):
    return wrap_pi(a - b)


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


def final_speed_limit(
    angle,
    max_step_rad,
):
    """
    前向 + 反向两次限速。
    避免单向EMA产生明显延迟。
    """

    out = angle.copy()

    # forward
    for i in range(
        1,
        len(out),
    ):

        d = (
            out[i]
            - out[i - 1]
        )

        if abs(d) > max_step_rad:

            out[i] = (
                out[i - 1]
                + np.sign(d)
                * max_step_rad
            )

    # backward
    for i in range(
        len(out) - 2,
        -1,
        -1,
    ):

        d = (
            out[i]
            - out[i + 1]
        )

        if abs(d) > max_step_rad:

            out[i] = (
                out[i + 1]
                + np.sign(d)
                * max_step_rad
            )

    return out


# ============================================================
# 读取 CSV
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

N = len(frames)

assert N == 661, N

frame_to_idx = {
    frame: i
    for i, frame in enumerate(
        frames
    )
}


print("=" * 72)
print("OpenLoong Full Hand Filter V5")
print("=" * 72)
print("frames:", N)
print(
    "time:",
    f"{frames[0]/FPS:.2f}",
    "~",
    f"{frames[-1]/FPS:.2f}",
)
print("=" * 72)


result = {}


# ============================================================
# 左右手独立处理
# ============================================================

for side in (
    "LEFT",
    "RIGHT",
):

    raw_detected = np.zeros(
        N,
        dtype=bool,
    )

    raw_angle = np.full(
        N,
        np.nan,
    )

    score = np.zeros(
        N,
        dtype=np.float64,
    )

    pose_wrist = np.full(
        (N, 2),
        np.nan,
    )

    hand_wrist = np.full(
        (N, 2),
        np.nan,
    )


    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    for r in rows:

        if r["side"] != side:
            continue

        i = frame_to_idx[
            int(r["frame"])
        ]


        # Pose wrist 每帧都有，用于最终箭头起点
        try:

            pose_wrist[i] = [
                float(
                    r["pose_wrist_x"]
                ),
                float(
                    r["pose_wrist_y"]
                ),
            ]

        except Exception:
            pass


        if int(
            r["detected"]
        ) != 1:
            continue


        try:

            dx = float(
                r["finger_dir_x"]
            )

            dy = float(
                r["finger_dir_y"]
            )

            hx = float(
                r["hand_wrist_x"]
            )

            hy = float(
                r["hand_wrist_y"]
            )

            conf = float(
                r["score"]
            )

        except Exception:
            continue


        norm = np.hypot(
            dx,
            dy,
        )

        if norm < 1e-8:
            continue


        dx /= norm
        dy /= norm


        raw_detected[i] = True

        raw_angle[i] = np.arctan2(
            dy,
            dx,
        )

        score[i] = conf

        hand_wrist[i] = [
            hx,
            hy,
        ]


    # ========================================================
    # 第一层：质量门控
    # ========================================================

    accepted = (
        raw_detected.copy()
    )

    reject_quality = np.zeros(
        N,
        dtype=bool,
    )


    for i in np.where(
        accepted
    )[0]:

        wrist_error = np.linalg.norm(
            hand_wrist[i]
            - pose_wrist[i]
        )

        if (
            score[i] < MIN_SCORE
            or wrist_error
            > MAX_WRIST_ERROR_PX
        ):

            accepted[i] = False

            reject_quality[i] = True


    # ========================================================
    # 第二层：局部多数方向
    #
    # 如果某一帧突然180°翻过去，而前后方向都正常，
    # 直接把这一帧丢掉。
    # ========================================================

    reject_local = np.zeros(
        N,
        dtype=bool,
    )


    for i in np.where(
        accepted
    )[0]:

        lo = max(
            0,
            i - 4,
        )

        hi = min(
            N,
            i + 5,
        )

        ids = np.array([
            j
            for j in range(
                lo,
                hi,
            )
            if accepted[j]
        ])


        if len(ids) < 3:
            continue


        weights = (
            score[ids] ** 2
        )


        cx = np.sum(
            weights
            * np.cos(
                raw_angle[ids]
            )
        )

        cy = np.sum(
            weights
            * np.sin(
                raw_angle[ids]
            )
        )


        # 如果方向互相抵消，扩大到17帧邻域
        strength = (
            np.hypot(
                cx,
                cy,
            )
            /
            (
                np.sum(weights)
                + 1e-12
            )
        )


        if strength < 0.25:

            lo = max(
                0,
                i - 8,
            )

            hi = min(
                N,
                i + 9,
            )

            ids = np.array([
                j
                for j in range(
                    lo,
                    hi,
                )
                if accepted[j]
            ])


            if len(ids) < 3:
                continue


            weights = (
                score[ids] ** 2
            )

            cx = np.sum(
                weights
                * np.cos(
                    raw_angle[ids]
                )
            )

            cy = np.sum(
                weights
                * np.sin(
                    raw_angle[ids]
                )
            )


        reference = np.arctan2(
            cy,
            cx,
        )


        error_deg = abs(
            np.rad2deg(
                angle_diff(
                    raw_angle[i],
                    reference,
                )
            )
        )


        if (
            error_deg
            > LOCAL_OUTLIER_DEG
        ):

            accepted[i] = False
            reject_local[i] = True


    # ========================================================
    # 根据有效点建立连续轨迹
    # ========================================================

    def build_trajectory(mask):

        valid_idx = np.where(
            mask
        )[0]

        if len(valid_idx) < 2:

            raise RuntimeError(
                f"{side}: "
                "有效手部数据太少"
            )


        # 这里只允许正常2π跨界展开。
        #
        # 不把180°翻转当成真实动作，
        # 因为前面已经把异常帧丢掉。
        unwrapped = np.unwrap(
            raw_angle[
                valid_idx
            ]
        )


        trajectory = np.interp(
            np.arange(N),
            valid_idx,
            unwrapped,
        )


        trajectory = (
            triangular_smooth(
                trajectory,
                SMOOTH_RADIUS,
                passes=2,
            )
        )


        return trajectory


    trajectory = build_trajectory(
        accepted
    )


    # ========================================================
    # 第三层：整条轨迹再次判断离群点
    #
    # 防止连续2~3帧错误方向绕过局部多数判断。
    # ========================================================

    reject_trajectory = np.zeros(
        N,
        dtype=bool,
    )


    for _ in range(3):

        newly_rejected = (
            np.zeros(
                N,
                dtype=bool,
            )
        )


        for i in np.where(
            accepted
        )[0]:

            error_deg = abs(
                np.rad2deg(
                    angle_diff(
                        raw_angle[i],
                        trajectory[i],
                    )
                )
            )


            if (
                error_deg
                > TRAJECTORY_OUTLIER_DEG
            ):

                accepted[i] = False

                newly_rejected[i] = True

                reject_trajectory[i] = True


        if not np.any(
            newly_rejected
        ):
            break


        trajectory = (
            build_trajectory(
                accepted
            )
        )


    # ========================================================
    # 最终重新建立一次连续轨迹
    # ========================================================

    trajectory = build_trajectory(
        accepted
    )


    # ========================================================
    # 第四层：最终方向变化速度限制
    # ========================================================

    trajectory = (
        final_speed_limit(
            trajectory,
            np.deg2rad(
                MAX_STEP_DEG
            ),
        )
    )


    direction = np.stack(
        [
            np.cos(
                trajectory
            ),

            np.sin(
                trajectory
            ),
        ],
        axis=1,
    )


    rejected = (
        reject_quality
        | reject_local
        | reject_trajectory
    )


    result[side] = {
        "direction":
            direction,

        "angle":
            trajectory,

        "pose_wrist":
            pose_wrist,

        "raw_angle":
            raw_angle,

        "raw":
            raw_detected,

        "accepted":
            accepted,

        "rejected":
            rejected,

        "reject_quality":
            reject_quality,

        "reject_local":
            reject_local,

        "reject_trajectory":
            reject_trajectory,
    }


    print()
    print(
        f"[{side}]"
    )

    print(
        "raw detected       :",
        int(
            raw_detected.sum()
        ),
        f"/ {N}",
    )

    print(
        "quality rejected   :",
        int(
            reject_quality.sum()
        ),
    )

    print(
        "local jumps rejected:",
        int(
            reject_local.sum()
        ),
    )

    print(
        "trajectory rejected:",
        int(
            reject_trajectory.sum()
        ),
    )

    print(
        "final anchors      :",
        int(
            accepted.sum()
        ),
    )

    print(
        "interpolated frames:",
        int(
            N
            - accepted.sum()
        ),
    )


# ============================================================
# 输出最终 CSV
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

        "left_raw_detected",
        "right_raw_detected",

        "left_accepted",
        "right_accepted",

        "left_rejected",
        "right_rejected",
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
                ]["rejected"][i]
            ),

            int(
                result[
                    "RIGHT"
                ]["rejected"][i]
            ),
        ])


print()
print(
    "filtered CSV:",
    OUT_CSV,
)


# ============================================================
# 可视化整个661帧
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


for i in range(N):

    ok, frame = cap.read()

    if not ok:
        break


    cv2.putText(
        frame,

        (
            f"FULL HAND FILTER "
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

        start = result[
            side
        ]["pose_wrist"][i]

        direction = result[
            side
        ]["direction"][i]


        if not np.isfinite(
            start
        ).all():

            continue


        # ----------------------------------------
        # 原始识别方向：细蓝线
        # ----------------------------------------

        if result[
            side
        ]["raw"][i]:

            raw_angle = result[
                side
            ]["raw_angle"][i]

            raw_dir = np.array([
                np.cos(
                    raw_angle
                ),
                np.sin(
                    raw_angle
                ),
            ])

            raw_end = (
                start
                + raw_dir * 100
            )


            cv2.arrowedLine(
                frame,

                tuple(
                    start.astype(int)
                ),

                tuple(
                    raw_end.astype(int)
                ),

                (
                    255,
                    255,
                    0,
                ),

                2,

                tipLength=0.20,
            )


        # ----------------------------------------
        # 最终方向：粗紫色
        # ----------------------------------------

        end = (
            start
            + direction * 170
        )


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

            tipLength=0.22,
        )


        # ----------------------------------------
        # 状态颜色
        #
        # 绿色：采用了真实检测
        # 红色：检测到但因为异常被拒绝
        # 黄色：遮挡/漏检，用轨迹补齐
        # ----------------------------------------

        if result[
            side
        ]["accepted"][i]:

            circle_color = (
                0,
                255,
                0,
            )

        elif result[
            side
        ]["rejected"][i]:

            circle_color = (
                0,
                0,
                255,
            )

        else:

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
                int(start[0]) + 8,
                int(start[1]) - 8,
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


# ============================================================
# H264
# ============================================================

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
