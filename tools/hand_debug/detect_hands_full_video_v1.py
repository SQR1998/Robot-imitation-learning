from pathlib import Path
import csv
import subprocess

import cv2
import mediapipe as mp
import numpy as np


VIDEO = Path(
    "dataset/openloong_master1/RGB_video_dataset.mp4"
)

POSE_MODEL = Path(
    "checkpoints/mediapipe/pose_landmarker_heavy.task"
)

HAND_MODEL = Path(
    "checkpoints/mediapipe/hand_landmarker.task"
)

OUT_DIR = Path(
    "output/openloong_hand_full_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TMP_VIDEO = OUT_DIR / "hand_full_tmp.mp4"
OUT_VIDEO = OUT_DIR / "hand_full_h264.mp4"
OUT_CSV = OUT_DIR / "hand_full.csv"

START_SEC = 0.0
END_SEC = 22.0

ROI_SIZE = 512


LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17),
]


def pixel(lm, w, h):
    return np.array(
        [
            lm.x * w,
            lm.y * h,
        ],
        dtype=np.float64,
    )


def make_crop(
    frame,
    elbow,
    wrist,
    center_ratio,
    size_ratio,
):
    h, w = frame.shape[:2]

    forearm = wrist - elbow

    length = float(
        np.linalg.norm(forearm)
    )

    if length < 8:
        return None

    direction = forearm / length

    center = (
        wrist
        + direction
        * length
        * center_ratio
    )

    size = float(
        np.clip(
            length * size_ratio,
            100,
            300,
        )
    )

    half = size / 2.0

    x0 = int(
        max(
            0,
            np.floor(center[0] - half),
        )
    )

    y0 = int(
        max(
            0,
            np.floor(center[1] - half),
        )
    )

    x1 = int(
        min(
            w,
            np.ceil(center[0] + half),
        )
    )

    y1 = int(
        min(
            h,
            np.ceil(center[1] + half),
        )
    )

    if x1 - x0 < 30 or y1 - y0 < 30:
        return None

    crop = frame[
        y0:y1,
        x0:x1
    ]

    resized = cv2.resize(
        crop,
        (ROI_SIZE, ROI_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    return {
        "crop": resized,
        "box": (
            x0,
            y0,
            x1,
            y1,
        ),
    }


def map_points(
    landmarks,
    box,
    flipped=False,
):
    x0, y0, x1, y1 = box

    bw = x1 - x0
    bh = y1 - y0

    pts = []

    for lm in landmarks:

        nx = float(lm.x)

        if flipped:
            nx = 1.0 - nx

        pts.append(
            np.array(
                [
                    x0 + nx * bw,
                    y0 + lm.y * bh,
                ],
                dtype=np.float64,
            )
        )

    return pts


def detect_candidate(
    detector,
    crop_info,
    pose_wrist,
    try_flip,
):
    crop = crop_info["crop"]

    if try_flip:
        image = cv2.flip(
            crop,
            1,
        )
    else:
        image = crop

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = detector.detect(
        mp_image
    )

    if not result.hand_landmarks:
        return None

    best = None

    for i, landmarks in enumerate(
        result.hand_landmarks
    ):

        pts = map_points(
            landmarks,
            crop_info["box"],
            flipped=try_flip,
        )

        wrist_dist = float(
            np.linalg.norm(
                pts[0]
                - pose_wrist
            )
        )

        score = 0.0

        if i < len(result.handedness):
            score = float(
                result.handedness[
                    i
                ][0].score
            )

        # 手骨架在原图里的尺寸
        hand_span = float(
            np.linalg.norm(
                pts[9] - pts[0]
            )
        )

        # wrist 越接近 Pose wrist 越好；
        # 置信度高、手骨架尺寸合理也加分
        cost = (
            wrist_dist
            - 25.0 * score
            - 0.05 * hand_span
        )

        candidate = {
            "pts": pts,
            "score": score,
            "cost": cost,
            "box": crop_info["box"],
            "flipped": try_flip,
        }

        if (
            best is None
            or candidate["cost"]
            < best["cost"]
        ):
            best = candidate

    return best


def detect_hand_multiroi(
    detector,
    frame,
    elbow,
    wrist,
):
    # 越靠前越优先。
    #
    # 小框：手在输入图里变得更大
    # 大框：Pose wrist 偏一点时仍能兜底
    proposals = [
        (0.20, 1.25),
        (0.35, 1.40),
        (0.50, 1.55),

        (0.20, 1.70),
        (0.40, 1.85),
        (0.60, 2.00),

        (0.20, 2.20),
        (0.45, 2.40),
        (0.70, 2.60),
    ]

    all_candidates = []

    for center_ratio, size_ratio in proposals:

        crop_info = make_crop(
            frame,
            elbow,
            wrist,
            center_ratio,
            size_ratio,
        )

        if crop_info is None:
            continue

        # 原图
        result = detect_candidate(
            detector,
            crop_info,
            wrist,
            False,
        )

        if result is not None:
            all_candidates.append(
                result
            )

        # 水平翻转再检测一次
        result_flip = detect_candidate(
            detector,
            crop_info,
            wrist,
            True,
        )

        if result_flip is not None:
            all_candidates.append(
                result_flip
            )

    if not all_candidates:
        return None

    all_candidates.sort(
        key=lambda x: x["cost"]
    )

    return all_candidates[0]


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

start_frame = int(
    round(
        START_SEC * fps
    )
)

end_frame = int(
    round(
        END_SEC * fps
    )
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame,
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


csv_f = open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.writer(
    csv_f
)

csv_writer.writerow([
    "frame",
    "time_sec",
    "side",
    "detected",

    "pose_wrist_x",
    "pose_wrist_y",

    "hand_wrist_x",
    "hand_wrist_y",

    "tip_center_x",
    "tip_center_y",

    "finger_dir_x",
    "finger_dir_y",

    "score",
    "flipped",
])


BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandOptions = (
    mp.tasks.vision.HandLandmarkerOptions
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


hand_options = HandOptions(
    base_options=BaseOptions(
        model_asset_path=str(
            HAND_MODEL
        )
    ),

    running_mode=RunningMode.IMAGE,

    num_hands=2,

    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20,
)


stats = {
    "frames": 0,
    "pose": 0,
    "left": 0,
    "right": 0,
    "both": 0,
}


with (
    PoseLandmarker.create_from_options(
        pose_options
    ) as pose_detector,

    HandLandmarker.create_from_options(
        hand_options
    ) as hand_detector
):

    frame_id = start_frame

    while frame_id <= end_frame:

        ok, frame = cap.read()

        if not ok:
            break

        stats["frames"] += 1

        timestamp_ms = int(
            round(
                frame_id
                / fps
                * 1000.0
            )
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        pose_result = (
            pose_detector.detect_for_video(
                mp_frame,
                timestamp_ms,
            )
        )

        detected = {
            "LEFT": False,
            "RIGHT": False,
        }

        cv2.putText(
            frame,
            (
                f"time="
                f"{frame_id/fps:.2f}s"
            ),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )


        if pose_result.pose_landmarks:

            stats["pose"] += 1

            pose = (
                pose_result
                .pose_landmarks[0]
            )

            sides = [
                (
                    "LEFT",
                    LEFT_ELBOW,
                    LEFT_WRIST,
                ),

                (
                    "RIGHT",
                    RIGHT_ELBOW,
                    RIGHT_WRIST,
                ),
            ]

            for (
                side,
                elbow_idx,
                wrist_idx,
            ) in sides:

                elbow = pixel(
                    pose[elbow_idx],
                    width,
                    height,
                )

                pose_wrist = pixel(
                    pose[wrist_idx],
                    width,
                    height,
                )

                best = (
                    detect_hand_multiroi(
                        hand_detector,
                        frame,
                        elbow,
                        pose_wrist,
                    )
                )

                if best is None:

                    csv_writer.writerow([
                        frame_id,
                        frame_id / fps,
                        side,
                        0,

                        pose_wrist[0],
                        pose_wrist[1],

                        "",
                        "",

                        "",
                        "",

                        "",
                        "",

                        "",
                        "",
                    ])

                    continue


                pts = best["pts"]

                # 过滤明显错误识别：
                # Hand wrist 必须离 Pose wrist 足够近
                wrist_error = float(
                    np.linalg.norm(
                        pts[0]
                        - pose_wrist
                    )
                )

                forearm_len = float(
                    np.linalg.norm(
                        pose_wrist
                        - elbow
                    )
                )

                max_error = max(
                    55.0,
                    forearm_len * 0.75,
                )

                if wrist_error > max_error:

                    csv_writer.writerow([
                        frame_id,
                        frame_id / fps,
                        side,
                        0,

                        pose_wrist[0],
                        pose_wrist[1],

                        "",
                        "",

                        "",
                        "",

                        "",
                        "",

                        best["score"],
                        int(
                            best["flipped"]
                        ),
                    ])

                    continue


                detected[side] = True


                for a, b in (
                    HAND_CONNECTIONS
                ):

                    cv2.line(
                        frame,

                        tuple(
                            pts[a]
                            .astype(int)
                        ),

                        tuple(
                            pts[b]
                            .astype(int)
                        ),

                        (
                            0,
                            255,
                            0,
                        ),

                        2,
                    )


                for idx, p in enumerate(
                    pts
                ):

                    radius = (
                        5
                        if idx in (
                            0,
                            4,
                            8,
                            12,
                            16,
                            20,
                        )
                        else 3
                    )

                    cv2.circle(
                        frame,

                        tuple(
                            p.astype(int)
                        ),

                        radius,

                        (
                            0,
                            0,
                            255,
                        ),

                        -1,
                    )


                hand_wrist = pts[0]

                tip_center = np.mean(
                    [
                        pts[8],
                        pts[12],
                        pts[16],
                    ],
                    axis=0,
                )

                direction = (
                    tip_center
                    - hand_wrist
                )

                norm = float(
                    np.linalg.norm(
                        direction
                    )
                )

                if norm > 1e-6:
                    direction /= norm
                else:
                    direction[:] = 0


                arrow_end = (
                    hand_wrist
                    + direction * 150
                )

                # 紫色箭头
                cv2.arrowedLine(
                    frame,

                    tuple(
                        hand_wrist
                        .astype(int)
                    ),

                    tuple(
                        arrow_end
                        .astype(int)
                    ),

                    (
                        255,
                        0,
                        255,
                    ),

                    6,

                    tipLength=0.25,
                )


                # 蓝色小圆 = Pose wrist
                cv2.circle(
                    frame,

                    tuple(
                        pose_wrist
                        .astype(int)
                    ),

                    8,

                    (
                        255,
                        0,
                        0,
                    ),

                    2,
                )


                cv2.putText(
                    frame,

                    (
                        f"{side} "
                        f"{best['score']:.2f}"
                        + (
                            " FLIP"
                            if best["flipped"]
                            else ""
                        )
                    ),

                    (
                        int(
                            hand_wrist[0]
                        ) + 10,

                        int(
                            hand_wrist[1]
                        ) - 10,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (
                        255,
                        255,
                        0,
                    ),

                    2,
                )


                csv_writer.writerow([
                    frame_id,
                    frame_id / fps,

                    side,
                    1,

                    pose_wrist[0],
                    pose_wrist[1],

                    hand_wrist[0],
                    hand_wrist[1],

                    tip_center[0],
                    tip_center[1],

                    direction[0],
                    direction[1],

                    best["score"],

                    int(
                        best["flipped"]
                    ),
                ])


        if detected["LEFT"]:
            stats["left"] += 1

        if detected["RIGHT"]:
            stats["right"] += 1

        if (
            detected["LEFT"]
            and detected["RIGHT"]
        ):
            stats["both"] += 1


        writer.write(frame)


        if (
            frame_id == start_frame
            or frame_id % 15 == 0
            or frame_id == end_frame
        ):

            print(
                f"frame={frame_id} "
                f"time={frame_id/fps:.2f}s "
                f"L={int(detected['LEFT'])} "
                f"R={int(detected['RIGHT'])}"
            )


        frame_id += 1


writer.release()
cap.release()
csv_f.close()


frames = stats["frames"]

print()
print("=" * 70)
print(
    "Multi-ROI Hand Detection Result"
)
print("=" * 70)

print(
    "processed frames:",
    frames,
)

print(
    "pose frames:",
    stats["pose"],
)

print(
    "left hand:",
    stats["left"],
    f"({100*stats['left']/frames:.1f}%)",
)

print(
    "right hand:",
    stats["right"],
    f"({100*stats['right']/frames:.1f}%)",
)

print(
    "both hands:",
    stats["both"],
    f"({100*stats['both']/frames:.1f}%)",
)

print(
    "csv:",
    OUT_CSV,
)

print("=" * 70)


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
    "H264 video:",
    OUT_VIDEO,
)
print("=" * 70)
