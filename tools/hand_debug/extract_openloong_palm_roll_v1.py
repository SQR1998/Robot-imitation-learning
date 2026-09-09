from pathlib import Path
import csv
import subprocess

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# OpenLoong Palm Roll Extractor V1
#
# Goal:
#   Estimate the HUMAN palm roll around the hand/finger long axis
#   from the same RGB video used by the V6 hand-direction pipeline.
#
# Important:
#   This script does NOT modify robot qpos.
#   It is a diagnostic / target-generation stage only.
#
# Calibration:
#   The user has visually verified that both palms face the ground
#   during 15~18 s. We use that interval ONLY to resolve the
#   +/- ambiguity of the palm-plane normal.
#   The rest of the video remains data-driven.
# ============================================================

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
    "output/openloong_palm_roll_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "palm_roll_v1.csv"
TMP_VIDEO = OUT_DIR / "palm_roll_v1_tmp.mp4"
OUT_VIDEO = OUT_DIR / "palm_roll_v1_h264.mp4"


LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

ROI_SIZE = 512

# Sign-calibration interval:
# user-verified "palm faces ground".
CALIB_START_SEC = 15.0
CALIB_END_SEC = 18.0

# Continuity / smoothing.
ANCHOR_STEP_DEG_PER_FRAME = 12.0
ANCHOR_MARGIN_DEG = 15.0
MAX_ANCHOR_JUMP_DEG = 90.0

SMOOTH_RADIUS = 4
FINAL_MAX_STEP_DEG = 8.0


# ============================================================
# Helpers
# ============================================================

def unit(v, fallback=None):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))

    if n < 1e-10:
        if fallback is None:
            return np.zeros_like(v)
        return unit(fallback)

    return v / n


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def signed_angle_deg(a, b, axis):
    """
    Signed angle from a -> b around `axis`.
    All vectors are expected to be approximately perpendicular to axis.
    """
    a = unit(a)
    b = unit(b)
    axis = unit(axis)

    c = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    s = float(
        np.dot(
            axis,
            np.cross(a, b),
        )
    )

    return float(
        np.rad2deg(
            np.arctan2(s, c)
        )
    )


def down_reference_perp(finger):
    """
    Build the "palm-down reference normal" in the image/camera coordinate
    system, while respecting that palm normal must be perpendicular to
    the hand/finger long axis.

    MediaPipe image coordinates:
        +x = image right
        +y = image down

    We therefore use [0,+1,0] as image-down.
    """
    finger = unit(
        finger,
        [1.0, 0.0, 0.0],
    )

    image_down = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    ref = (
        image_down
        - np.dot(image_down, finger) * finger
    )

    if np.linalg.norm(ref) < 1e-6:
        # Degenerate when the hand points almost straight down in image.
        camera_forward = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

        ref = (
            camera_forward
            - np.dot(camera_forward, finger) * finger
        )

    return unit(
        ref,
        [0.0, 0.0, 1.0],
    )


def triangular_smooth(values, radius, passes=2):
    values = np.asarray(values, dtype=np.float64)

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


def speed_limit_scalar(values, max_step):
    out = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    for i in range(1, len(out)):
        d = out[i] - out[i - 1]

        d = np.clip(
            d,
            -max_step,
            +max_step,
        )

        out[i] = out[i - 1] + d

    for i in range(
        len(out) - 2,
        -1,
        -1,
    ):
        d = out[i] - out[i + 1]

        d = np.clip(
            d,
            -max_step,
            +max_step,
        )

        out[i] = out[i + 1] + d

    return out


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

    if length < 8.0:
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
            100.0,
            300.0,
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

    crop = cv2.resize(
        crop,
        (ROI_SIZE, ROI_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    return {
        "crop": crop,
        "box": (x0, y0, x1, y1),
    }


def map_points_2d(
    landmarks,
    box,
    flipped,
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
                    y0 + float(lm.y) * bh,
                ],
                dtype=np.float64,
            )
        )

    return pts


def map_points_3d(
    landmarks,
    flipped,
):
    """
    Hand-landmark relative directions in crop-camera coordinates.

    Crop is square (512x512), therefore x/y scales are compatible enough
    for orientation estimation. MediaPipe z uses a comparable normalized
    hand scale.

    For a horizontally flipped detector input, restore x orientation.
    Translation does not matter because only vector differences are used.
    """
    pts = []

    for lm in landmarks:
        x = float(lm.x)

        if flipped:
            x = 1.0 - x

        pts.append(
            np.array(
                [
                    x,
                    float(lm.y),
                    float(lm.z),
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
        pts2 = map_points_2d(
            landmarks,
            crop_info["box"],
            try_flip,
        )

        pts3 = map_points_3d(
            landmarks,
            try_flip,
        )

        wrist_dist = float(
            np.linalg.norm(
                pts2[0]
                - pose_wrist
            )
        )

        score = 0.0

        if i < len(result.handedness):
            score = float(
                result.handedness[i][0].score
            )

        hand_span = float(
            np.linalg.norm(
                pts2[9]
                - pts2[0]
            )
        )

        cost = (
            wrist_dist
            - 25.0 * score
            - 0.05 * hand_span
        )

        candidate = {
            "pts2": pts2,
            "pts3": pts3,
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

    candidates = []

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

        for try_flip in (
            False,
            True,
        ):
            cand = detect_candidate(
                detector,
                crop_info,
                wrist,
                try_flip,
            )

            if cand is not None:
                candidates.append(
                    cand
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["cost"]
    )

    return candidates[0]


def compute_hand_geometry(
    candidate,
):
    pts3 = candidate["pts3"]

    wrist = pts3[0]

    # Same broad "finger-long-axis" idea as the V6 detector:
    # use the center of four fingertip landmarks.
    tip_center = np.mean(
        np.stack([
            pts3[8],
            pts3[12],
            pts3[16],
            pts3[20],
        ]),
        axis=0,
    )

    finger = unit(
        tip_center - wrist,
        [1.0, 0.0, 0.0],
    )

    # Across-palm axis: pinky MCP -> index MCP.
    across = unit(
        pts3[5] - pts3[17],
        [1.0, 0.0, 0.0],
    )

    normal = unit(
        np.cross(
            finger,
            across,
        ),
        [0.0, 0.0, 1.0],
    )

    down_ref = down_reference_perp(
        finger
    )

    raw_roll = signed_angle_deg(
        down_ref,
        normal,
        finger,
    )

    return {
        "finger": finger,
        "normal": normal,
        "down_ref": down_ref,
        "raw_roll": raw_roll,
    }


# ============================================================
# Video setup
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

frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

# The contest video is expected to be 661 frames.
N = frame_count

print("=" * 72)
print("OpenLoong Palm Roll Extractor V1")
print("=" * 72)
print("video :", VIDEO)
print("frames:", N)
print("fps   :", fps)
print(
    "calibration interval:",
    CALIB_START_SEC,
    "~",
    CALIB_END_SEC,
    "sec",
)


# ============================================================
# Storage
# ============================================================

sides = (
    "LEFT",
    "RIGHT",
)

detected = {
    side: np.zeros(
        N,
        dtype=bool,
    )
    for side in sides
}

score = {
    side: np.full(
        N,
        np.nan,
        dtype=np.float64,
    )
    for side in sides
}

raw_roll = {
    side: np.full(
        N,
        np.nan,
        dtype=np.float64,
    )
    for side in sides
}

raw_normal = {
    side: np.full(
        (N, 3),
        np.nan,
        dtype=np.float64,
    )
    for side in sides
}

raw_finger = {
    side: np.full(
        (N, 3),
        np.nan,
        dtype=np.float64,
    )
    for side in sides
}

wrist_2d = {
    side: np.full(
        (N, 2),
        np.nan,
        dtype=np.float64,
    )
    for side in sides
}


# ============================================================
# MediaPipe setup
# ============================================================

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


# ============================================================
# First pass: raw palm geometry
# ============================================================

with (
    PoseLandmarker.create_from_options(
        pose_options
    ) as pose_detector,

    HandLandmarker.create_from_options(
        hand_options
    ) as hand_detector,
):

    for frame_id in range(N):
        ok, frame = cap.read()

        if not ok:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(
            round(
                frame_id / fps * 1000.0
            )
        )

        pose_result = (
            pose_detector.detect_for_video(
                mp_frame,
                timestamp_ms,
            )
        )

        if not pose_result.pose_landmarks:
            continue

        pose = (
            pose_result.pose_landmarks[0]
        )

        def pix(idx):
            lm = pose[idx]

            return np.array(
                [
                    lm.x * width,
                    lm.y * height,
                ],
                dtype=np.float64,
            )

        side_info = [
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
        ) in side_info:

            elbow = pix(
                elbow_idx
            )

            p_wrist = pix(
                wrist_idx
            )

            cand = detect_hand_multiroi(
                hand_detector,
                frame,
                elbow,
                p_wrist,
            )

            if cand is None:
                continue

            # Same rejection idea as the existing full-video hand detector.
            hand_wrist_2d = cand["pts2"][0]

            wrist_error = float(
                np.linalg.norm(
                    hand_wrist_2d
                    - p_wrist
                )
            )

            forearm_len = float(
                np.linalg.norm(
                    p_wrist
                    - elbow
                )
            )

            max_error = max(
                55.0,
                forearm_len * 0.75,
            )

            if wrist_error > max_error:
                continue

            geom = compute_hand_geometry(
                cand
            )

            detected[side][frame_id] = True

            score[side][frame_id] = (
                cand["score"]
            )

            raw_roll[side][frame_id] = (
                geom["raw_roll"]
            )

            raw_normal[side][frame_id] = (
                geom["normal"]
            )

            raw_finger[side][frame_id] = (
                geom["finger"]
            )

            wrist_2d[side][frame_id] = (
                p_wrist
            )

        if (
            frame_id == 0
            or (frame_id + 1) % 100 == 0
            or frame_id == N - 1
        ):
            print(
                f"detected frame "
                f"{frame_id + 1}/{N}"
            )


cap.release()


# ============================================================
# Resolve palm-normal sign using 15~18 s only
# ============================================================

calib_lo = max(
    0,
    int(
        round(
            CALIB_START_SEC * fps
        )
    ),
)

calib_hi = min(
    N,
    int(
        round(
            CALIB_END_SEC * fps
        )
    ) + 1,
)


normal_sign = {}

signed_raw_roll = {}

signed_raw_normal = {}


for side in sides:
    valid = np.where(
        detected[side]
    )[0]

    assert len(valid) >= 2, (
        side,
        len(valid),
    )

    calib_valid = np.where(
        detected[side][
            calib_lo:calib_hi
        ]
    )[0] + calib_lo

    assert len(calib_valid) >= 2, (
        "not enough calibration detections",
        side,
        len(calib_valid),
    )

    candidate_cost = {}

    for sign in (
        +1.0,
        -1.0,
    ):
        vals = []

        for i in calib_valid:
            finger = raw_finger[
                side
            ][i]

            n = (
                sign
                * raw_normal[
                    side
                ][i]
            )

            ref = down_reference_perp(
                finger
            )

            angle = signed_angle_deg(
                ref,
                n,
                finger,
            )

            vals.append(
                abs(
                    wrap_deg(angle)
                )
            )

        candidate_cost[
            int(sign)
        ] = float(
            np.median(vals)
        )

    chosen = min(
        candidate_cost,
        key=candidate_cost.get,
    )

    normal_sign[side] = float(
        chosen
    )

    print()
    print(
        f"[{side}] detections:",
        int(
            detected[side].sum()
        ),
        "/",
        N,
    )

    print(
        f"[{side}] sign calibration:",
        candidate_cost,
        "-> chosen",
        chosen,
    )

    signed_raw_normal[
        side
    ] = (
        raw_normal[side]
        * float(chosen)
    )

    arr = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )

    for i in valid:
        finger = raw_finger[
            side
        ][i]

        ref = down_reference_perp(
            finger
        )

        arr[i] = signed_angle_deg(
            ref,
            signed_raw_normal[
                side
            ][i],
            finger,
        )

    signed_raw_roll[
        side
    ] = arr


# ============================================================
# Robust anchor continuity + interpolation
# ============================================================

filtered_roll = {}
anchor_mask = {}
reject_jump = {}


for side in sides:
    candidate_idx = np.where(
        np.isfinite(
            signed_raw_roll[
                side
            ]
        )
    )[0]

    keep = np.zeros(
        N,
        dtype=bool,
    )

    reject = np.zeros(
        N,
        dtype=bool,
    )

    unwrapped_anchor_idx = []
    unwrapped_anchor_val = []

    last_idx = None
    last_wrapped = None
    last_unwrapped = None

    for i in candidate_idx:
        value = float(
            wrap_deg(
                signed_raw_roll[
                    side
                ][i]
            )
        )

        if last_idx is None:
            keep[i] = True

            last_idx = i
            last_wrapped = value
            last_unwrapped = value

            unwrapped_anchor_idx.append(
                i
            )

            unwrapped_anchor_val.append(
                value
            )

            continue

        gap = i - last_idx

        allowed = min(
            MAX_ANCHOR_JUMP_DEG,
            (
                ANCHOR_STEP_DEG_PER_FRAME
                * gap
                + ANCHOR_MARGIN_DEG
            ),
        )

        delta = float(
            wrap_deg(
                value
                - last_wrapped
            )
        )

        if abs(delta) > allowed:
            reject[i] = True
            continue

        current_unwrapped = (
            last_unwrapped
            + delta
        )

        keep[i] = True

        unwrapped_anchor_idx.append(
            i
        )

        unwrapped_anchor_val.append(
            current_unwrapped
        )

        last_idx = i
        last_wrapped = value
        last_unwrapped = current_unwrapped

    unwrapped_anchor_idx = np.asarray(
        unwrapped_anchor_idx,
        dtype=int,
    )

    unwrapped_anchor_val = np.asarray(
        unwrapped_anchor_val,
        dtype=np.float64,
    )

    assert len(
        unwrapped_anchor_idx
    ) >= 2

    all_idx = np.arange(
        N
    )

    roll = np.interp(
        all_idx,
        unwrapped_anchor_idx,
        unwrapped_anchor_val,
    )

    roll = triangular_smooth(
        roll,
        SMOOTH_RADIUS,
        passes=2,
    )

    roll = speed_limit_scalar(
        roll,
        FINAL_MAX_STEP_DEG,
    )

    filtered_roll[
        side
    ] = roll

    anchor_mask[
        side
    ] = keep

    reject_jump[
        side
    ] = reject

    print()
    print(
        f"[{side}] kept anchors:",
        int(
            keep.sum()
        ),
    )

    print(
        f"[{side}] jump rejected:",
        int(
            reject.sum()
        ),
    )

    print(
        f"[{side}] filtered roll range:",
        f"{roll.min():.1f}",
        "~",
        f"{roll.max():.1f}",
        "deg",
    )


# ============================================================
# Useful segment statistics
# ============================================================

def segment_report(
    name,
    t0,
    t1,
):
    lo = max(
        0,
        int(round(t0 * fps)),
    )

    hi = min(
        N,
        int(round(t1 * fps)) + 1,
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    for side in sides:
        x = filtered_roll[
            side
        ][lo:hi]

        print(
            f"{side:5s} palm roll: "
            f"mean={np.mean(x):7.2f} "
            f"min={np.min(x):7.2f} "
            f"max={np.max(x):7.2f} deg"
        )


segment_report(
    "7~9 sec",
    7.0,
    9.0,
)

segment_report(
    "15~18 sec calibration check",
    15.0,
    18.0,
)


# ============================================================
# Save CSV
# ============================================================

with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "frame",
        "time_sec",

        "left_palm_roll_deg",
        "right_palm_roll_deg",

        "left_raw_roll_deg",
        "right_raw_roll_deg",

        "left_detected",
        "right_detected",

        "left_anchor",
        "right_anchor",

        "left_jump_rejected",
        "right_jump_rejected",

        "left_score",
        "right_score",

        "left_normal_sign",
        "right_normal_sign",
    ])

    for i in range(N):
        writer.writerow([
            i,
            i / fps,

            filtered_roll[
                "LEFT"
            ][i],

            filtered_roll[
                "RIGHT"
            ][i],

            (
                signed_raw_roll[
                    "LEFT"
                ][i]
                if np.isfinite(
                    signed_raw_roll[
                        "LEFT"
                    ][i]
                )
                else ""
            ),

            (
                signed_raw_roll[
                    "RIGHT"
                ][i]
                if np.isfinite(
                    signed_raw_roll[
                        "RIGHT"
                    ][i]
                )
                else ""
            ),

            int(
                detected[
                    "LEFT"
                ][i]
            ),

            int(
                detected[
                    "RIGHT"
                ][i]
            ),

            int(
                anchor_mask[
                    "LEFT"
                ][i]
            ),

            int(
                anchor_mask[
                    "RIGHT"
                ][i]
            ),

            int(
                reject_jump[
                    "LEFT"
                ][i]
            ),

            int(
                reject_jump[
                    "RIGHT"
                ][i]
            ),

            (
                score[
                    "LEFT"
                ][i]
                if np.isfinite(
                    score[
                        "LEFT"
                    ][i]
                )
                else ""
            ),

            (
                score[
                    "RIGHT"
                ][i]
                if np.isfinite(
                    score[
                        "RIGHT"
                    ][i]
                )
                else ""
            ),

            normal_sign[
                "LEFT"
            ],

            normal_sign[
                "RIGHT"
            ],
        ])


# ============================================================
# Visualization
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

assert cap.isOpened()
assert writer.isOpened()


for i in range(N):
    ok, frame = cap.read()

    if not ok:
        break

    cv2.putText(
        frame,
        (
            "PALM ROLL V1 "
            f"{i/fps:.2f}s"
        ),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    text_y = 78

    for side in sides:
        roll = filtered_roll[
            side
        ][i]

        cv2.putText(
            frame,
            (
                f"{side}: "
                f"roll={roll:+6.1f} deg "
                f"det={int(detected[side][i])}"
            ),
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (
                255,
                200,
                0,
            )
            if side == "LEFT"
            else (
                0,
                200,
                255,
            ),
            2,
        )

        text_y += 32

        if (
            detected[side][i]
            and np.isfinite(
                wrist_2d[
                    side
                ][i, 0]
            )
        ):
            w = wrist_2d[
                side
            ][i]

            n = (
                signed_raw_normal[
                    side
                ][i]
            )

            # Draw the image-plane projection of the calibrated raw normal.
            xy = np.array(
                [
                    n[0],
                    n[1],
                ],
                dtype=np.float64,
            )

            norm_xy = float(
                np.linalg.norm(xy)
            )

            if norm_xy > 1e-5:
                xy /= norm_xy

                end = (
                    w
                    + xy * 90.0
                )

                cv2.arrowedLine(
                    frame,
                    tuple(
                        np.round(w).astype(int)
                    ),
                    tuple(
                        np.round(end).astype(int)
                    ),
                    (
                        255,
                        120,
                        0,
                    )
                    if side == "LEFT"
                    else (
                        0,
                        120,
                        255,
                    ),
                    4,
                    tipLength=0.22,
                )

    writer.write(
        frame
    )


cap.release()
writer.release()


# ============================================================
# H264 encode if ffmpeg exists
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
print("CSV  :", OUT_CSV)
print("VIDEO:", OUT_VIDEO)
print()
print(
    "Interpretation: roll ~= 0 deg means palm normal is closest "
    "to image/ground-down after sign calibration."
)
