from pathlib import Path
import csv
import subprocess

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# OpenLoong Palm Roll V2
#
# Changes from V1:
#   1) Use HandLandmarker hand_world_landmarks to estimate palm plane.
#   2) Use PoseLandmarker pose_world_landmarks to estimate a fixed camera
#      "ground-down" direction from the known 15~18 s palm-down segment.
#   3) Use the already-validated V6 hand-direction anchors as preferred
#      trust anchors.
#   4) Resolve normal sign + zero offset from the known 15~18 s segment.
#
# This script ONLY estimates human palm roll.
# It does NOT modify robot qpos.
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

V6_CSV = Path(
    "output/openloong_hand_full_filter_v6/"
    "hand_full_filtered_v6.csv"
)

OUT_DIR = Path(
    "output/openloong_palm_roll_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "palm_roll_v2.csv"
TMP_VIDEO = OUT_DIR / "palm_roll_v2_tmp.mp4"
OUT_VIDEO = OUT_DIR / "palm_roll_v2_h264.mp4"


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

ROI_SIZE = 512

CALIB_START_SEC = 15.0
CALIB_END_SEC = 18.0

# Continuity filter
ANCHOR_STEP_DEG_PER_FRAME = 10.0
ANCHOR_MARGIN_DEG = 15.0
MAX_ANCHOR_JUMP_DEG = 80.0

SMOOTH_RADIUS = 4
FINAL_MAX_STEP_DEG = 6.0


# ============================================================
# Math helpers
# ============================================================

def unit(v, fallback=None):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-10:
        if fallback is None:
            return np.zeros_like(v)
        return unit(fallback)

    return v / n


def wrap_deg(x):
    return (
        np.asarray(
            x,
            dtype=np.float64,
        )
        + 180.0
    ) % 360.0 - 180.0


def signed_angle_deg(a, b, axis):
    """
    Signed angle a -> b around axis.
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
            np.arctan2(
                s,
                c,
            )
        )
    )


def project_perp(v, axis, fallback):
    axis = unit(axis)

    out = (
        np.asarray(v, dtype=np.float64)
        - np.dot(v, axis) * axis
    )

    return unit(
        out,
        fallback,
    )


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
    out = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    for i in range(
        1,
        len(out),
    ):
        d = (
            out[i]
            - out[i - 1]
        )

        d = np.clip(
            d,
            -max_step,
            max_step,
        )

        out[i] = (
            out[i - 1]
            + d
        )

    for i in range(
        len(out) - 2,
        -1,
        -1,
    ):
        d = (
            out[i]
            - out[i + 1]
        )

        d = np.clip(
            d,
            -max_step,
            max_step,
        )

        out[i] = (
            out[i + 1]
            + d
        )

    return out


# ============================================================
# Existing V6 anchors
# ============================================================

v6_anchor = {
    "LEFT": None,
    "RIGHT": None,
}

if V6_CSV.exists():
    with open(
        V6_CSV,
        "r",
        encoding="utf-8",
    ) as f:
        v6_rows = list(
            csv.DictReader(f)
        )

    if v6_rows:
        v6_anchor["LEFT"] = np.array(
            [
                int(r["left_anchor"])
                for r in v6_rows
            ],
            dtype=bool,
        )

        v6_anchor["RIGHT"] = np.array(
            [
                int(r["right_anchor"])
                for r in v6_rows
            ],
            dtype=bool,
        )

        print(
            "Loaded V6 anchors:",
            V6_CSV,
        )

else:
    print(
        "WARNING: V6 CSV not found.",
        "Palm V2 will use its own continuity filter only.",
    )


# ============================================================
# ROI / detection helpers
# ============================================================

def make_crop(
    frame,
    elbow,
    wrist,
    center_ratio,
    size_ratio,
):
    h, w = frame.shape[:2]

    forearm = (
        wrist
        - elbow
    )

    length = float(
        np.linalg.norm(
            forearm
        )
    )

    if length < 8.0:
        return None

    direction = (
        forearm
        / length
    )

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

    half = (
        size / 2.0
    )

    x0 = int(
        max(
            0,
            np.floor(
                center[0]
                - half
            ),
        )
    )

    y0 = int(
        max(
            0,
            np.floor(
                center[1]
                - half
            ),
        )
    )

    x1 = int(
        min(
            w,
            np.ceil(
                center[0]
                + half
            ),
        )
    )

    y1 = int(
        min(
            h,
            np.ceil(
                center[1]
                + half
            ),
        )
    )

    if (
        x1 - x0 < 30
        or y1 - y0 < 30
    ):
        return None

    crop = frame[
        y0:y1,
        x0:x1
    ]

    crop = cv2.resize(
        crop,
        (
            ROI_SIZE,
            ROI_SIZE,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    return {
        "crop": crop,
        "box": (
            x0,
            y0,
            x1,
            y1,
        ),
    }


def map_normalized_2d(
    landmarks,
    box,
    flipped,
):
    x0, y0, x1, y1 = box

    bw = (
        x1 - x0
    )

    bh = (
        y1 - y0
    )

    pts = []

    for lm in landmarks:
        x = float(
            lm.x
        )

        if flipped:
            x = (
                1.0
                - x
            )

        pts.append(
            np.array(
                [
                    x0 + x * bw,
                    y0 + float(lm.y) * bh,
                ],
                dtype=np.float64,
            )
        )

    return pts


def map_world_3d(
    landmarks,
    flipped,
):
    """
    Restore a horizontally-flipped detector input back to the original
    camera-axis convention.

    For world landmarks the hand origin is local to the detected hand,
    so restoring horizontal reflection is done by x -> -x.
    """
    pts = []

    for lm in landmarks:
        x = float(
            lm.x
        )

        if flipped:
            x = (
                -x
            )

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
    crop = (
        crop_info["crop"]
    )

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

    if not getattr(
        result,
        "hand_world_landmarks",
        None,
    ):
        raise RuntimeError(
            "This MediaPipe HandLandmarker result does not provide "
            "hand_world_landmarks."
        )

    best = None

    for i, landmarks in enumerate(
        result.hand_landmarks
    ):
        if i >= len(
            result.hand_world_landmarks
        ):
            continue

        pts2 = map_normalized_2d(
            landmarks,
            crop_info["box"],
            try_flip,
        )

        ptsw = map_world_3d(
            result.hand_world_landmarks[i],
            try_flip,
        )

        wrist_dist = float(
            np.linalg.norm(
                pts2[0]
                - pose_wrist
            )
        )

        hand_span = float(
            np.linalg.norm(
                pts2[9]
                - pts2[0]
            )
        )

        score = 0.0

        if i < len(
            result.handedness
        ):
            score = float(
                result.handedness[
                    i
                ][0].score
            )

        cost = (
            wrist_dist
            - 25.0 * score
            - 0.05 * hand_span
        )

        candidate = {
            "pts2": pts2,
            "ptsw": ptsw,
            "score": score,
            "cost": cost,
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

    for (
        center_ratio,
        size_ratio,
    ) in proposals:
        crop_info = make_crop(
            frame,
            elbow,
            wrist,
            center_ratio,
            size_ratio,
        )

        if crop_info is None:
            continue

        for flipped in (
            False,
            True,
        ):
            cand = detect_candidate(
                detector,
                crop_info,
                wrist,
                flipped,
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


# ============================================================
# MediaPipe setup
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

N = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print("=" * 72)
print("OpenLoong Palm Roll V2 - hand_world_landmarks")
print("=" * 72)
print("frames:", N)
print("fps   :", fps)


BaseOptions = mp.tasks.BaseOptions
RunningMode = (
    mp.tasks.vision.RunningMode
)

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

finger_world = {
    side: np.full(
        (N, 3),
        np.nan,
        dtype=np.float64,
    )
    for side in sides
}

palm_normal_world = {
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

torso_down_world = np.full(
    (N, 3),
    np.nan,
    dtype=np.float64,
)


# ============================================================
# First pass
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
                frame_id
                / fps
                * 1000.0
            )
        )

        pose_result = (
            pose_detector.detect_for_video(
                mp_frame,
                timestamp_ms,
            )
        )

        if (
            not pose_result.pose_landmarks
            or not pose_result.pose_world_landmarks
        ):
            continue

        pose2 = (
            pose_result.pose_landmarks[0]
        )

        posew = (
            pose_result.pose_world_landmarks[0]
        )

        def pix(idx):
            lm = pose2[idx]

            return np.array(
                [
                    float(lm.x) * width,
                    float(lm.y) * height,
                ],
                dtype=np.float64,
            )

        def pworld(idx):
            lm = posew[idx]

            return np.array(
                [
                    float(lm.x),
                    float(lm.y),
                    float(lm.z),
                ],
                dtype=np.float64,
            )

        shoulder_mid = (
            pworld(LEFT_SHOULDER)
            + pworld(RIGHT_SHOULDER)
        ) * 0.5

        hip_mid = (
            pworld(LEFT_HIP)
            + pworld(RIGHT_HIP)
        ) * 0.5

        torso_down_world[
            frame_id
        ] = unit(
            hip_mid
            - shoulder_mid,
            [0.0, 1.0, 0.0],
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

            hand_wrist = (
                cand["pts2"][0]
            )

            wrist_error = float(
                np.linalg.norm(
                    hand_wrist
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

            pts = cand[
                "ptsw"
            ]

            wrist = pts[0]

            # Long hand axis: wrist -> average fingertips
            tip_center = np.mean(
                np.stack([
                    pts[8],
                    pts[12],
                    pts[16],
                    pts[20],
                ]),
                axis=0,
            )

            finger = unit(
                tip_center
                - wrist,
                [0.0, -1.0, 0.0],
            )

            # Across palm: pinky MCP -> index MCP
            across = unit(
                pts[5]
                - pts[17],
                [1.0, 0.0, 0.0],
            )

            # Palm-plane normal
            normal = unit(
                np.cross(
                    finger,
                    across,
                ),
                [0.0, 0.0, 1.0],
            )

            detected[
                side
            ][frame_id] = True

            score[
                side
            ][frame_id] = (
                cand["score"]
            )

            finger_world[
                side
            ][frame_id] = (
                finger
            )

            palm_normal_world[
                side
            ][frame_id] = (
                normal
            )

            wrist_2d[
                side
            ][frame_id] = (
                p_wrist
            )

        if (
            frame_id == 0
            or (frame_id + 1) % 100 == 0
            or frame_id == N - 1
        ):
            print(
                f"processed "
                f"{frame_id + 1}/{N}"
            )


cap.release()


# ============================================================
# Fixed camera ground-down estimate
#
# Use torso world-landmark vertical from the known 15~18 s segment.
# This avoids using crop/image normalized z as a 3D direction source.
# ============================================================

calib_lo = max(
    0,
    int(
        round(
            CALIB_START_SEC
            * fps
        )
    ),
)

calib_hi = min(
    N,
    int(
        round(
            CALIB_END_SEC
            * fps
        )
    ) + 1,
)

calib_torso = (
    torso_down_world[
        calib_lo:calib_hi
    ]
)

valid_torso = np.all(
    np.isfinite(
        calib_torso
    ),
    axis=1,
)

assert np.count_nonzero(
    valid_torso
) >= 2

ground_down = unit(
    np.median(
        calib_torso[
            valid_torso
        ],
        axis=0,
    ),
    [0.0, 1.0, 0.0],
)

print()
print(
    "estimated fixed ground-down:",
    ground_down,
)


# ============================================================
# Raw palm roll, sign calibration
# ============================================================

raw_roll_by_sign = {
    side: {}
    for side in sides
}

chosen_sign = {}

for side in sides:
    valid = np.where(
        detected[
            side
        ]
    )[0]

    assert len(valid) >= 2

    for sign in (
        +1.0,
        -1.0,
    ):
        arr = np.full(
            N,
            np.nan,
            dtype=np.float64,
        )

        for i in valid:
            finger = (
                finger_world[
                    side
                ][i]
            )

            normal = (
                sign
                * palm_normal_world[
                    side
                ][i]
            )

            down_ref = project_perp(
                ground_down,
                finger,
                [0.0, 0.0, 1.0],
            )

            arr[i] = (
                signed_angle_deg(
                    down_ref,
                    normal,
                    finger,
                )
            )

        raw_roll_by_sign[
            side
        ][int(sign)] = (
            arr
        )

    # Choose the normal sign that gives the smallest median circular
    # distance to 0 deg in the known palm-down calibration segment.
    costs = {}

    for sign in (
        +1,
        -1,
    ):
        arr = (
            raw_roll_by_sign[
                side
            ][sign][
                calib_lo:calib_hi
            ]
        )

        arr = arr[
            np.isfinite(arr)
        ]

        assert len(arr) >= 2

        costs[sign] = float(
            np.median(
                np.abs(
                    wrap_deg(arr)
                )
            )
        )

    chosen = min(
        costs,
        key=costs.get,
    )

    chosen_sign[
        side
    ] = chosen

    print()
    print(
        f"[{side}] detections:",
        int(
            detected[
                side
            ].sum()
        ),
        "/",
        N,
    )

    print(
        f"[{side}] normal sign costs:",
        costs,
        "->",
        chosen,
    )


# ============================================================
# Anchor filtering + interpolation
# ============================================================

final_roll = {}
raw_chosen_roll = {}
anchor_mask = {}
jump_reject = {}
zero_offset = {}


for side in sides:
    raw = (
        raw_roll_by_sign[
            side
        ][
            chosen_sign[
                side
            ]
        ]
    )

    raw_chosen_roll[
        side
    ] = raw.copy()

    candidates = np.isfinite(
        raw
    )

    # Prefer the already-validated V6 anchors when available.
    if (
        v6_anchor[
            side
        ] is not None
        and len(
            v6_anchor[
                side
            ]
        ) == N
    ):
        preferred = (
            candidates
            & v6_anchor[
                side
            ]
        )

        # If V6 gives too few anchors for some reason, gracefully fall back.
        if np.count_nonzero(
            preferred
        ) >= 20:
            candidates = (
                preferred
            )

            print(
                f"[{side}] using V6 anchors:",
                int(
                    candidates.sum()
                ),
            )

    candidate_idx = np.where(
        candidates
    )[0]

    keep = np.zeros(
        N,
        dtype=bool,
    )

    reject = np.zeros(
        N,
        dtype=bool,
    )

    anchor_idx = []
    anchor_unwrapped = []

    last_idx = None
    last_wrapped = None
    last_unwrapped = None

    for i in candidate_idx:
        value = float(
            wrap_deg(
                raw[i]
            )
        )

        if last_idx is None:
            keep[i] = True

            anchor_idx.append(
                i
            )

            anchor_unwrapped.append(
                value
            )

            last_idx = i
            last_wrapped = value
            last_unwrapped = value
            continue

        gap = (
            i
            - last_idx
        )

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

        anchor_idx.append(
            i
        )

        anchor_unwrapped.append(
            current_unwrapped
        )

        last_idx = i
        last_wrapped = value
        last_unwrapped = current_unwrapped

    anchor_idx = np.asarray(
        anchor_idx,
        dtype=int,
    )

    anchor_unwrapped = np.asarray(
        anchor_unwrapped,
        dtype=np.float64,
    )

    assert len(
        anchor_idx
    ) >= 2

    all_idx = np.arange(
        N
    )

    roll = np.interp(
        all_idx,
        anchor_idx,
        anchor_unwrapped,
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

    # Final zero offset:
    # known 15~18 sec palm-down -> physical roll = 0.
    calib_segment = (
        roll[
            calib_lo:calib_hi
        ]
    )

    zero = float(
        np.median(
            calib_segment
        )
    )

    zero_offset[
        side
    ] = zero

    roll = (
        roll
        - zero
    )

    final_roll[
        side
    ] = wrap_deg(
        roll
    )

    anchor_mask[
        side
    ] = keep

    jump_reject[
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
        f"[{side}] zero offset:",
        f"{zero:.2f} deg",
    )

    print(
        f"[{side}] physical range:",
        f"{final_roll[side].min():.1f}",
        "~",
        f"{final_roll[side].max():.1f}",
        "deg",
    )


# ============================================================
# Reports
# ============================================================

time_sec = (
    np.arange(N)
    / fps
)


def segment_report(
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

    for side in sides:
        x = (
            final_roll[
                side
            ][sel]
        )

        print(
            f"{side:5s} roll: "
            f"mean={np.mean(x):7.2f} "
            f"median={np.median(x):7.2f} "
            f"min={np.min(x):7.2f} "
            f"max={np.max(x):7.2f} deg"
        )


segment_report(
    "7~9 sec",
    7.0,
    9.0,
)

segment_report(
    "15~18 sec ZERO check",
    15.0,
    18.0,
)


for side in sides:
    x = final_roll[
        side
    ]

    step = np.abs(
        wrap_deg(
            np.diff(x)
        )
    )

    print()
    print(
        f"[{side}] max circular step:",
        f"{step.max():.2f}",
        "deg/frame",
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

        "left_palm_roll_v2_deg",
        "right_palm_roll_v2_deg",

        "left_raw_world_roll_deg",
        "right_raw_world_roll_deg",

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

        "left_zero_offset_deg",
        "right_zero_offset_deg",

        "ground_down_x",
        "ground_down_y",
        "ground_down_z",
    ])

    for i in range(N):
        writer.writerow([
            i,
            time_sec[i],

            final_roll[
                "LEFT"
            ][i],

            final_roll[
                "RIGHT"
            ][i],

            (
                raw_chosen_roll[
                    "LEFT"
                ][i]
                if np.isfinite(
                    raw_chosen_roll[
                        "LEFT"
                    ][i]
                )
                else ""
            ),

            (
                raw_chosen_roll[
                    "RIGHT"
                ][i]
                if np.isfinite(
                    raw_chosen_roll[
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
                jump_reject[
                    "LEFT"
                ][i]
            ),

            int(
                jump_reject[
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

            chosen_sign[
                "LEFT"
            ],

            chosen_sign[
                "RIGHT"
            ],

            zero_offset[
                "LEFT"
            ],

            zero_offset[
                "RIGHT"
            ],

            ground_down[0],
            ground_down[1],
            ground_down[2],
        ])


# ============================================================
# Overlay video
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
    ok, image = cap.read()

    if not ok:
        break

    cv2.putText(
        image,
        (
            "PALM ROLL V2 WORLD  "
            f"{time_sec[i]:.2f}s"
        ),
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        image,
        (
            "LEFT  "
            f"{final_roll['LEFT'][i]:+7.1f} deg "
            f"A={int(anchor_mask['LEFT'][i])}"
        ),
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 200, 0),
        2,
    )

    cv2.putText(
        image,
        (
            "RIGHT "
            f"{final_roll['RIGHT'][i]:+7.1f} deg "
            f"A={int(anchor_mask['RIGHT'][i])}"
        ),
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
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
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
        )

    writer.write(
        image
    )


cap.release()
writer.release()


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
