#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)


# ======================================================================
# INPUT / OUTPUT
# ======================================================================

V1_CSV = (
    ROOT
    / "output/linglong20_arm_v1"
    / "linglong20_arm_v1_qpos37.csv"
)

XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

SMPLX_FILE = (
    ROOT
    / "output/openloong_v2_kneeguide"
    / "stream_demo"
    / "gmr_smplx_results.npz"
)

BODY_MODEL_DIR = ROOT / "assets/body_models"

OUT_DIR = (
    ROOT
    / "output/linglong20_arm_v15_strong_extension"
)

OUT_CSV = (
    OUT_DIR
    / "linglong20_arm_v15_strong_extension_qpos37.csv"
)


LEFT_ARM = np.arange(23, 30)
RIGHT_ARM = np.arange(30, 37)


# ======================================================================
# OPTIMIZATION POLICY
# ======================================================================

# Main skeleton tracking.
ELBOW_WEIGHT = 35.0
WRIST_WEIGHT = 30.0

# Prefer the already good V1 solution unless collision requires change.
V1_ARM_POSTURE_WEIGHT = 0.035
V1_WRIST_POSTURE_WEIGHT = 0.080

# Temporal continuity.
CONTINUITY_WEIGHT = 0.060

# Collision avoidance.
#
# < 15 mm:
#     increasingly strong repulsion
#
# > 15 mm:
#     zero collision penalty
CLEARANCE = 0.015

# Start correcting slightly before the actual clearance threshold.
ACTIVATION_DISTANCE = 0.030

# Expand collision windows so corrections fade in/out smoothly.
WINDOW_EXPANSION = 8

# Stronger than the position tracking term.
COLLISION_WEIGHT = 180.0

# mj_geomDistance only needs to search locally.
DISTANCE_MAX = 0.10

MAX_NFEV = 120


# ======================================================================
# HELPERS
# ======================================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise RuntimeError("Degenerate vector")
    return v / n


def chest_frame(frame):

    ls = np.asarray(frame["left_shoulder"][0], dtype=np.float64)
    rs = np.asarray(frame["right_shoulder"][0], dtype=np.float64)

    spine = np.asarray(frame["spine3"][0], dtype=np.float64)
    neck = np.asarray(frame["neck"][0], dtype=np.float64)

    # +Y left
    y = normalize(ls - rs)

    # +Z up
    z0 = normalize(neck - spine)
    z = normalize(z0 - np.dot(z0, y) * y)

    # +X forward
    x = normalize(np.cross(y, z))

    # Re-orthogonalize
    y = normalize(np.cross(z, x))

    return np.column_stack((x, y, z))


def body_id(model, name):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(f"Body not found: {name}")

    return int(bid)


def body_geoms(model, body_name):

    bid = body_id(model, body_name)

    return [
        gid
        for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) == bid
    ]


def geoms_can_collide(model, g1, g2):

    c1 = int(model.geom_contype[g1])
    a1 = int(model.geom_conaffinity[g1])

    c2 = int(model.geom_contype[g2])
    a2 = int(model.geom_conaffinity[g2])

    return bool(
        (c1 & a2)
        or
        (c2 & a1)
    )


def make_body_pair(
    model,
    arm_body,
    target_body,
):

    arm_geoms = body_geoms(
        model,
        arm_body,
    )

    target_geoms = body_geoms(
        model,
        target_body,
    )

    geom_pairs = []

    for g1 in arm_geoms:
        for g2 in target_geoms:

            if geoms_can_collide(
                model,
                g1,
                g2,
            ):
                geom_pairs.append(
                    (
                        int(g1),
                        int(g2),
                    )
                )

    if not geom_pairs:
        return None

    return {
        "arm_body": arm_body,
        "target_body": target_body,
        "pairs": geom_pairs,
    }


def build_collision_pairs(
    model,
    side,
):

    # We intentionally do NOT constrain shoulder bodies:
    # they are attached close to the torso by design.
    #
    # These links cover the forearm/wrist region that actually
    # produced the V1 contacts.
    arm_bodies = [
        f"{side}_elbow_link",
        f"{side}_wrist_roll_link",
        f"{side}_wrist_pitch_link",
        f"{side}_wrist_yaw_link",
    ]

    # Torso + both hips.
    #
    # Both sides are included so crossing the midline cannot
    # create a new collision while fixing the original one.
    target_bodies = [
        "base_link",
        "waist_yaw_link",
        "waist_pitch_link",

        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",

        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
    ]

    specs = []

    for a in arm_bodies:
        for b in target_bodies:

            spec = make_body_pair(
                model,
                a,
                b,
            )

            if spec is not None:
                specs.append(spec)

    return specs


def body_pair_distance(
    model,
    data,
    spec,
):

    best = DISTANCE_MAX

    # mj_geomDistance returns signed distance:
    #
    #   > 0 : separated
    #   = 0 : touching
    #   < 0 : penetration
    #
    # fromto receives the nearest points.
    fromto = np.zeros(
        6,
        dtype=np.float64,
    )

    for g1, g2 in spec["pairs"]:

        d = float(
            mujoco.mj_geomDistance(
                model,
                data,
                g1,
                g2,
                DISTANCE_MAX,
                fromto,
            )
        )

        if d < best:
            best = d

    return best


def all_clearance_distances(
    model,
    data,
    specs,
):

    return np.asarray(
        [
            body_pair_distance(
                model,
                data,
                spec,
            )
            for spec in specs
        ],
        dtype=np.float64,
    )


def joint_bounds(
    model,
    indices,
):

    lows = []
    highs = []

    for qidx in indices:

        found = False

        for jid in range(model.njnt):

            if int(
                model.jnt_qposadr[jid]
            ) == int(qidx):

                lo, hi = model.jnt_range[jid]

                lows.append(float(lo))
                highs.append(float(hi))

                found = True
                break

        if not found:
            raise RuntimeError(
                f"joint not found for qpos {qidx}"
            )

    return (
        np.asarray(lows),
        np.asarray(highs),
    )


def joint_name(
    model,
    qidx,
):

    for jid in range(model.njnt):

        if int(
            model.jnt_qposadr[jid]
        ) == int(qidx):

            return mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )

    return "UNKNOWN"


def expand_mask(
    mask,
    radius,
):

    out = mask.copy()

    indices = np.flatnonzero(mask)

    for i in indices:

        lo = max(
            0,
            i - radius,
        )

        hi = min(
            len(mask),
            i + radius + 1,
        )

        out[
            lo:hi
        ] = True

    return out


# ======================================================================
# TARGET GENERATION
# ======================================================================

def morphology_lengths(
    model,
    data,
    reference_q,
    side,
    arm_indices,
):

    shoulder_id = body_id(
        model,
        f"{side}_shoulder_pitch_link",
    )

    elbow_id = body_id(
        model,
        f"{side}_elbow_link",
    )

    wrist_id = body_id(
        model,
        f"{side}_wrist_roll_link",
    )

    q = reference_q.copy()

    q[
        arm_indices
    ] = 0.0

    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    shoulder = data.xpos[
        shoulder_id
    ].copy()

    elbow = data.xpos[
        elbow_id
    ].copy()

    wrist = data.xpos[
        wrist_id
    ].copy()

    upper_len = float(
        np.linalg.norm(
            elbow - shoulder
        )
    )

    fore_len = float(
        np.linalg.norm(
            wrist - elbow
        )
    )

    return (
        upper_len,
        fore_len,
    )


def arm_target_for_frame(
    model,
    data,
    q_reference,
    frame,
    side,
    shoulder_id,
    upper_len,
    fore_len,
):

    # Human directions in human chest coordinates.
    Ch = chest_frame(frame)

    hs = np.asarray(
        frame[
            f"{side}_shoulder"
        ][0],
        dtype=np.float64,
    )

    he = np.asarray(
        frame[
            f"{side}_elbow"
        ][0],
        dtype=np.float64,
    )

    hw = np.asarray(
        frame[
            f"{side}_wrist"
        ][0],
        dtype=np.float64,
    )

    upper_local = (
        Ch.T
        @ normalize(
            he - hs
        )
    )

    fore_local = (
        Ch.T
        @ normalize(
            hw - he
        )
    )


    # Robot chest / shoulder anchor.
    data.qpos[:] = q_reference

    mujoco.mj_forward(
        model,
        data,
    )

    chest_id = body_id(
        model,
        "waist_pitch_link",
    )

    Cr = (
        data.xmat[
            chest_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    shoulder = (
        data.xpos[
            shoulder_id
        ]
        .copy()
    )


    target_elbow = (
        shoulder
        + Cr
        @ upper_local
        * upper_len
    )

    target_wrist = (
        target_elbow
        + Cr
        @ fore_local
        * fore_len
    )

    return (
        target_elbow,
        target_wrist,
    )


# ======================================================================
# PRE-COLLISION SCAN
# ======================================================================

def pre_scan(
    model,
    data,
    qpos,
    specs,
):

    mins = np.zeros(
        len(qpos),
        dtype=np.float64,
    )

    for i, q in enumerate(qpos):

        data.qpos[:] = q
        mujoco.mj_forward(
            model,
            data,
        )

        d = all_clearance_distances(
            model,
            data,
            specs,
        )

        mins[i] = float(
            np.min(d)
        )

    return mins


# ======================================================================
# SOLVER
# ======================================================================

def solve_side(
    side,
    model,
    data,
    frames,
    input_qpos,
    arm_indices,
):

    print()
    print("=" * 110)
    print(
        side.upper(),
        "ARM V1.5 STRONG EXTENSION SOLVE",
    )
    print("=" * 110)


    shoulder_id = body_id(
        model,
        f"{side}_shoulder_pitch_link",
    )

    elbow_id = body_id(
        model,
        f"{side}_elbow_link",
    )

    wrist_id = body_id(
        model,
        f"{side}_wrist_roll_link",
    )


    low, high = joint_bounds(
        model,
        arm_indices,
    )


    upper_len, fore_len = (
        morphology_lengths(
            model,
            data,
            input_qpos[0],
            side,
            arm_indices,
        )
    )


    print(
        "upper-arm length:",
        upper_len,
    )

    print(
        "forearm length:",
        fore_len,
    )


    specs = build_collision_pairs(
        model,
        side,
    )


    print(
        "collision body-pair constraints:",
        len(specs),
    )


    if not hasattr(
        mujoco,
        "mj_geomDistance",
    ):
        raise RuntimeError(
            "This MuJoCo Python build does not expose "
            "mj_geomDistance."
        )


    # ----------------------------------------------------------
    # V1 pre-scan
    # ----------------------------------------------------------

    pre_min = pre_scan(
        model,
        data,
        input_qpos,
        specs,
    )


    raw_active = (
        pre_min
        < ACTIVATION_DISTANCE
    )

    active = expand_mask(
        raw_active,
        WINDOW_EXPANSION,
    )

    # ======================================================
    # V1.4 HUMAN ARM EXTENSION MASK
    #
    # Collision-only optimization is insufficient:
    # a visually wrong bent elbow may never enter the solver.
    #
    # Measure the HUMAN elbow bend directly.
    # 0 deg = straight.
    # ======================================================
    
    human_bend_deg_seq = np.zeros(
        len(frames),
        dtype=np.float64,
    )
    
    for _fi, _fr in enumerate(frames):
    
        _hs = np.asarray(
            _fr[f"{side}_shoulder"][0],
            dtype=np.float64,
        )
    
        _he = np.asarray(
            _fr[f"{side}_elbow"][0],
            dtype=np.float64,
        )
    
        _hw = np.asarray(
            _fr[f"{side}_wrist"][0],
            dtype=np.float64,
        )
    
        _upper = normalize(
            _he - _hs
        )
    
        _fore = normalize(
            _hw - _he
        )
    
        _dot = float(
            np.clip(
                np.dot(
                    _upper,
                    _fore,
                ),
                -1.0,
                1.0,
            )
        )
    
        human_bend_deg_seq[_fi] = (
            np.degrees(
                np.arccos(_dot)
            )
        )
    
    
    # Start solving before the arm becomes completely straight.
    #
    # < 30 deg:
    #   extension-aware IK is active.
    extension_trigger = (
        human_bend_deg_seq
        < 45.0
    )
    
    extension_active = (
        extension_trigger.copy()
    )
    
    # Add a short temporal margin around extension events.
    for _idx in np.flatnonzero(
        extension_trigger
    ):
        _lo = max(
            0,
            int(_idx) - 3,
        )
    
        _hi = min(
            len(extension_active),
            int(_idx) + 4,
        )
    
        extension_active[
            _lo:_hi
        ] = True
    
    
    _collision_active_count = int(
        np.count_nonzero(active)
    )
    
    active = np.logical_or(
        active,
        extension_active,
    )
    
    print(
        f"{side} V1.4 human near-straight frames:",
        int(
            np.count_nonzero(
                extension_trigger
            )
        ),
    )
    
    print(
        f"{side} V1.4 extension-active frames:",
        int(
            np.count_nonzero(
                extension_active
            )
        ),
    )
    
    print(
        f"{side} V1.4 total active after union:",
        int(
            np.count_nonzero(active)
        ),
    )

    # ======================================================
    # V1.3 END-OF-SEQUENCE WINDOW CONTINUITY
    #
    # V1.2 correctly limits motion while a frame is active,
    # but immediately restoring V1 outside the mask can
    # create a very large boundary jump.
    #
    # If a collision window already reaches the final
    # 12 frames, simply keep solving until the sequence ends.
    # There is no reason to force a return to V1 immediately
    # before the video finishes.
    # ======================================================
    
    _active_ids = np.flatnonzero(active)
    
    if _active_ids.size > 0:
        _last_active = int(
            _active_ids[-1]
        )
    
        if (
            _last_active
            >= len(active) - 12
            and
            _last_active
            < len(active) - 1
        ):
            _tail_start = (
                _last_active + 1
            )
    
            active[
                _tail_start:
            ] = True
    
            print(
                f"{side} V1.3 tail window extended: "
                f"{_tail_start}..{len(active) - 1}"
            )


    print(
        "V1 penetration frames:",
        int(
            np.sum(
                pre_min < 0.0
            )
        ),
    )

    print(
        "V1 below 15mm frames:",
        int(
            np.sum(
                pre_min < CLEARANCE
            )
        ),
    )

    print(
        "V1 below 30mm frames:",
        int(
            np.sum(
                raw_active
            )
        ),
    )

    print(
        "expanded optimization frames:",
        int(
            np.sum(active)
        ),
    )

    print(
        "V1 min signed distance:",
        float(
            np.min(pre_min)
            * 1000.0
        ),
        "mm",
    )


    result_qpos = input_qpos.copy()


    # Previous corrected solution.
    prev = result_qpos[
        0,
        arm_indices,
    ].copy()


    elbow_error = np.zeros(
        len(frames)
    )

    wrist_error = np.zeros(
        len(frames)
    )



    # ==============================================================
    # V1.2 TEMPORAL STATE
    # ==============================================================
    
    # Robot joint velocity history.
    prev_delta = None
    
    # Human elbow/wrist target history.
    prev_target_elbow = None
    prev_target_wrist = None
    
    prev_target_elbow_delta = None
    prev_target_wrist_delta = None
    
    last_target_frame = None
    
    low_conf_elbow_frames = []
    low_conf_wrist_frames = []
    
    # Stronger continuity for redundant wrist DoFs.
    temporal_scale = np.asarray(
        [
            1.0,
            1.0,
            1.4,
            1.0,
            1.5,
            2.0,
            2.0,
        ],
        dtype=np.float64,
    )
    
    # Hard per-frame IK trust region.
    #
    # shoulder / elbow:
    #   5.16 deg = 0.090 rad
    #
    # wrist:
    #   5.73 deg = 0.100 rad
    #
    hard_step = np.asarray(
        [
            0.090,
            0.090,
            0.090,
            0.090,
            0.100,
            0.100,
            0.100,
        ],
        dtype=np.float64,
    )
    
    
    def target_confidence(
        error_m,
        soft_m,
        hard_m,
    ):
        """Convert motion-prediction error to [0, 1] confidence."""
    
        error_m = float(error_m)
    
        if error_m <= soft_m:
            return 1.0
    
        if error_m >= hard_m:
            return 0.0
    
        return float(
            (hard_m - error_m)
            / (hard_m - soft_m)
        )
    

    # ======================================================
    # V1.5 MODEL-SPECIFIC STRAIGHT ELBOW CALIBRATION
    #
    # Never assume elbow_joint == 0 means mechanically straight.
    #
    # Sweep the real LingLong elbow range and find the joint
    # value that minimizes the geometric angle:
    #
    #   shoulder -> elbow
    #   elbow    -> wrist
    #
    # 0 deg geometric bend = perfectly straight.
    # ======================================================

    elbow_joint_name = f"{side}_elbow_joint"

    elbow_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        elbow_joint_name,
    )

    if elbow_joint_id < 0:
        raise RuntimeError(
            f"joint not found: {elbow_joint_name}"
        )

    elbow_qpos_addr = int(
        model.jnt_qposadr[
            elbow_joint_id
        ]
    )

    arm_index_list = [
        int(v)
        for v in arm_indices
    ]

    if elbow_qpos_addr not in arm_index_list:
        raise RuntimeError(
            f"{elbow_joint_name} qpos={elbow_qpos_addr} "
            f"not in arm_indices={arm_index_list}"
        )

    elbow_local_index = (
        arm_index_list.index(
            elbow_qpos_addr
        )
    )

    if bool(
        model.jnt_limited[
            elbow_joint_id
        ]
    ):
        elbow_lo = float(
            model.jnt_range[
                elbow_joint_id,
                0,
            ]
        )

        elbow_hi = float(
            model.jnt_range[
                elbow_joint_id,
                1,
            ]
        )
    else:
        elbow_lo = -1.6
        elbow_hi = +1.6

    calibration_q = (
        input_qpos[0].copy()
    )

    straight_elbow_q = None
    straight_geometric_bend = float("inf")

    for candidate_q in np.linspace(
        elbow_lo,
        elbow_hi,
        401,
    ):
        calibration_q[
            elbow_qpos_addr
        ] = float(candidate_q)

        data.qpos[:] = calibration_q

        mujoco.mj_forward(
            model,
            data,
        )

        cal_s = np.asarray(
            data.xpos[
                shoulder_id
            ],
            dtype=np.float64,
        )

        cal_e = np.asarray(
            data.xpos[
                elbow_id
            ],
            dtype=np.float64,
        )

        cal_w = np.asarray(
            data.xpos[
                wrist_id
            ],
            dtype=np.float64,
        )

        cal_upper = normalize(
            cal_e - cal_s
        )

        cal_fore = normalize(
            cal_w - cal_e
        )

        cal_dot = float(
            np.clip(
                np.dot(
                    cal_upper,
                    cal_fore,
                ),
                -1.0,
                1.0,
            )
        )

        cal_bend = float(
            np.degrees(
                np.arccos(
                    cal_dot
                )
            )
        )

        if (
            cal_bend
            < straight_geometric_bend
        ):
            straight_geometric_bend = (
                cal_bend
            )

            straight_elbow_q = float(
                candidate_q
            )

    if straight_elbow_q is None:
        raise RuntimeError(
            "V1.5 straight elbow calibration failed"
        )

    print(
        f"{side} V1.5 straight elbow calibration | "
        f"local_idx={elbow_local_index} | "
        f"joint_q={np.degrees(straight_elbow_q):.2f} deg | "
        f"geometric_bend={straight_geometric_bend:.2f} deg"
    )


    # V1.4 general active->inactive transition bridge.
    boundary_bridge_frames = []
    for i, frame in enumerate(frames):

        v1_seed = input_qpos[
            i,
            arm_indices,
        ].copy()


        # ------------------------------------------------------
        # Outside the expanded danger window:
        # preserve V1 exactly.
        # ------------------------------------------------------


        # ======================================================
        # V1.4 GENERAL BOUNDARY ANTI-FLIP
        #
        # V1.3 can still jump when an active window ends because
        # it immediately copies the old V1 pose.
        #
        # If the previous optimized arm is still far from V1,
        # keep solving this frame as a release/bridge frame.
        # ======================================================
        is_boundary_bridge = False
        
        if not active[i]:
        
            bridge_error = np.abs(
                v1_seed - prev
            )
        
            bridge_needed = bool(
                np.any(
                    bridge_error
                    > 0.75 * hard_step
                )
            )
        
            if not bridge_needed:
                result_qpos[
                    i,
                    arm_indices,
                ] = v1_seed
        
                prev = v1_seed.copy()
                prev_delta = None
        
                continue
        
            # Do NOT jump to V1.
            # Continue solving and gradually converge back.
            is_boundary_bridge = True
            active[i] = True
            boundary_bridge_frames.append(
                int(i)
            )


        base_q = result_qpos[
            i
        ].copy()


        (
            target_elbow,
            target_wrist,
        ) = arm_target_for_frame(
            model,
            data,
            base_q,
            frame,
            side,
            shoulder_id,
            upper_len,
            fore_len,
        )

        # ==========================================================
        # V1.2 HUMAN TARGET RELIABILITY
        #
        # A hidden/occluded arm may still receive a WHAM/SMPL-X
        # estimate.  Do not blindly chase a sudden 3-D jump.
        # ==========================================================
        
        raw_target_elbow = np.asarray(
            target_elbow,
            dtype=np.float64,
        ).copy()
        
        raw_target_wrist = np.asarray(
            target_wrist,
            dtype=np.float64,
        ).copy()
        
        frame_i = int(i)
        
        # If we jump between separated collision windows,
        # do not extrapolate human motion across the gap.
        if (
            last_target_frame is not None
            and frame_i != last_target_frame + 1
        ):
            prev_target_elbow = None
            prev_target_wrist = None
            prev_target_elbow_delta = None
            prev_target_wrist_delta = None
            prev_delta = None
        
        elbow_conf = 1.0
        wrist_conf = 1.0
        
        elbow_prediction_error = 0.0
        wrist_prediction_error = 0.0
        
        
        # -----------------------
        # ELBOW reliability
        # -----------------------
        
        if prev_target_elbow is not None:
            if prev_target_elbow_delta is None:
                predicted_elbow = (
                    prev_target_elbow.copy()
                )
            else:
                predicted_elbow = (
                    prev_target_elbow
                    + 0.70
                    * prev_target_elbow_delta
                )
        
            elbow_prediction_error = float(
                np.linalg.norm(
                    raw_target_elbow
                    - predicted_elbow
                )
            )
        
            # Start distrusting at 45 mm.
            # Fully reject at 90 mm.
            elbow_conf = target_confidence(
                elbow_prediction_error,
                0.045,
                0.090,
            )
        
            target_elbow = (
                elbow_conf
                * raw_target_elbow
                + (1.0 - elbow_conf)
                * predicted_elbow
            )
        
            if elbow_conf < 0.999:
                low_conf_elbow_frames.append(
                    (
                        frame_i,
                        elbow_prediction_error,
                        elbow_conf,
                    )
                )
        
        
        # -----------------------
        # WRIST reliability
        # -----------------------
        
        if prev_target_wrist is not None:
            if prev_target_wrist_delta is None:
                predicted_wrist = (
                    prev_target_wrist.copy()
                )
            else:
                predicted_wrist = (
                    prev_target_wrist
                    + 0.70
                    * prev_target_wrist_delta
                )
        
            wrist_prediction_error = float(
                np.linalg.norm(
                    raw_target_wrist
                    - predicted_wrist
                )
            )
        
            # Wrist is more vulnerable to occlusion.
            #
            # Start distrusting at 60 mm.
            # Fully reject at 120 mm.
            wrist_conf = target_confidence(
                wrist_prediction_error,
                0.060,
                0.120,
            )
        
            target_wrist = (
                wrist_conf
                * raw_target_wrist
                + (1.0 - wrist_conf)
                * predicted_wrist
            )
        
            if wrist_conf < 0.999:
                low_conf_wrist_frames.append(
                    (
                        frame_i,
                        wrist_prediction_error,
                        wrist_conf,
                    )
                )
        
        
        # Explicitly diagnose the old V1.1 failure frame.
        if frame_i == 422:
            print(
                f"{side} frame 422 target confidence | "
                f"elbow={elbow_conf:.3f} "
                f"err={1000.0 * elbow_prediction_error:.2f} mm | "
                f"wrist={wrist_conf:.3f} "
                f"err={1000.0 * wrist_prediction_error:.2f} mm"
            )
        
        
        # Update FILTERED target velocity.
        if prev_target_elbow is None:
            next_elbow_delta = None
        else:
            next_elbow_delta = (
                np.asarray(
                    target_elbow,
                    dtype=np.float64,
                )
                - prev_target_elbow
            )
        
        if prev_target_wrist is None:
            next_wrist_delta = None
        else:
            next_wrist_delta = (
                np.asarray(
                    target_wrist,
                    dtype=np.float64,
                )
                - prev_target_wrist
            )
        
        prev_target_elbow = np.asarray(
            target_elbow,
            dtype=np.float64,
        ).copy()
        
        prev_target_wrist = np.asarray(
            target_wrist,
            dtype=np.float64,
        ).copy()
        
        prev_target_elbow_delta = (
            None
            if next_elbow_delta is None
            else next_elbow_delta.copy()
        )
        
        prev_target_wrist_delta = (
            None
            if next_wrist_delta is None
            else next_wrist_delta.copy()
        )
        
        last_target_frame = frame_i

        # ======================================================
        # V1.4 EXTENSION GATE
        #
        # Human bend:
        #   >= 30 deg -> no explicit extension constraint
        #   30 -> 8 deg -> progressively stronger
        #   <= 8 deg -> full extension constraint
        #
        # If elbow/wrist target reliability drops, weaken this
        # constraint rather than trusting an occluded arm.
        # ======================================================
        
        human_elbow_bend_deg = float(
            human_bend_deg_seq[i]
        )
        
        extension_gate = float(
            np.clip(
                (
                    45.0
                    - human_elbow_bend_deg
                )
                / (
                    45.0 - 15.0
                ),
                0.0,
                1.0,
            )
        )
        
        extension_gate *= float(
            min(
                elbow_conf,
                wrist_conf,
            )
        )
        
        # Boundary bridge frames prioritize returning to the
        # nearby V1 branch smoothly instead of continuing to
        # chase a now-inactive spatial target.
        bridge_track_scale = (
            0.20
            if is_boundary_bridge
            else 1.0
        )
        
        bridge_v1_scale = (
            4.0
            if is_boundary_bridge
            else 1.0
        )


        def residual(x):

            q = base_q.copy()

            q[
                arm_indices
            ] = x


            data.qpos[:] = q

            mujoco.mj_forward(
                model,
                data,
            )


            actual_elbow = (
                data.xpos[
                    elbow_id
                ]
            )

            actual_wrist = (
                data.xpos[
                    wrist_id
                ]
            )


            # Main skeleton.
            r_elbow = (
                (ELBOW_WEIGHT * bridge_track_scale)
                * (
                    actual_elbow
                    - target_elbow
                )
            )

            r_wrist = (
                (WRIST_WEIGHT * bridge_track_scale)
                * (
                    actual_wrist
                    - target_wrist
                )
            )


            # Stay close to the already visually meaningful
            # Arm V1 posture.
            r_v1 = (
                (V1_ARM_POSTURE_WEIGHT * bridge_v1_scale)
                * (
                    x - v1_seed
                )
            )

            # V1.5:
            # Human straight-arm shape has priority over the
            # old V1 bent-elbow posture.
            r_v1[
                elbow_local_index
            ] *= max(
                0.03,
                1.0
                - 0.97
                * extension_gate,
            )


            # Additional wrist-posture preservation.
            r_v1_wrist = (
                (V1_WRIST_POSTURE_WEIGHT * bridge_v1_scale)
                * (
                    x[4:7]
                    - v1_seed[4:7]
                )
            )


            # Temporal continuity.
            r_cont = (
                CONTINUITY_WEIGHT
                * (
                    x - prev
                )
            )

            # V1.5:
            # Let the elbow progressively open while still
            # respecting the hard trust-region bounds.
            r_cont[
                elbow_local_index
            ] *= max(
                0.15,
                1.0
                - 0.85
                * extension_gate,
            )


            # --------------------------------------------------
            # Geometry clearance.
            #
            # One residual per BODY PAIR, not per geom pair,
            # preventing duplicate mesh geoms from multiplying
            # the penalty.
            # --------------------------------------------------

            distances = (
                all_clearance_distances(
                    model,
                    data,
                    specs,
                )
            )

            violation = np.maximum(
                0.0,
                CLEARANCE
                - distances,
            )

            r_collision = (
                COLLISION_WEIGHT
                * violation
            )



            # ======================================================
            # V1.2 acceleration-like temporal continuity
            #
            # V1.1 already contains r_cont = position continuity.
            # This new term prevents sudden IK branch switching.
            # ======================================================
            
            if prev_delta is None:
                r_accel = np.zeros(
                    7,
                    dtype=np.float64,
                )
            else:
                delta_now = x - prev
            
                predicted_delta = (
                    0.70 * prev_delta
                )
            
                accel_like = (
                    delta_now
                    - predicted_delta
                )
            
                r_accel = (
                    1.25
                    * temporal_scale
                    * accel_like
                )

                r_accel[
                    elbow_local_index
                ] *= max(
                    0.25,
                    1.0
                    - 0.75
                    * extension_gate,
                )

            # ======================================================
            # V1.4 ROBOT ARM SHAPE / ELBOW EXTENSION
            #
            # Do not rely only on elbow/wrist XYZ.
            # When the human arm is straight, explicitly align:
            #
            #   robot shoulder -> elbow
            #              with
            #   robot elbow -> wrist
            #
            # This is independent of left/right elbow joint sign.
            # ======================================================
            
            actual_shoulder = np.asarray(
                data.xpos[
                    shoulder_id
                ],
                dtype=np.float64,
            )
            
            robot_upper_dir = normalize(
                actual_elbow
                - actual_shoulder
            )
            
            robot_fore_dir = normalize(
                actual_wrist
                - actual_elbow
            )
            
            r_extension = (
                5.0
                * extension_gate
                * (
                    robot_upper_dir
                    - robot_fore_dir
                )
            )
            # ==================================================
            # V1.5 DIRECT ELBOW-JOINT EXTENSION
            #
            # Geometric collinearity alone can still compromise
            # with the old posture.
            #
            # Explicitly drive the actual elbow joint toward the
            # model-calibrated straight configuration.
            # ==================================================

            r_elbow_joint_extension = np.asarray(
                [
                    15.0
                    * extension_gate
                    * (
                        x[
                            elbow_local_index
                        ]
                        - straight_elbow_q
                    )
                ],
                dtype=np.float64,
            )

            return np.concatenate(
                (
                    r_elbow,
                    r_wrist,
                    r_extension,
                    r_elbow_joint_extension,
                    r_v1,
                    r_v1_wrist,
                    r_cont,
                    r_accel,
                    r_collision,
                )
            )


        # Starting from Arm V1 normally gives the optimizer
        # the correct IK branch.
        x0 = np.minimum(
            np.maximum(
                v1_seed,
                low,
            ),
            high,
        )

        # ======================================================
        # V1.2 HARD TRUST REGION
        #
        # IMPORTANT:
        # This is inside least_squares bounds.
        # It is NOT post-hoc qpos clipping.
        # ======================================================
        
        solve_low = np.maximum(
            low,
            prev - hard_step,
        )
        
        solve_high = np.minimum(
            high,
            prev + hard_step,
        )
        
        # Keep initial guess strictly feasible.
        x0 = np.minimum(
            np.maximum(
                x0,
                solve_low + 1.0e-8,
            ),
            solve_high - 1.0e-8,
        )


        sol = least_squares(
            residual,
            x0,
            bounds=(solve_low, solve_high),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=MAX_NFEV,
        )

        # ======================================================
        # V1.3 COLLISION EMERGENCY RE-SOLVE
        #
        # First keep the normal V1.2 continuity limit:
        #   proximal = 0.090 rad
        #   wrist    = 0.100 rad
        #
        # Only if the resulting pose STILL penetrates another
        # body do we allow a slightly larger one-frame search.
        # ======================================================
        
        # Force MuJoCo state to the actual chosen normal solution.
        _ = residual(
            sol.x
        )
        
        normal_min_clearance = float(
            np.min(
                all_clearance_distances(
                    model,
                    data,
                    specs,
                )
            )
        )
        
        if normal_min_clearance < 0.0:
            # Small emergency relaxation only.
            #
            # proximal:
            #   0.120 rad = 6.88 deg
            #
            # distal wrist:
            #   0.140 rad = 8.02 deg
            emergency_step = np.asarray(
                [
                    0.120,
                    0.120,
                    0.120,
                    0.120,
                    0.140,
                    0.140,
                    0.140,
                ],
                dtype=np.float64,
            )
        
            em_low = np.maximum(
                low,
                prev - emergency_step,
            )
        
            em_high = np.minimum(
                high,
                prev + emergency_step,
            )
        
            em_x0 = np.minimum(
                np.maximum(
                    np.asarray(
                        sol.x,
                        dtype=np.float64,
                    ),
                    em_low + 1.0e-8,
                ),
                em_high - 1.0e-8,
            )
        
            em_sol = least_squares(
                        residual,
                        em_x0,
                        bounds=(em_low, em_high),
                        method="trf",
                        ftol=1e-8,
                        xtol=1e-8,
                        gtol=1e-8,
                        max_nfev=MAX_NFEV,
                    )
        
            # Evaluate actual geometry at emergency solution.
            _ = residual(
                em_sol.x
            )
        
            emergency_min_clearance = float(
                np.min(
                    all_clearance_distances(
                        model,
                        data,
                        specs,
                    )
                )
            )
        
            if (
                emergency_min_clearance
                > normal_min_clearance
            ):
                print(
                    f"{side} frame {i:03d} "
                    f"V1.3 emergency collision solve | "
                    f"{1000.0 * normal_min_clearance:.2f} "
                    f"-> "
                    f"{1000.0 * emergency_min_clearance:.2f} mm"
                )
        
                sol = em_sol
        
            else:
                # Restore MuJoCo state to normal solution if
                # emergency solve was not actually better.
                _ = residual(
                    sol.x
                )


        result_qpos[
            i,
            arm_indices,
        ] = sol.x


        # Diagnostics.
        data.qpos[:] = result_qpos[i]

        mujoco.mj_forward(
            model,
            data,
        )


        elbow_error[i] = float(
            np.linalg.norm(
                data.xpos[
                    elbow_id
                ]
                - target_elbow
            )
        )

        wrist_error[i] = float(
            np.linalg.norm(
                data.xpos[
                    wrist_id
                ]
                - target_wrist
            )
        )



        # V1.2 save actual joint delta for next frame.
        delta_now_v12 = (
            np.asarray(
                sol.x,
                dtype=np.float64,
            )
            - prev
        )
        
        prev_delta = delta_now_v12.copy()
        prev = sol.x.copy()


        if (
            i == 0
            or (i + 1) % 50 == 0
            or i == len(frames) - 1
        ):

            distances = (
                all_clearance_distances(
                    model,
                    data,
                    specs,
                )
            )

            print(
                f"{side:5s} "
                f"frame {i:03d}/660 | "
                f"elbow={elbow_error[i]*1000:6.2f} mm | "
                f"wrist={wrist_error[i]*1000:6.2f} mm | "
                f"clear={np.min(distances)*1000:7.2f} mm"
            )



    print()
    print(
        f"{side} low-confidence elbow target frames:",
        len(low_conf_elbow_frames),
    )
    
    print(
        f"{side} low-confidence wrist target frames:",
        len(low_conf_wrist_frames),
    )
    
    if low_conf_elbow_frames:
        worst = max(
            low_conf_elbow_frames,
            key=lambda item: item[1],
        )
    
        print(
            f"{side} worst elbow target jump | "
            f"frame={worst[0]} | "
            f"error={1000.0 * worst[1]:.2f} mm | "
            f"confidence={worst[2]:.3f}"
        )
    
    if low_conf_wrist_frames:
        worst = max(
            low_conf_wrist_frames,
            key=lambda item: item[1],
        )
    
        print(
            f"{side} worst wrist target jump | "
            f"frame={worst[0]} | "
            f"error={1000.0 * worst[1]:.2f} mm | "
            f"confidence={worst[2]:.3f}"
        )

    print()
    print(
        f"{side} V1.4 boundary bridge frames:",
        len(boundary_bridge_frames),
    )
    
    if boundary_bridge_frames:
        print(
            f"{side} V1.4 boundary bridge sample:",
            boundary_bridge_frames[:40],
        )
    
    _straight_ids = np.flatnonzero(
        human_bend_deg_seq <= 15.0
    )
    
    print(
        f"{side} human strong-extension "
        f"(<=15deg) frames:",
        len(_straight_ids),
    )
    
    if len(_straight_ids) > 0:
    
        _robot_bends = []
    
        for _idx in _straight_ids:
    
            data.qpos[:] = (
                result_qpos[
                    int(_idx)
                ]
            )
    
            mujoco.mj_forward(
                model,
                data,
            )
    
            _s = np.asarray(
                data.xpos[
                    shoulder_id
                ],
                dtype=np.float64,
            )
    
            _e = np.asarray(
                data.xpos[
                    elbow_id
                ],
                dtype=np.float64,
            )
    
            _w = np.asarray(
                data.xpos[
                    wrist_id
                ],
                dtype=np.float64,
            )
    
            _ru = normalize(
                _e - _s
            )
    
            _rf = normalize(
                _w - _e
            )
    
            _c = float(
                np.clip(
                    np.dot(
                        _ru,
                        _rf,
                    ),
                    -1.0,
                    1.0,
                )
            )
    
            _robot_bends.append(
                float(
                    np.degrees(
                        np.arccos(_c)
                    )
                )
            )
    
        _robot_bends = np.asarray(
            _robot_bends,
            dtype=np.float64,
        )
    
        print(
            f"{side} robot bend on human-straight "
            f"frames | "
            f"mean={_robot_bends.mean():.2f} deg | "
            f"median={np.median(_robot_bends):.2f} deg | "
            f"p95={np.percentile(_robot_bends,95):.2f} deg | "
            f"max={_robot_bends.max():.2f} deg"
        )
    return (
        result_qpos,
        specs,
        pre_min,
        active,
        elbow_error,
        wrist_error,
    )


# ======================================================================
# FINAL AUDIT
# ======================================================================

def audit_side(
    side,
    model,
    data,
    qpos,
    specs,
):

    min_dist = np.zeros(
        len(qpos),
        dtype=np.float64,
    )

    actual_contact_frames = []

    arm_body_ids = {
        body_id(
            model,
            f"{side}_elbow_link",
        ),
        body_id(
            model,
            f"{side}_wrist_roll_link",
        ),
        body_id(
            model,
            f"{side}_wrist_pitch_link",
        ),
        body_id(
            model,
            f"{side}_wrist_yaw_link",
        ),
    }

    target_body_ids = {
        body_id(model, "base_link"),
        body_id(model, "waist_yaw_link"),
        body_id(model, "waist_pitch_link"),

        body_id(model, "left_hip_pitch_link"),
        body_id(model, "left_hip_roll_link"),
        body_id(model, "left_hip_yaw_link"),

        body_id(model, "right_hip_pitch_link"),
        body_id(model, "right_hip_roll_link"),
        body_id(model, "right_hip_yaw_link"),
    }


    for i, q in enumerate(qpos):

        data.qpos[:] = q
        mujoco.mj_forward(
            model,
            data,
        )


        distances = (
            all_clearance_distances(
                model,
                data,
                specs,
            )
        )

        min_dist[i] = float(
            np.min(distances)
        )


        has_contact = False

        for ci in range(
            data.ncon
        ):

            con = data.contact[ci]

            b1 = int(
                model.geom_bodyid[
                    int(con.geom1)
                ]
            )

            b2 = int(
                model.geom_bodyid[
                    int(con.geom2)
                ]
            )


            if (
                (
                    b1 in arm_body_ids
                    and
                    b2 in target_body_ids
                )
                or
                (
                    b2 in arm_body_ids
                    and
                    b1 in target_body_ids
                )
            ):
                has_contact = True
                break


        if has_contact:
            actual_contact_frames.append(i)

    # V1.3 explicit frame IDs for any remaining failures.
    penetration_frame_ids = (
        np.flatnonzero(
            min_dist < 0.0
        )
        .astype(int)
        .tolist()
    )
    
    if penetration_frame_ids:
        print(
            f"{side} penetration frame ids:",
            penetration_frame_ids,
        )
    
    if actual_contact_frames:
        print(
            f"{side} actual contact frame ids:",
            actual_contact_frames,
        )


    print()
    print("=" * 110)
    print(
        side.upper(),
        "FINAL CLEARANCE AUDIT",
    )
    print("=" * 110)


    print(
        "penetration frames:",
        int(
            np.sum(
                min_dist < 0.0
            )
        ),
    )

    print(
        "below 15mm frames:",
        int(
            np.sum(
                min_dist < CLEARANCE
            )
        ),
    )

    print(
        "actual MuJoCo contact frames:",
        len(
            actual_contact_frames
        ),
    )

    print(
        "minimum signed distance:",
        float(
            np.min(min_dist)
            * 1000.0
        ),
        "mm",
    )


    return min_dist


def report_tracking(
    side,
    active,
    elbow_error,
    wrist_error,
):

    mask = active

    print()
    print("=" * 110)
    print(
        side.upper(),
        "TRACKING COST IN OPTIMIZED WINDOWS",
    )
    print("=" * 110)


    if not np.any(mask):

        print(
            "No active frames."
        )

        return


    for name, x in [
        (
            "elbow error mm",
            elbow_error[
                mask
            ]
            * 1000.0,
        ),
        (
            "wrist error mm",
            wrist_error[
                mask
            ]
            * 1000.0,
        ),
    ]:

        print(
            f"{name:20s} "
            f"mean={np.mean(x):7.2f}  "
            f"median={np.median(x):7.2f}  "
            f"p95={np.percentile(x,95):7.2f}  "
            f"max={np.max(x):7.2f}"
        )


def report_joint_changes(
    model,
    old,
    new,
    indices,
):

    print()
    print("=" * 110)
    print(
        "V1 -> V1.1 ARM JOINT CHANGE"
    )
    print("=" * 110)


    for qidx in indices:

        change = np.degrees(
            new[
                :,
                qidx,
            ]
            - old[
                :,
                qidx,
            ]
        )

        print(
            f"{joint_name(model,qidx):28s} "
            f"max_abs_change="
            f"{np.max(np.abs(change)):7.2f} deg"
        )


    steps = np.degrees(
        np.abs(
            np.diff(
                new[
                    :,
                    indices,
                ],
                axis=0,
            )
        )
    )

    where = np.unravel_index(
        np.argmax(steps),
        steps.shape,
    )

    qidx = int(
        indices[
            where[1]
        ]
    )


    print()
    print(
        "V1.5 max arm step:",
        float(
            steps[where]
        ),
        "deg",
    )

    print(
        "frame:",
        int(
            where[0] + 1
        ),
    )

    print(
        "joint:",
        joint_name(
            model,
            qidx,
        ),
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    v1 = np.loadtxt(
        V1_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert v1.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(v1)
    )


    (
        smplx_data,
        body_model,
        smplx_output,
        human_height,
    ) = load_smplx_file(
        str(SMPLX_FILE),
        str(BODY_MODEL_DIR),
        coord_fix="auto",
    )


    frames, fps = (
        get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=30,
        )
    )

    assert len(frames) == 661


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(
        model
    )


    print("=" * 110)
    print(
        "LINGLONG ARM V1.5 STRONG EXTENSION IK"
    )
    print("=" * 110)

    print(
        "frames:",
        len(frames),
    )

    print(
        "human height:",
        human_height,
    )

    print(
        "fps metadata:",
        fps,
    )

    print()

    print(
        "clearance:",
        CLEARANCE
        * 1000.0,
        "mm",
    )

    print(
        "activation distance:",
        ACTIVATION_DISTANCE
        * 1000.0,
        "mm",
    )

    print(
        "collision weight:",
        COLLISION_WEIGHT,
    )


    # =========================================================
    # LEFT
    # =========================================================

    (
        q_left,
        left_specs,
        left_pre,
        left_active,
        left_elbow_error,
        left_wrist_error,
    ) = solve_side(
        "left",
        model,
        data,
        frames,
        v1.copy(),
        LEFT_ARM,
    )


    # =========================================================
    # RIGHT
    #
    # Start from already corrected left side.
    # =========================================================

    (
        q_final,
        right_specs,
        right_pre,
        right_active,
        right_elbow_error,
        right_wrist_error,
    ) = solve_side(
        "right",
        model,
        data,
        frames,
        q_left.copy(),
        RIGHT_ARM,
    )


    # =========================================================
    # VERIFY NON-ARM PRESERVATION
    # =========================================================

    non_arm_diff = float(
        np.max(
            np.abs(
                q_final[
                    :,
                    :23,
                ]
                - v1[
                    :,
                    :23,
                ]
            )
        )
    )


    print()
    print("=" * 110)
    print(
        "NON-ARM PRESERVATION"
    )
    print("=" * 110)

    print(
        "qpos[0:23] max diff:",
        non_arm_diff,
    )

    assert non_arm_diff < 1e-12


    # =========================================================
    # TRACKING REPORT
    # =========================================================

    report_tracking(
        "left",
        left_active,
        left_elbow_error,
        left_wrist_error,
    )

    report_tracking(
        "right",
        right_active,
        right_elbow_error,
        right_wrist_error,
    )


    # =========================================================
    # FINAL COLLISION AUDIT
    # =========================================================

    left_post = audit_side(
        "left",
        model,
        data,
        q_final,
        left_specs,
    )

    right_post = audit_side(
        "right",
        model,
        data,
        q_final,
        right_specs,
    )


    print()
    print("=" * 110)
    print(
        "PRE -> POST CLEARANCE"
    )
    print("=" * 110)

    print(
        "LEFT:"
    )

    print(
        "  penetration frames:",
        int(
            np.sum(
                left_pre < 0.0
            )
        ),
        "->",
        int(
            np.sum(
                left_post < 0.0
            )
        ),
    )

    print(
        "  min distance:",
        float(
            np.min(left_pre)
            * 1000.0
        ),
        "->",
        float(
            np.min(left_post)
            * 1000.0
        ),
        "mm",
    )


    print(
        "RIGHT:"
    )

    print(
        "  penetration frames:",
        int(
            np.sum(
                right_pre < 0.0
            )
        ),
        "->",
        int(
            np.sum(
                right_post < 0.0
            )
        ),
    )

    print(
        "  min distance:",
        float(
            np.min(right_pre)
            * 1000.0
        ),
        "->",
        float(
            np.min(right_post)
            * 1000.0
        ),
        "mm",
    )


    # =========================================================
    # ARM CHANGE / CONTINUITY
    # =========================================================

    report_joint_changes(
        model,
        v1,
        q_final,
        np.concatenate(
            (
                LEFT_ARM,
                RIGHT_ARM,
            )
        ),
    )


    assert q_final.shape == (
        661,
        37,
    )

    assert np.all(
        np.isfinite(q_final)
    )


    np.savetxt(
        OUT_CSV,
        q_final,
        delimiter=",",
        fmt="%.12f",
    )


    print()
    print("=" * 110)
    print(
        "OUTPUT"
    )
    print("=" * 110)

    print(
        "shape:",
        q_final.shape,
    )

    print(
        "finite:",
        bool(
            np.all(
                np.isfinite(q_final)
            )
        ),
    )

    print(
        "saved:",
        OUT_CSV,
    )

    print()

    print(
        "LINGLONG ARM V1.1 "
        "COLLISION-AWARE IK: PASS"
    )


if __name__ == "__main__":
    main()
