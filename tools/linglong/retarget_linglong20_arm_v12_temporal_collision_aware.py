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
    / "output/linglong20_arm_v12_temporal_collision_aware"
)

OUT_CSV = (
    OUT_DIR
    / "linglong20_arm_v12_temporal_collision_aware_qpos37.csv"
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
        "ARM V1.1 COLLISION-AWARE SOLVE",
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
    
    for i, frame in enumerate(frames):

        v1_seed = input_qpos[
            i,
            arm_indices,
        ].copy()


        # ------------------------------------------------------
        # Outside the expanded danger window:
        # preserve V1 exactly.
        # ------------------------------------------------------

        if not active[i]:

            result_qpos[
                i,
                arm_indices,
            ] = v1_seed

            prev = v1_seed.copy()
            prev_delta = None

            continue


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
                ELBOW_WEIGHT
                * (
                    actual_elbow
                    - target_elbow
                )
            )

            r_wrist = (
                WRIST_WEIGHT
                * (
                    actual_wrist
                    - target_wrist
                )
            )


            # Stay close to the already visually meaningful
            # Arm V1 posture.
            r_v1 = (
                V1_ARM_POSTURE_WEIGHT
                * (
                    x - v1_seed
                )
            )


            # Additional wrist-posture preservation.
            r_v1_wrist = (
                V1_WRIST_POSTURE_WEIGHT
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
            return np.concatenate(
                (
                    r_elbow,
                    r_wrist,
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
        "V1.2 max arm step:",
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
        "LINGLONG ARM V1.2 TEMPORAL COLLISION-AWARE IK"
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
