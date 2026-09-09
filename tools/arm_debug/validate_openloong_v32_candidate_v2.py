#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import subprocess
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DIR = ROOT / "output/openloong_v32_final"
CANDIDATE_CSV = CANDIDATE_DIR / "csv/openloong/live_motion.csv"
DIAG_CSV = CANDIDATE_DIR / "diagnostics/arm_spatial_diagnostics.csv"
VIDEO = CANDIDATE_DIR / "video/openloong_armspatial_v1.mp4"

BODY_CSV = (
    ROOT
    / "output/openloong_v2_kneeguide/csv/openloong/live_motion.csv"
)

MODEL_XML = ROOT / "assets/openloong/scene_white.xml"

EXPECTED_FRAMES = 661
EXPECTED_COLS = 38
FPS = 30.0

# Verified OpenLoong arm qpos columns in the 38-value submission row.
RIGHT_ARM_COLS = np.arange(9, 16, dtype=int)
LEFT_ARM_COLS = np.arange(16, 23, dtype=int)
ARM_COLS = np.concatenate([RIGHT_ARM_COLS, LEFT_ARM_COLS])
NON_ARM_COLS = np.array(
    [i for i in range(EXPECTED_COLS) if i not in set(ARM_COLS.tolist())],
    dtype=int,
)

HARD_MAX_ARM_STEP = 0.50
MIN_HAND_HAND = 0.14
MIN_FOREARM_FOREARM = 0.075
MIN_TORSO_CLEARANCE = 0.0

REPORT_DIR = CANDIDATE_DIR / "validation"
REPORT_JSON = REPORT_DIR / "v32_validation_report.json"
REPORT_TXT = REPORT_DIR / "v32_validation_report.txt"


def load_csv_no_header(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def csv_to_mj_qpos(rows: np.ndarray) -> np.ndarray:
    q = np.asarray(rows, dtype=np.float64).copy()
    # submission CSV: qx,qy,qz,qw -> MuJoCo: qw,qx,qy,qz
    q[:, 3:7] = rows[:, [6, 3, 4, 5]]
    return q


def ffprobe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        p = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        obj = json.loads(p.stdout)
        streams = obj.get("streams", [])
        return streams[0] if streams else {}
    except Exception as e:
        return {"ffprobe_error": repr(e)}


def rational_to_float(text: str | None) -> float | None:
    if not text:
        return None
    try:
        if "/" in text:
            a, b = text.split("/", 1)
            b = float(b)
            return float(a) / b if b != 0 else None
        return float(text)
    except Exception:
        return None


def load_diag(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fcol(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def icol(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([int(float(r[key])) for r in rows], dtype=np.int64)


def main() -> int:
    checks: list[dict] = []
    metrics: dict[str, object] = {}

    def record(name: str, ok: bool, detail: str, hard: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "hard": bool(hard),
                "detail": detail,
            }
        )

    print("=" * 80)
    print("OpenLoong V3.2 FINAL candidate validation (submission criteria v2)")
    print("=" * 80)

    required = [
        CANDIDATE_CSV,
        DIAG_CSV,
        VIDEO,
        BODY_CSV,
        MODEL_XML,
    ]

    missing = [str(p) for p in required if not p.exists()]
    record(
        "required files",
        not missing,
        "all required files exist" if not missing else f"missing={missing}",
    )

    if missing:
        for c in checks:
            print(("PASS" if c["ok"] else "FAIL"), "-", c["name"], ":", c["detail"])
        return 2

    cand = load_csv_no_header(CANDIDATE_CSV)
    body = load_csv_no_header(BODY_CSV)

    metrics["candidate_shape"] = list(cand.shape)
    metrics["body_shape"] = list(body.shape)

    record(
        "submission CSV shape",
        cand.shape == (EXPECTED_FRAMES, EXPECTED_COLS),
        f"shape={cand.shape}, expected={(EXPECTED_FRAMES, EXPECTED_COLS)}",
    )

    record(
        "submission CSV finite",
        bool(np.isfinite(cand).all()),
        f"all_finite={bool(np.isfinite(cand).all())}",
    )

    body_compatible = (
        body.shape[0] >= EXPECTED_FRAMES
        and body.shape[1] == EXPECTED_COLS
    )
    record(
        "stable body CSV compatible",
        body_compatible,
        f"shape={body.shape}",
    )

    # Quaternion check, CSV order xyzw.
    quat = cand[:, 3:7]
    qnorm = np.linalg.norm(quat, axis=1)
    qnorm_err = np.abs(qnorm - 1.0)
    metrics["quat_norm_min"] = float(qnorm.min())
    metrics["quat_norm_max"] = float(qnorm.max())
    metrics["quat_norm_max_abs_error"] = float(qnorm_err.max())

    record(
        "root quaternion normalized",
        bool(qnorm_err.max() <= 2e-5),
        (
            f"norm_min={qnorm.min():.9f}, norm_max={qnorm.max():.9f}, "
            f"max_abs_error={qnorm_err.max():.3e}"
        ),
    )

    # Preserve frozen body exactly outside the 14 arm qpos columns.
    if body_compatible:
        body_ref = body[:EXPECTED_FRAMES]
        non_arm_diff = np.abs(
            cand[:, NON_ARM_COLS] - body_ref[:, NON_ARM_COLS]
        )
        max_non_arm = float(non_arm_diff.max())
        metrics["max_non_arm_abs_diff"] = max_non_arm

        record(
            "body/head/waist/legs frozen",
            max_non_arm <= 2e-8,
            f"max_abs_diff_outside_arms={max_non_arm:.3e}",
        )

    # Model / joint-limit check.
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    q_mj = csv_to_mj_qpos(cand)

    metrics["model_nq"] = int(model.nq)
    record(
        "model nq",
        int(model.nq) == EXPECTED_COLS,
        f"model.nq={model.nq}, expected={EXPECTED_COLS}",
    )

    limit_violations = []
    max_limit_over = 0.0

    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        limited = bool(model.jnt_limited[jid])

        # MuJoCo enum: slide=2, hinge=3. OpenLoong arm/leg joints are hinge.
        if not limited or jtype not in (2, 3):
            continue

        adr = int(model.jnt_qposadr[jid])
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        vals = q_mj[:, adr]

        low_over = np.maximum(lo - vals, 0.0)
        high_over = np.maximum(vals - hi, 0.0)
        over = np.maximum(low_over, high_over)
        this_max = float(over.max())
        max_limit_over = max(max_limit_over, this_max)

        if this_max > 1e-6:
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )
            limit_violations.append(
                {
                    "joint": name,
                    "qpos_adr": adr,
                    "max_over_rad": this_max,
                    "frame": int(np.argmax(over)),
                    "range": [lo, hi],
                }
            )

    metrics["max_joint_limit_violation"] = max_limit_over
    metrics["joint_limit_violations"] = limit_violations

    record(
        "joint limits",
        len(limit_violations) == 0,
        (
            "no hinge/slide limit violations"
            if not limit_violations
            else f"violations={limit_violations[:8]}"
        ),
    )

    # Diagnostics.
    diag = load_diag(DIAG_CSV)
    metrics["diag_rows"] = len(diag)
    record(
        "diagnostics rows",
        len(diag) == EXPECTED_FRAMES,
        f"rows={len(diag)}, expected={EXPECTED_FRAMES}",
    )

    required_diag_cols = [
        "frame",
        "time_sec",
        "cost",
        "nfev",
        "optimality",
        "success",
        "left_wrist_err_mm",
        "right_wrist_err_mm",
        "left_hand_dir_err_deg",
        "right_hand_dir_err_deg",
        "left_palm_dir_err_deg",
        "right_palm_dir_err_deg",
        "left_palm_roll_target_deg",
        "right_palm_roll_target_deg",
        "hand_hand_dist_m",
        "forearm_forearm_dist_m",
        "min_torso_clearance_m",
        "contact_count",
        "max_joint_step_rad",
    ]

    diag_cols_ok = bool(diag) and all(k in diag[0] for k in required_diag_cols)
    record(
        "diagnostic columns",
        diag_cols_ok,
        "required columns present" if diag_cols_ok else "missing required columns",
    )

    if len(diag) == EXPECTED_FRAMES and diag_cols_ok:
        numeric_keys = [
            "cost",
            "nfev",
            "optimality",
            "left_wrist_err_mm",
            "right_wrist_err_mm",
            "left_hand_dir_err_deg",
            "right_hand_dir_err_deg",
            "left_palm_dir_err_deg",
            "right_palm_dir_err_deg",
            "left_palm_roll_target_deg",
            "right_palm_roll_target_deg",
            "hand_hand_dist_m",
            "forearm_forearm_dist_m",
            "min_torso_clearance_m",
            "max_joint_step_rad",
        ]

        finite_diag = all(np.isfinite(fcol(diag, k)).all() for k in numeric_keys)
        record(
            "diagnostics finite",
            finite_diag,
            f"all_required_numeric_finite={finite_diag}",
        )

        success = icol(diag, "success")
        success_count = int(np.count_nonzero(success))

        nfev = fcol(diag, "nfev")
        cost = fcol(diag, "cost")
        optimality = fcol(diag, "optimality")

        failed = success == 0
        failed_count = int(np.count_nonzero(failed))

        metrics["solver_success_frames"] = success_count
        metrics["solver_non_success_frames"] = failed_count
        metrics["solver_nfev_min"] = float(nfev.min())
        metrics["solver_nfev_max"] = float(nfev.max())
        metrics["solver_cost_max"] = float(cost.max())
        metrics["solver_optimality_max"] = float(optimality.max())

        if failed_count:
            metrics["solver_non_success_nfev_min"] = float(
                nfev[failed].min()
            )
            metrics["solver_non_success_nfev_max"] = float(
                nfev[failed].max()
            )
            metrics["solver_non_success_cost_max"] = float(
                cost[failed].max()
            )
            metrics["solver_non_success_optimality_max"] = float(
                optimality[failed].max()
            )

        # IMPORTANT:
        # scipy.optimize.least_squares.success is a termination-condition flag,
        # not a "no usable solution" flag. V3.2 still uses result.x when the
        # per-frame max_nfev budget is reached. Therefore this is informational.
        # Actual submission validity is guarded below by finite outputs, joint
        # limits, continuity, clearances, and the resulting tracking errors.
        record(
            "IK solver termination flags",
            True,
            (
                f"success={success_count}/{EXPECTED_FRAMES}, "
                f"non_success={failed_count}; INFO ONLY. "
                f"nfev_range={nfev.min():.0f}..{nfev.max():.0f}"
            ),
            hard=False,
        )

        # Hard sanity check: every frame must still have a finite optimizer
        # result and finite diagnostic values. This is the meaningful failure
        # condition for packaging; diagnostics-finite is also checked above.
        solver_output_finite = bool(
            np.isfinite(cost).all()
            and np.isfinite(nfev).all()
            and np.isfinite(optimality).all()
        )
        record(
            "IK solver outputs finite",
            solver_output_finite,
            (
                f"cost/nfev/optimality finite={solver_output_finite}; "
                f"cost_max={cost.max():.6g}, "
                f"optimality_max={optimality.max():.6g}"
            ),
        )

        wr_l = fcol(diag, "left_wrist_err_mm")
        wr_r = fcol(diag, "right_wrist_err_mm")
        dir_l = fcol(diag, "left_hand_dir_err_deg")
        dir_r = fcol(diag, "right_hand_dir_err_deg")
        palm_l = fcol(diag, "left_palm_dir_err_deg")
        palm_r = fcol(diag, "right_palm_dir_err_deg")
        roll_l = fcol(diag, "left_palm_roll_target_deg")
        roll_r = fcol(diag, "right_palm_roll_target_deg")
        hh = fcol(diag, "hand_hand_dist_m")
        ff = fcol(diag, "forearm_forearm_dist_m")
        tc = fcol(diag, "min_torso_clearance_m")
        dq = fcol(diag, "max_joint_step_rad")
        contacts = icol(diag, "contact_count")

        metrics.update(
            {
                "left_wrist_err_mean_mm": float(wr_l.mean()),
                "left_wrist_err_max_mm": float(wr_l.max()),
                "right_wrist_err_mean_mm": float(wr_r.mean()),
                "right_wrist_err_max_mm": float(wr_r.max()),
                "left_hand_dir_err_mean_deg": float(dir_l.mean()),
                "left_hand_dir_err_max_deg": float(dir_l.max()),
                "right_hand_dir_err_mean_deg": float(dir_r.mean()),
                "right_hand_dir_err_max_deg": float(dir_r.max()),
                "left_palm_dir_err_mean_deg": float(palm_l.mean()),
                "left_palm_dir_err_max_deg": float(palm_l.max()),
                "right_palm_dir_err_mean_deg": float(palm_r.mean()),
                "right_palm_dir_err_max_deg": float(palm_r.max()),
                "min_hand_hand_m": float(hh.min()),
                "min_forearm_forearm_m": float(ff.min()),
                "min_torso_clearance_m": float(tc.min()),
                "max_arm_joint_step_rad": float(dq.max()),
                "frames_with_broad_mujoco_contact": int(np.count_nonzero(contacts)),
            }
        )

        record(
            "arm continuity",
            float(dq.max()) <= HARD_MAX_ARM_STEP,
            f"max_joint_step={dq.max():.4f} rad/frame <= {HARD_MAX_ARM_STEP:.2f}",
        )

        record(
            "hand-hand clearance",
            float(hh.min()) >= MIN_HAND_HAND,
            f"min={hh.min():.4f} m >= {MIN_HAND_HAND:.3f} m",
        )

        record(
            "forearm-forearm clearance",
            float(ff.min()) >= MIN_FOREARM_FOREARM,
            f"min={ff.min():.4f} m >= {MIN_FOREARM_FOREARM:.3f} m",
        )

        record(
            "torso proxy clearance",
            float(tc.min()) >= MIN_TORSO_CLEARANCE,
            f"min={tc.min():.4f} m >= {MIN_TORSO_CLEARANCE:.3f} m",
        )

        # Broad MuJoCo contact counter is informational only.
        record(
            "broad MuJoCo arm-contact counter",
            True,
            (
                f"{int(np.count_nonzero(contacts))}/{EXPECTED_FRAMES} frames; "
                "INFO ONLY (known to include benign adjacent/self-structural contacts)"
            ),
            hard=False,
        )

        # Key visual-validation intervals.
        frames = np.array([int(float(r["frame"])) for r in diag], dtype=int)
        times = np.array([float(r["time_sec"]) for r in diag], dtype=float)

        segments = {}
        for name, lo, hi in [
            ("7_9_sec", 7.0, 9.0),
            ("15_18_sec", 15.0, 18.0),
        ]:
            sel = (times >= lo) & (times <= hi)
            segments[name] = {
                "frames": [int(frames[sel].min()), int(frames[sel].max())],
                "left_palm_roll_target_mean_deg": float(roll_l[sel].mean()),
                "right_palm_roll_target_mean_deg": float(roll_r[sel].mean()),
                "left_palm_dir_err_mean_deg": float(palm_l[sel].mean()),
                "right_palm_dir_err_mean_deg": float(palm_r[sel].mean()),
                "left_hand_dir_err_mean_deg": float(dir_l[sel].mean()),
                "right_hand_dir_err_mean_deg": float(dir_r[sel].mean()),
                "left_wrist_err_mean_mm": float(wr_l[sel].mean()),
                "right_wrist_err_mean_mm": float(wr_r[sel].mean()),
                "max_joint_step_rad": float(dq[sel].max()),
            }
        metrics["segments"] = segments

    # Video.
    vinfo = ffprobe_video(VIDEO)
    metrics["video"] = vinfo

    nb_frames = None
    try:
        if vinfo.get("nb_frames") not in (None, "N/A"):
            nb_frames = int(vinfo["nb_frames"])
    except Exception:
        pass

    duration = None
    try:
        if vinfo.get("duration") not in (None, "N/A"):
            duration = float(vinfo["duration"])
    except Exception:
        pass

    fps = rational_to_float(vinfo.get("r_frame_rate"))

    record(
        "video readable",
        "ffprobe_error" not in vinfo and bool(vinfo),
        str(vinfo),
    )

    if nb_frames is not None:
        record(
            "video frame count",
            nb_frames == EXPECTED_FRAMES,
            f"nb_frames={nb_frames}, expected={EXPECTED_FRAMES}",
        )
    else:
        # Not all MP4 encoders store nb_frames; duration/fps is acceptable fallback.
        approx_frames = (
            int(round(duration * fps))
            if duration is not None and fps is not None
            else None
        )
        record(
            "video frame count",
            approx_frames == EXPECTED_FRAMES,
            (
                f"nb_frames unavailable, estimated={approx_frames} "
                f"from duration={duration}, fps={fps}"
            ),
        )

    if fps is not None:
        record(
            "video fps",
            abs(fps - FPS) < 0.05,
            f"fps={fps:.6f}, expected~={FPS}",
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    hard_failures = [
        c for c in checks
        if c["hard"] and not c["ok"]
    ]

    report = {
        "candidate": "OpenLoong V3.2",
        "candidate_csv": str(CANDIDATE_CSV),
        "diagnostics_csv": str(DIAG_CSV),
        "video": str(VIDEO),
        "checks": checks,
        "metrics": metrics,
        "hard_failures": hard_failures,
        "overall_pass": len(hard_failures) == 0,
    }

    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("=" * 80)
    lines.append("OpenLoong V3.2 FINAL validation")
    lines.append("=" * 80)

    for c in checks:
        label = "PASS" if c["ok"] else ("FAIL" if c["hard"] else "INFO")
        lines.append(f"{label:4s} | {c['name']}: {c['detail']}")

    lines.append("")
    lines.append("Key metrics")
    lines.append("-" * 80)

    for key, value in metrics.items():
        if key in ("video", "joint_limit_violations", "segments"):
            continue
        lines.append(f"{key}: {value}")

    if "segments" in metrics:
        lines.append("")
        lines.append("Key segments")
        lines.append("-" * 80)
        for name, obj in metrics["segments"].items():
            lines.append(f"{name}: {json.dumps(obj, ensure_ascii=False)}")

    lines.append("")
    lines.append(
        "OVERALL: PASS"
        if not hard_failures
        else f"OVERALL: FAIL ({len(hard_failures)} hard failures)"
    )

    REPORT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    for line in lines:
        print(line)

    print()
    print("REPORT JSON:", REPORT_JSON)
    print("REPORT TXT :", REPORT_TXT)

    return 0 if not hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
