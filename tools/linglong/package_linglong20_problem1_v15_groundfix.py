#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SRC_CSV = (
    ROOT
    / "output/linglong20_arm_v15_strong_extension"
    / "linglong20_arm_v15_strong_extension_qpos37.csv"
)

URDF = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0.urdf"
)

OUT_DIR = (
    ROOT
    / "output/linglong20_v15_groundfix"
)

FIXED_INTERNAL = (
    OUT_DIR
    / "linglong20_v15_groundfix_qpos37_wxyz.csv"
)

SUBMISSION_DIR = (
    ROOT
    / "submission/linglong20_problem1_v15_groundfix"
)

DST_CSV = (
    SUBMISSION_DIR
    / "linglong20_action_sequence.csv"
)

DST_VIDEO = (
    SUBMISSION_DIR
    / "linglong20_mujoco_demo.mp4"
)

DST_README = (
    SUBMISSION_DIR
    / "README.md"
)


FPS = 30
WIDTH = 640
HEIGHT = 480

# 最低脚部几何体保持在真实地面上方 5 mm。
CLEARANCE = 0.005

# root-z 修正每帧最多变化 4 mm。
# 30 FPS 下相当于 0.12 m/s。
MAX_LIFT_STEP = 0.004


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def build_model():

    spec = mujoco.MjSpec.from_file(
        str(URDF)
    )

    spec.compiler.meshdir = str(
        URDF.resolve().parent
    )

    base = spec.body(
        "base_link"
    )

    if base is None:
        raise RuntimeError(
            "base_link not found"
        )

    base.add_freejoint(
        name="submission_root"
    )

    floor = (
        spec.worldbody.add_geom()
    )

    floor.name = "ground_plane"
    floor.type = (
        mujoco.mjtGeom.mjGEOM_PLANE
    )

    floor.size = [
        5.0,
        5.0,
        0.1,
    ]

    # IMPORTANT:
    # GroundFix submission uses REAL z=0 ground.
    floor.pos = [
        0.0,
        0.0,
        0.0,
    ]

    floor.rgba = [
        0.35,
        0.35,
        0.35,
        1.0,
    ]

    model = spec.compile()

    if model.nq != 37:
        raise RuntimeError(
            f"Expected nq=37, got {model.nq}"
        )

    return model


def body_name(model, bid):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        bid,
    )

    return name or ""


def find_foot_geoms(model):

    plane = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "ground_plane",
    )

    left = []
    right = []

    print()
    print("=" * 90)
    print("FOOT / ANKLE GEOMETRY")
    print("=" * 90)

    for gid in range(
        model.ngeom
    ):

        if gid == plane:
            continue

        bid = int(
            model.geom_bodyid[gid]
        )

        name = body_name(
            model,
            bid,
        )

        lname = name.lower()

        if not any(
            key in lname
            for key in (
                "ankle",
                "foot",
                "toe",
            )
        ):
            continue

        print(
            f"geom={gid:3d} "
            f"body={name}"
        )

        if "left" in lname:
            left.append(gid)

        if "right" in lname:
            right.append(gid)

    if not left or not right:
        raise RuntimeError(
            "foot geoms not found"
        )

    print()
    print("left :", left)
    print("right:", right)

    return (
        plane,
        left,
        right,
    )


def geom_distance(
    model,
    data,
    gid,
    plane_gid,
):

    fromto = np.zeros(
        6,
        dtype=np.float64,
    )

    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            int(gid),
            int(plane_gid),
            2.0,
            fromto,
        )
    )


def measure_ground(
    model,
    qseq,
    plane_gid,
    left_geoms,
    right_geoms,
):

    data = mujoco.MjData(
        model
    )

    left = []
    right = []

    for q in qseq:

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data,
        )

        ld = min(
            geom_distance(
                model,
                data,
                gid,
                plane_gid,
            )
            for gid in left_geoms
        )

        rd = min(
            geom_distance(
                model,
                data,
                gid,
                plane_gid,
            )
            for gid in right_geoms
        )

        left.append(ld)
        right.append(rd)

    left = np.asarray(
        left,
        dtype=np.float64,
    )

    right = np.asarray(
        right,
        dtype=np.float64,
    )

    lowest = np.minimum(
        left,
        right,
    )

    return (
        left,
        right,
        lowest,
    )


def smooth_upper_envelope(
    required,
    max_step,
):
    """
    Minimal correction >= required while enforcing

        |lift[i+1] - lift[i]| <= max_step

    This avoids post-processing below the required
    anti-penetration correction.
    """

    required = np.asarray(
        required,
        dtype=np.float64,
    )

    n = len(required)

    idx = np.arange(
        n,
        dtype=np.float64,
    )

    out = np.empty(
        n,
        dtype=np.float64,
    )

    # Exact minimal Lipschitz majorant:
    #
    # lift[i] =
    #   max_j(required[j] - r * |i-j|)
    #
    # N=661, O(N^2) is trivial.
    for i in range(n):

        out[i] = np.max(
            required
            - max_step
            * np.abs(
                idx - i
            )
        )

    out = np.maximum(
        out,
        required,
    )

    return out


def make_groundfix(
    model,
    qsrc,
    plane_gid,
    left_geoms,
    right_geoms,
):

    (
        pre_left,
        pre_right,
        pre_low,
    ) = measure_ground(
        model,
        qsrc,
        plane_gid,
        left_geoms,
        right_geoms,
    )

    # Required vertical translation to ensure
    # every lowest foot/ankle point >= CLEARANCE.
    raw_lift = np.maximum(
        0.0,
        CLEARANCE
        - pre_low,
    )

    lift = smooth_upper_envelope(
        raw_lift,
        MAX_LIFT_STEP,
    )

    qfix = qsrc.copy()

    # ========================================================
    # ONLY ROOT Z IS MODIFIED.
    # ========================================================
    qfix[:, 2] += lift

    (
        post_left,
        post_right,
        post_low,
    ) = measure_ground(
        model,
        qfix,
        plane_gid,
        left_geoms,
        right_geoms,
    )

    print()
    print("=" * 90)
    print("GROUND FIX AUDIT")
    print("=" * 90)

    print(
        "pre lowest min mm    :",
        pre_low.min() * 1000.0,
    )

    print(
        "pre lowest median mm :",
        np.median(pre_low)
        * 1000.0,
    )

    print(
        "pre penetration frames:",
        int(
            np.sum(
                pre_low < 0
            )
        ),
    )

    print()
    print(
        "raw lift min/median/max mm:",
        raw_lift.min() * 1000.0,
        np.median(raw_lift) * 1000.0,
        raw_lift.max() * 1000.0,
    )

    print(
        "final lift min/median/max mm:",
        lift.min() * 1000.0,
        np.median(lift) * 1000.0,
        lift.max() * 1000.0,
    )

    print(
        "max lift step mm/frame:",
        np.max(
            np.abs(
                np.diff(lift)
            )
        ) * 1000.0,
    )

    print()
    print(
        "post lowest min mm    :",
        post_low.min() * 1000.0,
    )

    print(
        "post lowest median mm :",
        np.median(post_low)
        * 1000.0,
    )

    print(
        "post penetration frames:",
        int(
            np.sum(
                post_low < -1e-6
            )
        ),
    )

    # Everything except root-z must be identical.
    mask = np.ones(
        37,
        dtype=bool,
    )

    mask[2] = False

    other_diff = float(
        np.max(
            np.abs(
                qfix[:, mask]
                - qsrc[:, mask]
            )
        )
    )

    print()
    print(
        "non-root-z max diff:",
        other_diff,
    )

    if other_diff != 0.0:
        raise RuntimeError(
            "Unexpected change outside root z"
        )

    if post_low.min() < -1e-5:
        raise RuntimeError(
            "GroundFix still penetrates ground"
        )

    if (
        np.max(
            np.abs(
                np.diff(lift)
            )
        )
        > MAX_LIFT_STEP
        + 1e-9
    ):
        raise RuntimeError(
            "GroundFix lift is not smooth"
        )

    return (
        qfix,
        lift,
        post_low,
    )


def write_submission_csv(
    qfix,
):

    # Internal:
    # xyz + qw qx qy qz + 30 joints
    #
    # Submission:
    # xyz + qx qy qz qw + 30 joints

    out = qfix.copy()

    out[:, 3:7] = (
        qfix[
            :,
            [4, 5, 6, 3],
        ]
    )

    np.savetxt(
        DST_CSV,
        out,
        delimiter=",",
        fmt="%.9f",
    )

    check = np.loadtxt(
        DST_CSV,
        delimiter=",",
    )

    if check.shape != (
        661,
        37,
    ):
        raise RuntimeError(
            check.shape
        )

    if not np.isfinite(
        check
    ).all():
        raise RuntimeError(
            "NaN/Inf in submission"
        )

    return out


def render_video(
    model,
    qseq,
):

    data = mujoco.MjData(
        model
    )

    renderer = mujoco.Renderer(
        model,
        height=HEIGHT,
        width=WIDTH,
    )

    cam = mujoco.MjvCamera()

    mujoco.mjv_defaultCamera(
        cam
    )

    cam.distance = 3.0
    cam.azimuth = 145.0
    cam.elevation = -12.0

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(DST_VIDEO),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
    )

    try:

        for i, q in enumerate(
            qseq
        ):

            data.qpos[:] = q
            data.qvel[:] = 0

            mujoco.mj_forward(
                model,
                data,
            )

            cam.lookat[:] = [
                float(q[0]),
                float(q[1]),
                float(q[2]) + 0.55,
            ]

            renderer.update_scene(
                data,
                camera=cam,
            )

            frame = (
                renderer.render()
            )

            proc.stdin.write(
                np.ascontiguousarray(
                    frame,
                    dtype=np.uint8,
                ).tobytes()
            )

            if (
                i == 0
                or (i + 1) % 50 == 0
                or i == len(qseq) - 1
            ):
                print(
                    f"render "
                    f"{i:03d}/"
                    f"{len(qseq)-1}"
                )

    finally:

        renderer.close()

        if proc.stdin:
            proc.stdin.close()

    ret = proc.wait()

    if ret != 0:
        raise RuntimeError(
            "ffmpeg failed"
        )


def write_readme(
    lift,
    post_low,
):

    text = f"""# LingLong2.0 赛题一提交候选 - V1.5 GroundFix

本目录包含三个文件：

- `linglong20_action_sequence.csv`
- `linglong20_mujoco_demo.mp4`
- `README.md`

## 动作版本

基础动作：

`LingLong2.0 Arm Retarget V1.5 Strong Extension`

基础 checkpoint：

`3c70730`

基础 tag：

`linglong20-v15-checkpoint-20260911`

## GroundFix

本提交候选没有重新求解任何关节。

与冻结的 V1.5 相比：

- root x 不变；
- root y 不变；
- root quaternion 不变；
- 30 个机器人关节完全不变；
- 仅调整 floating root 的 z；
- 根据每帧脚/踝最低几何点计算向上修正；
- 修正使用速度受限的平滑上包络；
- MuJoCo 演示地面位于真实 z=0。

Ground clearance target：

`{CLEARANCE * 1000:.1f} mm`

Root-Z correction：

- minimum: `{lift.min() * 1000:.3f} mm`
- median: `{np.median(lift) * 1000:.3f} mm`
- maximum: `{lift.max() * 1000:.3f} mm`
- maximum step: `{np.max(np.abs(np.diff(lift))) * 1000:.3f} mm/frame`

Ground audit：

- minimum final clearance:
  `{post_low.min() * 1000:.3f} mm`
- penetration frames:
  `{int(np.sum(post_low < -1e-6))}`

## CSV

661 rows × 37 columns。

顺序：

- root xyz: 3
- root quaternion xyzw: 4
- LingLong2.0 joints: 30

## Video

- 640 × 480
- 30 FPS
- 661 frames
- MuJoCo
- ground z = 0
"""

    DST_README.write_text(
        text,
        encoding="utf-8",
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SUBMISSION_DIR.exists():
        shutil.rmtree(
            SUBMISSION_DIR
        )

    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qsrc = np.loadtxt(
        SRC_CSV,
        delimiter=",",
    )

    if qsrc.shape != (
        661,
        37,
    ):
        raise RuntimeError(
            qsrc.shape
        )

    if not np.isfinite(
        qsrc
    ).all():
        raise RuntimeError(
            "source contains NaN/Inf"
        )

    model = build_model()

    (
        plane_gid,
        left_geoms,
        right_geoms,
    ) = find_foot_geoms(
        model
    )

    (
        qfix,
        lift,
        post_low,
    ) = make_groundfix(
        model,
        qsrc,
        plane_gid,
        left_geoms,
        right_geoms,
    )

    np.savetxt(
        FIXED_INTERNAL,
        qfix,
        delimiter=",",
        fmt="%.9f",
    )

    write_submission_csv(
        qfix
    )

    render_video(
        model,
        qfix,
    )

    write_readme(
        lift,
        post_low,
    )

    print()
    print("=" * 90)
    print("FINAL SUBMISSION")
    print("=" * 90)

    for p in (
        DST_CSV,
        DST_VIDEO,
        DST_README,
    ):

        print(
            p.name,
            p.stat().st_size,
            "bytes"
        )

        print(
            "sha256:",
            sha256(p),
        )

    print()
    print(
        "submission dir:",
        SUBMISSION_DIR
    )

    print(
        "GROUND-FIXED "
        "SUBMISSION: READY"
    )


if __name__ == "__main__":
    main()
