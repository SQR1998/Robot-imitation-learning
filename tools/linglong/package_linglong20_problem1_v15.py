#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = "3c70730"

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

INPUT_VIDEO = (
    ROOT
    / "dataset/openloong_master1"
    / "RGB_video_dataset.mp4"
)

SUBMISSION_DIR = (
    ROOT
    / "submission/linglong20_problem1_v15"
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


EXPECTED_SHAPE = (661, 37)

FPS = 30
WIDTH = 640
HEIGHT = 480


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_pitch_joint",

    "head_yaw_joint",
    "head_pitch_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git_value(*args: str) -> str:

    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    except Exception:
        return "unknown"


def verify_checkpoint_source():

    rel = SRC_CSV.relative_to(ROOT)

    checkpoint_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f"{CHECKPOINT}:{rel}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    local_blob = subprocess.run(
        [
            "git",
            "hash-object",
            str(SRC_CSV),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    print(
        "checkpoint blob:",
        checkpoint_blob,
    )

    print(
        "local blob     :",
        local_blob,
    )

    if checkpoint_blob != local_blob:
        raise RuntimeError(
            "Current V1.5 CSV is NOT identical to "
            f"checkpoint {CHECKPOINT}. Stop packaging."
        )

    print(
        "V1.5 CHECKPOINT SOURCE: EXACT MATCH"
    )


def build_model():

    spec = mujoco.MjSpec.from_file(
        str(URDF)
    )

    # LingLong URDF contains meshes/... already.
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
        name="submission_root_free_joint"
    )

    # Visual-only ground.
    ground = spec.worldbody.add_geom()

    ground.name = "submission_ground"
    ground.type = (
        mujoco.mjtGeom.mjGEOM_PLANE
    )

    ground.size = [
        5.0,
        5.0,
        0.1,
    ]

    # Same visual offset used by the validated viewer.
    ground.pos = [
        0.0,
        0.0,
        -0.100,
    ]

    ground.rgba = [
        0.35,
        0.35,
        0.35,
        1.0,
    ]

    # Video reference only:
    # do not introduce physical collision.
    ground.contype = 0
    ground.conaffinity = 0

    model = spec.compile()

    if model.nq != 37:
        raise RuntimeError(
            f"Expected nq=37, got {model.nq}"
        )

    return model


def verify_joint_mapping(model):

    if len(JOINT_NAMES) != 30:
        raise RuntimeError(
            "Expected 30 LingLong joints"
        )

    print()
    print("=" * 80)
    print("JOINT MAPPING")
    print("=" * 80)

    for offset, name in enumerate(
        JOINT_NAMES,
        start=7,
    ):

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Missing joint: {name}"
            )

        qadr = int(
            model.jnt_qposadr[jid]
        )

        print(
            f"{qadr:2d} "
            f"expected={offset:2d} "
            f"{name}"
        )

        if qadr != offset:
            raise RuntimeError(
                f"Wrong qpos mapping: "
                f"{name}: {qadr} != {offset}"
            )

    print()
    print(
        "LINGLONG30 JOINT MAPPING: PASS"
    )


def load_internal_qpos():

    q = np.loadtxt(
        SRC_CSV,
        delimiter=",",
        dtype=np.float64,
    )

    if q.ndim == 1:
        q = q[None, :]

    if tuple(q.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"source shape={q.shape}, "
            f"expected={EXPECTED_SHAPE}"
        )

    if not np.isfinite(q).all():
        raise RuntimeError(
            "source contains NaN/Inf"
        )

    quat_norm = np.linalg.norm(
        q[:, 3:7],
        axis=1,
    )

    if np.max(
        np.abs(
            quat_norm - 1.0
        )
    ) > 1e-5:
        raise RuntimeError(
            "source root quaternion "
            "is not normalized"
        )

    return q


def write_submission_csv(
    internal_qpos,
):

    # ========================================================
    # INTERNAL MuJoCo:
    #
    # x y z qw qx qy qz joint...
    #
    # SUBMISSION:
    #
    # x y z qx qy qz qw joint...
    #
    # Only quaternion column order changes.
    # ========================================================

    out = internal_qpos.copy()

    out[:, 3:7] = (
        internal_qpos[
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

    if tuple(check.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"submission CSV "
            f"shape={check.shape}"
        )

    if not np.isfinite(check).all():
        raise RuntimeError(
            "submission CSV contains NaN/Inf"
        )

    # Round-trip submission XYZW -> MuJoCo WXYZ.
    roundtrip = check.copy()

    roundtrip[:, 3:7] = (
        check[
            :,
            [6, 3, 4, 5],
        ]
    )

    max_diff = float(
        np.max(
            np.abs(
                roundtrip
                - internal_qpos
            )
        )
    )

    print(
        "submission round-trip "
        "max diff:",
        max_diff,
    )

    if max_diff > 2e-8:
        raise RuntimeError(
            "submission quaternion "
            "conversion failed"
        )

    return out


def render_video(
    model,
    qseq,
):

    if shutil.which(
        "ffmpeg"
    ) is None:
        raise RuntimeError(
            "ffmpeg not found"
        )

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

        "-vcodec",
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

        "-preset",
        "medium",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(DST_VIDEO),
    ]

    print()
    print("=" * 80)
    print("RENDER VIDEO")
    print("=" * 80)

    print(
        f"{WIDTH}x{HEIGHT} "
        f"@ {FPS} FPS"
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
    )

    try:

        for i, q in enumerate(
            qseq
        ):

            data.qpos[:] = q

            data.qvel[:] = 0.0

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

            frame = renderer.render()

            if (
                frame.shape
                != (
                    HEIGHT,
                    WIDTH,
                    3,
                )
            ):
                raise RuntimeError(
                    f"bad frame shape: "
                    f"{frame.shape}"
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
            f"ffmpeg failed: {ret}"
        )

    if (
        not DST_VIDEO.exists()
        or DST_VIDEO.stat().st_size == 0
    ):
        raise RuntimeError(
            "MP4 was not created"
        )


def probe_video():

    cmd = [
        "ffprobe",
        "-v",
        "error",

        "-select_streams",
        "v:0",

        "-show_entries",
        (
            "stream="
            "width,height,"
            "r_frame_rate,"
            "avg_frame_rate,"
            "nb_frames,duration"
        ),

        "-of",
        "json",

        str(DST_VIDEO),
    ]

    obj = json.loads(
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    streams = obj.get(
        "streams",
        []
    )

    if not streams:
        raise RuntimeError(
            "ffprobe found no video stream"
        )

    info = streams[0]

    print()
    print("=" * 80)
    print("VIDEO PROBE")
    print("=" * 80)

    print(
        json.dumps(
            info,
            indent=2,
        )
    )

    width = int(
        info.get(
            "width",
            -1,
        )
    )

    height = int(
        info.get(
            "height",
            -1,
        )
    )

    nb_frames = int(
        info.get(
            "nb_frames",
            -1,
        )
    )

    if (
        width != WIDTH
        or height != HEIGHT
        or nb_frames != EXPECTED_SHAPE[0]
    ):
        raise RuntimeError(
            "video validation failed: "
            f"{info}"
        )

    return info


def write_readme(
    video_info,
):

    branch = git_value(
        "branch",
        "--show-current",
    )

    head = git_value(
        "rev-parse",
        "--short",
        "HEAD",
    )

    readme = f"""# LingLong2.0 赛题一提交说明 - V1.5

## 1. 提交文件

本目录包含且仅包含三个正式提交文件：

- `linglong20_action_sequence.csv`
- `linglong20_mujoco_demo.mp4`
- `README.md`

## 2. 输入视频

正式输入视频：

`dataset/openloong_master1/RGB_video_dataset.mp4`

- 720 × 1280
- 30 FPS
- 661 frames
- 约 22.06 s

## 3. 机器人

目标机器人：

`LingLong2.0_20260616`

机器人模型：

`dataset/openloong_master1/robot/LingLong2.0_20260616/LingLong2.0.urdf`

机器人具有 30 个可动关节。

## 4. 动作序列格式

`linglong20_action_sequence.csv`

- 无表头
- 661 行
- 每行 37 个浮点数
- root position 3 维：`x, y, z`
- root quaternion 4 维：`qx, qy, qz, qw`（xyzw）
- LingLong2.0 关节位置 30 维
- 正式尺寸：`661 × 37`

内部 MuJoCo qpos 使用 `qw,qx,qy,qz`；
提交 CSV 仅将 root quaternion 重排为
`qx,qy,qz,qw`，关节动作数值不做修改。

### 30 个关节顺序

1. left_hip_pitch_joint
2. left_hip_roll_joint
3. left_hip_yaw_joint
4. left_knee_joint
5. left_ankle_pitch_joint
6. left_ankle_roll_joint
7. right_hip_pitch_joint
8. right_hip_roll_joint
9. right_hip_yaw_joint
10. right_knee_joint
11. right_ankle_pitch_joint
12. right_ankle_roll_joint
13. waist_yaw_joint
14. waist_pitch_joint
15. head_yaw_joint
16. head_pitch_joint
17. left_shoulder_pitch_joint
18. left_shoulder_roll_joint
19. left_shoulder_yaw_joint
20. left_elbow_joint
21. left_wrist_roll_joint
22. left_wrist_pitch_joint
23. left_wrist_yaw_joint
24. right_shoulder_pitch_joint
25. right_shoulder_roll_joint
26. right_shoulder_yaw_joint
27. right_elbow_joint
28. right_wrist_roll_joint
29. right_wrist_pitch_joint
30. right_wrist_yaw_joint

## 5. 当前动作版本

本提交使用冻结的：

`LingLong2.0 Arm Retarget V1.5 Strong Extension`

源轨迹：

`output/linglong20_arm_v15_strong_extension/linglong20_arm_v15_strong_extension_qpos37.csv`

Git checkpoint：

`{CHECKPOINT}`

Git tag：

`linglong20-v15-checkpoint-20260911`

## 6. 动作生成流程

1. RGB 视频恢复 SMPL-X 人体运动；
2. General Motion Retargeting 生成 LingLong2.0 全身动作；
3. 下肢映射经过 LingLong2.0 专用校准；
4. 腰部及头部进行独立映射；
5. 左右双臂根据人体肩、肘、腕空间方向进行重定向；
6. 使用机器人实际上臂和前臂长度重建目标；
7. 双臂 IK 包含时间连续性约束；
8. 使用 collision-aware IK 避免明显自碰撞；
9. 使用模型标定的肘关节机械伸直位置改善伸臂动作；
10. 最终输出完整 661 帧 LingLong2.0 qpos37 动作。

## 7. 演示视频

`linglong20_mujoco_demo.mp4`

- width: {video_info.get("width")}
- height: {video_info.get("height")}
- FPS: {FPS}
- frames: {video_info.get("nb_frames")}
- duration: {video_info.get("duration")} s

视频由 MuJoCo 直接加载同一份 V1.5 qpos37 轨迹生成。

## 8. 软件环境

Conda：

`openloong_il`

主要组件：

- Python 3.10
- PyTorch 2.1.2 + CUDA 11.8
- MuJoCo 3.12.0
- OpenCV 4.11.0
- SMPL-X
- General Motion Retargeting

## 9. 版本信息

Packaging branch:

`{branch}`

Packaging HEAD:

`{head}`

Frozen motion checkpoint:

`{CHECKPOINT}`
"""

    DST_README.write_text(
        readme,
        encoding="utf-8",
    )


def main():

    for path in (
        SRC_CSV,
        URDF,
        INPUT_VIDEO,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # Start with an exact three-file directory.
    if SUBMISSION_DIR.exists():
        shutil.rmtree(
            SUBMISSION_DIR
        )

    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "LINGLONG2.0 V1.5 "
        "SUBMISSION PACKAGER"
    )
    print("=" * 80)

    print(
        "source:",
        SRC_CSV,
    )

    print(
        "checkpoint:",
        CHECKPOINT,
    )

    verify_checkpoint_source()

    model = build_model()

    verify_joint_mapping(
        model
    )

    q_internal = (
        load_internal_qpos()
    )

    print()
    print(
        "internal qpos shape:",
        q_internal.shape,
    )

    print(
        "internal finite:",
        bool(
            np.isfinite(
                q_internal
            ).all()
        ),
    )

    submission_csv = (
        write_submission_csv(
            q_internal
        )
    )

    render_video(
        model,
        q_internal,
    )

    video_info = (
        probe_video()
    )

    write_readme(
        video_info
    )

    files = sorted(
        [
            p.name
            for p in SUBMISSION_DIR.iterdir()
            if p.is_file()
        ]
    )

    expected = sorted(
        [
            DST_CSV.name,
            DST_VIDEO.name,
            DST_README.name,
        ]
    )

    if files != expected:
        raise RuntimeError(
            "Submission directory "
            "does not contain exactly "
            "the expected 3 files: "
            f"{files}"
        )

    print()
    print("=" * 80)
    print(
        "LINGLONG2.0 V1.5 "
        "SUBMISSION PACKAGE: READY"
    )
    print("=" * 80)

    print(
        "DIR:",
        SUBMISSION_DIR,
    )

    print()

    for path in (
        DST_CSV,
        DST_VIDEO,
        DST_README,
    ):
        print(
            f"{path.name:36s} "
            f"{path.stat().st_size:10d} bytes"
        )

        print(
            "  sha256:",
            sha256(path),
        )

    print()
    print(
        "CSV shape:",
        submission_csv.shape,
    )

    print(
        "Exactly 3 formal files "
        "were created."
    )


if __name__ == "__main__":
    main()
