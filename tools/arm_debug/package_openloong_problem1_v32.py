#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DIR = ROOT / "output/openloong_v32_final"
SRC_CSV = CANDIDATE_DIR / "csv/openloong/live_motion.csv"
SRC_VIDEO = CANDIDATE_DIR / "video/openloong_armspatial_v1.mp4"
VALIDATION_JSON = CANDIDATE_DIR / "validation/v32_validation_report.json"

SUBMISSION_DIR = ROOT / "submission/openloong_problem1_v32"

DST_CSV = SUBMISSION_DIR / "openloong_action_sequence.csv"
DST_VIDEO = SUBMISSION_DIR / "openloong_mujoco_demo.mp4"
DST_README = SUBMISSION_DIR / "README.md"

EXPECTED_SHAPE = (661, 38)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
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


def main() -> int:
    for p in (SRC_CSV, SRC_VIDEO, VALIDATION_JSON):
        if not p.exists():
            print(f"ERROR: missing required file: {p}")
            return 2

    validation = json.loads(
        VALIDATION_JSON.read_text(encoding="utf-8")
    )

    if not validation.get("overall_pass", False):
        print("ERROR: V3.2 validation did not PASS. Refusing to package.")
        print("See:", VALIDATION_JSON)
        return 3

    arr = np.loadtxt(SRC_CSV, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]

    if tuple(arr.shape) != EXPECTED_SHAPE:
        print(f"ERROR: CSV shape={arr.shape}, expected={EXPECTED_SHAPE}")
        return 4

    if not np.isfinite(arr).all():
        print("ERROR: CSV contains NaN/Inf")
        return 5

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SRC_CSV, DST_CSV)
    shutil.copy2(SRC_VIDEO, DST_VIDEO)

    branch = git_value("branch", "--show-current")
    commit = git_value("rev-parse", "--short", "HEAD")

    readme = f"""# OpenLoong Master 赛题一提交说明

## 1. 提交文件

本目录包含赛题一正式提交所需的 3 个文件：

- `openloong_action_sequence.csv`
- `openloong_mujoco_demo.mp4`
- `README.md`

## 2. 输入视频

正式输入视频：

`dataset/openloong_master1/RGB_video_dataset.mp4`

视频参数：720 × 1280，约 30 FPS，共 661 帧，时长约 22.06 秒。

## 3. 动作序列格式

`openloong_action_sequence.csv`：

- 无表头；
- 共 661 行；
- 每行 38 个浮点数；
- 顺序为：
  - 根节点位置 3 维：`x, y, z`
  - 根节点四元数 4 维：`qx, qy, qz, qw`（xyzw）
  - OpenLoong 31 个关节位置；
- 正式尺寸：`661 × 38`。

## 4. 动作生成流程

本提交版本为 **OpenLoong V3.2**。

整体流程：

1. 从输入 RGB 视频恢复人体 SMPL-X / 身体运动；
2. 使用稳定身体基线生成 OpenLoong 身体、腰部、头部和腿部动作；
3. 双臂采用从第 0 帧开始的双侧 14-DoF 联合空间 IK；
4. 肩、肘、腕位置采用人体局部空间方向重建，并按 OpenLoong 实际上臂/前臂长度映射；
5. 使用 V6 手方向结果确定手/手指长轴方向；
6. 使用 Palm Roll V2 结果确定绕手指长轴的逐帧翻掌角；
7. OpenLoong 手指长轴使用左右手 `Link_arm_*_07` 的局部 ±Y 轴；
8. OpenLoong 掌心法向使用视觉标定后的局部 +X 轴；
9. 手指方向和掌心方向与双臂空间位置一起进入同一个 14-DoF 优化；
10. 求解包含时间连续性、关节限位以及双臂/躯干距离约束；
11. 未使用后处理式 wrist delta，也未使用逐帧关节硬裁剪。

## 5. 软件环境

### 机器人动作生成

Conda 环境：

`openloong_il`

主要用于：

- SMPL-X / GMR
- MuJoCo
- OpenLoong V3.2 双臂 14-DoF IK
- 正式动作 CSV 和 MuJoCo 演示视频生成

### 手部视觉感知

Conda 环境：

`openloong_hand`

主要用于：

- MediaPipe Pose / Hand Landmarker
- V6 手指方向
- Palm Roll V2 掌心翻转角

已验证 MediaPipe 版本：`1.0.1`。

## 6. V3.2 正式生成命令

```bash
cd ~/桌面/Robot-imitation-learning
conda activate openloong_il

rm -rf output/openloong_v32_final

python tools/arm_debug/remap_openloong_arms_spatial_v32.py \\
  --out-dir output/openloong_v32_final \\
  --render
```

该命令处理完整 661 帧，不使用 `--max-frames` 截断。

## 7. 提交前验证

正式候选版本经过以下检查：

- CSV 尺寸为 `661 × 38`；
- 所有数值均为 finite，无 NaN/Inf；
- 根节点四元数模长约为 1；
- 除左右双臂 14 个 qpos 外，身体/头/腰/腿与稳定身体基线保持不变；
- MuJoCo 模型 qpos 数量为 38；
- 所有限位 hinge/slide 关节均检查物理关节范围；
- IK diagnostics 共 661 帧；
- 双臂连续性检查；
- hand-hand、forearm-forearm、torso proxy clearance 检查；
- 正式演示视频帧数和 FPS 检查；
- 7～9 秒及 15～18 秒关键手部姿态区间单独统计；
- MuJoCo 宽泛 arm-contact 计数仅作为参考，因为其中包含相邻结构/正常自接触，不作为单独淘汰条件。

验证报告位于：

`output/openloong_v32_final/validation/v32_validation_report.json`

## 8. 版本信息

Git branch: `{branch}`

Git commit at packaging time: `{commit}`
"""

    DST_README.write_text(readme, encoding="utf-8")

    # Final exact-copy checks.
    dst = np.loadtxt(DST_CSV, delimiter=",", dtype=np.float64)
    if tuple(dst.shape) != EXPECTED_SHAPE or not np.isfinite(dst).all():
        print("ERROR: packaged CSV failed final check")
        return 6

    print("=" * 80)
    print("OpenLoong V3.2 submission package READY")
    print("=" * 80)
    print("DIR   :", SUBMISSION_DIR)
    print("CSV   :", DST_CSV)
    print("VIDEO :", DST_VIDEO)
    print("README:", DST_README)
    print()
    print("CSV shape :", dst.shape)
    print("CSV sha256:", sha256(DST_CSV))
    print("MP4 sha256:", sha256(DST_VIDEO))
    print("README sha256:", sha256(DST_README))
    print()
    print("Only 3 formal submission files were created in this directory.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
