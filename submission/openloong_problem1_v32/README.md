# OpenLoong Master 赛题一提交说明

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

python tools/arm_debug/remap_openloong_arms_spatial_v32.py \
  --out-dir output/openloong_v32_final \
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

Git branch: `openloong-v32-handpose`

Git commit at packaging time: `9c39139`
