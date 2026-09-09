# OpenLoong 赛题一提交说明 - V3.3.4

## 1. 输入视频

比赛输入视频：

`dataset/openloong_master1/RGB_video_dataset.mp4`

视频参数：

- 720 × 1280
- 30 FPS
- 661 frames
- 约 22.06 s

## 2. 输出文件

本提交目录包含且仅包含正式提交所需的三个文件：

1. `openloong_action_sequence.csv`
2. `openloong_mujoco_demo.mp4`
3. `README.md`

## 3. CSV格式

`openloong_action_sequence.csv`

尺寸：

`661 × 38`

每行包括：

- Root position：3
- Root quaternion：4，顺序为 qx, qy, qz, qw
- OpenLoong joint qpos：31

CSV无表头。

## 4. 动作生成流程

整体流程：

RGB视频
→ WHAM / SMPL-X人体动作恢复
→ GMR映射到OpenLoong
→ 稳定身体与腿部修正
→ V6手指方向估计
→ Palm Roll掌心姿态估计
→ 双臂14DoF空间IK
→ 连续性约束
→ 人体肘部伸展引导
→ V3.3.4自然姿态先验
→ OpenLoong动作序列

V3.3.4主要优化双臂动作连续性和自然姿态。

在存在多个可行IK解时，优先保证：

1. 双臂运动连续，无明显跳变
2. 整条手臂姿态自然
3. 手腕和肘部位置尽量准确
4. 手指方向尽量正确
5. 掌心方向作为较低优先级软约束

掌心方向允许在必要时让步，以避免整条手臂出现明显瞬时跳变或不自然扭曲。

## 5. V3.3.4代码版本

Git branch：

`openloong-v33-optimization`

代码 commit：

`322c8a17b1c4f904e8b09cbb7f660d88f0361238`

short commit：

`322c8a1`

核心脚本：

`tools/arm_debug/remap_openloong_arms_spatial_v334.py`

## 6. 软件环境

主要环境：

`openloong_il`

主要组件包括：

- Python 3.10
- PyTorch
- PyTorch3D
- WHAM
- SMPL-X
- General Motion Retargeting
- MuJoCo

手部视觉检测使用独立环境：

`openloong_hand`

## 7. 正式生成命令

```bash
conda activate openloong_il

python tools/arm_debug/remap_openloong_arms_spatial_v334.py \
  --out-dir output/openloong_v334_natural_posture_full \
  --render
```

正式动作序列：

`output/openloong_v334_natural_posture_full/csv/openloong/live_motion.csv`

正式MuJoCo演示视频：

`output/openloong_v334_natural_posture_full/video/openloong_armspatial_v1.mp4`
