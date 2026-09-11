# LingLong2.0 赛题一提交候选 - V1.5 GroundFix

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

`5.0 mm`

Root-Z correction：

- minimum: `0.000 mm`
- median: `209.582 mm`
- maximum: `280.400 mm`
- maximum step: `4.000 mm/frame`

Ground audit：

- minimum final clearance:
  `5.000 mm`
- penetration frames:
  `0`

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
