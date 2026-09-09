# OpenLoong V2 Body + Hand Direction Checkpoint

日期：2026-09-09

这是 OpenLoong 赛题一当前已验证成果的冻结节点。

## 已验证内容

### 1. 身体 / 下肢

基于：

- commit `1fde381`
- OpenLoong V2.3 knee guidance

已视觉确认：

- 机器人能够站直
- 腿部姿态明显改善
- 腰部状态正常
- 足底地面修正正常
- 身体整体动作可继续作为后续双臂重定向的基础

### 2. 机器人手方向 POC

基于：

- commit `5b3b5b6`
- `tools/hand_debug/map_head_direction_to_openloong_wrist.py`

在 15~18 秒实验区间已经确认：

- OpenLoong 手部长轴方向可以通过 arm_06 / arm_07 调整
- 指尖方向能够与目标方向正确对齐
- 该 POC 用于证明机器人腕部自由度能够表达正确的手方向

注意：

该 POC 只验证了 15~18 秒，不是最终全视频双臂映射方案。

### 3. 真人手部二维方向

当前稳定流程：

RGB 视频
→ MediaPipe Pose
→ Multi-ROI Hand Landmarker
→ 手腕匹配
→ Hand21 关键点
→ 原始指尖方向
→ 异常检测
→ 防 180° 翻转
→ 遮挡插值
→ 前臂相对角约束
→ 时序平滑

已视觉确认：

- 全视频手方向总体正确
- 遮挡阶段能够连续补偿
- 单帧错误识别不会直接造成 180° 翻转
- 7~9 秒曾出现的左手 360° 假旋转已在 V6 中解决
- 手部真实动作整体是连续变化的

## 本 checkpoint 保存的手部脚本

- `detect_hands_roi_15_18_v2.py`
  - 已验证的 Multi-ROI 手部检测基础版本

- `filter_hand_direction_v4.py`
  - 15~18 秒防突变 / 防 180° 翻转版本

- `detect_hands_full_video_v1.py`
  - 全 661 帧手部检测

- `filter_hand_full_v5.py`
  - 全视频检测结果的中间质量门控和锚点过滤
  - V5 的最终角度轨迹本身曾出现 360° unwrap 问题
  - 这里只作为 V6 所需的中间锚点结果

- `filter_hand_full_v6_forearm.py`
  - 当前视觉验证通过的全视频二维手方向版本
  - 使用手方向相对前臂角度
  - 不再使用无限角度 unwrap
  - 已解决虚假 360° 旋转

## 当前尚未解决

最终 OpenLoong 双臂映射尚未完成。

之前的全视频 3D 腕部映射和 collision-aware V2.5 属于实验版本，存在：

- 双臂映射位置不合理
- 手方向与避碰发生冲突
- 个别关节解在相邻帧之间跳变

因此这些版本不属于本 checkpoint 的已验证成果。

后续工作应从本 checkpoint 的身体动作和手方向结果出发，
重新设计左右双臂的空间位置重定向，而不是继续对失败的腕部结果做补丁。
