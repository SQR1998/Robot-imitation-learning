# OpenLoong 赛题一提交说明

## 1. 提交文件

本次提交包含以下三个文件：

1. `openloong_action_sequence.csv`
   - OpenLoong 机器人动作序列

2. `openloong_mujoco_demo.mp4`
   - 动作序列在 MuJoCo 中的演示视频

3. `README.md`
   - 使用说明及动作生成流程

---

## 2. 输入视频

使用正式比赛 RGB 视频：

`dataset/openloong_master1/RGB_video_dataset.mp4`

视频参数：

- 分辨率：720 × 1280
- 帧率：30 FPS
- 总帧数：661
- 时长：22.033 秒

---

## 3. 动作生成方法

完整流程：

RGB 视频  
→ YOLO 人体检测  
→ ViTPose 人体关键点估计  
→ HMR2 / WHAM 三维人体运动恢复  
→ SMPL-X 人体运动表示  
→ General Motion Retargeting（GMR）  
→ OpenLoong 全身动作序列  
→ MuJoCo 可视化

机器人使用 OpenLoong MuJoCo 模型。

人体到机器人的动作重定向配置：

`general_motion_retargeting/ik_configs/smplx_to_openloong.json`

---

## 4. 动作序列格式

动作序列文件：

`openloong_action_sequence.csv`

CSV 不包含表头。

每一行为一帧 OpenLoong qpos，共 38 个数值。

其中：

- 第 1～3 列：根节点位置 x、y、z
- 第 4～7 列：根节点姿态四元数 qx、qy、qz、qw（xyzw 顺序）
- 第 8～38 列：31 个机器人关节位置

正式动作序列尺寸：

`661 × 38`

动作序列已经检查：

- 帧数：661
- 每帧 qpos 数量：38
- 无 NaN
- 无 Inf
- 根节点四元数模长约为 1

---

## 5. 软件环境

测试环境：

- Ubuntu 22.04
- Python 3.10
- PyTorch 2.1.2
- CUDA 11.8
- NVIDIA GeForce RTX 4090
- MuJoCo
- WHAM
- General Motion Retargeting

Conda 环境：

`openloong_il`

---

## 6. 正式视频处理命令

进入项目：

```bash
cd ~/桌面/Robot-imitation-learning
conda activate openloong_il
```

执行：

```bash
OUTPUT_ROOT=output/openloong_full_v1 \
ROBOT=openloong \
VIDEO=dataset/openloong_master1/RGB_video_dataset.mp4 \
RECORD_WHAMVIDEO=0 \
RECORD_GMRVIDEO=0 \
GMR_TORCH_DEVICE=cuda \
GMR_MAX_ITER=5 \
bash run.sh
```

主要输出：

```text
output/openloong_full_v1/csv/openloong/live_motion.csv
output/openloong_full_v1/pkl/my_motion.pkl
```

---

## 7. MuJoCo 演示视频

MuJoCo 演示视频由最终生成的机器人动作序列逐帧重放产生。

演示参数：

- 帧数：661
- 帧率：30 FPS
- 视频分辨率：608 × 1072
- 机器人：OpenLoong

文件：

`openloong_mujoco_demo.mp4`

---

## 8. 当前代码版本

Git 分支：

`openloong-groundfix-local-validated-20260907`

Git Commit：

`0d9ad0626a2bd657db55b61fca589f5e954532e3`

该版本保持原 WHAM → GMR 动作重定向方法不变，仅针对 OpenLoong 足底几何与 MuJoCo 地面高度进行了对齐修正。

当前未对双手接近时的动作进行额外约束或修正，以尽可能保留原始视频动作的重定向结果。
