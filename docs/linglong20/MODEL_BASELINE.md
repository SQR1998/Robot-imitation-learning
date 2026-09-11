# LingLong 2.0 Model Baseline

## Source model

Competition model:

dataset/openloong_master1/robot/LingLong2.0_20260616/LingLong2.0.urdf

SHA256:

eb2a6341eb9c511b0d378d06fb320a0da30eed82dd2948c787f56562a6f890f8

Do NOT use:

dataset/openloong_master1/robot/LingLong2.0_20260616/LingLong_L1_V1.4.urdf

LingLong_L1_V1.4 SHA256:

31a1d2a5da904fa50055be9ef32e5d977bb267e165c6cba4c0e504ce0b2639bf

## MuJoCo verification

Original LingLong2.0 URDF:

- nq = 30
- nv = 30
- njnt = 30
- nbody = 31

The original URDF is fixed-base when loaded directly into MuJoCo.

Expected future floating-base LingLong 2.0 model:

- robot joints = 30
- floating root qpos = 7
- expected nq = 37
- expected nv = 36

## 30 DoF joint order

### Left leg

0. left_hip_pitch_joint
1. left_hip_roll_joint
2. left_hip_yaw_joint
3. left_knee_joint
4. left_ankle_pitch_joint
5. left_ankle_roll_joint

### Right leg

6. right_hip_pitch_joint
7. right_hip_roll_joint
8. right_hip_yaw_joint
9. right_knee_joint
10. right_ankle_pitch_joint
11. right_ankle_roll_joint

### Waist

12. waist_yaw_joint
13. waist_pitch_joint

### Head

14. head_yaw_joint
15. head_pitch_joint

### Left arm

16. left_shoulder_pitch_joint
17. left_shoulder_roll_joint
18. left_shoulder_yaw_joint
19. left_elbow_joint
20. left_wrist_roll_joint
21. left_wrist_pitch_joint
22. left_wrist_yaw_joint

### Right arm

23. right_shoulder_pitch_joint
24. right_shoulder_roll_joint
25. right_shoulder_yaw_joint
26. right_elbow_joint
27. right_wrist_roll_joint
28. right_wrist_pitch_joint
29. right_wrist_yaw_joint

## Important

Previous OpenLoong V3.3.4 work used:

assets/openloong/AzureLoong.xml

That model is QingLong / AzureLoong, NOT LingLong 2.0.

The AzureLoong V3.3.4 result is retained only as an algorithm checkpoint.
All future competition retargeting should use LingLong2.0.

## Internal motion format

Current development standard:

661 rows x 37 columns.

MuJoCo qpos order:

0. root_x
1. root_y
2. root_z
3. root_qw
4. root_qx
5. root_qy
6. root_qz

7-36: LingLong 2.0 30 robot joints in model qpos order.

This is the internal development format.
It is NOT yet claimed to be the official competition submission schema.

All retargeting, validation and rendering should operate on this
37-D representation. A final exporter may later convert it to the
competition-required schema if necessary.
