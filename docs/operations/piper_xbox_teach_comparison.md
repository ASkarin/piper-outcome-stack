# Xbox 与拖动示教的四起点对照

本轮先各用四起点样本做实用性比较，不训练模型，不将四条样本当作统计结论。

使用同一绿色物块、相机视角和左侧胶带放置区。相机视角下编号为 P1 左上、P2 右上、P3 左下、P4 右下。Xbox 与示教分别保存 Dataset，不混合续录。

## Xbox capture

Use the fixed piper record entry (50 Hz control/Dataset, 640×480 RGBD60); X is already measured for the current controller. Default XYZ
controls the internal grasp center with wrist-priority IK; horizontal right stick
is unused. X selects fixed-orientation translation with yaw; RB selects orientation.

In preparation, A/Y are optional. After confirmed hold, use `start P1`, then
`end` and `save success` or `save failure <reason>`. Raw save seals the current
attempt and returns to preparation; `start P2` is then available directly.
`redo` retains the discarded attempt. After four attempts or `quit`, run the
printed `xbox-convert` command separately. No video encoding or depth compression
runs between episodes. Do not interrupt sealing with Ctrl+C.

Preparation and A/Y do not enter demonstrations. Saved RGBD retains source timing;
model inputs remain RGB and seven-value state. Xbox actions remain actual sent
waypoints/retained holds (N rows), never teaching's next-state labels (N-1 rows).
The previously recorded half-P1 remains a partial trial, not a full demonstration.

## 比较内容

- 任务：是否抓住、搬运、放下、释放，是否滑落或碰撞，重试次数。
- 画面：指尖、物块和接触过程的可见性；手部遮挡与照明变化。
- 运动：依据实测状态和实际时间比较移动、停顿、回退和无意义整理动作。
- 时间质量：帧龄、图像与状态偏差、丢失和重复情况。
- 操作成本：操作者耗时、修正次数、保存等待和疲劳。
- 标签：分别核对 Xbox 实际下发目标和示教下一帧实测目标的来源。

不要直接比较两种 action 列的平滑度来评判优劣：它们的生成语义不同。已有示教数据的初始姿态和尾段也不完全一致，总时长需结合任务阶段解释。先找出明显的可用性差异；若后续要判断 ACT 学习效果，需要等量数据和统一条件的训练、执行对照，另行安排。

## 当前比较边界（2026-09-21）

历史20Hz示教/Xbox与新50Hz数据要披露频率、输入和执行参考差异，不能只按帧数或动作平滑度判断训练优劣。任务结果、数据有效性与人工遮挡各自统计；成功且有效的完整回合用于现有行为克隆选择，失败和恢复片段另行设计。

近期仍完成绿色方块A闭环，再引入B物体/容器和语言。single_task是会话文本，起点编号不代表指令。C抽屉收纳/有限恢复为条件研究任务，不因本次四点比较而宣称具备规划、避障或恢复能力。
