# 阶段 C：Push‑T

当前从操作者定义的第三种起始姿态 `start-03` 推到有三个角标的 G。位置和朝向均对齐，夹爪撤开后 T 保持对齐，记录为成功。待这一条件下的策略取得实机成功后，再扩展起点。

在控制机现场终端启动采集：

```bash
piper record c
```

默认用途为 train。开发回合另开会话，用 `piper record c --split development`。准备界面可输入 `fine`，按参考图复位后输入 `3` 开始。沿用 Xbox 的 LB/B 操作，`end` 结束。审阅时 `1` 保存成功，`2 原因` 保存失败，`3 原因` 保存取消。`q` 在准备界面提前结束。菜单编号与开始位置、实际任务文本一起进入每回合记录。

采集使用活动 operator profile 中的 `record_c_config`。D435 彩色流和状态/实际下发动作保留，阶段 C 部署配置关闭深度流和深度导出。默认控制/Dataset 50 Hz，彩色流 640×480 / 60 fps。记录文件位于各自根目录的 `task-c/train/` 或 `task-c/development/`。终端打印的 Session 是含 config.json 的会话目录。

```bash
piper convert <Session目录>
piper audit <Session目录>
```

`python -m piper_outcome_stack.task_c manifest --session <Session目录> --output <新文件>` 可汇总已转换会话。每段保留来源、起始/目标参考及真实任务文本，成功且有效的示范进入行为克隆候选。失败、取消和接管分别记录。旧 G 的三次开发试推归档在原场景。

策略资产到位后，使用 `piper c check` 在文件参考上检查，再由操作者执行 `piper c run`。两条命令使用同一活动 C 配置。checkpoint、VLM 资源及 `policy.reference_inputs.push_to_G` 需要来自本轮模型与参考输入。参考 PNG 描述物体起止布局，模型核验另用含 images/states/expected_absolute_chunks/tasks 的 NPZ。实机执行沿用现有 RTC、实时动作检查、LB/B 和保持流程。程序结束与物理任务结果单独标注。
