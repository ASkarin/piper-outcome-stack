# 任务C：等货期间的软件准备

2026-10-30用户批准三技能定义、标注、采集与单技能执行接口，并明确沿用B收尾的RTC配方。抽屉和杯子已选定、等待到货。本文件描述软件能力；尚无C实物可行性、训练模型或实机结果。

## 三技能契约

|skill_id|训练/执行文本|起始条件|成功条件|
|---|---|---|---|
|open_drawer|打开抽屉，松开把手并退出。|抽屉关闭、夹爪空闲，处于约定起始范围|达到约定开度，释放把手，退出后续取杯区域|
|store_cup|把{物体名称}放入抽屉，松开并退出。|抽屉打开，目标杯在取物范围，机械臂处于本段起始范围|正确杯子稳定留在抽屉内，释放并退出关闭路径|
|close_drawer|关闭抽屉并退出。|杯子放稳，夹爪空闲且在抽屉外，关闭路径可用|达到约定关闭状态，杯子未挤出，机械臂退出|

一条示范只教一个技能；store_cup内部抓取/搬运/释放/退出不拆成四个模型。任务文本由已登记杯子名称生成并原样传给SmolVLA，不声称自由语言解析已完成。单个C checkpoint服务三个技能；初始权重、数据量、训练步数及评测预算另行冻结，当前不启动训练。

相邻技能的终态/起始范围需在到货后衔接。单技能从操作者就位后的当前姿态运行，不自动回A或回零。人工停止和事后标注支持开发检查，不等于已实现自主阶段切换或完整C成功。

## 配置

从`configs/tasks/c.template.json`建立控制机专用配置：

- `robot`、`teleop`采用当前实际配置结构，记录新的C场景及实测工作区/视野；不把B场景和抽屉碰撞空间视为相同。
- `task_c.drawer_id`记录固定抽屉；`objects`将物体ID映射到实际文本名称；`position_id`记录实际起点范围编号。
- `task_c.duration_s`分别设置三个技能的正数时限；`dataset.num_episodes`明确本会话回合数。模板留空，未到货不猜数值。
- `dataset.fps=50`，复用原始RGBD采集与官方Dataset转换；模型输入仍是RGB和七维rad/m状态。
- `policy.checkpoint`和`policy.vlm_assets`指定C模型/处理器资源；`policy.reference_inputs`按`open_drawer`、`store_cup:<object_id>`、`close_drawer`映射离线参考NPZ。NPZ沿用images/states/expected_absolute_chunks/tasks及可选seeds结构，任务文本必须一致。

`policy`中的相对资源路径以C配置文件目录为基准；robot/teleop配置沿用现有路径规则。等货期间模板中的设备配置、场景、对象、时限、回合数和模型资源均未填写；检查明确失败并列出首个缺项，不生成“已验收”标志。

## 采集与数据标注

控制机operator profile的`record_c_config`指向该配置。命令示例（实物与配置准备完成后由操作者运行）：

```bash
piper record c --skill open_drawer --split development
piper record c --skill store_cup --object cup01 --split development
piper record c --skill close_drawer --split development
```

raw与Dataset各自写入既有根目录下的`task-c/`。菜单每次只有本技能任务，保留现有开始/结束、1成功/2失败/3取消、重录和封存流程。失败不计成功示范，完整有效失败仍保存；数据有效性与物理成功分开。只在真实串联试采时使用`--chain-id <一次完整任务ID>`，此时每技能每会话仅一回合；独立技能采集不填写chain_id。

技能上下文由同一配置生成，写入raw.json的capture_context及episode_started/save_requested/outcome事件：task_family、skill_id、scene_id、drawer_id、可选object_id/object_name/chain_id、起始与成功定义。沿用既有菜单task_id及actual task文本，不把它们重定义为技能ID，也不改变七维策略向量。转换保留原遥测；重录/取消及来源帧索引仍可审计。

完成转换后可汇总多个operator会话，输出供后续训练spec使用的显式成功段清单：

```bash
python -m piper_outcome_stack.task_c manifest \
  --session <第一份operator会话目录> --session <第二份operator会话目录> \
  --output <新manifest.json>
```

manifest读取实际Dataset与事件，核对标签/任务文本/上下文，仅选择success且data_valid的段；失败/取消列为excluded。源会话或同一chain跨train/development会拒绝。输出sources.train/development的segments可直接供现有RetainedDataset使用；未来帧仍止于每个连续段边界。该清单不启动训练，也不自动把开发数据转为训练数据。

## 单技能执行

```bash
piper c check --config <C配置.json> --skill store_cup --object cup01
piper c run --config <C配置.json> --skill store_cup --object cup01
```

check使用指定模型和录制参考进行离线验证，不构造/打开CAN、相机或手柄；会进行模型加载/推理，缺模型时不能通过。run也先做参考核对，再由操作者确认连接/使能，保持确认后按LB开始。松LB结束本段，B停止、实时输入/动作验证与故障锁存均复用run_policy_trial。结果及RGB索引写入独立task-c会话。

RTC固定使用收尾配方：50Hz、25tick请求、10步去噪、native_joint6、初始延迟20tick、seed42、无关节低通、既有夹爪zero投影与连续参考，以及7/32维结构padding。共享执行器显式选择joint_lowpass=False；原通用5Hz调用默认保持原行为。各技能开始/结束清除旧推理任务、队列、前缀及模型缓存，硬件侧已接受的夹爪参考仍由原生命周期维护。

程序结果segment_ended_held只表示已结束并保持；物理结果独立标注。C成功标签不能覆盖程序失败或未确认保持。结果中保留技能、物体、实际任务文本和中途人工介入；汇总按场景/技能/物体区分，另列辅助次数及无中途介入成功率，范围始终是单技能。结束后的结果标注不是中途介入。

```bash
python -m piper_outcome_stack.policy_trials prompt <trial.json>
python -m piper_outcome_stack.policy_trials summarize <各trial.outcome.json>
```

本轮没有完整自主串联、自动抽屉状态判断或恢复模型。到货后确认开闭/释放/退出的实际可观察依据，再实现自动阶段切换；不凭固定计时或命令发出认定成功。

## 软件验证范围

验证三技能任务选择、实物/预算缺项拒绝、实际CLI解析、假设备采集/重录/转换/续录、原始与Dataset标签一致、成功段训练读取、跨会话chain隔离、check不接设备、当前位置起跑、清理错误记录以及RTC与收尾快照的逐值等价。共用控制循环的LB/B/时效/夹爪/故障回归同时覆盖。

数值配置仍待到货后的现场测量，C新入口未执行真实硬件验证。没有新增训练、采集预算或抽屉sim2real任务。
