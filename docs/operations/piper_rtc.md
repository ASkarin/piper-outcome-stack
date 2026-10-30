# PiPER SmolVLA RTC

2026-10-17交接：下文为既有开发记录；阶段3按[固定接口](stage2_interface_contract_20261017.md)准备共同数据与评测预算。本次不自动重启历史运行或已删除的监控。

## 本轮基准与范围

2026-10-03用户批准根据Black等《Real-Time Execution of Action Chunking Flow Policies》接入RTC，随后明确纠正基准选择。本轮固定原始绝对动作SmolVLA：`smolvla-a-20261002-seed42-r1/checkpoints-output/checkpoints/010000/pretrained_model`。ACT40k是已有实机证据的参考基线；sqrt_gripper只是保留的探索实验，不是默认基准。RTC与朴素异步使用相同R1权重，没有训练、表示改动、控制机部署或硬件I/O。负夹爪目标仍原样记录并拒绝。

## 论文对应

项目内`rtc.py`实现论文Eq.2–5及Algorithm1。旧块未来前缀在模型标准化空间参与引导；先开启动作输入梯度，再计算denoiser与最终动作估计，使用完整VJP。SmolVLA从1降至0的时间方向对应`clean=x-time*v`及`guided_velocity=v-strength*VJP`，积分步长为负。β=5，只限制引导系数，不裁剪物理动作。真实七维参与输出误差掩码，32维内部填充不作为引导目标。

当前安装的LeRobot RTC先计算速度、再设置输入requires_grad；线性测试返回2.0，而论文完整VJP应为1.625。本项目复用SmolVLA现有rtc_processor接口，替换为项目内实现，不改共享包。模型参数冻结，VLM缓存由no_grad路径生成；局部梯度仅对动作求导，每个去噪步释放计算图。无前缀RTC与同噪声普通推理必须一致。

`SmolVLAChunkPredictor.predict_chunk`返回`ActionChunk(actions, model_actions, noise)`：物理rad/m用于执行和真实标签误差，原始标准化预测用于RTC前缀。旧`predict`仍返回物理数组。不得从解码后的平方夹爪反推模型原始符号；历史sqrt checkpoint的处理器仍可读取，但本轮不使用它。

## 时间轴与仿真接口

仿真新增`--execution rtc-measured`，配合`--action-steps 25`、`--denoising-steps 10|5`、`--rtc-initial-delay <profile所得tick>`及`--temporal-ensemble-coeff none`。H=50，周期20ms，延迟窗口10项取最大值。软掩码末端是旧块剩余长度H−s，不是s；首次没有旧块时不引导。

RTCTimeline在锁内复制观测与旧块前缀；新结果按实际完成tick生效并跳过过时项。推理延迟只有在模拟完成时才进入历史，不能从尚未生效的未来结果提前估计下一次延迟。首次等待明确保持初始目标；过期、耗尽、异常或数值边界失败均停止。s与预测延迟必须满足论文可执行时域。

rtc-measured是**测得耗时驱动的离散事件仿真**：推理在线程中先算出，结果在逻辑完成时刻前不可用，其间按旧队列推进物理。渲染耗时不推进物理；相机、通信、真实50Hz控制线程未验收。它不能被称为实机RTC实时控制通过。

每次预测保存输入RGB/state、完整新旧块、标准化/物理预测、噪声、预计/实际延迟和索引；步骤记录区分当前测量时刻与策略观测时刻。

## 固定验证

`rtc_experiment`使用独立spec入口，结果目录必须不存在。主run为`smolvla-a-rtc-absolute-20261003-seed42`，配置位于训练端`/workspace/piper/staging/csm/rtc-absolute-20261003/spec.json`，输出在run下`results/`，GPU总预算2700秒。

先测10步去噪，5次引导预热后20次计时；最大耗时若超过500ms，唯一备选为5步，并对两方法统一使用该步数。仍超时就停止。之后使用13开发回合各8对、相隔25个保留帧的固定观测，共104对；新块共享噪声、旧块、观测与延迟条件。旧块重叠一致性是**模型参考proxy**，真实动作误差另由不变的开发标签计算。

闭环为朴素异步与RTC各seed42/43/44、每回合最多40模拟秒，使用相同权重、步数和s_min=25，分别计入各自实际耗时。负值、越界、队列失败和未成功回合均保留，不放宽限位、不换seed、不追加训练。重启失败/完成run必须使用新ID并记入原总预算，禁止覆盖结果。

36项相关回归已在分阶段运行中通过：完整VJP、有限差分、符号、掩码、参数不变、原始预测保存、时间对齐与已有仿真行为。完整性审阅及实际实验结果另见`artifacts/rtc-absolute-20261003/`。

## 当前结果与官方实现定位

本轮完整VJP RTC与朴素异步对照已完成，均使用原始绝对动作R1；根据冻结规则改用5步去噪，六个仿真回合均因负夹爪目标停止。离线重叠proxy改善，真实标签误差没有总体改善。详见[结果报告](../../artifacts/rtc-absolute-20261003/RESULTS.md)。

官方Issue2511作者说明单位雅可比行为最初可能源于调试，后来因实机有效希望保留；PR2499仍提供更快近似选项，尚不能推断其可靠性保证。本项目完整VJP是论文对照变体，**未与官方近似做效果比较，不自动推荐为默认方案**。本轮未启用用户讨论的夹爪max(0,g)约束，也未追加实验。

## 最新范围与显式执行约束

2026-10-03用户批准max(0,g)执行端处理和三组比较，并强调当前仿真有缺陷、ACT实机基线尚未复现。已实现显式zero模式及投影后RTC承诺前缀，默认strict保留；三组效果比较改用真实记录，仿真只作接口回归。不是闭环成功验证，不据此选择默认算法。详见[执行约束](piper_rtc_execution_constraint.md)与[真实记录结果](../../artifacts/rtc-projection-offline-20261003/RESULTS.md)。
