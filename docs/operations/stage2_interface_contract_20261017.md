# 阶段2固定接口与阶段3接入契约（2026-10-17）

本契约固定当前已实现行为，供ACT、SmolVLA与独立仿真仓库衔接；不修改数值安全限制、历史数据或既有冻结评测协议。

## 数据与坐标

- 公共state/action按J1—J6、夹爪顺序，七维；关节rad，夹爪两指总开口m。Dataset动作是绝对真实下发目标或已确认保持目标。
- 当前B策略输入：`observation.images.d435`，640×480 RGB；采集RGBD60，控制/Dataset50Hz。模型接口接收HWC uint8 RGB和7维state；批量接口为NCHW float RGB[0,1]与N×7state。相机配置的一般能力不限于一个名称/数量。
- 数值深度与RGB独立；原始Z16、尺度、时序与外参保留。深度不作为当前ACT输入；native深度不冒充RGB对齐或基座外参。
- Xbox原始封存与Dataset转换分开。转换只使用保存回合，并依据现有操作者/保护暂停标记形成连续段；正常任务等待不启发式删除。分段不能跨缺口取未来标签，padding mask必须保留。
- train/development按原会话和source trajectory隔离；同一示范的多个片段不能当作独立示范，不得跨划分。SDK目标、原始帧索引、来源会话和派生规则可追溯。

## 策略接口与冻结执行

- 纯策略模块归`packages/piper_policy`，真实观测新鲜度/硬件校验归主应用与PiPER插件；仿真在独立`piper-simulation`仓库，通过固定wheel/数据接口衔接。
- 当前四ACT是positive-gripper ACT，序列化注册名`act_positive_gripper`，七维relative处理器及原normalization随checkpoint保存。`predict`输出绝对50×7动作块；每块在其生成state处解码，不能重新锚定到后来的state。
- 已选模型执行50Hz、每步推理、绝对动作TE0.01，单回合最多60秒；目标仍须通过实时反馈/工作区/步长/时间/模式检查。不静默clip、补字段、复用过期状态或自动通信重试。
- `execution_chunk`负责生成观测与动作索引，`select_target`负责时间对齐/TE；重置清空旧块。返回的候选不是SDK发送证据。
- 固定模型资产位于控制机`/home/fff/piper-runs/b-act-fixed-20261016`，以该目录`freeze.json`及源码快照为运行权威。常用入口`piper act <skill> <1|2>`读取operator profile的显式`act_bundle`，委托冻结入口；`--verify-only`只查文件，不连接硬件。
- skill为green-left/green-right/blue-left/blue-right。这是规范文本对应的显式技能选择，不是自由语言理解；阶段3若加入解析器，解析错误进入系统级评测。
- 命令、原始动作块、SDK下发、RGB索引/文件、退出码及人工标签独立保留。1成功/2失败直接保存，3外部无效需要原因。缺人工标签不应被汇总器忽略：报告程序故障与待标注，启动前取消单列。

## 生命周期与机器边界

冻结入口仍由操作者确认启动、使能、回A及LB执行。保持/电子停止不提供断电防坠保证。import/无设备check不发真实CAN或运动；ROS不进入控制热路径。

训练只在piper-training，控制/设备只在piper-local；数据和模型由控制机通过自身已验证alias直传。环境保持主机独立；同步源码不复制venv、不修改驱动/服务/设备权限或历史release。

## 阶段3接入要求

SmolVLA输入任务文本、同一RGB/7维状态，官方处理器和真实预训练来源随run固定。内部padding不是新的机器人维度；输出解码后仍通过相同绝对七维检查。异步/RTC需要记录生成观测、延迟与有效执行范围，不能把过期块当新动作，不自动绕过队列耗尽/超限故障。

ACT与SmolVLA公平对照需重新冻结共同示范清单、模型来源、训练历史/成本、回合预算与成功语义。当前四ACT使用不同历史数据量与续训次数，不能直接与任意一次SmolVLA训练声称等预算。10月15日最终测试及后续已看见的开发场景，不再称未见holdout。

sim2real保持sim/real来源分域、相同七维物理单位；模拟观测不能制造SDK/D435实测证据。R与S+R必须用相同真实子集；S的真实标定成本单列。仿真内成功不替代S2实机到达和S3抓取迁移。
