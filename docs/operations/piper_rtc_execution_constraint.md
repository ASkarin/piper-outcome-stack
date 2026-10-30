# 原始 SmolVLA 的显式夹爪约束与真实记录对照

2026-10-17交接：下文为既有开发记录；阶段3按[固定接口](stage2_interface_contract_20261017.md)准备共同数据与评测预算。本次不自动重启历史运行或已删除的监控。

2026-10-03用户批准推进执行端`max(0,g)`及RTC三组对照，同时指出仿真尚未复现ACT实机成功基线，参考价值有限。因此本轮以**真实采集记录的离线推理**为主要证据；没有训练、控制机部署、相机/CAN/Xbox或实机命令。仿真只做使用假预测器的接口回归，不开展抓取效果评测，不用仿真成功/失败选择默认方案。

## 执行约束

`--gripper-lower-bound strict|zero`默认strict，只有显式zero启用非负投影。project_gripper先要求所有原始七维值有限，再仅对夹爪执行max(0,g)。不改原数组，不修改其余六维，不限制夹爪上限。投影后仍执行原关节/夹爪范围与单步变化检查，通过后才提交目标。

预测结果分别保留raw physical actions、raw normalized model actions、execution_actions、committed_model_actions。RTC对齐的是将执行目标：使用原checkpoint保存的Normalizer重新编码execution_actions，而不是把模型空间的夹爪直接写成0。raw保留用于GT误差及追踪；queue消费execution_actions，RTC剩余前缀消费committed_model_actions。当前投影重编码仅支持原始absolute checkpoint，显式拒绝历史sqrt处理器，防止语义混用。

仿真日志区分raw_target、target、submitted_sim_target与包含执行器延迟的applied_target；拒绝候选不产生submitted_sim_target。chunk日志区分旧/新原始与承诺前缀。candidate_projection记录候选截断数和幅度，不是实机发送统计。

## 三组算法

- naive：原模型无RTC。
- identity：LeRobot式单位雅可比近似g=e，不反传denoiser；沿用本项目统一的EXP、β=5及七维输出掩码。
- full-vjp：论文完整雅可比向量积。

identity的核心已与安装的LeRobot在真实七维、相同参数下验证一致，但它不是原样使用官方默认参数/填充维度策略。三组均使用原始R1 10k绝对动作权重、5步去噪和显式zero，不将完整VJP或identity自动升级为默认选择。

## 离线协议与解释

run为`smolvla-a-rtc-projection-offline-20261003`，预算900秒。复用已独立核验的父run `smolvla-a-rtc-absolute-20261003-seed42/results/pairs/`：13个真实开发回合各8对，间隔25帧，总104对。旧预测、RGB/state、采样噪声和真实GT均固定；旧预测先投影，再通过保存的action Normalizer转换成相同前缀供两种RTC使用。每对轮换方法测量顺序，各方法先5次预热。

公共比较延迟固定19tick（380ms），用于隔离同一边界条件下的引导差异；这不是保证实际deadline。另保存每次真实耗时、超过公共延迟的次数。GT指标始终使用raw预测，区分全块与第19–49项可用后缀。连续性参考为投影后的旧**模型预测proxy**，不当作GT。截断次数是预测位置/候选目标的统计；本轮实际硬件命令数为0。

真实观测离线对照仍不能说明策略执行后机器人会进入哪些状态，不能提供抓取成功率。结果用于评估输入/输出契约、误差与计算代价，并帮助选择后续真实场景验证候选，而不替代实机闭环。
