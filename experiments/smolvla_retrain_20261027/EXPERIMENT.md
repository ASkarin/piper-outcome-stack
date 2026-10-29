# 原始SmolVLA与平坦FM：从官方预训练源重新训练

用户随后要求数据准备并行提速：控制机已由`prepare_parallel_20261027.py`接管，复用前6来源和第7来源既有直传；实际确认7/9完成、最后两个来源同时转换，转换结束各自立即直传。全局状态路径不变，既有数据ready发布与训练coordinator不变；模型初始化、两组50k预算、239/39划分及padding处理保持。

用户于2026-10-27批准准备完整B数据并同时训练两组，各50,000步，明确32维中只有前7维有效。范围、数据、初始化、损失与固定预算见规划仓库`docs/protocols/piper_experiment_preregistration_PR-20261027-01.md`。

训练根：`piper-training:/workspace/piper/staging/csm/smolvla-retrain-20261027`；唯一后台coordinator为`pipeline.py`，实际launch/PID和当前阶段在同目录`pipeline-launch.json`、`pipeline-status.json`。控制机数据工作位于`/home/fff/piper-runs/smolvla-multitask-recollection-20261023`，`data-preparation-status.json`记录转换/直传进度。

顺序为：9来源恢复/转换/纠正/直传完成 → 训练机完整清单核对 → 两组各4步witness → 两组从原始base重新各50,000步 → 同39条development离线评估。原始20k/22k模型、optimizer和processor均不继承。数据准备与训练进程不依赖Mac SSH连接的生命周期。

`train.py`复用官方LeRobot训练、checkpoint和Dataset生命周期。`temporal_loss.py`原样复用既有平坦辅助项；两组同一结构、同seed、同完整窗口采样与train-only归一化。两组均使用原生七维loss及既有结构padding修正。flat只多固定六关节时序项，夹爪和尾25维不受辅助项监督。

已完成部署前独立静态复核；两个新采样测试和既有padding回归在训练机实际通过：`10 passed in11.93s`。GPU更新、checkpoint重载和十步尾维路径的真实验证由独立witness run执行，尚未发生时不称通过。

main runs：

- `smolvla-b-original-fresh-20261027-seed42`
- `smolvla-b-flat-fresh-20261027-seed42`

各run权威输出位于`/workspace/piper/runs/csm/<run_id>/`；`initialization.json`证明原始base张量和step0，`training-status.json`记录实际步数/结果，`objective.jsonl`记录损失，`checkpoints-output/checkpoints/050000/`为固定终点，`evaluation/`保存统一GT比较。witness run额外有`-witness`后缀，权重不用于main。

不自动重跑失败、不新增模型/seed、不进行硬件I/O；训练完成或失败后保留全部记录。模型推理必须通过项目的32维结构padding路径，不能用未约束尾维的默认模型采样替代。
