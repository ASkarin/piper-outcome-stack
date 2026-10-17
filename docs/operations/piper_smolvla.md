# PiPER SmolVLA 开发训练与测试

2026-10-17交接：下文为既有开发记录；阶段3按[固定接口](stage2_interface_contract_20261017.md)准备共同数据与评测预算。本次不自动重启历史运行或已删除的监控。

用户于2026-10-02批准搭建SmolVLA、使用现有A40k数据微调并测试。当前只做训练机上的开发微调、离线动作评估及逻辑时间仿真；实机测试由现场操作执行，不沿用仿真成功作为运动验收。

## 固定输入与来源

- 机器人预训练来源：`lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28`。官方站连接超时，本轮通过显式 `https://hf-mirror.com` 下载，来源记录在stage的`sources.json`。
- 内部VLM保持 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`；单独固定其配置/分词器资源，VLM和动作专家都读取机器人checkpoint。训练前smoke及推理重载显式使用strict；r1训练经官方factory的默认非strict入口，独立核对确认源权重与初始化checkpoint的500个键及shape一致，抽检视觉/语言/专家/投影9处权重一致，训练日志无缺失键告警。代码设置`load_vlm_weights=False`仅避免再次单独加载VLM权重，不代表随机初始化动作专家。
- 训练沿用A40k最终52回合、62030个保留帧；开发集仍是13回合、16724个保留帧，来源会话不重叠。原始视频和记录不改写，逐回合未来action不跨边界，尾部填充有mask。动作是原记录的绝对七维rad/m。
- 模型只消费`observation.images.d435`的640×480 RGB、七维状态及源任务文本 `Grasp the green block, lift, place and release.`，不做图像别名、空相机补齐或Aloha变换。模型内部32维padding是官方结构；输出解码后只保留真实七维。
- 正规化仅从所选训练帧计算；SmolVLA使用官方预处理和flow matching。冻结VLM，微调动作专家和投影；当前总参数450046176，可训练99880992。

stage根为 `/workspace/piper/staging/csm/smolvla-a-20261002`。`spec.json`记录验证配置；当前训练为`spec-r1.json`，固定来源、划分、seed42、batch32、10000步、4小时上限；每1000步记录208个开发锚点的loss，每2500步保存。首发在官方trainer的eval_split配置验证阶段退出，未执行训练更新；修正版声明13/65的开发回合比例，但实际读入仍为原来的独立52/13视图，没有重新划分。没有按开发集延长预算或选择final test；只评最终完成checkpoint。相同最终数据也不等于相同训练历史、预训练或计算预算，不能由此宣称正式公平优势。

## 环境与独立验证命令

alias为`piper-training`，实现目录 `/workspace/projects/piper-outcome-stack`，解释器为该目录`.venv/bin/python`。复用已安装Torch2.11.0+cu128、LeRobot0.6.0、Transformers5.5.4、Accelerate1.14.0，无共享依赖修改。环境说明为stage的`env-spec.json`。

以下命令在训练机执行。它读取真实预训练权重和数据，做一次batch32反向更新、检查参数变化、保存/重载并用相同随机种子比较50×7输出；只生成独立验证工件，不是正式训练或任务成功评估。指定输出和run ID必须尚不存在：

```bash
cd /workspace/projects/piper-outcome-stack
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 /workspace/piper/bin/piper-gpu-run \
  --gpus 0 --run-id smolvla-a-doc-witness-20261002 \
  --repo /workspace/projects/piper-outcome-stack \
  --python /workspace/projects/piper-outcome-stack/.venv/bin/python \
  --config /workspace/piper/staging/csm/smolvla-a-20261002/spec.json -- \
  /workspace/projects/piper-outcome-stack/.venv/bin/python -m piper_outcome_stack.smolvla_training smoke \
  --spec /workspace/piper/staging/csm/smolvla-a-20261002/spec.json \
  --output /workspace/piper/staging/csm/smolvla-a-20261002/doc-witness
```

通过条件：命令返回0，输出`SMOLVLA_WITNESS`，`doc-witness/summary.json`为`status=passed`、`decoded_shape=[50,7]`、`state_projection_update_max>0`、保存重载差≤1e-5且梯度有限。发现问题保留输出，诊断后使用新run ID，不覆盖失败证据。

## 训练和评估入口

独立验证通过后，在GPU runner管理下运行：

```bash
/workspace/projects/piper-outcome-stack/.venv/bin/python -m piper_outcome_stack.smolvla_pipeline \
  --spec /workspace/piper/staging/csm/smolvla-a-20261002/spec-r1.json
```

本会话由训练机独立后台启动runner，进程和日志落盘；不会依赖Mac持续SSH。pipeline依次调用官方训练、初始化/微调后/ACT40k三个相同开发锚点的离线评估，再运行一次40模拟秒诊断。每阶段失败立即报告并停止，不自动重训。

官方trainer仍负责优化、scheduler和checkpoint生命周期。本项目只提供保留帧视图、PiPER处理器、固定预算及来源记录。`smolvla_training smoke`中的一次参数更新不会用于正式微调；训练重新从原机器人checkpoint开始。

离线结果保存逐锚点完整预测、真实action、padding mask、state及源索引。报告关节角度MAE、夹爪毫米MAE、首动作的记录数值边界违反数和3090推理延迟；这些不等于真实抓取成功率，也不构成整个action chunk的硬件验收。训练/开发原始同源片段不会跨划分。训练期间官方eval_loss按batch等权平均，末批16个样本与其他32样本批次权重相同；它只作过程监控，最终另按保存的有效目标计算物理单位误差。

仿真执行每次预测50步、每10个控制步重新推理，不做ACT的时间集成，不裁剪动作。场景明确设为零观测到指令延迟的逻辑时间诊断，未模拟SmolVLA真实推理延迟；3090延迟也不能当作3060实机延迟。ACT入口默认仍为每步推理、TE=0.01。

当前checkpoint的配置/处理器引用训练机stage的固定`vlm-assets`目录。跨机部署时需要同时传输这份资源、按真实目标目录显式重绑定并复核；本轮不声称只复制权重即可在控制机部署，也不访问CAN、相机或Xbox。

## 本轮完成结果（2026-10-03）

R1 已完成10,000步及全部预定诊断，总耗时约64.5分钟。微调首动作关节/夹爪MAE为0.7135°/2.4029mm，ACT40k为0.1049°/1.4295mm；微调版3/208首动作夹爪越界。仿真在1.64秒因负夹爪目标被拒绝，任务失败；没有实机测试。3090推理P50/P95约240/305ms，尚不能直接接入50Hz同步控制。详见[结果报告](../../artifacts/smolvla-a-20261002/RESULTS.md)。

上述R1命令已执行，不应重复启动；现有pipeline/training输出会在写入前被拒绝。新运行应明确提供新的run ID和输出目录，不覆盖已有结果。R1完成后添加的启动保护不改变已保存的实验结果。

## 2026-10-03 后续修正

新增显式 `action_encoding=sqrt_gripper` 训练表示：夹爪标签先取平方根、随后使用训练帧统计量正规化，预测反正规化后平方还原成米。编码/解码步骤随官方processor一起保存和加载；旧absolute checkpoint保留其原处理器。它需要新的适配训练，不能对旧权重输出直接平方来冒充修复。数值边界检查仍作用于物理action。

仿真可显式选择 `--execution async-measured`，配合SmolVLA、`--action-steps 10`和`--temporal-ensemble-coeff none`。默认同步模式不变。新模式按每次实测推理耗时确定动作块生效tick，跳过观测之后已过时的动作，等待时继续执行上一块尚有效的动作；首次等待保持初始目标，队列耗尽或结果过期即停止。它是离散事件仿真，未验证真实50Hz控制线程；不包含相机/通信延迟，场景渲染耗时不计入物理时间。

原始推理耗时保存在新的离线 `predictions.npz/inference_seconds` 与仿真 `summary.json/inference_events`，旧结果不补造。修正实现及验证范围见[本轮说明](../../artifacts/smolvla-a-sqrt-20261003/IMPLEMENTATION.md)。

本轮1000步适配与单次异步事件仿真已经结束：208个离线首动作均未越界，但首动作夹爪MAE增至2.6557mm；仿真在1.72秒的新旧块切换处因第1关节目标与state相差11.2613°而拒绝。未抓取成功，未放宽边界或追加训练。详见[结果](../../artifacts/smolvla-a-sqrt-20261003/RESULTS.md)。

## RTC接入与基准纠正（2026-10-03）

用户明确要求原始绝对动作SmolVLA R1 10k作为本轮模型比较基准。sqrt_gripper仅保留为探索记录，不作为默认方案；ACT40k仍是已有实机成功证据的参考。RTC与朴素异步均使用原R1相同权重，不新增训练；负夹爪输出继续记录并拒绝。实现和实验范围见[RTC说明](piper_rtc.md)。
