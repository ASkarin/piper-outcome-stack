# SmolVLA真实场景只读采样与3060影子推理

2026-10-17交接：下文为既有开发记录；阶段3按[固定接口](stage2_interface_contract_20261017.md)准备共同数据与评测预算。本次不自动重启历史运行或已删除的监控。

2026-10-03。用户要求继续真实场景影子验证，并明确当前仿真未复现ACT实机成功基线，因此本步骤不采用仿真效果作为判断依据。

## 当前交付与状态

控制机任务目录为`/home/fff/piper-runs/smolvla-shadow-20261003`。新代码在该目录的`src/`，不覆盖已验证ACT开发源码、活动采集配置或release，不安装/升级包。已核对现有私有Python使用Torch2.11.0+cu128、LeRobot0.6.0、Transformers5.5.4、Accelerate1.14.0，GPU为RTX3060 12GB。

`robot-read-only.json`来自当前operator record配置的独立副本：显式改为read_only，并复制原安全数值；相机、场景和实测capture_timing保持来源值。无设备配置检查已通过。模型R1 10k、VLM配置/分词器和一对既有记录输入，由控制机自身的piper-training alias直接拉取，后台独立运行，不经Mac中转。

模型仍在传输时不能称为预检完成。transfer-status.json记录直传状态；独立的prepare-after-transfer.py等待完成后运行一次记录输入GPU预检。该预检不连接相机或机器人；发现传输错误、GPU占用或预检错误时保留日志，不自动重试。preparation-status.json为ready_for_operator_capture只代表已可准备只读现场采样，不是运动验收。

已按用户要求建立每5分钟只读完成通知（automation smolvla）。完成或明确错误时通知并暂停自身；正常、等待或短暂SSH不可达不反复通知。旧B数据automation piper保持关闭。

## 现场入口——收到准备完成通知后执行

在控制机终端运行：

```bash
python3 /home/fff/piper-runs/smolvla-shadow-20261003/run-live.py
```

该入口需要现场sudo认证进入既有CAN namespace。依据规划仓库`docs/operations/piper_admin_guide.md`与`piper_query_access.md`，助手只可直接调用固定piper-query，不能代执行通用piper-socketcan exec；因此实时采样由现场操作者启动。

程序不使能、不接管保持、不发送位置目标。connect会按既有SDK读取固件/反馈，可能发送固件查询，并连接D435；不改变CAN、驱动、模式或机械臂运行参数。使用现场已准备好的机械臂状态，不把影子程序当作保持控制器。

## 两阶段流程

1. `shadow_capture capture`使用明确read_only配置，以50Hz读取现有SDK的时间校验观测。默认采样8对RGB/state，间隔25个采样tick，每对起点相隔100tick，共726次读取，约14.52秒，另有设备连接和暖机时间。只保留选中的16幅RGB，不构造action GT；保存全部观测时序、源序列与实际pair间隔。写压缩文件安排在采样后。过期、重复序列、形状错误或SDK异常直接失败。
2. 关闭设备连接后，普通用户环境运行`smolvla_shadow captured`。三种方法使用相同选中输入和采样噪声；原始绝对动作R1、5步去噪、显式夹爪zero约束。RTC前缀是**从未执行过的假设计划**，不是实际发送历史；公共delay19只用于对齐对照，不能称为现场控制deadline验证。

输出在任务目录`runs/<时间戳>/`，包含实际源代码/环境快照、命令、采样配置/时序、RGB/state、完整raw/model/execution/committed预测与噪声，以及推理时间。原始模型误差没有现场action GT可计算，因此不报告精度或抓取成功率。

--safety只做现有安全配置下的离线候选检查（索引19对照捕获state，含工作区检查）；不是未来真实状态检查或运动许可。记录被拒绝原因，不发送目标。

## 如何解释预检

preflight使用直传的已有记录输入，明确不是现场新采样。主要核对本机模型/资源路径、3060执行、投影与三种推理接口；仅一对数据，不能将其P95当作可靠尾延迟统计。模型路径和tokenizer路径通过显式vlm-assets绑定，不改写原始checkpoint。

源码相关15项不同测试在分阶段运行中通过，其中5项专门覆盖只读采样、不调用运动接口、过期/重复观测和三方法共同输入/噪声。Ruff通过。一项既有仿真约束测试最初在无MuJoCo的训练环境中缺依赖，改用已有仿真环境后通过，无安装或源码修复。

本流程是新鲜现场输入的**采样后影子推理**，不是同步现场策略控制，也不是仿真或实机闭环成功。完成后应根据真实输入、3060耗时和候选检查决定下一次现场验证范围。

## 网络断点续传专项授权（2026-10-03）

用户随后明确要求自动恢复断点续传。automation `smolvla`已恢复ACTIVE，每5分钟检查本轮状态；仅通过控制机固定resume-once.py，在无活跃收发/准备进程、日志确认网络中断且直连恢复后续传。旧失败日志/PID/状态归档于resume-attempts，分片不删除。已开始过的GPU预检及非网络错误不自动重试；该变更不允许现场采样或运动。当前规则见规划仓库docs/automation/piper_smolvla_shadow_20261003.md。
