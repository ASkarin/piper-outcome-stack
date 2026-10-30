# 部署前有限独立审查（2026-10-27）

由`experiment-bridge`要求的独立Codex reviewer执行，继承主线程模型、medium推理。只审查本轮数据准备与两个fresh训练入口，未替代实际GPU witness。审查者未修改文件或启动任务。

数据准备审查结论：未发现阻断。连续15条封存恢复的重编号为identity；官方转换的source→segment episode映射匹配；frame task_index与episode tasks元数据修正形状正确；只写新Dataset，保留原始数据/telemetry；控制机自身alias直传并排除训练不消费的原始telemetry合理。

训练入口审查结论：未发现启动阻断。采样器与固定LeRobot实际调用匹配；完整窗口与239条原示范覆盖检查正确；两组明确fresh、空optimizer与首步1；原版native loss、flat辅助梯度及7/32维处理符合用户要求；434训练段/282138帧与72开发段/39来源计数一致；coordinator先完成两个4步witness，再创建两组各50k main。

作者在部署前另收紧了重载检查：使用完全相同显式noise，将4步训练后的live模型与独立checkpoint重载模型比较（1e−5容差），再检查重载重复预测；这替换了仅比较重载模型两次输出的不足。十步尾维`t×epsilon`仍要求精确一致。其他训练行为未变。

实际离线回归：训练机执行`test_sampling.py`和既有`test_smolvla_padding.py`，10项通过。GPU witness仍验证完整真实数据、实际环境、参数更新与模型重载；仅文件存在或import成功不算通过。

用户批准并行提速后的有界增量复核：只审查`prepare_parallel_20261027.py`，未发现阻断。第7来源接手原发送进程并核对接收端文件大小，第8/9来源独立转换直传；六条既有记录加三worker结果的9来源/108条/113473帧聚合及manifest发布顺序正确。未重复审查未变化的训练代码。实际启动后已确认第7来源完成，最后两个worker同时转换且无错误；原始数据和训练预算未改变。
