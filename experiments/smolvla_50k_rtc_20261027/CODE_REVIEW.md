# RTC准备代码复核与实际修复

2026-10-27，按已有部署准备流程由独立Codex reviewer执行有限静态复核，medium推理，未运行硬件或修改文件。

- 发现准备脚本读取`cycle_work_seconds_p50_p95_max`但新trial未生成该字段。已在实际preflight循环记录工作时长并汇总，统计不包含sleep；复核确认字段与读取一致。
- 其余未发现阻断：RTC保留原关节目标、既有时间轴/过期/异常退出、committed归一化前缀、夹爪连续参考及逐步检查；check在设备构造前返回；真实padding检查覆盖两次十步去噪与25步既有执行前缀。无需全CI。
- 实际第一次check暴露控制机采集源码的夹爪约束接口较早、缺`last_gripper_target`。新候选中已同步两份此前通过验证的`execution_constraints.py`和`gripper_reference.py`；没有增加兼容分支、去掉参数或修改数值限值。采集checkout/旧候选保持。
- 23项RTC/timeline/padding/原始关节保持测试通过（4.53s）；上述接口对齐后9项夹爪事务及禁止设备构造的check入口回归通过（3.69s）。第一次失败trial与prepare_controller-initial.py、日志保留；仅从实际模型check阶段继续，没有重新传模型或重复已通过的RTC测试。

后续实际3060模型检查和RTC/padding结果以`preparation-status.json`、`checks/`与`rtc-validation.json`为准。静态复核和软件检查不构成实机成功证据。

实际模型检查阶段现已完成：8项入口全部结束，每项2次预测和1次RTC前缀发布，固定输入候选无拒绝；实际两个10步推理的尾维误差均0。恢复驱动拆分时另出现一次局部spec变量未传入，已改为显式operator任务映射，F821检查通过；该失败日志保留，随后从相同模型check阶段完成，未重传模型。冷准备路径也已包含两个已验证plugin覆盖文件及9项接口回归，避免重新复制较早接口。
