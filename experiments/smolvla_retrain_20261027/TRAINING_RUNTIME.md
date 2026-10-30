# 本轮运行入口

环境复用`piper-training:/workspace/projects/piper-outcome-stack/.venv`，既有ledger为主实现仓库`.aris/compute/piper-training.md`。没有构建或修改依赖；本轮实际使用的主项目与纯策略代码冻结在stage的`source/`，由`PYTHONPATH`选择。

两个main为从step0开始的官方SmolVLA训练，固定run ID、独立GPU锁和50000更新。`pipeline.py`已经独立后台启动，**不要再次启动同一coordinator或同名run**。查询当前状态：

```bash
ssh piper-training 'cat /workspace/piper/staging/csm/smolvla-retrain-20261027/pipeline-status.json'
```

数据准备完成后，coordinator生成并保存每个实际spec。它通过以下形式调用既有GPU runner（参数中的run_id/spec/gpu由保存的launch.json给出）：

```text
/workspace/piper/bin/piper-gpu-run --gpus <free_gpu> --run-id <run_id>
  --repo /workspace/projects/piper-outcome-stack
  --python /workspace/projects/piper-outcome-stack/.venv/bin/python
  --config /workspace/piper/staging/csm/smolvla-retrain-20261027/<run_id>.json
  -- /workspace/projects/piper-outcome-stack/.venv/bin/python
  /workspace/piper/staging/csm/smolvla-retrain-20261027/train.py
  --spec /workspace/piper/staging/csm/smolvla-retrain-20261027/<run_id>.json
```

先执行各4步的`-witness` run，两者必须正常完成，并且首批序列、数据/统计相同。main不加载witness checkpoint。每个run的`training-status.json`、`initialization.json`、registry/runner manifest和resolved config是权威证据；进程存在不算训练完成。

实际通过的无GPU采样/padding命令：

```bash
ssh piper-training 'cd /workspace/piper/staging/csm/smolvla-retrain-20261027 && PYTHONPATH=source /workspace/projects/piper-outcome-stack/.venv/bin/python -m pytest -q test_sampling.py test_smolvla_padding.py'
```

2026-10-27结果：10 passed in11.93s。数据准备、coordinator、两个训练run均有独立日志；网络失败或非网络错误保留原始日志并交互诊断，不重新执行旧34条/12条任务。
