# PiPER 仿真 S0/S1 独立操作

本功能使用 MuJoCo 3.12.0 / Python 3.12.13，仅连接虚拟模型。当前覆盖模型转换、几何核验、桌面窗口、历史反馈回放和未辨识伺服预览，不训练策略、不调用真实 CAN/相机/手柄，也不替代实机安全验收。

## 1. 安装与运行位置

local 的独立源码目录为管理员 `$HOME/src/piper-simulation-s0-s1`。在该目录运行：

```bash
bash infra/simulation/setup.sh
.venv-sim/bin/piper-outcome-stack sim doctor --headless --output artifacts/simulation/my-doctor-01
```

安装明确使用 `.venv-sim`，不会写入原机械臂 `.venv`、共享训练环境或 release。普通 Python 修改后重启仿真进程即可；依赖变更重新运行安装脚本。环境检查拒绝硬件/训练插件被误装入仿真环境。不要加 sudo，不要在真实 CAN namespace 里运行仿真。

每次使用新的输出目录；已存在目录不会覆盖。`summary.json` 保存结果与来源；失败也留下原因。显示窗口必须从 local 已登录的图形桌面终端运行，SSH 没有 DISPLAY 时使用 `--headless`；本项目不自动配置远程桌面。

## 2. 看模型和指定姿态

```bash
.venv-sim/bin/piper-outcome-stack sim view --config configs/simulation/tabletop.json
.venv-sim/bin/piper-outcome-stack sim target \
  --joint-deg -42.488 132.772 -106.826 -4.781 43.002 66.067 \
  --gripper-mm 65 --duration 0 --output artifacts/simulation/my-target-01
```

鼠标拖动可旋转视角、滚轮缩放；窗口文字显示六轴角度、夹爪总宽度和基座系法兰坐标。关闭窗口退出。`target` 是运动学姿态展示，不是实机命令，也不是伺服到位测试；`--duration 0` 让桌面窗口持续显示。

SSH 下只保存截图：

```bash
.venv-sim/bin/piper-outcome-stack sim target --headless \
  --joint-deg -42.488 132.772 -106.826 -4.781 43.002 66.067 \
  --gripper-mm 65 --output artifacts/simulation/my-target-image-01
```

示意桌面/虚拟相机的尺寸、位姿和内参在 `configs/simulation/tabletop.json`。`simulation_only=true`、`illustrative_unidentified` 必须保留：参数未经过现场标定。公共图像名为 d435，但它是虚拟 RGB，不冒充真实设备帧。

## 3. 复现最后一次实机往返

```bash
REPORT="$HOME/piper-hardware-acceptance/20260907-final-roundtrip-02/result.json"
.venv-sim/bin/piper-outcome-stack sim replay --report "$REPORT" \
  --mode measured --output artifacts/simulation/my-measured-01
.venv-sim/bin/piper-outcome-stack sim replay --report "$REPORT" \
  --mode commanded --output artifacts/simulation/my-commanded-01
```

- `measured`：按原始反馈时间插值展示，包括实机短暂越出目标限位的反馈；原数据不裁剪，报告列出越界幅度与样本数。它不提供“控制跟踪误差”，因为画面本身由反馈设定。
- `commanded`：按原始命令时间切换关节/夹爪目标，物理步长固定 2 ms。六轴与总宽度伺服按零位质量矩阵及 2 Hz 临界阻尼设定；不补偿重力、不模拟已辨识控制器，误差可能明显。
- 原始目标仍严格验证范围与有限值；允许展示越界反馈不意味着允许下发越界目标。
- 空格暂停/继续，Q 或关闭窗口提前结束；提前结束记为 interrupted，保留已有数据，不标为完整回放。
- 显示目标 30 FPS；实际速度以报告为准。渲染卡顿不会改变物理步长，但可能让窗口慢于真实时间。

## 4. 一次生成两类结果与对照图

```bash
.venv-sim/bin/piper-outcome-stack sim compare --report "$REPORT" \
  --output artifacts/simulation/my-compare-01
```

此命令默认无窗口。输出包括：

- `summary.json`：源报告、实际源码/配置/版本、几何检查和两类回放结果；
- `measured.csv`、`commanded.csv`：模拟时间、插值的参考七维反馈和仿真七维状态；插值行不计成新的实测样本；
- `*-summary.json`：完成情况、时间因子、渲染时间、接触计数/最大穿透与误差；
- `measured-0/1/2.png`、`commanded-0/1/2.png`：开始、代表性中间时刻和末尾截图；不是三个端点验收证书；
- `tracking.png`：蓝线为原始反馈插值，红线为未辨识伺服结果，关节 rad、夹爪 m。

命令驱动预览的误差不应解释为真实控制器误差；接触次数是物理步上的接触记录累计，不是独立碰撞事件数。MuJoCo 警告、时间重置或非有限状态会中止并保留结果。

## 5. 模型与测试

```bash
.venv-sim/bin/python -m piper_outcome_stack.sim.model
MUJOCO_GL=egl .venv-sim/bin/python -m pytest -q tests/test_simulation.py
```

转换来源、MIT 许可证、夹爪映射及 SDK FK 数值参考见 `assets/piper/README.md`。URDF 与 MJCF 几何一致性要求 1e-6 m / 1e-6 rad；SDK MDH 差异另列，不修改模型掩盖差异。模拟关节范围取官方模型与已有控制器读数的交集，尤其 J6 保留模型更窄的范围；这些不是新实机安全限值。

原始模型惯量保留，但惯量、执行器/接触、桌面、相机尚未整体辨识或现场校准。通过 S0/S1 软件验收不等于 S2 策略迁移完成；后续训练和实机迁移需独立任务。
