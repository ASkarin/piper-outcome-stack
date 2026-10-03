# 独立 PiPER 仿真

仿真已迁入 `piper-simulation` 独立 Git 仓库，入口为 `piper-sim`。

- Mac：`/Users/csming/Documents/CoderLife/piper-simulation`
- 训练机：`/workspace/projects/piper-simulation`
- 安装、命令、接口与验证见新仓库 `README.md`、`docs/interfaces.md`。

主项目负责真实数据、训练业务、设备 SDK 与控制；共享纯策略由 `packages/piper_policy` 提供，仿真安装固定 wheel。原 `piper-outcome-stack sim`、`simulation` extra、`infra/simulation/setup.sh` 已移除。

只读文件导入现有实机记录；历史数据、模型、视频和日志仍在原位置，源代码和环境记录见迁移证据 `artifacts/simulation/split-20261003`（训练机 `/workspace/piper/staging/csm/simulation-split-20261003`）。历史快照不代表当前入口。本轮不部署 `piper-local` 或操作实机。
