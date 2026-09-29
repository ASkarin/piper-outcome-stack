# 训练容器开发与实验

训练机支持直接安装、editable 调试与个人环境训练。普通开发不要求 PR、干净 main、release 或镜像构建；代码合并与正式结果各自保留必要验证。

## 账号与安装

管理员维护 `/workspace/piper/python-env`；协作者可读取共享依赖，在个人环境自由安装和覆盖包，无 sudo 或共享环境写权限。获授权的交互任务中，助手可直接使用训练容器内 sudo 安装系统工具；宿主机、驱动、Docker 和实机权限独立。

```bash
# 容器内管理员按需安装，无需先改 lockfile 或构建镜像。
sudo apt-get update
sudo apt-get install <needed-tool>
piper-python install <package>
# 同样允许直接指定目标解释器：
uv pip install --python /workspace/piper/python-env/bin/python <package>
```

`piper-python` 是便利入口，保存操作结果和包清单，不是唯一包管理器。稳定采用的项目依赖再更新 pyproject/lock；不要对运行中的任务修改其依赖，改用个人环境开展实验。

## 一次性建立 editable 环境

管理员当前源码位于 `/workspace/projects/piper-outcome-stack`，允许未提交修改；保留已有内容。协作者使用 `/workspace/users/$USER/src/piper-outcome-stack`。共享环境不绑定任何人的 editable 源码。

在自己的开发目录执行一次：

```bash
uv venv --python /workspace/piper/python-env/bin/python .venv
# 私有包优先；显式复用共享环境的 site-packages，避免重新安装 PyTorch。
.venv/bin/python - <<'PIPER_DEVELOPMENT_PATH'
import pathlib
import subprocess
import sysconfig
shared = subprocess.check_output([
    '/workspace/piper/python-env/bin/python', '-c',
    'import sysconfig; print(sysconfig.get_path("purelib"))',
], text=True).strip()
(pathlib.Path(sysconfig.get_path('purelib')) / 'piper-shared.pth').write_text(shared + '\n')
PIPER_DEVELOPMENT_PATH
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
python -c 'import piper_outcome_stack; print(piper_outcome_stack.__file__)'
piper-outcome-stack --help
```

这里的 `.pth` 只添加共享依赖搜索路径，不复制环境、不连接其他 checkout。原生 venv 的 `--system-site-packages` 并不会继承另一个 venv 的包，因此不能代替上述共享依赖路径。新增包直接安装到 `.venv`；需要完全独立依赖时可使用不带该路径的个人环境。

修改普通 Python 源码后重启进程即可。依赖、包布局或 console entry point 改变时才重新安装。直接调用 `.venv/bin/python` 或激活后调用 CLI，避免误用其他目录源码。运行期间不要修改正式任务使用的代码和依赖。

## GPU 开发与正式实验

```bash
# 自动生成 run ID，允许未提交代码。编号和 UUID 均可用。
piper-gpu-run --gpus 0 -- python -m your_training_module
piper-gpu-run --gpus 0,1 --python "$PWD/.venv/bin/python" -- \
  python -m torch.distributed.run --standalone --nproc-per-node=2 train.py

# 正式运行可以使用尚未合并的个人分支提交；不要求 PR、main 或 release。
piper-gpu-run --formal --gpus 0 --run-id EXP-A001 --repo "$PWD" \
  --python "$PWD/.venv/bin/python" --config /path/to/config.json \
  --dataset-manifest /path/to/versioned-dataset/manifest.json -- \
  piper-outcome-stack train --config_path=/path/to/config.json
```

`--formal` 要求干净提交、配置和数据 Manifest；研究协议、数据版本与划分的实质有效性仍由实验准备和数据审计负责，入口不冒充研究验收。开发与正式运行均保留 GPU 协作锁，不终止其他人的进程。

`--python` 默认为激活环境或 PATH 中 Python。Python 命令、Python console script 和 torchrun 使用选定解释器；shell 包装脚本应继承 PATH，不在内部切换到未记录的环境。`--repo` 同时指定 Git 来源与执行目录。

每次运行保存在 `/workspace/piper/runs/<user>/<run-id>`：Git commit/dirty 状态、binary diff、未跟踪源码/配置副本、实际 Python 与包清单、GPU UUID、命令、传入配置/Manifest 副本、合并 stdout/stderr 日志及退出结果。未跟踪源码归档包含 Python、shell、JSON/YAML/TOML/INI/CFG；数据、模型、编辑器状态和大工件不混入源码快照。需要记录的其他运行输入通过配置和数据版本显式引用。旧 run ID 不覆盖。

## 数据、网络和诊断

开发可以直接使用已完成写入的本地数据、checkpoint 和 Hugging Face 缓存。正式实验引用协议要求的固定数据版本，不强制经过 sudo 晋升或重复复制。只有发布共享只读数据/模型时才使用 `piper-artifact-promote`；保留已有工件和失败证据。

官方 `hf download`/Python API 可直接使用。`piper-artifact-fetch --repo owner/name --type model --revision main` 接受 commit、分支或 tag，先解析精确 commit，再下载并记录来源。默认官方 endpoint，`HF_ENDPOINT` 或 `--endpoint` 可显式选镜像；不自动换源。实际下载失败明确报告，空间不足不由固定 200 GB 门槛猜测。

登录与运行器不强制设置 `HF_HUB_OFFLINE`。需要离线时自行设置；W&B 默认 offline，显式设置的 `WANDB_MODE` 不被覆盖。普通训练不自动上传数据或结果。

```bash
piper-env-doctor --repo "$PWD" --python "$PWD/.venv/bin/python" --json
piper-env-doctor --network --json   # 需要诊断联网时再运行
```

doctor 记录实际环境、可见 GPU、磁盘和共享内存；不要求使用三卡或连通指定镜像站。磁盘余量/共享内存建议只作提示，真实不可用资源明确报错。

## 工具更新、镜像与持久化

- 经相关测试后直接安装更新到 `/workspace/piper/bin`、`lib`、`profile.sh`，无需项目 release 或重建镜像。初始化脚本只补缺失工具，不覆盖已更新版本。
- `/workspace` 中的源码、个人环境、共享依赖和运行结果跨 restart/recreate 保留。容器层的 apt 包普通 restart 保留，recreate 会丢失；新增系统工具的包名和安装命令记在现有操作记录，重建时照此安装。无需每次安装都发布镜像。
- 本次流程优化没有新增系统包。基础镜像中已有的工具清单见 Dockerfile；不创建空的额外包管理框架。
- `container-build` 在构建输入变化或 workflow_dispatch 时构建；无关源码/文档变化只完成配置检查并返回成功，保留必需检查名称。镜像发布仍由管理员显式触发。
- 日常验证运行相关测试；阶段合并/正式发布执行必需检查。完整多卡/NCCL 验收不是每次调试前置条件。

## 宿主机部署边界

仅明确授权的宿主机维护任务使用 `dell`。`.env` 中的实际用户名、地址、端口、挂载和凭据不进入 Git，权限为 0600。沿用现有普通镜像引用与 Compose 隔离；不开放 privileged、host IPC、Docker socket 或实机设备。

```bash
./piper-compose config
./piper-compose pull       # 显式下载镜像
./piper-compose up         # 不自动拉取
./piper-compose restart    # 不拉取、不重建
./piper-compose recreate   # 显式重建；先核对运行任务与容器内额外工具
```

基础设施验收见 [acceptance/README.md](acceptance/README.md)，不是日常开发启动清单。
