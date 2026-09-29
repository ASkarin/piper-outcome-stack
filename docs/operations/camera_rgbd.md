# 相机配置、RGBD 采集与模型输入

2026-09-21现状：默认Xbox为640×480 RGBD60／控制与Dataset50Hz；本文其他尺寸和历史20Hz只代表对应示例/会话。Xbox原始RGB/Z16快速暂存，离线转换再编码/压缩；示教保存仍走独立流程。默认模型消费RGB与七维状态，不因保存深度自动增加模型输入。

当前配备一台 D435，是默认示例，不是代码中的设备/名称/数量限制。复用固定 LeRobot RealSense 后端，支持配置多个 RealSense、序列号或唯一设备名，以及 RGB、深度或 RGBD。没有增加其他相机后端；设备必须实际支持所选 profile。

## 采集与导出选择（2026-09-11）

日常调试默认RGB；重要示范可设置`robot.cameras.<name>.use_depth=true`保留RGBD。项目`record`默认只生成RGB视觉列；已采集深度以一份Z16、尺度及时间/几何元数据归档。`--export_depth=true`才额外生成米制TIFF列。旧含深度Dataset续录时需显式匹配该选项，不能静默改列。

深度可按需读取：`depth_m = raw_z16.astype(np.float32) * depth_scale_m`。示教数据使用`teach-convert --include-depth`离线导出RGBD Dataset；默认转换为RGB基线，不复制Z16。已有数据不删除、不重写。详见[示教说明](piper_teach_collection.md)。

## 配置

`configs/cameras/d435_rgbd.json` 是 **cameras 配置片段**，不是可以直接运行的整份机器人/录制配置。将其内容放入现有配置的 `robot.cameras`：

```json
{
  "d435": {
    "type": "intelrealsense",
    "serial_number_or_name": "Intel RealSense D435",
    "width": 640,
    "height": 480,
    "fps": 30,
    "use_rgb": true,
    "use_depth": true,
    "color_mode": "rgb"
  }
}
```

`d435` 可改为 `front` 等逻辑名；型号名也可替换成实际序列号。多个同名型号连接时，用序列号明确选择设备。相机配置的 fps 可以高于 Dataset fps；不能低于需要的采样率。实际供帧、帧龄、图像/状态偏差继续检查，不重复图像凑帧。没有相机数量或必须数字序列号的额外审批条件。

## 数据是什么

以逻辑名 `front` 为例：

| 数据 | 内容 |
|---|---|
| Robot `front` / Dataset `observation.images.front` | RGB uint8；后端即使设置 BGR，机器人观测仍转为 RGB |
| Robot `front.depth` / 可选Dataset `observation.images.front.depth` | float32米，H×W×1；显式导出深度时使用官方TIFF保存 |
| telemetry 内 `raw_depth/attempt-*/frame-*-front.depth.npz` | 原始Z16 uint16计数，录制阶段未压缩、离线无损压缩后读回验证；保留图像旋转排列 |
| 逐帧 `raw_depth` 记录 | 文件路径、形状、设备实际 `depth_scale_m` |
| 每流相机元数据 | 帧号、设备时间/域、接收/发布时间、实际序列号、像素格式、原始内参、深度到彩色外参、图像旋转 |

RGB 可以继续编码为视频，深度不使用上游默认的 12-bit 视频量化。原始计数与实际Dataset行关联，用于精确追溯；米图按需生成，默认不保存两份深度。额外原始文件会增加空间/写入开销，正式采集前测量当前配置的实际吞吐。

深度零值保留，表示没有有效深度；不进行孔洞填补或伪彩色替代。保存的是 **native 深度网格**，不是已经对齐到彩色的像素网格。内参对应旋转前的 native profile；若设置图像旋转，做几何计算前应将坐标转换回该网格。`depth_to_color.rotation_column_major` 与 `translation_m` 是设备提供的两传感器外参，不是相机到机械臂的外参，也不证明硬同步。

## 读取与审计

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id, root=dataset_root, depth_output_unit="m")
frame = dataset[0]  # 仅适用于已显式导出深度列的Dataset
depth_m = frame["observation.images.front.depth"]  # (1,H,W), metres
valid = depth_m > 0
```

固定官方 LeRobot 的默认读取单位是 mm；想使用米请明确传 `depth_output_unit="m"`。原始数据可用 `np.load(path, allow_pickle=False)["depth"]` 读取，再乘对应行的 `depth_scale_m`。

继续使用 `piper-outcome-stack record` 与 `audit-dataset`。审计检查已采集深度的Z16文件、尺度与dtype/形状；若导出TIFF，再核对原始计数→米图的一致性。没有TIFF列不代表已采集的Z16可以丢失。失败/中断不能标为完整；重录保留弃用 attempt 的原始文件，审计只接受保存的 attempt。新增深度字段不能直接续录到旧 RGB-only schema；新建匹配的数据版本，不补造历史深度。

## 模型输入与采集分离

推荐使用：

```bash
piper-outcome-stack train --config_path=/absolute/path/train.json
```

它只在从头训练且未设置 `policy.input_features` 时默认选择七维状态和非深度 RGB 图像，再调用官方 LeRobot `train` 函数。显式输入、预训练路径和 resume 不会被自动改写；没有复制优化器、训练循环或 checkpoint 框架。

也可以继续使用原生 `lerobot-train`，但含深度数据时应明确配置输入，例如：

```json
{
  "input_features": {
    "observation.state": {"type": "STATE", "shape": [7]},
    "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]}
  }
}
```

这是 `policy` 子配置。使用当前示例名时把 `front` 改为 `d435`；键名和尺寸应匹配自己的 Dataset/checkpoint。

深度可以用于几何处理或后续 RGBD 模型，但当前没有新增深度编码器，也没有把深度复制为三通道或拼成第四通道喂给 RGB checkpoint。`select_policy_inputs(features, image_keys=[...])` 支持显式选择可用图像字段；选择深度不代表原 RGB 网络已经兼容一通道输入。

## 验证边界

软件测试覆盖 RGBD/深度单流、设备尺度、内外参、帧重复/缺失、任意逻辑名、RGB/深度混合保存、原始像素核对、重录/续录及训练输入选择。合成测试不冒充实际 D435 的帧率、对齐或采集稳定性；实机相机检查独立记录。本次不操作机械臂，不恢复硬件验收文件或 no-drop 启动条件。

## 2026-09-09 本轮结果

- 修正已写入 Mac 工作区，并以相机范围补丁同步到控制机开发目录 `/home/fff/piper-hardware-acceptance/20260906-capture-timing/development-src`。既有其他修改保留；没有提交、发布或激活 release。
- 候选完整离线测试：323 passed / 1 skipped；相关独立仿真测试：27 passed。随后 doctor 枚举修正的相关回归：15 passed。Ruff、格式和锁文件检查通过，shell 修改通过语法检查。
- 实机只读相机验证未通过：能枚举 D435，USB 描述为 3.2，但项目 1 秒/3 秒启动等待均无首帧；原生 RealSense pipeline 等待 10 秒也无帧。不能据此确定接线是根因，也不能宣称 RGBD 实机采集成功。
- 用户不在现场，拔插检查暂缓；已停止相机读取，没有操作机械臂、CAN 或重置设备。
- 相机补丁、离线日志和失败 JSON 在 `artifacts/camera-correction/`；远端候选及日志在 `piper-local:/home/fff/piper-rgbd-9r5mq5h0/`。这是开发验证，不代替正式采集或运动验收。

### 重新连接后的结果

2026-09-09 重新连接后已恢复 RGBD。原生流在预热期间出现深度帧重复和计数重启，现按启动期处理；重复组合帧不作为新观测发布。37 项相关测试通过，真实连续 60 组读取通过。实际画面倒置且深度零值较多，工作区质量及长期稳定性仍待验证。见 `artifacts/camera-correction/reconnect-20260909/result.md`。
