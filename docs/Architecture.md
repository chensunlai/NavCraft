# NavCraft 架构归档

本文件描述**已实现的基础实验框架**，用于定位代码、理解接口及约束后续开发。后续设想留在 [设计文档](Design/Architecture.md)；修改公共接口或模块职责时同步更新本归档。

## 1. 快速入口

- 运行：[`scripts/run.sh`](../scripts/run.sh)、[`scripts/smoke.sh`](../scripts/smoke.sh)；无 GPU／服务联调：[`scripts/fake_smoke.sh`](../scripts/fake_smoke.sh)。
- Python 入口：[`src/bench/run.py`](../src/bench/run.py)，`run_experiment(config)` 返回运行目录和汇总。
- 装配入口：[`src/harness/build.py`](../src/harness/build.py)，`load_builtins()` 显式注册，`build(...)` 创建一个 episode 的环境、任务、模型、runtime 和 session。
- 配置入口：[`src/common/config.py`](../src/common/config.py)，`load_config(paths)` 读取 Python CLI 显式传入的五类 YAML 路径。
- 环境：统一使用现有 `vln` Conda 环境，脚本不负责激活、不固定 Python 路径。
- 验证：`python -m pytest -q`；测试发现范围由 `pytest.ini` 限定为 `tests/`。

```bash
conda activate vln
bash scripts/run.sh         # 单任务真实运行
bash scripts/smoke.sh       # ObjectNav、R2R、RxR、GOAT 各首个固定样本
bash scripts/fake_smoke.sh  # fake + scripted；仍需读取已安装的任务数据
python -m pytest -q
```

切换模型时修改脚本顶部 `MODEL_CONFIG`：`config/model/deepseek.yaml`、`sol.yaml` 或 `qwen.yaml`。所有实验参数放在五类 YAML 中。

```text
bench.run：读取配置／选任务／组织 episode／汇总
  └─ build：按注册名装配
       ├─ AgentRuntime ↔ ModelBackend
       └─ NavigationSession → Tool → Environment → TaskAdapter
```

真实 Habitat episode 使用独立 worker，隔离 Habitat 的原生崩溃；fake 在单 worker 时直接运行，并发时也使用独立 worker。父进程调度和汇总结果，每个 worker 独立拥有当前 episode 的环境、模型连接、上下文和观测文件。

## 2. 目录与代码导航

```text
src/
├── common/                     # Registry、五类配置、JSON／脱敏
├── dataset/episodes/           # EpisodeSpec、选择清单、三类任务读取器
├── environment/
│   ├── base.py                 # 公开观测、目标、Transition、Environment 契约
│   ├── fake.py                 # 轻量环境，复用真实任务生命周期
│   ├── habitat.py              # 任务规则与模拟器交互
│   ├── simulator.py            # Habitat-Sim 原生边界；真实 reset 时才导入
│   └── tasks/                  # TaskAdapter、ObjectNav／VLN-CE／GOAT、指标
├── harness/
│   ├── agent/                  # 单次模型后端、追加式上下文、工具循环
│   ├── tools/                  # 工具声明与参数检查
│   ├── session.py              # 公开操作入口、帧索引、调用去重
│   ├── trace.py                # 事件写入、模型决策读取
│   ├── build.py                # 依赖装配与关闭
│   ├── prompts.py              # 系统约束、观测消息
│   └── map.py                  # MapAsset 类型，尚未接入工具／模型输入
└── bench/                      # episode 执行、评估、汇总、worker 隔离
config/{task,environment,model,harness,run}/
scripts/                        # 配置引用与 Python 启动；不放实验逻辑
tests/                          # CPU 契约／集成回归测试
doc/Design/                     # 后续方案；不代表已实现能力
data/、model/、runs/             # 外部资源／实验产物，可采用符号链接
```

`train` 及其他尚空的预留目录没有可运行训练实现。

## 3. 类与接口索引

| 类／接口 | 源码 | 主要接口与状态归属 |
|---|---|---|
| `Registry` | [common/register.py](../src/common/register.py) | `register/get/create/list`；只保存工厂，不保存运行实例 |
| `ExperimentConfig` | [common/config.py](../src/common/config.py) | 五类有效配置；`snapshot()` 脱敏；`load_config(paths)` 校验并解析路径 |
| `EpisodeSpec`、`EpisodeSource` | [dataset/episodes/base.py](../src/dataset/episodes/base.py) | 私有初始化／评分信息；`load(selection=None)` 返回指定 episode |
| 三类 `EpisodeSource` | [dataset/episodes/readers.py](../src/dataset/episodes/readers.py) | ObjectNav、R2R／RxR、GOAT 格式解码与 canonical ID |
| `Observation`、`PublicTask`、`Transition` | [environment/base.py](../src/environment/base.py) | 公开 RGB／深度引用、位姿、当前目标及动作反馈；`public_dict()` |
| `Environment` | [environment/base.py](../src/environment/base.py) | `reset/observe/step/finish_goal/finalize/metrics/close` |
| `TaskAdapter` | [environment/tasks/base.py](../src/environment/tasks/base.py) | `reset(episode,point,distance)`；`on_step/finish` 返回 `TaskUpdate`；拥有目标序号、预算、私有距离与评分 |
| `HabitatEnvironment` | [environment/habitat.py](../src/environment/habitat.py) | 将任务更新转换为公开 Transition；拥有模拟器、当前观测与目标图片 |
| `HabitatSimulator` | [environment/simulator.py](../src/environment/simulator.py) | `position/distance/step/pose/capture/target_image/close`；唯一原生 Habitat 依赖边界 |
| `NavigationSession` | [harness/session.py](../src/harness/session.py) | `start/execute/close`；拥有工具、已观测帧索引、调用 ID 去重记录 |
| `ToolSpec`、`ToolCall`、`ToolResult` | [harness/agent/base.py](../src/harness/agent/base.py) | 工具 schema／调用／公开结果与图片引用；不携带评分注释 |
| `ToolContext`、四个工具 | [harness/tools/navigation.py](../src/harness/tools/navigation.py) | `spec()`、`execute(args,context)`；工具获得公开操作回调 |
| `ModelBackend`、`ModelRequest`、`ModelTurn` | [harness/agent/base.py](../src/harness/agent/base.py) | `complete(request)`；request 包含实际上下文、工具、可选剩余超时；turn 保留完整协议输出 |
| `ResponsesModel`、`ScriptedModel` | [harness/agent/model.py](../src/harness/agent/model.py) | Responses HTTP／确定性 fake 后端；图片在发送边界编码 |
| `AgentContext` | [harness/agent/context.py](../src/harness/agent/context.py) | `snapshot/append_output/append_tool_result`；追加式上下文与字符预算统计 |
| `AgentRuntime` | [harness/agent/loop.py](../src/harness/agent/loop.py) | `run(initial,tools,trace,limits)` 返回 `AgentResult`；仅管理模型／工具循环 |
| `TraceWriter`、`TraceReader` | [harness/trace.py](../src/harness/trace.py) | `emit/close` 与 `events/decisions`；决策读取配对实际请求和返回 |
| `MapAsset` | [harness/map.py](../src/harness/map.py) | 仅保存 scene、资源路径、坐标约定；没有地图构建／查询功能 |
| `EpisodeWorker`、episode 执行入口 | [bench/worker.py](../src/bench/worker.py) | `start/poll/wait/close` 管理独立 spawn 进程；`run_episode` 运行／记录／关闭；`execute_episode` 保留单 episode 阻塞入口 |

注册表名称：`episode_registry`、`env_registry`、`task_registry`、`model_registry`、`runtime_registry`、`tool_registry`。注册时重名报错，无自动目录扫描、无隐式插件发现。

## 4. 一次运行怎样执行

1. Python 入口解析五个必填配置路径，`load_config(paths)` 按固定归属读取 YAML，校验任务／环境／模型与预算，数据路径统一相对仓库根目录解析。`EpisodeSource` 读取明确的 canonical IDs，写出有效配置和实际清单。
2. 装配 `TaskAdapter`、环境、模型和 session。真实环境加载场景、设置初始位姿，返回当前 RGB-D／位姿及当前目标；模型输入由独立 `PublicTask` 构造，同时公开可执行动作、前进步长、转向角度与滑动设置。
3. Runtime 记录实际请求，调用模型，将**完整** Responses output 追加到上下文，包括 opaque reasoning。工具按返回顺序执行；参数错误作为工具结果反馈，环境异常作为失败退出。
4. `nav_step` 执行 1–6 个原子动作；每步保存观测和动作事实，反馈最新观测。目标切换或 episode 结束时提前停止批次。`nav_finish` 执行任务 STOP，不代替模型判断成功。
5. 工具结果序列化为 `function_call_output`；图片单独追加 user 观测消息。GOAT 新目标与目标图片进入下一轮，位置和前序上下文保留。
6. 正常结束、预算耗尽或异常后，评估器 `finalize(reason)` 完结评分状态，保存私有状态与结果，关闭环境和连接。主进程持续写入逐条结果与汇总。

worker 的启动失败、原生退出、缺失或损坏的结果记为 `worker_error`，超过 worker 预算记为 `worker_timeout`，其日志保存在 episode 的 `worker.log`。`worker.json` 记录进程、实际分配的 GPU 和 EGL 配置；`egl_vendor_file` 在导入 Habitat 前生效。worker 总预算在模型运行预算外加环境启动预算及清理余量，从实际启动时计时，不包含排队时间。失败时恢复可读取的私有进度，保留在固定选择清单分母中，不自动重跑。

### Episode 并发与多 GPU

`run.num_workers` 控制同时运行的 episode 总数，默认为 `1`。父进程管理固定槽位，任何槽位完成或失败后立即补入下一个 episode。每次运行创建新的 `spawn` 子进程，不复用模拟器；Agent 内部的模型请求、工具调用和移动动作仍然顺序执行。

环境配置可使用单卡 `environment.gpu`，或非空、不重复的多卡列表 `environment.gpus`，两者互斥。GPU 编号沿用 Habitat 的现有解释方式，不改写 `CUDA_VISIBLE_DEVICES`。GPU 固定绑定槽位，按列表循环分配：4 个 worker 配合 `[2, 3]` 使用 `2、3、2、3`。各 episode 获得独立配置副本；fake 可验证分配记录，但不使用 GPU 渲染。

```yaml
# config/run/*.yaml：在现有 run 配置中添加
run:
  num_workers: 4
```

```yaml
# config/environment/*.yaml：将现有 gpu 替换为 gpus
environment:
  gpus: [2, 3]
```

以上只展示新增字段，保留原配置中的其余参数。并发数不自动限制每卡资源或 API 请求速率；这两个配置只控制当前实验的调度。

episode 目录按原始选择序号固定，完成顺序不影响 `episodes/0000` 等路径。子进程只写自身产物，父进程在每次完成后更新全局结果与汇总；`results.json` 中已完成结果按选择顺序排列。Ctrl+C 停止派发、回收全部活跃子进程，保存已收集结果并以 `130` 退出；未完成任务不计入 `finished`，`selected` 仍为原清单长度。

## 5. 必须保持的约束

### 公开／私有数据

`EpisodeSpec.goals` 中目标坐标、目标相机以及 `reference_path/gt_path` 仅用于环境初始化、目标图片渲染和评分。模型只接收公开目标、观测、工具执行事实；工具反馈不包含距离、SR、SPL、参考路径或未来目标。

GT 评分状态独立保存在 `private/state.json`。当前 Harness 不提供文件读取／shell 工具，模型不能借此读取私有状态。

### 坐标、深度与动作

- 世界坐标：Habitat XYZ，Y 向上；agent／sensor 本地向前为 −Z。
- 四元数：`rotation_xyzw`，顺序 X、Y、Z、W；同时提供 agent 与相机位姿。
- 图像像素：左上原点，`[u,v]`；光学坐标 X 右、Y 下、Z 前。
- `camera.K` 给出内参，`world_from_optical` 将光学坐标转换为世界坐标。深度为米制光轴 Z，不是欧氏射线距离。
- 前进／转向步长由环境 YAML 控制，并在初始化时提供给模型。R2R 仅前进、左右转；其他任务允许上下看。没有导航 mesh follower 或全局寻路代走。
- 渲染 GOAT 目标图片使用私有绝对相机位姿，随后恢复 agent 和已有传感器状态。

### 调用、上下文与记录

同一 `call_id` 和参数重复调用复用结果，不重复移动；相同 ID 改变参数返回错误。目标结束后不执行同轮剩余工具调用。模型纯文本声明完成只产生 `model_finished`，不会自动 STOP。

上下文目前完全追加，无摘要压缩、图片裁剪或每轮重组。达到请求／时间／字符预算明确退出。图片文件路径保存在语义上下文与轨迹中，仅在发送模型请求时转为 base64。

`TraceReader.decisions()` 配对原始 `model_request/model_response`，不补入后来的地图或总结，不丢弃 opaque reasoning。API 鉴权从轨迹、错误和配置快照中脱敏；模型 YAML 按用户要求正常纳入版本管理。

## 6. 配置、协议和评分

| Bash 变量 | YAML 内容 | 控制内容 |
|---|---|---|
| `TASK_CONFIG` | `episodes`、`task` | 数据源、选择清单、协议名称、成功距离、动作预算 |
| `ENV_CONFIG` | `environment` | fake／Habitat、场景、单卡 GPU 或多卡列表、EGL vendor、启动预算、传感器、动作步长 |
| `MODEL_CONFIG` | `model` | Responses／scripted、服务、模型、鉴权、输出与 HTTP 预算 |
| `HARNESS_CONFIG` | `harness` | runtime、可用工具 |
| `RUN_CONFIG` | `run` | 输出目录、episode 并发数、请求／时间／上下文预算 |

配置不做任意顺序合并、不隐式继承。Bash 不接受外部参数，顶部定义五个大写变量，再通过 Python 的 `--task-config/--env-config/--model-config/--harness-config/--run-config` 传入；Python 不从环境变量获取这些路径。默认模型 DeepSeek；可切换 Sol／Qwen 配置。

本版协议明确命名为 AstraBench 派生协议：ObjectNav 阈值 0.1 m；VLN-CE／GOAT 3 m；GOAT 55 词描述过滤及预算显式写入 YAML。协议参数随结果保存，不能混称所有官方默认条件。

VLN-CE nDTW 仅使用对应 GT 的 `locations`；缺失时返回不可计算，不用稀疏 reference path 替换。汇总同时报告可计算指标的样本数，以及包含失败样本的 `score_over_selected`；GOAT 使用 `partial_success`，其他任务使用 `success`。

## 7. 扩展和开发风格

- 新数据源：实现 `EpisodeSource._read`，在 `dataset/episodes` 显式注册，复用严格 ID 选择；不引入 simulator 依赖。
- 新任务：扩展 `TaskAdapter`，集中实现目标推进／预算／评分；不要把分支散落在 Agent 循环中。
- 新模型：实现 `ModelBackend.complete`，保留可继续交互的完整返回项；注册并在 `load_builtins` 显式导入。新增配置名称时同步调整配置校验。
- 新工具：实现 `spec/execute` 并注册；通过 `ToolContext` 增加必要的公开操作，session 管理状态和去重；在 YAML 中启用。
- 新空间记忆：后续放在 session 拥有的外部状态中；不要将其与原始对话记录或评估器 GT 混为同一状态。

公共接口用类型标注，数据结构就近定义，小型领域类，显式依赖注入。重依赖延迟导入；不吞掉异常、不多次采样择优、不将实验参数散写进源码。关闭逻辑覆盖正常退出、参数错误、模型错误和环境异常。

目前没有完整 coding-agent workspace、MCP／skill／命令执行、语义地图、空间记忆、上下文压缩、SFT／RL／OPD 实现。基础框架验收关注可靠运行、退出、记录和评分；模型导航成功率由后续实验评估。

## 8. 基础验收记录（2026-10-11）

- `vln` 环境中 54 项测试通过，覆盖数据筛选、公开／私有隔离、工具幂等、动作后异常、GOAT 切换、模型协议、评分失败及 worker 异常清理。
- GPU 2 上的 ObjectNav、R2R、RxR、GOAT 原生环境检查通过；GOAT 十个目标覆盖三种模态，目标图片渲染及目标切换保持机器人／相机位姿。
- 三个指定模型通过真实图片输入、工具调用和结果续接。Qwen 要求显式 `input_image.detail`，后端统一补 `auto` 并保留用户指定值；Qwen 的 reasoning 返回项已在真实续接中验证。

| 真实模型运行 | 请求数 | 退出状态 | 运行目录（`runs/` 下） |
|---|---:|---|---|
| DeepSeek／R2R | 32 | `request_budget` | `20261011T115346Z-smoke-r2r-deepseek-v4-flash-vision-27a6e4d2` |
| DeepSeek／RxR | 32 | `request_budget` | `20261011T115653Z-smoke-rxr-deepseek-v4-flash-vision-15805454` |
| DeepSeek／GOAT | 32 | `request_budget` | `20261011T120049Z-smoke-goat-deepseek-v4-flash-vision-21b198a1` |
| Qwen／ObjectNav | 4 | `terminated` | `20261011T120239Z-smoke-objnav-Qwen3.6-35B-A3B-4e403012` |
| Sol／ObjectNav | 22 | `terminated` | `20261011T120409Z-smoke-objnav-gpt-6.1-sol-552bde0d` |

以上运行的实际请求／返回配对、图片文件存在性和记录脱敏已检查。`terminated` 表示执行了任务 STOP，不等于成功；`request_budget` 表示按预算结束，成功与否须查看私有评分。首次 DeepSeek／ObjectNav 运行中目录被删除并重建，造成历史帧丢失，失败结果已保留；后续运行未再次发生，不将该次错误归因于模型服务。

## 9. Episode 并发验收（2026-10-11）

- `vln` 环境完整回归：86 项测试通过；`ruff check src tests --select F` 通过。
- CPU fake 环境与本地 Responses 服务通过同步屏障确认两路 episode 请求同时运行，四个 episode 使用独立子进程、上下文和产物；scripted 串行／并发评分一致。
- 覆盖并发上限、完成即补位、GPU 槽位配置、乱序汇总、独立超时、启动失败、损坏文件及评分字段、Ctrl+C 回收和已完成结果保存。
- GPU 分配通过配置与进程记录验证；本次未运行真实 Habitat 多 GPU 渲染，也未使用付费模型 API。
