# 项目架构与 Harness

当前使用自研 Python Responses Agent，以预构建的地图、场景描述和证据帧辅助 ObjectNav、自建 HM3D 指令与 GOAT 连续目标导航。模型自主选择查询、观察、定位与行动顺序；当前没有在线语义地图编辑、MCP／shell 或训练实现。研究目标见 [Report](Report.md)，数据设计见 [Tasks](Tasks.md)，协议与成绩见 [Experiments](Experiments.md)。

<a id="harness"></a>

## 运行流程

```text
bench.run → build：按配置和注册名装配一个 episode
              ├─ AgentRuntime ↔ ModelBackend
              │       ↕ AgentContext：历史、图片剪枝与摘要
              └─ NavigationSession
                   ├─ ToolSet → Environment → TaskAdapter
                   └─ SceneAssets：场景资料与本次运行证据
```

1. **初始化。** 读取五类配置与固定 episode 清单，加载环境、当前公开目标和工具。视觉模式提供 RGB、场景资料引用，以及真实起始位置／朝向箭头；后续不公开实时 pose、相机矩阵或 depth。
2. **决策。** Runtime 将实际上下文交给模型，追加完整 Responses output，并按返回顺序执行工具。模型可以先查询、先移动或回看证据，没有固定阶段调度。
3. **反馈。** 文本作为 `function_call_output`，图片追加为 user 观测消息；参数错误返回模型修正，环境异常退出。运动后保存真实动作和观测，不自动规划或代走。
4. **维护上下文。** 完整工具交互结束后剪枝旧图片，达到阈值时压缩历史。GOAT 切换目标时保留位置、会话和证据。
5. **结束与记录。** STOP、预算耗尽或异常后，评估器完结评分，保存 `trace.jsonl`、结果及私有状态，释放环境。纯文本声明完成只产生 `model_finished`，不会自动 STOP。

`config/harness/visual_map.yaml` 使用下面六个工具；`minimal.yaml` 是可提供位姿和深度查询、不开启压缩的原基线，二者不是相同输入条件。

## 工具与空间记忆

| 工具 | 输入 → 输出与约束 |
|---|---|
| `nav_observe` | `mode: single/panorama` → RGB；全景返回向右 0/60/120/180/240/300° 六视角拼图及原图引用，不改变最终姿态或动作计数 |
| `read_text` | 文本资源引用、可选行范围 → 场景描述或运行摘要，证据链接可直接用于查询图片 |
| `view_images` | 最多十个图片引用、可选裁剪／拼图 → 对应图像；裁剪坐标为归一化 `[left,top,right,bottom]` |
| `mark_pose` | `map_ref`、归一化位置、朝向点及依据 → 带模型估计箭头的新地图；不读取 GT 自动纠偏 |
| `nav_step` | 1–6 个原子动作 → 动作事实、碰撞反馈与末帧 RGB；每步保存图像，碰撞／目标切换可中断批次 |
| `nav_finish` | STOP → 当前目标结束或 GOAT 下一目标；不返回成功分数 |

地图坐标以引用图片左上角为原点，向右／下增加；朝向用第二个点表示。裁剪保留原图映射，拼图不能直接作为位置标记坐标系。世界坐标内部使用 Habitat XYZ、Y 向上、本地前方 −Z；四元数为 XYZW。深度是米制光轴 Z，以上内部几何不进入视觉模式的后续模型输入。

**空间资料与对话分开保存：** 场景地图、房间／物体描述和证据帧是只读先验；本次观察、估计箭头及摘要写入 episode 资源目录。当前仅能新增位置标记，尚不能增删房间、物体或关系。`SceneAssets` 只开放注册资料和运行资源，拒绝 `raw/`、深度、私有评分及任意系统路径。

**上下文增量追加：** `visual_map.yaml` 保留最近十张普通图片，另保留最新地图和当前目标图；移出的图像仍能按引用重开，文本和工具协议保留。报告输入达到 100k token 时触发压缩，缺少用量时回退至 400k 字符；这是本地触发阈值，不代表已验证的服务容量。摘要读取已闭合上下文，替换旧历史并保留最近五个完整工具交互；不能把计划或猜测写成已确认事实。压缩由同一模型执行，计入请求与时间预算。

原始文件不随剪枝删除，图片仅在发送时编码；缓存收益以服务实际用量为准。`trace.jsonl` 保存实际输入、完整输出、工具返回、动作和压缩，私有轨迹／评分放在 `private/`。训练导出只监督可见回答与工具调用，不把不可读 reasoning 当训练文本，不补入未来证据。

<a id="interfaces"></a>

## 模块边界与扩展

| 目录／入口 | 职责及核心接口 |
|---|---|
| `src/common/` | `Registry` 与五类配置；`register/get/create/list`，只注册工厂，不存运行实例 |
| `src/dataset/episodes/` | `EpisodeSource.load` 读取 `EpisodeSpec` 与 canonical ID；不启动模拟器 |
| `src/environment/` | `Environment.reset/observe/panorama/step/finish_goal/finalize/close`；`simulator.py` 隔离 Habitat 原生依赖 |
| `src/environment/tasks/` | `TaskAdapter.reset/on_step/finish` 持有目标推进、预算与私有评分 |
| `src/harness/agent/` | `ModelBackend.complete`、`AgentContext`、`AgentRuntime.run`；仅组织模型／工具循环 |
| `src/harness/tools/` | 工具 `spec/execute`；通过 `ToolContext` 使用公开回调，不自行创建环境 |
| `src/harness/session.py` | `NavigationSession.start/execute/close`；信息过滤、资源与帧索引、调用去重 |
| `src/harness/assets.py`、`map.py` | 资源权限、裁剪／拼图／估计箭头；地图校准仅绘制初始化真实箭头 |
| `src/harness/build.py`、`trace.py` | 显式注册和装配；记录及配对当时的模型请求／响应 |
| `src/dataset/atomic/` | 原子任务构造、真实交互、私有核验与训练导出 |
| `src/bench/` | 配置入口、独立 worker 调度、评估与报告 |

六个注册表为 `episode_registry/env_registry/task_registry/model_registry/runtime_registry/tool_registry`，由 `load_builtins()` 显式导入，无自动扫描，重名报错。新增组件的最短路径：

- **数据源／任务：** 在对应域实现读取器或 `TaskAdapter` 并注册；目标推进和评分集中在任务适配器，不散落到 Agent 循环。
- **模型：** 同 Responses 协议只换模型配置；新协议实现 `ModelBackend.complete` 并保留完整可续接输出，必要时同步配置校验。
- **工具：** 实现 `spec/execute`、注册并在 Harness YAML 启用；新增公开操作经过 session，状态和调用幂等仍由 session 管理。
- **地图更新／训练：** 后续地图状态由 session 持有，更新绑定真实证据与版本；教师采集、学生运行和训练复用同一工具语义。通用训练 Dataset、SFT／RL／OPD 尚未实现。

同一 `call_id` 和参数重复调用复用结果，不能重复移动；相同 ID 改参报错。目标结束后不执行同轮剩余调用。目标真值、参考路径、未来目标、成功距离及 SR／SPL 只供初始化和评分，不得进入工具反馈。公共接口用类型标注和显式依赖注入，重依赖延迟加载。

## 配置、运行与产物

| 配置目录 | 内容 |
|---|---|
| `config/task/` | 数据源、固定选择清单、协议、成功距离与动作预算 |
| `config/environment/` | fake／Habitat、场景、GPU、传感器、步长与环境启动预算 |
| `config/model/` | 模型协议、服务、凭据、输出与 HTTP 预算 |
| `config/harness/` | runtime、工具、场景资料与上下文策略 |
| `config/run/` | 输出位置、并发数、请求／时间／上下文预算 |
| `config/dataset/` | 独立的数据构造参数 |

`load_config(paths)` 接收五个显式 YAML 路径，数据路径相对仓库根目录解析，无隐式继承或任意顺序合并。基础脚本顶部设置 `TASK_CONFIG/ENV_CONFIG/MODEL_CONFIG/HARNESS_CONFIG/RUN_CONFIG`，传给 Python CLI；脚本只负责启动，业务逻辑放在 `src/`。

```bash
conda activate vln
bash scripts/run.sh                # 基础真实任务
bash scripts/fake_smoke.sh         # fake + scripted；仍读取已安装任务数据
python -m pytest -q
python -m src.bench.probe_models --model-config config/model/qwen27.yaml
python -m src.bench.visual --stage smoke --model qwen27
python -m src.bench.visual --stage develop --auto
python -m src.bench.visual --stage validate --model sol
python -m src.bench.visual_report --output cache/visual-harness-v1/VisualHarnessResults.md
```

`scripts/visual.sh` 转发视觉 CLI；`--families objnav instruction goat` 选择任务，`--limit/--max-requests` 仅用于调试。已有冻结清单可直接运行；首次准备使用 `python -m src.dataset.prepare_visual`，新清单指定 `--output`，仅显式重建才用 `--overwrite`。`--fine-actions` 选择 ObjectNav 的 0.1 m 步长，默认 0.25 m，二者分开对照。协议、冻结条件及正式验收见 [Experiments](Experiments.md)。

**并发隔离。** `run.num_workers` 默认为 1；真实 Habitat 每个 episode 使用新的 spawn worker，父进程调度与汇总，Agent 内部调用仍串行。`environment.gpu` 与 `gpus` 互斥，多卡按槽位循环分配，例如四个 worker 配 `[2,3]` 使用 `2,3,2,3`。并发不会自动限制 API 速率；GPU／EGL 配置记录在 `worker.json`。原生失败／损坏结果计为 `worker_error`，超时计为 `worker_timeout`，保留在选定样本分母中，不自动重试。Ctrl+C 回收 worker、保存已有结果并退出。

**资源位置。** `data/` 为整理后的数据，`model/` 为模型资产，二者及 `runs/` 可使用符号链接；`runs/<run>/episodes/` 保存可追溯交互、`worker.log` 和结果。`cache/` 放可重建清单、生成报告及分类历史归档，不保存唯一原始记录。`tests/` 是常规测试入口；`docs/` 只保留报告、架构、任务、实验四份正文及示例资源。
