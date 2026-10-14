# Python navigation Harness

Python 主导任务、环境、空间记忆、工具和实验记录；[TS Core](../agent-core/README.md) 保留 Pi Agent 执行能力。详细架构见 [Architecture](../../docs/Architecture.md)。

在已安装 Habitat、PyYAML、Pillow 的 Python 环境中，从仓库根目录运行：

```bash
npm ci --ignore-scripts
npm run build
python -m src.harness doctor
python -m src.harness run --case local
python -m src.harness smoke
```

`doctor` 默认离线；`doctor --online` 运行 Coding 测试。快速 smoke 顺序运行 Coding、局部 IA、压缩续接；`smoke --suite full` 增加查询 QA、ObjectNav、指令、GOAT 和局部构图。

运动中压缩单独运行 `python -m src.harness run --case moving_compaction`（24 请求／10 分钟）。使用预建地图，在同一环境与同一会话中先真实短移、记录带帧引用的判断；目标未完成时将触发点临时降至 512 token，压缩后恢复正常配置并继续完成局部事件。私有轨迹、图片、实际工具消息、配置回执和正常关闭共同核验。它不代表自然达到 100k 的实测。

```bash
python -m src.harness --config config/harness-pi/default.yaml run --case query
python -m src.harness run --case objnav --max-requests 32 --seconds 1800
```

沿用现有 `config/harness-pi` 配置。模型服务凭据由 Python 从本地私有配置读取，显式传入 Core；不写入公开 workspace。默认 high reasoning、8192 输出 token、300 秒请求超时。按用户提供的服务能力配置 128000 token 窗口，估算上下文超过 100000 token 时触发 Pi 压缩，近期交互预算为 20000 token；三项均可在配置中调整。压缩专项临时降低触发阈值，完成后恢复正常配置。历史真实测试使用 65536 窗口，不作为新阈值的实测证据。

## 代码入口

- `__main__.py`：doctor、单案例、快速／完整 smoke。
- `runner.py`：初始化与运行，连接 Core、任务和评分，保存结果。
- `session.py`：直接持有 Habitat；观察、运动、提交、公开过滤与关闭在同一线程执行。
- `tools/`：五个扩展工具的名称、参数和执行函数。
- `memory.py`：文件化地图、卡片、图片的初始化／检查／绘制。
- `core_client.py`：持续接收 Core 事件，在主执行线程处理工具回调。
- `trace.py`：实际输入输出、工具反馈和文件变化的统一记录。
- `worker.py`：episode 超时监督和后代进程回收。

查询任务只读取固定资料；局部 IA 按真实事件结束；完整导航需要 `nav_finish`，GOAT 切换目标后重新决策。模型自主决定工具顺序，程序不自动定位或规划。

工作空间为 `scene/observations/memory/scratch`，前两者只读、后两者可写。私有评分、Core 会话、请求账本和 audit 放在相邻 `private/`，模型不可访问。每次运行有独立请求账本，本轮无全局请求总限额；单 episode 限额包含摘要及失败尝试。

旧 Python Harness、旧数据生成和旧评测保留在 `src/legacy/harness`，旧入口维持原行为。新应用不依赖旧 Agent 循环。

## 连续多轮任务

`sequence` 是正式多轮入口，不拆成多个独立 episode。`sequence.py` 持有一个环境与一个 Core，会话历史、文件记忆和图片引用跨轮保留；每轮独立设置当前指令、动作／请求／时间预算及评分。`nav_finish` 仅结束当前轮，未执行的同批调用仍得到配对回执；下一轮才下发新目标。普通切换继承实际结束位姿，指定传送只修改环境位姿并提供新 RGB，历史判断保留原时刻。

私有 sequence JSON 定义起点、环境、`rounds` 和核验资料；每轮含 `id`、`kind`、`instruction`、预算、私有目标视点、可选 `teleport` 与 `route_events`。模型只收到当前任务，全部后续任务、目标坐标、门坐标和评分都留在 `private/`。当前多轮条件为无地图：初始记忆为空，提供原生 Coding 工具、`view_images` 和三个导航工具，允许自行写记忆和绘图；不调用预建地图转换或自动定位。

```bash
python -m src.bench.astra_campaign --output cache/harness-pi/astra-campaign-v2/scenario.json
python -m src.harness --config config/harness-pi/astra.yaml sequence \
  --scenario cache/harness-pi/astra-campaign-v2/scenario.json
```

`src/bench/astra_campaign.py` 只准备这次已确认的 12 轮清单；通用运行器读取任意符合上述格式的清单，私有路线评分位于 `src/bench/sequence.py`。Astra 的凭据文件 `.env.harness-astra` 只在本地保存。每轮限额含摘要和失败尝试；轮间扩展累计请求上限，保留原账本条目。预算／动作耗尽后保留失败并从实际当前位置继续下一个任务；服务／协议错误中止整条链，避免在未知执行状态下重发运动。每轮独立结果、记忆版本及传送前后真实位姿均保存，整条链不作为独立样本的成功率评测。
