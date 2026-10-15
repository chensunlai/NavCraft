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
- `runtime.py`：正式持久会话接口；一次初始化，连续接收 prompt，显式关闭。
- `session.py`：直接持有 Habitat；观察、运动、提交、公开过滤与关闭在同一线程执行。
- `tools/`：五个扩展工具的名称、参数和执行函数。
- `memory.py`：文件化地图、卡片、图片的初始化／检查／绘制。
- `core_client.py`：持续接收 Core 事件，在主执行线程处理工具回调。
- `trace.py`：实际输入输出、工具反馈和文件变化的统一记录。
- `worker.py`：episode 超时监督和后代进程回收。

查询任务只读取固定资料；局部 IA 按真实事件结束；完整导航需要 `nav_finish`，GOAT 切换目标后重新决策。模型自主决定工具顺序，程序不自动定位或规划。

`nav_step` 支持 `move_forward`、`move_left`、`move_right`、`move_backward` 四个机体相对方向的平移动作，以及 `turn_left`／`turn_right`；允许俯仰的任务还提供 `look_up`／`look_down`。四个平移动作共用配置 `forward_step`（默认 0.25 m），公开状态用 `translation_metres` 声明，并保留原 `forward_metres` 字段。左右、后退保持机体朝向和相机俯仰；碰撞可导致部分位移并中断批次，动作数与实际路程照常计入任务预算。例如：`nav_step({"actions":["move_left","move_backward"]})`。扩展动作集合与先前只有前进／转向的实验应分开记录。

2026-10-15 平移验收：Python 全量 248 项通过；真实 Qa Habitat 场景检查两个机体朝向下的四方向位移、俯仰保持及三种新动作的碰撞，共 11 项通过，无模型请求。[验收记录](../../cache/harness-pi/translation-actions-20261015/result.json)

工作空间为 `scene/observations/memory/scratch`，前两者只读、后两者可写。私有评分、Core 会话、请求账本和 audit 放在相邻 `private/`，模型不可访问。每次运行有独立请求账本，本轮无全局请求总限额；单 episode 限额包含摘要及失败尝试。

2026-10-15 暂时移除 `src/legacy/` 及依赖它的旧批量评测、模型探测、原子数据采集／重放入口和旧启动脚本。当前运行统一使用 `python -m src.harness`。已有场景、样例与实验结果保留；原子数据 schema、几何判定、导出和旧结果离线报告仍可使用。旧实现及配套测试可从 Git 提交 `92ae3e0` 恢复；暂不将旧数据采集器迁移到 Pi。

## 持久会话接口

外部 bench 和未来交互终端共用 `open_session`。调用方准备现有工作空间、工具和任务；同一会话可以反复追问或继续行动，不需要每次重新打开 Core：

```python
from src.harness import PromptLimits, open_session

with open_session(repo, config, directory, workspace, system_prompt, tools,
                  max_requests=8) as session:
    first = session.prompt("熟悉当前环境", images=initial_images)
    second = session.prompt("根据刚才的观察，说明主要出入口")
    third = session.prompt("继续观察，核实不确定的地方",
                           limits=PromptLimits(max_requests=6, seconds=300))
```

`prompt(text, images=(), limits=None, on_event=None)` 返回 `PromptResult`，包含本次 `prompt_id`、回答、结束原因、本次及累计实际请求数、用时和记录位置。默认使用打开时的每次请求额度与 600 秒限制；调用方可逐次指定限额。摘要与失败请求也计数，账本不清零。实际请求、响应和文件变化保存在共同 trace，每次调用另外保存 `private/prompts/<prompt_id>.json` 与 trace 字节范围。

一次 prompt 返回、导航目标完成和整个会话关闭分别处理。普通追问保留原任务、实际位置、工具实例、Pi 历史及模型记忆；`nav_finish` 沿用原语义，普通文字不代替提交。bench 显式准备下一任务并更新工具公开状态后，可调用 `session.update_checkpoint()` 再发送下一条消息。传送也由 bench 决定。每次消息不触发额外压缩，继续沿用 Pi 阈值。

执行期间再次 prompt 或 close 会报 `SessionBusyError`；跨线程可调用 `abort()` 请求取消，等待原 prompt 收齐回执返回后再继续或关闭。导航工具仍在 prompt 调用线程执行，初始化、导航调用与环境关闭应使用同一线程。工具执行状态不明或通信未完整结束时，禁止继续使用会话。`close()` 幂等并释放 Core 与工具持有的环境；外部传入的 trace 仍由调用方关闭。

2026-10-14 验证：Python 全量 349 项、Core 47 项测试和构建通过。真实 Astra 在一个 Pi 会话中完成 3 次 prompt，使用原生文件／Python 工具和 `view_images`，总计 12 次实际请求；第 3 次调用触发一次强制压缩，之后继续读取文件、查看图片并回答，8 个工具调用全部配对，Core 正常关闭。[真实记录](../../runs/harness-pi/20261014T132745Z-persistent-coding/result.json)仅验证 Coding／图片工具和压缩续接；导航同目标追问、提交后追问、显式换任务、传送与环境保留由本地 fixture 回归验证，不将此前启动的旧序列运行计为新接口的真实导航验收。

## 连续任务案例

`sequence` 是调用上述接口的批量案例入口，不拆成多个独立 episode，也不维护第二套 Core 生命周期。案例准备一个环境与工作空间，多次调用同一 `session.prompt()`，逐任务设置指令、预算及评分。`nav_finish` 仅结束当前轮，未执行的同批调用仍得到配对回执；下一轮才下发新目标。普通切换继承实际结束位姿，指定传送只修改环境位姿并提供新 RGB，历史判断保留原时刻。

私有 sequence JSON 定义起点、环境、`rounds` 和核验资料；每轮含 `id`、`kind`、`instruction`、预算、私有目标视点、可选 `teleport` 与 `route_events`。模型只收到当前任务，全部后续任务、目标坐标、门坐标和评分都留在 `private/`。当前多轮条件为无地图：初始记忆为空，提供原生 Coding 工具、`view_images` 和三个导航工具，允许自行写记忆和绘图；不调用预建地图转换或自动定位。

```bash
python -m src.bench.astra_campaign --output cache/harness-pi/astra-campaign-v2/scenario.json
python -m src.harness --config config/harness-pi/astra.yaml sequence \
  --scenario cache/harness-pi/astra-campaign-v2/scenario.json
```

`src/bench/astra_campaign.py` 只准备这次已确认的 12 轮清单；通用运行器读取任意符合上述格式的清单，私有路线评分位于 `src/bench/sequence.py`。Astra 的凭据文件 `.env.harness-astra` 只在本地保存。每轮限额含摘要和失败尝试；轮间扩展累计请求上限，保留原账本条目。预算／动作耗尽后保留失败并从实际当前位置继续下一个任务；服务／协议错误中止整条链，避免在未知执行状态下重发运动。每轮独立结果、记忆版本及传送前后真实位姿均保存，整条链不作为独立样本的成功率评测。

## 提示词入口

- `navigation_prompt.md`：稳定的基础提示词，描述角色、资源边界、工具真实语义、坐标／证据要求及任务结束规则。
- `navigation_experience.md`：可迭代的导航经验，包括目标与入口辨认、局部通行、碰撞恢复、记忆更新及按需检索。
- `config.load_navigation_prompt()` 按“基础提示词＋导航经验”拼接为一份系统提示词；导航、空间 QA 和 `sequence` 共用此入口。
- 任务消息、`scene/index.md` 和实际工具列表说明本次条件：有没有地图／初始位姿、能否运动、如何结束。`sequence` 只负责同一会话中依次下发任务，不维护另一套导航提示词。Coding 基座检查使用其独立的通用 Coding 指令。
- `../agent-core/pi/packages/coding-agent/src/core/compaction/compaction.ts` 的 `NAVIGATION_SUMMARY_INSTRUCTIONS`：导航摘要要求，同时用于初次、更新和 split-turn 压缩；修改后执行 `npm run build`。

2026-10-15 根据 12 轮轨迹补强通用提示词，并移除 sequence 和查询案例中的独立系统提示词：从可走地面选择接近方式、按实际动作回执更新、反复受阻后改变接近位置、按发现和纠错维护记忆、缺少历史证据时定向检索。摘要保留有效空间知识及未完成的局部状态，合并旧动作日志。不强制全景／读写频率或固定恢复动作，不包含场景路线与私有评分点。这是待验证的提示词修订，不代表已减少碰撞或提高成功率；后续对照需在同一 runner 上分别运行旧／新提示词，历史旧 runner 的成绩不能直接作为纯提示词对照。
