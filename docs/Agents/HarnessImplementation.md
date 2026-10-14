# Pi Harness 实施记录

**现行版本：Python 主程序＋独立 TS Core 已迁移完成。** 最新命令、提交与真实验证见 [Python 主导迁移回传](Message/20261014-Harness-to-Main-Python主导迁移回传.md)。下文旧 TS 主导实现保留为历史记录，原 191/200 账本与旧运行结果不改写。

依据 [当前交接](Message/latest.md)、[Architecture](../Architecture.md) 和 [SpatialMemory](../SpatialMemory.md)。2026-10-14 用户授权实现，并确认先提交相关设计，再逐阶段 commit；论文及其他对话的既有改动不混入。

## 初版固定条件（历史）

Pi v1.2.0 完整 coding-agent 本地源码、根 npm workspaces；Node 内嵌 SDK，常驻 Python Habitat 通过 JSONL 通信。全部工具串行。Coding 使用 bwrap 和已有 vln Python，scene/observations 只读，memory/scratch 可写。默认 high reasoning、8192 输出 token、300 秒请求超时。

全程最多 200 次实际模型 HTTP 请求，包含探针、压缩、失败与复测。默认上下文运行预算 64k，预留 16k，近期内容约 20k；沿用 Pi 压缩并临时追加导航要求，不滚动删图。

## 阶段与验收

| 阶段 | 提交主题 | 验收 |
|---|---|---|
| 0 | docs(harness): freeze pi implementation baseline | 仅相关设计与交接 |
| 1 | build(pi): vendor coding-agent and configure workspaces | 本地包构建、实际解析、空会话 |
| 2 | feat(harness-pi): add isolated coding runtime and model tracing | 权限、Python、图片、真实工具续接 |
| 3 | feat(harness-pi): add editable spatial memory and map tools | 资料转换、图元检查、新增修订删除、坐标 |
| 4 | feat(harness-pi): connect persistent habitat navigation | fresh frame、全景恢复、批次中断、公开边界 |
| 5 | test(harness-pi): add atomic and navigation smoke cases | QA/IA、三任务族、局部构图真实记录 |
| 6 | feat(harness-pi): adapt native compaction for navigation | 一次真实压缩与继续工具调用 |
| 7 | docs(harness): record validation and return implementation handoff | 回归、可复现命令、结果与限制 |

每阶段通过检查再提交；已提交功能修复另作 fix commit。普通功能使用已有地图，构图仅做独立局部专项；历史地图、manifest 和运行产物不覆盖。

## 初版真实测试预算（历史）

接口/Coding 12、A03 查询 QA 8、A04 局部 IA 12、ObjectNav 32、指令 32、GOAT 两目标 52、局部构图 16、压缩 12、定向复测 24，共 200。未用额度可转后续阶段；总额不突破。软件失败、模型判断错误、预算结束分开报告。

## 进度

- 阶段 0：`e56a774`，相关设计已冻结。
- 阶段 1：`e43ce13`，Pi v1.2.0 单包 SDK 构建通过，本地解析与空会话检查通过；普通依赖统一安装到根目录。上游 commit 为 `06a466935536f8c9bcacb2d5efc0fc021be1bdff`。

- 阶段 2：`fbb9128`，真实 Coding 闭环通过，多图识别、脚本执行、JSON 与生成图片重读均完成。服务配置原标识 `gpt6.1sol` 返回 404；公开列表核验后显式使用 `gpt-6.1-sol`，Responses/high/8192 保持。累计请求 8 次（含无效标识与模型列表），真实成功闭环 6 次。产物：`runs/harness-pi/2026-10-14T01-54-08-585Z-coding/`。

- 阶段 3：`6a06114`，Qa prebuilt 55 实体/120 证据、partial 2 实体/4 证据、empty 0 均可检查与绘图。PNG-only 查询不要求额外图元，全景拼图标注相对方向。构图专项进一步剥离旧证据，仅保留墙层和两个区域。

- 阶段 4：`81b8164`，常驻 JSONL 环境桥完成；新帧、六视图恢复、批次预校验/中断、重复调用和目标 ID 测试通过。真实 A04 IA 修正后由 Sol 执行 5 动作/1 米，实际穿门和到达通过，无终止错误。初次输入遗漏 T 图、第二次终止 abort 错误保留为开发诊断；最终记录 `runs/harness-pi/2026-10-14T02-07-01-691Z-local/`。

- 阶段 5：`ec06216`，固定案例、独立 QA/IA/构图核验和失败结果保存完成，真实结果见下表。
- 阶段 6：`ea99fd0`，Pi 原生压缩接入完成，覆盖普通与 split-turn 摘要，缺失公开状态与图片可补回。GOAT 自然触发两次压缩后继续行动；专项另触发一次并重读笔记、观察和修订。压缩模块先于案例入口提交，确保入口的依赖已在版本控制中。
- 修复提交：`2da1dd1` 请求账本锁与按运行计数；`b6e1c39` 最新观察及文件审计；`eef27b5` 实际预算拒绝、进程清理和初始箭头边界；`c1f8909` 引用检查及派生图片软链接边界。
- 最后修复：`c0c079a`，环境关闭幂等、JSON 原子替换、等待子进程正常退出，损坏的私有结果不再阻止保存模型轨迹。
- 阶段 7：运行说明和[回传](Message/20261014-Harness-to-Main-Pi实施回传.md)已整理；实现批次最终账本 **191/200**（190 次模型 HTTP 尝试及保守计入的 1 次模型列表查询），无需继续消耗剩余额度。

## 最终软件检查

`npm run build`、`npm run test:pi`（31 项）与 vln Python 全量 `pytest -q`（248 项）通过。离线 `doctor` 确认 SDK 解析到本地源码构建产物，bwrap 中 NumPy/Pillow 可运行。验收仅运行单 worker，不干扰原后台实验。

## 真实结果与入口

以下目录均在 `runs/harness-pi/`；每条包含 `result.json`、公开 workspace、私有 trace 和环境结果。开发中修订前的记录完整保留，不合并成一次成功轨迹。

| 案例 | 最终/主要记录目录 | 请求 | 核验结果 |
|---|---|---:|---|
| Coding | `2026-10-14T01-54-08-585Z-coding` | 6 | 多图、读写、Python、结果重读通过 |
| A03 查询 QA | `2026-10-14T01-59-39-187Z-query` | 3 | D03，答案 anchor 核验通过 |
| A04 局部 IA | `2026-10-14T02-07-01-691Z-local` | 5 | 实际穿指定门并到达，5 动作/1 米 |
| ObjectNav | `2026-10-14T02-08-46-892Z-objnav` | 28 | 86 动作，显式提交成功，距离 0.0183 米，SPL 0.8672 |
| 自建 HM3D 指令 | `2026-10-14T02-15-35-386Z-instruction` | 31 | 115 动作，终点/有序途经约束通过，SPL 0.9944 |
| GOAT 前两个描述目标 | `2026-10-14T02-22-18-058Z-goat` | 52 | 第一个成功，第二个在额度用尽时未完成；链成功 0、目标完成 1/2 |
| 局部构图首次 | `2026-10-14T02-38-35-500Z-mapping` | 12 | 未通过：最终 geometry 为 null，提示与评分契约不充分 |
| 局部构图定向复测 | `2026-10-14T02-52-13-642Z-mapping` | 14 | P01 新建/新视角修订/读回重绘通过；中心误差 0.0302 米、端点误差 0.0669 米；连接/证据与图像复核通过 |
| 压缩专项首次 | `2026-10-14T02-47-41-745Z-compaction` | 12 | 压缩及后续 read/view/observe/edit 已执行；最终文字未输出，单案例额度用尽 |
| 压缩定向复测 | `2026-10-14T02-57-10-107Z-compaction` | 14 | 压缩/六次续接工具/最终回答完成；退出归档失败，另存有来源标注的 `recovered_result.json` |

构图复测仅补充最终需要近似 `segment` 的要求，未提供私有门坐标或改评分标准。图像人工核查保存在该记录的 `private/manual_review.json`；构图不要求穿门，原种子的 `local_event:false` 不代表构图失败。

GOAT 首次运行的 `last_error:"Connection error."` 是 SDK 包装后的预算拒绝，52 次实际请求后没有再发送。其较早代码将环境原因写成 `model_finished`，原记录不改写；现在按实际拒绝标记 `request_budget`。GOAT 当时的压缩检查点还可能携带旧全景；已修复，但没有将旧轨迹当作修复后的验证。

不调用模型的 GOAT 图像目标生成测试在 `cache/harness-pi/image-goal-smoke-20261014/`：512×512 目标图和相机状态恢复通过。这不是图像目标导航成功。

压缩定向复测使用剩余额度，将单案例上限设为 16，实际使用 14。最后退出时触发重复关闭写入竞态，原私有 `result.json` 为 0 字节；保留原文件，从同次运行有效的内部状态文件及 trace 恢复独立记录，不能标成已修复版本的完整成功运行。`c0c079a` 后在真实 Habitat 上无模型执行初始化→fresh observe→finalize→两次 close，结果完整且释放正常，记录在 `cache/harness-pi/shutdown-smoke-20261014/result.json`。修复后未重跑模型整段；摘要和后续模型行为使用上述真实记录核查。

## 限制与交接

- 本轮只验证 Qa 的转换及少量固定案例，不提供成功率或跨场景结论。预建地图含允许的语义信息，只有初始化位姿提示，后续无实时真值 pose/depth。
- 地图检查器核验格式与引用，不能代替空间正确性。当前局部构图的准确落点得益于已知墙层、区域和初始箭头，不代表从零重建。
- 原生压缩暂只做导航提示适配及公开检查点补充；文字无法完整保存视觉细节。摘要明确“最新”只限被压缩段，更晚尾部/检查点优先；补回地图携带原版本及 `pose_at_frame`，不伪装实时定位。
- 实际上下文与 HTTP trace 内嵌图片，长交互产物较大；后续可做内容寻址去重，当前优先保证输入可核查。
- 运行产物、依赖目录和密钥均不提交。其他对话的论文及文档改动保留在工作区。入口和使用方法见 [Harness README](../../src/harness/README.md)。

## 后续源码边界修正（2026-10-14）

用户确认 Python 主导、保留独立 TS Core，并明确要求压缩提示直接修改本地 Pi 源码。已将导航摘要要求从外围扩展移到 `pi/coding-agent/src/core/compaction/compaction.ts` 的共享摘要请求构造入口，覆盖首次摘要、已有摘要更新和 split-turn 前缀；外围不再追加提示，仍负责统一预算、审计、失败取消和公开图片/状态恢复。原生切点、格式、文件跟踪与保存机制保留；分支切换摘要未改。

这次只调整提示归属和相关设计说明；Python 启动并主导 TS Core 的控制方向尚未迁移。先前真实运行结果保留原版本含义，不作为本次源码变更后的实测。本地 Pi 构建、Harness TypeScript 检查和 36 项 Node 测试通过，其中 5 项直接调用 Core 验证首次、更新、两路 split、prefix-only 和额外指令；每个请求只含一份导航要求。未增加模型请求，预算账本仍为 191/200。

## Python 主导迁移（2026-10-14，现行版本）

按用户批准的计划先测试再迁目录：`90945db` 贯通 Python 工具回调并实际测试；`56d9052` 直接运行 Habitat 局部 IA；`5fde029` 将 Python 主程序统一到 `src/harness`、完整 TS Core 移至 `src/agent-core`、旧实现放 `src/legacy/harness`；`a7cb9e5` 补齐收尾与运行配置回归，`8667207` 将 Core 默认总次数限制改为无上限，单 episode 限制保留。

导航工具的名称、JSON Schema、执行函数均在 Python；Core 仅通过通用回调执行。Habitat 在 Python worker 主线程初始化、运动和关闭。原生 Coding 工具、会话历史、模型协议与压缩留在 Core；`private/trace.jsonl` 由 Python 唯一写入，保存实际输入输出、回执、图片及文件版本。

本轮不设全局模型请求总上限；每个 episode 独立计数，摘要与失败尝试共同计入。原预算账本仍为 191/200。本轮 6 个真实运行合计 41 次请求，详细路径及核验见 [回传](Message/20261014-Harness-to-Main-Python主导迁移回传.md)。最终入口 Coding 6 次、局部 IA 5 次、压缩 14 次、查询 QA 3 次均通过对应验收；压缩诊断不是目标导航成功。

Python 全量 290 项、Core 41 项回归通过，本地 Pi 与 Core 构建成功。ObjectNav／指令／GOAT／构图入口迁移，未重跑这些完整模型案例；旧结果不能直接作为新架构的成绩。另一对话的论文改动未混入本轮提交。
