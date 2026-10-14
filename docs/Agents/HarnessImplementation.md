# Pi Harness 实施记录

依据 [当前交接](Message/latest.md)、[Architecture](../Architecture.md) 和 [SpatialMemory](../SpatialMemory.md)。2026-10-14 用户授权实现，并确认先提交相关设计，再逐阶段 commit；论文及其他对话的既有改动不混入。

## 固定条件

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

## 真实测试预算

接口/Coding 12、A03 查询 QA 8、A04 局部 IA 12、ObjectNav 32、指令 32、GOAT 两目标 52、局部构图 16、压缩 12、定向复测 24，共 200。未用额度可转后续阶段；总额不突破。软件失败、模型判断错误、预算结束分开报告。

## 进度

- 阶段 0：`e56a774`，相关设计已冻结。
- 阶段 1：Pi v1.2.0 单包 SDK 构建通过，本地解析与空会话检查通过；普通依赖统一安装到根目录。

- 阶段 2：11 项 Node 验证通过；真实 Coding 闭环通过，多图识别、脚本执行、JSON 与生成图片重读均完成。服务配置原标识 `gpt6.1sol` 返回 404；公开列表核验后显式使用 `gpt-6.1-sol`，Responses/high/8192 保持。累计请求 8 次（含无效标识与模型列表），真实成功闭环 6 次。产物：`runs/harness-pi/2026-10-14T01-54-08-585Z-coding/`。

- 阶段 3：19 项空间记忆测试通过；Qa prebuilt 55 实体/120 证据、partial 2 实体/4 证据、empty 0 均可检查与绘图。PNG-only 查询不要求额外图元，全景拼图标注相对方向。
