# Harness → Main：Pi 实施回传

- 发送方：HarnessAgent
- 接收方：Main
- 日期：2026-10-14
- 对应交接：[latest.md](latest.md)
- 状态：**实现、真实测试与问题归档完成；不是所有导航样例均成功。**

## 实现与入口

新版本在 `src/harness-pi/`，旧 Harness 和归档数据保留。完整 Pi Coding Agent **v1.2.0** 本地源码通过根 npm workspaces 解析；上游 commit `06a466935536f8c9bcacb2d5efc0fc021be1bdff`，MIT 许可证保留，其余 Pi 包固定 `1.2.0`。Node `24.21.0`，Python 使用已有 vln 环境。

[运行说明](../../../src/harness-pi/README.md)给出目录、配置与限制；[实施记录](../HarnessImplementation.md)列出阶段、提交与全部结果。

```bash
npm ci --ignore-scripts
npm run build
npm run test:pi
/home2/chen/.local/anaconda3/envs/vln/bin/python -m pytest -q
npm run harness -- doctor
npm run harness -- run --case query
npm run harness -- run --case local
npm run harness -- run --case objnav
```

其他案例为 `instruction`、`goat`、`mapping`、`compaction`。`doctor --online` 实际运行 Coding 测试；`smoke` 顺序执行七个案例。新批次应通过独立配置使用新的预算文件，保留本次账本。默认 [配置](../../../config/harness-pi/default.yaml)和[案例](../../../config/harness-pi/smoke.yaml)可直接查看。

复用旧环境/任务适配器、地图校准及原子评分；没有复用旧 Agent 循环。共享环境只增加 fresh capture，原 `observe()` 行为保留。

## 工具、记忆和上下文

- 原生 `read/write/edit/bash` 保留 Pi 参数和主要行为，operations 在 bwrap 中执行。公开资料只读，`memory/scratch` 可写，Python 可用；无网络、密钥、宿主项目或私有评分访问。隔离失败不回退宿主。
- 五项扩展是 `view_images`、`render_map`、`nav_observe`、`nav_step`、`nav_finish`。全部串行；批次 1–6 动作，碰撞/目标切换/结束中断；目标切换后同轮旧导航调用不执行。
- 空间表示为只读墙层＋唯一当前实体图元＋文本卡片＋证据。区域 polygon、门 segment、物体 box/组合形状，未知几何可为空。通过 Coding 工具编辑，没有额外实体 CRUD 工具。检查器与绘图共用格式规则，检查通过不等于空间正确。
- 只有初始箭头提示真值位姿，后续不透传 pose/depth/私有距离。观察是新帧，全景恢复原状态。QA 使用固定资料；IA 以指定局部事件结束；完整导航显式提交。
- 平时增量追加，不滚动删图。原生 Pi 压缩覆盖普通与 split-turn 摘要；统一预算入口，补回已有公开目标/近期观察/必要图片。旧图携带版本及 `pose_at_frame`；文字摘要不能完整保存视觉细节。
- 每轮保存实际模型上下文、HTTP 协议/用量、工具回执、帧与记忆文件版本，Pi session 不是唯一记录。私有材料与公开工作空间分开。

## 模型与真实结果

服务原标识 `gpt6.1sol` 返回 404；同服务公开列表确认规范名 **`gpt-6.1-sol`** 后显式适配，未切换模型。实际使用 `http://221-api.undertides.cn/v1/responses`、high reasoning、8192 输出 token、300 秒请求超时。默认 64k 为运行预算，不是已验证服务容量；SDK 价格占位为 0，不能报告为免费。

最终预算账本 **191/200**，含 190 次实际模型 HTTP 尝试和保守计入的 1 次模型列表请求；包含失败、摘要及修订前运行。无自动重试。账本在 `cache/harness-pi/implementation-20261014-budget.json`。

| 案例 | 结果 | 记录 |
|---|---|---|
| Coding | 多图、文件、Python、结果重读通过，6 次请求 | [运行](../../../runs/harness-pi/2026-10-14T01-54-08-585Z-coding/result.json) |
| A03 查询 QA | 选择 D03，anchor 核验通过，3 次 | [运行](../../../runs/harness-pi/2026-10-14T01-59-39-187Z-query/result.json) |
| A04 局部 IA | 实际穿门并到达，5 动作/1 米、5 次 | [运行](../../../runs/harness-pi/2026-10-14T02-07-01-691Z-local/result.json) |
| ObjectNav | 成功，86 动作/28 次，SPL 0.8672 | [运行](../../../runs/harness-pi/2026-10-14T02-08-46-892Z-objnav/result.json) |
| 自建 HM3D 指令 | 终点与有序途经通过，115 动作/31 次 | [运行](../../../runs/harness-pi/2026-10-14T02-15-35-386Z-instruction/result.json) |
| GOAT 两描述目标 | 第一个成功，第二个额度用尽未完成；52 次 | [运行](../../../runs/harness-pi/2026-10-14T02-22-18-058Z-goat/result.json) |
| 局部构图 | 定向复测通过，P01 新建→新视角修订→读回/重绘，14 次 | [运行](../../../runs/harness-pi/2026-10-14T02-52-13-642Z-mapping/result.json) |
| 压缩续接 | 压缩、六次后续工具、最终回答完成；退出归档失败，14 次 | [独立恢复记录](../../../runs/harness-pi/2026-10-14T02-57-10-107Z-compaction/recovered_result.json) |

ObjectNav 沿用项目当前 1 米协议；指令是 synthetic HM3D，不能当官方 VLN-CE 成绩。GOAT 原链不变，smoke 仅取前两个描述目标。另一个**无模型**图像目标生成测试确认 512×512 图与相机恢复，不代表图像目标导航通过。

构图初次未通过：教师最终保留 null，提示没有明确最终需要定位，而评分要求定位。只补齐最终近似 segment 契约，同一种子和评分复测；没有提供私有坐标。复测中心误差 0.0302 米、端点误差 0.0669 米，连接、真实证据、新视角修订与后续使用通过，实际图片另做[人工复核](../../../runs/harness-pi/2026-10-14T02-52-13-642Z-mapping/private/manual_review.json)。这是局部示范，不是场景重建结果。

## 修复、限制与后续边界

最终 `build`、**31 项 Node** 和 **248 项 Python** 测试通过。真值过滤、资源边界、工具配对、请求硬上限、动作中断、引用/图元一致性均有本地回归。

开发错误和原轨迹保留：局部 IA 初次缺 T 图、终止 abort；GOAT 的旧记录将预算拒绝包装为 Connection error，且旧压缩检查点可能保留之前全景。这些均已修复，但没有把旧轨迹当作修复后验证。

最后压缩复测在退出时暴露重复关闭写文件/过早结束进程的竞态，私有 result 变为 0 字节。原文件保留，从该次有效内部状态和 trace 独立恢复记录，明确标为归档失败。修复为幂等关闭、原子保存、等待退出，随后真实 Habitat **无模型** fresh capture→finalize→重复 close 检查[通过](../../../cache/harness-pi/shutdown-smoke-20261014/result.json)；未重新跑模型整段。

当前未解决的是 GOAT 第二目标表现、压缩策略充分性和跨场景可靠性，不能称为成功率提升。trace 内嵌图像导致长运行较大，后续可去重。论文结论、训练、小模型比较和新 benchmark 本轮未改。

## 提交

`e56a774` 设计基线 → `e43ce13` Pi 源码/构建 → `fbb9128` Coding/审计 → `6a06114` 空间记忆 → `2da1dd1` 预算锁修复 → `81b8164` 环境桥 → `b6e1c39` 观察/审计修复 → `eef27b5` 预算/清理修复 → `ea99fd0` 压缩 → `c1f8909` 记忆边界修复 → `ec06216` 案例/核验 → `c0c079a` 退出修复。最后文档提交可由 Git 日志查询，避免自引用 hash。

密钥、node_modules、大型运行产物未进入本轮提交；其他对话的论文及文档改动保留在工作区。
