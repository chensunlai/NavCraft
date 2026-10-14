# NavCraft 文档

维护以下五份正文。空间记忆按本轮要求独立成文，其余新内容优先补充到对应文件。

| 文档 | 内容 |
|---|---|
| [Paper.md](Paper.md) | 论文完整归档：三层动机、数字 Agent 训练迁移、两类中间数据与 Harness、AgentVLN 等近邻、实验矩阵、摘要及写作大纲 |
| [Architecture.md](Architecture.md) | Harness 目标架构、[Coding Agent 基座](Architecture.md#framework)、工具输入输出、工作空间与[上下文管理](Architecture.md#context-management) |
| [SpatialMemory.md](SpatialMemory.md) | 平面图、文本、证据图片的具体形式、字段和样例，三者关联、局部在线构建与修订、运行判断，以及训练样本的信息条件 |
| [Tasks.md](Tasks.md) | 空间任务输入输出、场景素材、数据构造、核验与展示集 |
| [Experiments.md](Experiments.md) | 评测协议、关键结果、失败诊断与验收条件 |

另保留可整体分享的[图文示例包](examples/README.md)，图片及编辑源文件都在 `examples/` 内。

精简针对重复内容和文件数量；论文核心论证在 `Paper.md` 中完整维护，不以项目进度摘要替代。

2026-10-14 论文讨论已统一归档到 [Paper.md](Paper.md)：[完整动机](Paper.md#motivation)、[方法路线](Paper.md#method)、[相关工作](Paper.md#evidence)和[确认／未定项](Paper.md#decisions)。协作身份与交接记录位于 [Agents/](Agents/)，不另建重复的论文正文。

Harness 与空间记忆设计分别维护在 [Architecture](Architecture.md#harness) 和 [SpatialMemory](SpatialMemory.md)。正文聚焦论文、目标架构与训练任务；具体安装、代码组织和重构安排见[当前交接](Agents/Message/latest.md)，正文不重复工程细节。

历史实验明细见[结果记录](../cache/visual-harness-v1/VisualHarnessResults.md)。场景素材、样例与实验结果保留为研究依据，目标设计与已验证结论分别表述。
