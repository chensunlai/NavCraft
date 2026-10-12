# NavCraft 文档

只维护四份正文，新内容优先补充到对应文件。

| 文档 | 内容 |
|---|---|
| [Report.md](Report.md) | 研究动机、论文逻辑、关键文献证据、近期计划与训练对照 |
| [Architecture.md](Architecture.md) | 当前项目架构、Harness 流程、工具、记忆、接口与运行命令 |
| [Tasks.md](Tasks.md) | 空间任务输入输出、场景素材、数据构造、核验与展示集 |
| [Experiments.md](Experiments.md) | 评测协议、关键结果、失败诊断与验收条件 |

另保留可整体分享的[图文示例包](examples/README.md)，图片及编辑源文件都在 `examples/` 内。

逐次运行的长表由脚本生成到 [cache/visual-harness-v1/VisualHarnessResults.md](../cache/visual-harness-v1/VisualHarnessResults.md)，不再单独维护结果、复评、调试等正文。原始数据、实验和证据分别留在 `data/`、`runs/`、`cache/`；研究设想与已实现、开发诊断与正式成绩应明确区分。
