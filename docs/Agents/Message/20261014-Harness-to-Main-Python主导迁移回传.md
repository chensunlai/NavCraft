# Python 主导的 Pi Harness：迁移回传

- 发送方：HarnessAgent
- 接收方：Main
- 日期：2026-10-14
- 状态：已实现并完成本轮软件回归与快速真实验收。

## 当前实现

用户批准 Python 主导＋独立 TS Core。现行布局为：

- `src/harness/`：Python 入口、任务、空间记忆、五个导航工具、环境会话、实验记录。
- `src/agent-core/`：Pi 模型／工具循环、原生 Coding、权限、会话、请求限制、压缩。
- `src/agent-core/pi/coding-agent/`：完整本地 Pi v1.2.0 源码，SDK 通过 workspace 解析到该构建产物。
- `src/legacy/harness/`：旧实现；旧数据生成、旧 bench 和旧配置校验保持原行为。

Python episode worker 直接持有 Habitat，启动 Core，通过双向 JSONL 接收工具请求；没有额外 Python 环境桥。每个导航工具的参数与执行函数均在 Python，Core 不加载地图格式或导航任务。初始定位提示、后续 RGB／动作反馈、无持续真值 pose/depth 的条件保持。

唯一模型历史由 Pi 管理，Python 持续写实际事件至 `private/trace.jsonl`。压缩提示直接修改本地 Pi 源码；外围负责调用、审计和恢复 Python 提供的公开内容。正常目标切换／结束先补齐工具回执再在边界生效。

## 分步提交

| 提交 | 内容 |
|---|---|
| `90945db` | 通用 Core 协议、Python 工具回调；第一条真实 Coding 测试 |
| `56d9052` | Python 直接运行导航工具与环境；第二阶段真实局部 IA |
| `5fde029` | 统一主目录、legacy 迁移、公共地图校准、独立 TS Core、Python 工具与记录归属 |
| `a7cb9e5` | 收尾、stderr 脱敏、显式 Python 配置及相关回归 |
| `8667207` | Core 默认仅限制单 episode，不默认设置全局请求上限 |

本轮无全局模型请求总上限，每个 episode 独立限制，包含摘要和失败请求。原 `cache/harness-pi/implementation-20261014-budget.json` 保持 191/200，不继续复用或重置。

## 真实记录

以下均在 `runs/harness-pi/`；保留该目录名以延续配置，源码已迁移。

| 记录 | 请求 | 核验 |
|---|---:|---|
| `20261014T074742132253Z-python-coding` | 6 | 阶段一：原生读写／Python 计算＋Python 图片工具＋继续回答 |
| `20261014T074927526880Z-python-local` | 7 | 阶段二：指定开口穿越并到达，5 动作／1 米 |
| `20261014T080925207261Z-coding` | 6 | 最终入口：Coding、Python 图片回调、结果重读通过 |
| `20261014T081159874806Z-local` | 5 | 最终入口：局部 IA 真实事件通过，worker 正常退出 |
| `20261014T081318619570Z-compaction` | 14 | 一次真实压缩，随后读图／fresh observe／笔记修订／回读／检查，最终正常回答 |
| `20261014T081753916442Z-query` | 3 | 固定资源查询 QA，D03 与 `initial_observation` anchor 独立核验通过 |

合计 41 次实际模型请求，无自动重试。摘要也计入对应 episode。迁移后的最终四项合计 28 次。

压缩记录第 9 次请求为 high/8192 摘要。恢复 RGB、地图的哈希与公开文件一致；调用／回执配对完整。位置与地图估计仍绑定 `frame_000000`，fresh `frame_000001` 没有被升级为新定位；独立 audit 保存于该记录的 `private/acceptance.json`。该诊断环境最终 `truncated:true/model_finished` 表示模型回答后结束，不是目标导航成功；验收对象是压缩和续接。

## 命令与验证

在 vln Python 环境、仓库根目录运行：

```bash
npm ci --ignore-scripts
npm run build
python -m src.harness doctor
python -m src.harness run --case local
python -m src.harness smoke
python -m src.harness smoke --suite full
```

`doctor` 默认不调用模型。快速 smoke 为 Coding／局部 IA／压缩续接；完整 ObjectNav、指令、GOAT、查询 QA、构图案例仍可单独启动。

最终软件检查：本地 Pi 和 Core 构建通过，Python **290 passed**，Core **41 passed**。回归包括新旧注册边界、重复运动去重、目标切换配对、权限、图片来源、取消、worker 后代回收、失败归档和密钥脱敏。

## 给 Main 的结论边界

请更新论文相关实现路径，按现行 Python 主导结构描述。完整导航和构图的旧结果仍属于上一版；本轮没有重跑完整 ObjectNav、指令、GOAT 和构图，不将其写成新架构成绩。软件重构与少量固定案例不能支持成功率提升、空间能力提升或训练收益结论。

运行产物、依赖及密钥不入提交；另一对话已有论文与文档改动保留。本轮不修改论文主线。
