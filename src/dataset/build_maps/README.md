# 批量场景地图构建

Python 调度 Codex app-server（默认串行，可配置并发），每个场景一个独立 thread。Codex 使用指定 skill 取证、编辑和复查地图；本程序保存过程并独立运行归档校验，不包含自动补墙或房间识别算法，不依赖导航 Harness / Bench。

## 运行参数

所有入口均为 `python -m src.dataset.build_maps`。在已有 Habitat 环境中运行（本机为 `conda activate vln`）；校验器与 Codex 的渲染环境说明自动沿用启动程序的 Python，无需另传解释器路径。

| 参数 | 作用 |
|---|---|
| `--env` | 必填，读取 dotenv，不执行 shell、不展开变量 |
| `--model` | 模型名；省略时读取 env 中的 `NAVCRAFT_TEST_MODEL` |
| `--effort` | 必填，原样传给 Codex，不替换或降低 |
| `--codex-home` | 必填，独立 Codex 状态目录，包含原生会话；不复制用户主目录的凭据 |
| `--base-url` | 可选，覆盖 env 中的 API base；保留指定路径，不自动追加 `/v1` |
| `--base-url-env / --api-key-env / --model-env` | 自定义 dotenv 字段名，默认 `NAVCRAFT_TEST_BASE_URL / NAVCRAFT_TEST_API_KEY / NAVCRAFT_TEST_MODEL` |
| `--queue` | 默认 `data/samples/scene_build_queue.json` |
| `--skill` | 默认同目录 `build_scenes_map_skill_v2`，传入含 `SKILL.md` 的目录 |
| `--stage` | `full` 完整地图（默认），或 `walls` 仅墙体与楼梯 |
| `--scene` | 按 ID 选择，可重复；仍保持清单顺序 |
| `--start / --limit` | 在 scene 过滤后截取，从 1 开始 |
| `--gpu` | 串行时的物理 GPU 0 或 1，默认 0 |
| `--concurrency / --gpus` | 并发槽位数（默认 1）；并发时按槽位循环分配 GPU，默认 `0 1` |
| `--min-free-gb` | 并发派发前检查工作及归档磁盘，默认剩余不足 20 GiB 时停止新派发，正在执行的场景继续 |
| `--output` | 必填，批次计划、汇总和总日志目录 |
| `--scene-workspaces` | 每个场景直接在自身归档的 `raw/` 工作；脚本、prompt、事件和状态保存在该目录，原生线程 cwd 同步设置 |
| `--max-turns` | 每次调用中，每个场景最多新增几轮，默认 3 |
| `--turn-timeout` | 每轮等待完成的秒数，默认 3600 |
| `--resume` | 读取同一 output 的计划和 thread，继续未完成场景 |
| `--repo / --codex` | 仓库根目录、Codex 可执行文件 |

三个命令：`plan` 只生成计划和逐场景提示；`probe` 发一次简短模型请求；`run` 执行构建。`plan` 不启动 Codex、不渲染；使用 `--scene-workspaces` 时会创建场景 `raw/` 并写入任务提示。

## 当前 221 服务示例

```bash
# 检查 Responses、指定模型和 effort，不构建地图。
python -m src.dataset.build_maps probe \
  --env .env.harness-astra --model gpt-6-astra --effort high \
  --codex-home data/scene_maps/codex \
  --output cache/build-maps/probe-221 --turn-timeout 120

# 按现有清单顺序准备全部场景；去掉 --limit 即处理全部。
python -m src.dataset.build_maps plan \
  --env .env.harness-astra --model gpt-6-astra --effort high \
  --codex-home data/scene_maps/codex --gpu 0 --limit 1 \
  --output cache/build-maps/first-scene

# 执行上一步已生成的计划；参数保持一致，使用 --resume。
python -m src.dataset.build_maps run \
  --env .env.harness-astra --model gpt-6-astra --effort high \
  --codex-home data/scene_maps/codex --gpu 0 --limit 1 \
  --output cache/build-maps/first-scene --resume
```

也可以直接 `run` 写入新的空 output，不必先执行 `plan`。再次执行相同 `run --resume` 会续接未完成场景；已达到 `ready_for_review` 的场景会重新进行结构检查，通过则跳过。除并发上限外，参数、清单内容或 skill 文件发生变化时拒绝混入旧计划；密钥可以在同一 env 文件中更新。

## 一次场景如何执行

1. 冻结队列、公开配置和 skill 文件哈希，按队列顺序选择场景。
2. 启动独立 app-server；使用指定 Codex Home，通过自定义 Responses provider 从子进程环境读取密钥。密钥不写入配置、命令行或本程序日志。已有 Home 的配置不覆盖。
3. 新建或恢复该场景 thread，核对 Codex 返回的模型、provider 和 effort。以原生 `skill` 输入显式传入指定 SKILL.md。
4. Codex 检查已有归档，必要时在场景工作目录备份，然后执行 skill。只允许编辑该场景与该工作目录，不改其他场景、共享索引或项目源码。
5. 模型返回 reviewed / partial / blocked 报告；主程序另用当前 Python 执行 `validate_archive.py` 或 `validate_wall_stage.py`。报告格式、证据路径或结构校验失败时，将检查反馈给同一 thread 继续修正。
6. 模型报告 reviewed、证据文件存在且结构检查通过，才记为 `ready_for_review` 并处理下一场景。该状态**不是独立视觉验收**。临时服务错误按下述策略等待续接。blocked、其他错误、超时或修订轮数耗尽时保留现场，串行停止，并发模式继续该槽下一场景。

默认串行。使用 `--concurrency 16 --gpus 0 1` 时，场景按固定槽位分配，各槽使用 `<codex-home>/worker-XX`；每个场景仍使用独立 thread。并发模式下单场景失败留档后继续该槽后续场景，最终退出码反映是否全部通过。串行仍在首个失败处停止。同一个批次 Codex Home 加进程锁，避免重复派发；续跑可调整并发上限，原槽位、Codex Home 和 GPU 分配保持不变；有效并发不超过最初槽位数量。中断会关闭本程序创建的 app-server 进程组；不会主动操作其他模型服务。服务器返回需要人类回答/审批的请求时明确失败，不伪造批准。

GPU 环境与提示限制使用 0/1；这是进程配置和执行约束，不是硬件隔离。Codex 使用 `danger-full-access` 与 `approvalPolicy=never`，为原生渲染、文件和命令提供访问能力；写入范围由任务指令约束，不宣称是文件系统沙箱。Coding shell 默认仅继承基础环境并排除密钥变量。

## 产物与复核

```text
<output>/
├── plan.json                 # 场景、模型、effort、skill 哈希，无密钥
├── summary.json              # 各场景状态
└── <顺序>-<数据集>-<场景>/
    ├── prompt.md
    ├── events.jsonl          # app-server RPC、工具及通知流
    ├── stderr.log
    ├── result.json           # thread/turn ID、最终回答、独立校验
    └── ...                   # Codex 的临时脚本、备份和工作文件
```

最终地图使用队列内 `output_dir`，原生 Codex 会话另保存在 `--codex-home` 下。`events.jsonl` 是 app-server 事件，不是声称记录了所有上游 HTTP 请求。临时服务故障先由 Codex 重试，再由运行器等待并恢复原会话，详见服务重试。

2026-10-25 验证：本机 Codex 0.164.0，221 的 `gpt-6-astra / high` 短探针成功。`.env.harness-test` 中的旧别名 `gpt6.1sol` 被服务返回 404；不会自动换名。144 场景墙体批次已启动；实际进度以批次日志和场景状态为准，不以本说明作为完成证明。

```bash
python -m pytest -q tests/test_build_maps.py
```

协议参考：[Codex App Server](https://developers.openai.com/codex/app-server)。实现按本机 `codex app-server generate-json-schema` 核对，不内置复制一份协议框架。

使用场景内工作空间时，`raw/events.jsonl`、`raw/result.json`、`raw/stderr.log`、`raw/prompt.md` 为调度器保留文件，校验和工具排除这些运行记录。几何和证据照常校验，不再增加 build 子目录。

### 服务重试

Codex 内部 HTTP 和流中断各最多重试 5 次；仍遇到 503、429 或流断开等临时错误时，关闭当前客户端进程，等待 300 秒后续接原线程，外层次数不限。等待期间结果为 `retry_wait`，含 `retry_count` 和 `next_retry_at`；Ctrl-C 可取消。失败请求不消耗地图修订轮次，等待不计入下一次 turn 的超时。认证/配置错误及构建校验失败不做此类无限重试。
