# 运行脚本

先激活已有 Habitat 环境并完成 `npm ci --ignore-scripts && npm run build`。所有脚本使用当前环境的 `python -m src...`，不指定解释器路径，也不接收命令行参数。**路径、配置文件放在顶部大写变量中；并发、GPU、模式等简单参数直接改 `python -m` 命令，然后执行 `bash scripts/xxx.sh`。** 从任意目录调用均会先进入仓库根目录。

## 入口

| 脚本 | 用途与默认行为 |
|---|---|
| `doctor.sh` | 检查冻结样本、场景路径、现有地图和本地依赖，不请求模型 |
| `run_episode.sh` | Sol 单条 ObjectNav；选择命令中的模型、顶部任务 ID 和工具配置 |
| `run_pilot.sh` | 三模型共用 10 条 ObjectNav/R2R 交替任务，各自并发 1/4/2 |
| `run_sequence.sh` | Astra + Super 工具执行 Qa 连续任务，保持同一会话和记忆 |
| `compare_sequences.sh` | DeepSeek、27B、Sol、Astra 依次执行相同连续任务，产物和会话独立 |
| `check_harness.sh` | `local` 模式 检查真实动作与全景；`probe` 检查模型与工具；`compaction` 检查压缩续接 |
| `check_geometry.sh` | 重放 pilot 几何与动作可达性检查，不请求模型 |
| `check_super_tools.sh` | 真实环境中的 Super 工具检查，不请求模型 |
| `build_maps.sh` | 默认 Sol high、16 并发，按队列构建 wall/wall_real 与楼梯，使用场景内 `raw/` 作为工作空间，结果平铺在场景根目录 |
| `build_maps_nonhssd100_full.sh` | 续接新增100个非HSSD场景的原会话，从现有墙图补齐完整地图；Sol high、16并发、GPU0/1 |
| `compare_map_builds.sh` | Sol/Astra high 同场景完整构图，各占 GPU 0/1，独立 Home、工作空间和场景产物 |
| `report.sh` | 从已有 pilot 记录重新计算并生成报告 |
| `export_viewer.sh` | 从已有连续任务记录导出静态图文网页 |
| `serve_viewer.sh` | 提供静态网页，默认 `http://10.82.1.221:8770`；也可修改 DIRECTORY 指向归档的 3D 网页 |

示例路径对应以前实际执行过的任务。换实验时修改 `RUN`、`OUTPUT`、`DIRECTORY` 等顶部变量；脚本不自动挑选“最近一次”而误用别的运行。新实验默认使用纳秒时间戳目录，不覆盖旧实验。所有脚本前台运行，需要后台运行时由调用方使用 tmux/nohup。

## 配置组合

| 方向 | 路径 | 负责内容 |
|---|---|---|
| 服务 | `config/providers/221.yaml` | 凭据文件、API 协议，不保存密钥 |
| 模型 | `config/models/{deepseek,qwen27,sol,astra}.yaml` | 模型名、effort、模型并发 |
| 模型集合 | `config/models/{pilot,all}.yaml` | pilot 三模型或包含 Astra 的四模型注册表 |
| 地图 | `config/maps/scene_dataset.yaml` | 已发布地图根目录和 walls / wall_real 图层 |
| 环境 | `config/environments/habitat.yaml` | 场景根目录、GPU 池、随机种子 |
| 任务 | `config/tasks/` | 冻结 episode 列表、连续任务与地图路径、场景构建队列 |
| 预算 | `config/budgets/navigation.yaml` | 完整导航的动作、请求、时间限制 |
| 工具 | `config/tools/{basic,super}.yaml` | 工具集合及局部规划方式 |
| 上下文 | `config/runtime/navigation.yaml` | 上下文容量与压缩阈值 |
| 构图 | `config/build_maps/{runtime,walls,full}.yaml` | 运行限制、GPU 池、构建阶段和工作空间模式 |

导航脚本用多个 `--config` 合并任务/环境/地图/预算/模型注册表，用多个 `--model-config` 组合实际模型配置。连续任务使用同一模型配置格式；其轮次预算来自冻结 scenario，不用 pilot 预算覆盖。构图脚本复用同一模型与服务配置，读取其中 model/effort/credentials，不继承模型推理并发作为场景构建并发。

合并规则：后面的文件覆盖前面的同名字段；字典递归合并，列表整体替换，最后显式传入的模块参数优先。`extends` 相对当前 YAML 解析，其余路径相对仓库根目录。`config/pilot.yaml` 仅作为原有模块命令的默认组合入口，不再重复维护样本。

切换模型或工具通常只需更换顶部 `MODEL_CONFIG`、`TOOLS_CONFIG`。批量 pilot 通过命令中的 `--model` 选择模型注册表中的名称；选 Astra 时将 `MODELS_CONFIG` 改成 `config/models/all.yaml`。221 当前从 `.env.harness-test` 读取，切换服务修改 provider 配置即可。子进程和工具里的 Python 跟随启动脚本的当前解释器。

## 构图新建、预览与恢复

`build_maps.sh` 的模块命令支持 `plan`（仅写计划）、`probe`（真实服务探针）、`run`。默认全队列；需要筛选时直接在命令中增加 `--scene ID`、`--start N`、`--limit N`。并发、GPU 直接修改 `--concurrency`、`--gpu`。更换 `STAGE_CONFIG` 可使用完整构图。

恢复时在命令末尾增加 `--resume`，把 `OUTPUT` 与 `BUILD_CODEX_HOME` 改为原批次的准确路径，并保留原模型、服务、任务范围、阶段与运行环境。恢复会核对冻结计划，不自动迁移参数，也不会代替调用方停止已运行的同批次进程。当前后台批次不因本次脚本整理而重启。运行器沿用现有服务故障每 300 秒重试的机制，内部重试策略未改。

完整构图对比输出到 `data/scene_maps/comparisons/<本次ID>/<模型>/`，避免两个模型写同一个场景目录。旧墙体提取实现和一次性数据修复不作为当前执行入口恢复；已有墙图由任务配置直接指定。

## 进度显示

有限任务统一使用 tqdm 在 stderr 原地刷新进度条，显示耗时、预计剩余时间与状态计数；并发更新由锁保护。通过 `2>&1 | tee` 也保留原地刷新，保存的日志包含回车控制符，可用支持终端控制符的查看方式读取。批量评测按 episode、连续任务按轮次、构图按场景统计；检查、报告和网页导出按检查项或模型统计。`ended` 表示执行已结束，不等于成功；`ready/finished`、`partial`、`failed`、`running`、`waiting`、`pending` 分开显示。模型任务成功率仍以评分报告为准。网页服务器是持续服务，没有虚构的完成百分比。

在 tmux 中直接运行任务脚本即可看到任务自身输出的进度条，不需要额外查看程序。已经启动的旧进程不会热加载新代码，本次不为此中断任务；后续启动或恢复时生效。

## 场景原档接入

正式 Bench 直接读取 `data/scene_maps/<dataset>/<split?>/<scene>/` 的 `scene.json`、`walls.json` 等原始文件；无需发布或复制；`raw/` 仅为原始工作区，Bench 不从中寻找缺失结果。`MAP_CONFIG` 指定目录，`maps.layer` 选择 `walls` 或 `wall_real`。`doctor.sh` 检查加载条件；缺图、缺楼层或高度匹配不明确均报错。

读取时由 Dataset 统一选择楼层并换算公开坐标，公开几何保存在当次运行 `private/map.json`，源文件哈希记录于 `private/map_source.json`；随后仅允许的几何进入模型工作空间。原始标注不被改写，既有运行的输入不随场景更新改变。构图过程日志和 Codex Home 放在 `cache/build-maps/`。

新增非 HSSD 100 景清单冻结后，可运行 `bash scripts/build_maps_nonhssd100.sh`；复用 provider/model/runtime/stage 配置，只换 `config/tasks/scene_maps_nonhssd100.yaml`。启动前确认同批次没有其他进程正在写入。
