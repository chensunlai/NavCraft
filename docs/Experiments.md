# 实验与评测

更新：2026-10-14。本文保留协议、关键成绩及影响解释的诊断；逐次指标、成本和逐目标结果见[结果记录](../cache/visual-harness-v1/VisualHarnessResults.md)。原始轨迹与评分不因文档整理或阈值调整而改写。

**现有记录主要用于探索性诊断，尚无模型训练收益结论。** 历史尝试、离线复评与固定条件的实验分别报告。

<a id="protocol"></a>

## 评测口径

本文现有实验中的视觉 Harness 提供完整场景地图、描述和 RGB 证据；真实位姿只用于初始箭头，后续无实时 pose 或 depth。当前属于地图辅助的自定义设置，不代表已验证只给墙图的在线构图条件。

| 任务 | 当前执行判据 | 原始 benchmark 公开实现／主要差异 |
|---|---|---|
| ObjectNav | 主动 STOP，到目标类别任一成功视点的最短测地距离 `<1 m` | HM3D ObjectNav 为成功视点 `<0.1 m` |
| GOAT | subtask STOP，到当前目标成功视点的最短测地距离 `<1 m` | GOAT 为成功视点 `<0.25 m`；旧 NavCraft 开发运行曾用 3 m |
| 指令导航 | STOP，到终点的最短测地距离 `<3 m`，另统计有序途经约束 | VLN-CE 同为 3 m，但本轮为自建 HM3D 指令，不是官方 MP3D VLN-CE |

来源：[ObjectNav 配置](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/habitat/task/objectnav.yaml)、[距离与成功测量](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/nav/nav.py)、[VLN-CE 配置](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/vlnce_task.yaml)、[GOAT 配置](https://github.com/Ram81/goat-bench/blob/74c41d19d4a4c3608d1575b512087b5a529aee0e/config/tasks/goat_stretch_hm3d.yaml#L45-L55)及[测量](https://github.com/Ram81/goat-bench/blob/74c41d19d4a4c3608d1575b512087b5a529aee0e/goat_bench/measurements/nav.py#L266-L294)。

成功视点是预采样的合格观察位置，**视点距离不是物体中心距离**。将视点容差改为 1 m 会扩大成功区域，不能称为原始 benchmark 的复现。当前机器人为 0.88 m 高／0.18 m 半径，HM3D v2 Challenge 的 Stretch 为 1.41 m／0.17 m；输入、相机、动作和 NavMesh 也须统一才能比较。

GOAT 主指标采用成功子目标／全部预定子目标，并报告整链成功率；未尝试目标计失败。官方按 episode 内成功比例再取均值，还过滤超过 55 词的描述；下述旧视觉 Harness 开发记录保留完整链，聚合与任务集合不同。文末 Pi 回归仅取两目标前缀，单独报告。[论文中“物体 1 m”](https://arxiv.org/html/2404.06609v1#S3)不能替代代码中的视点判据。

已核查的相关实现也不统一：

| 方法 | 已确认条件 | 使用边界 |
|---|---|---|
| [BSC-Nav](https://github.com/Heathcliff-saku/BSC-Nav/blob/ad4c544f7d1feb980ffad33d4eefa8ad2e39fad0/env.py#L552) | 默认参数覆盖 YAML 后为 `VIEW_POINTS@1m + STOP` | RGB-D、持续位姿与局部规划器；是发布代码默认值核查 |
| [OmniNav 快系统](https://github.com/amap-cvlab/OmniNav/blob/e8b485953c6555e841a3283c3e3a4f7d8d0f24c3/infer_ovon/ovon/configs/ovon_pano.yaml#L32) | `VIEW_POINTS@1m + STOP`，连续 waypoint 执行 | [慢快系统自定义计分](https://github.com/amap-cvlab/OmniNav/blob/e8b485953c6555e841a3283c3e3a4f7d8d0f24c3/infer_ovon_slowfast/run_nav_ovon_omni.py#L301)无额外 STOP 门槛，不能混用版本 |
| NavHarness／HarnessVLN／SpaceVLN | NavHarness 主表半径未确认；后两者论文写 1 m | 未确认距离参照时，不推定为成功视点 1 m；论文来源见 [论文主线](Paper.md#evidence) |

对照需同时记录距离参照、阈值、STOP、输入与低层执行器；仅统一一个数字不足以建立可比实验。

## 现有结果

所有失败、服务中断与未尝试目标保留在选定集合分母；不同尝试不取最优结果合并。以下样本和版本不同，不能据此作严格模型排名。

| 记录 | ObjectNav | 自建指令 | GOAT 子目标／整链 | 性质 |
|---|---:|---:|---:|---|
| Sol 历史开发轨迹，离线统一视点 1 m | 3/6（50.0%） | — | 16/27（59.3%）／0/3 | 非新运行，原运行有中途条件变化或主动终止 |
| Qwen3.8-27B 完整开发集，当前协议 | 1/6（16.7%） | 2/6（33.3%） | 11/27（40.7%）／0/3 | 开发结果，非冻结验收 |
| Sol 冻结指令 smoke | — | 0/1 | — | 指纹核验通过，单样本不能代表总体性能 |

Qwen 原始结果：[ObjectNav](../runs/20261012T035905Z-visual-develop-objnav-qwen27-manifest-Qwen3.8-27B-9fb95da7/summary.json)、[指令](../runs/20261012T043321Z-visual-develop-instruction-qwen27-manifest-Qwen3.8-27B-4159e006/summary.json)、[GOAT](../runs/20261012T060951Z-visual-develop-goat-qwen27-manifest-Qwen3.8-27B-3a06cdb3/summary.json)。

<a id="sol-1m"></a>

### Sol 离线复评与阈值敏感性

同一批历史轨迹按不同成功视点容差复评，均要求 STOP；未重新运行模型，不覆盖原始成绩。

| 阈值 | ObjectNav／6 | GOAT／27 |
|---|---:|---:|
| 0.1 m | 0 | 0 |
| 0.25 m | 0 | 6 |
| 0.5 m | 0 | 11 |
| 1 m | 3 | 16 |
| 3 m | 3 | 18 |

1 m 下 GOAT 三条链分别 5/7、8/10、3/10；episode 宏平均为 60.5%，全部子目标加权为 59.3%。按描述／图片／类别拆分为 5/14、4/6、7/7。ObjectNav 两条服务／预算失败和 GOAT 第三链三个未尝试目标均计零。原 3 m 的 18/27 不是官方 GOAT 成绩。

证据：[逐目标复评及原结果 SHA-256](../cache/visual-harness-v1/sol-success-1m.json)、[阈值诊断](../cache/visual-harness-v1/evaluation-protocol-audit.json)、[ObjectNav 原 run](../runs/20261011T224116Z-visual-develop-objnav-sol-manifest-gpt-6.1-sol-877ef173/summary.json)、[GOAT 原 run](../runs/20261011T222324Z-visual-develop-goat-sol-manifest-gpt-6.1-sol-5cfc6451/summary.json)。

<a id="qwen-history"></a>

### 历史记录的证据范围

Qwen 部分历史尝试因服务或输出限制中断；另一次 GOAT 运行在前三个目标成功后主动中止，3/7 仅是部分进度。不同条件下的重试不能合并成一次导航成绩。

原始条件与结果见[首次尝试](../cache/visual-harness-v1/qwen-four-concurrent-20261012.json)、[重试记录](../cache/visual-harness-v1/qwen-timeout-retries-20261012.json)、[后续尝试](../cache/visual-harness-v1/qwen-8192-low-20261012.json)。[归档表](../cache/visual-harness-v1/runs-archive-latest.json)记录保留证据；更早三组 smoke 原轨迹已删除，仅有[清理清单](../cache/visual-harness-v1/runs-cleanup-20261012.json)等汇总，不能严格复评。

<a id="harness-diagnostics"></a>

## 导航现象与解释边界

| 现象 | 研究中需要区分什么 | 不能据此推断什么 |
|---|---|---|
| 摘要将餐桌记成目标床并延续错误判断 | 当前观察、原始证据和摘要判断应分别检查 | 后续 STOP 也见到近期 RGB，不能断言旧摘要是唯一原因 |
| 未调用图片工具被摘要写成“没看地图” | 区分自动提供图片与显式查询，检查真实输入 | 不能推断模型内部是否关注地图 |
| 跨目标沿用旧箭头、门口反复碰撞、识别目标后过早 STOP | 分别检查位置更新、局部通行判断与完成判断 | 提示模型检查这些事项，不等于已经解决问题 |

局部可达性与评分几何的核验材料见[场景诊断](../cache/visual-harness-v1/geometry-diagnosis/report.json)、[视点核验](../cache/visual-harness-v1/geometry-diagnosis/acceptance-05.json)。这些材料不作为模型导航成绩。

Sol 旧指令开发集为 6/6，但运行条件中途变化，只作诊断；后续固定条件的 smoke 为 0/1、332 动作、途经约束 1/3、终点距离 10.53 m，并耗尽请求时间预算。样本与条件不同，不能归因某项改动导致退化。

原 GOAT 第三链被主动停止；已有成功和未尝试目标继续按原协议保留，不能将中止后的目标移出分母。

<a id="pi-128k-regression"></a>

## Python＋Pi：128k 导航与运动压缩回归

2026-10-14，在同一 Qa 场景、预建地图和初始位姿提示条件下，顺序重新运行三个固定导航案例与一个运动压缩专项。使用 `gpt-6.1-sol`、Responses、high reasoning、8192 输出上限；正常配置为 128000 上下文、100000 估算触发阈值、20000 近期预算。后续只返回 RGB 与公开动作反馈，无实时真值 pose 或 depth。源码基线为 `413f7a8`，新增运动压缩专项为 `bc15d82`；各次源文件／构建产物哈希与中途开发差异另存，不沿用旧版成绩。

| 案例与本次记录 | 实际请求／上限 | 原子动作 | 实际路程 | 独立结果 | 压缩次数 |
|---|---:|---:|---:|---|---:|
| [ObjectNav](../runs/harness-pi/20261014T102335238401Z-objnav/result.json) | 23／32 | 73 | 10.916 m | 成功，SPL 0.8521，成功视点距离 0.0463 m | 0 |
| [自建 HM3D 指令](../runs/harness-pi/20261014T102941345738Z-instruction/result.json) | 31／32 | 111 | 13.275 m | 成功、两项有序途经满足；nDTW 仅 0.2430 | 0 |
| [GOAT 两描述目标前缀](../runs/harness-pi/20261014T103604830437Z-goat/result.json) | 52／52 | 143 | 17.770 m | 第一成功，第二预算耗尽；partial success 0.5，整段失败 | 0 |
| [运动中压缩](../runs/harness-pi/20261014T104912521812Z-moving_compaction/result.json) | 17／24 | 5 | 1.000 m | 压缩前 0.25 m，后 0.75 m；指定开口穿越与到达均通过 | 1，强制触发 |

动作数包含转向及实际执行的 STOP；路程由私有连续位姿核验。四项合计 **123 次实际请求**，包括摘要；无全局请求总限额，原 191/200 账本未改动。ObjectNav／GOAT 使用本项目成功视点 1 m 判据，不能替代官方成绩；指令是自建任务，GOAT 仅截取原链前两目标。这些是功能回归与诊断，不作模型排名或成功率提升结论。

**碰撞、结束与记忆使用。** ObjectNav／指令／GOAT 分别记录 1／3／11 次碰撞，后续有转向或成功平移。GOAT 第一目标提交后切换到第二目标并重新决策，但未在 52 次请求内完成；末尾 SDK `Connection error.` 实为第 53 次发送前的预算拒绝，独立账本与 HTTP 记录一致，实际 HTTP 错误为零。另有一次请求不存在图片的工具错误，模型改正引用后继续，原错误保留。

三项导航均读取预建资料并写入记忆，但新写内容没有后续读取／重绘；ObjectNav 与指令在结束前写入，GOAT 切换后查询的仍是原场景资料。不能据此证明新增外部记忆改善导航。ObjectNav 末尾单个位置估计对私有位姿误差约 1.355 m，到达成功也不等于地图定位精确。

GOAT 第二目标还有**描述与实例证据不一致的疑点**：原目标为 `picture_153`，描述“三面框镜、风口上方”，已有客厅图显示四幅蓝色框画位于可见风口下方；模型在读取该图后明确拒绝匹配。这可能涉及画内内容、另一个风口或标注问题，尚不能断言标注错误或失败根因。下一步先核验原实例与描述，再分析局部通行和对应能力，不修改原标签或成绩。[疑点材料与实际决策](../cache/harness-pi/128k-regression-audit/goat-goal1-annotation-review.json)。

**运动压缩的证据边界。** 专项使用 `A04-interactive_ia-S`，同一环境与 Pi 会话先右转、实际前移 0.25 m；目标未完成时，将触发阈值临时降为 512 token、近期预算改为 1200 token。真实压缩后，模型重读 `memory/moving_compaction.md`，查看旧／新 RGB 和任务图、取得全景，再执行三次前移，共 0.75 m。私有重放验证实际经过 D01 并到达 T；最新观察为 `frame_000002`，旧地图箭头仍注明 `frame_000000`。自动检查与人工图像／摘要核查均通过；Core 与外层 worker 正常退出。

该次摘要只总结 split-turn 前段，空间笔记保留在近期历史，首个续接模型输入实际包含笔记与最新 RGB；不能把“摘要片段未见笔记”误读为丢失记忆。配置恢复采用跨进程事件回调，本次在首个续接 HTTP 发出后约 14 ms 生效，早于首次回复、工具和后续轮次；恢复值确认为预留 28000、近期 20000。这不是请求前的同步屏障。

三项导航最大实际输入约 67.4k，未自然达到 100k；运动专项只证明强制触发后的续接，不证明自然长上下文导航或视觉信息压缩已经充分验证。指令原 CLI supervisor 曾被外部中断，同一个 worker 继续运行并单独确认退出码 0、无遗留子进程；独立监督记录保存这一例外，没有补写原 result 的 worker 字段。

本地构建通过，Python **315 passed**、Core **43 passed**。新增测试检查实际平移、动作／回执与模型消息配对、真实图片、配置回显、事件评分和正常关闭，避免仅凭新 frame ID 或模型自称完成验收。运动专项入口为 `python -m src.harness run --case moving_compaction`，其余案例将 case 分别改为 `objnav`、`instruction`、`goat`。

完整入口与证据：[本轮汇总](../cache/harness-pi/128k-regression-20261014/summary.json)、[导航离线审计](../cache/harness-pi/128k-regression-audit/navigation-audit-summary.json)、[运动专项独立核验](../runs/harness-pi/20261014T104912521812Z-moving_compaction/private/independent_verification.json)、[图像与摘要人工复核](../runs/harness-pi/20261014T104912521812Z-moving_compaction/private/manual_review.json)。各 run 的 `private/trace.jsonl` 保存实际模型内容、工具回执与文件版本，`workspace/observations/` 保存公开图片，`private/environment/` 保存私有评分材料。

## 正式比较要求

一组对照固定场景、任务、模型条件、输入资料、动作与评分协议；不同条件分别报告。服务中断、输出截断、预算耗尽与真实导航失败分开解释，成本包含失败尝试。保留完整分母和重复尝试记录，不挑选最好一次作为总体成绩。
