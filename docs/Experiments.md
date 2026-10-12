# 实验与评测

更新：2026-10-13。本文保留协议、关键成绩及影响解释的诊断；逐 run 指标、成本和逐目标结果见[生成报告](../cache/visual-harness-v1/VisualHarnessResults.md)。原始轨迹与评分不因文档合并或阈值调整而改写。

**当前已能记录完整导航交互，尚未通过稳定教师系统的冻结验收，也没有模型训练收益结论。** 开发诊断、离线复评、接口探针和冻结实验分别报告。

<a id="protocol"></a>

## 评测口径

视觉 Harness 提供完整场景地图、描述和 RGB 证据；真实位姿只用于初始箭头，后续无实时 pose 或 depth。当前属于地图辅助的自定义设置。

| 任务 | 当前执行判据 | 原始 benchmark 公开实现／主要差异 |
|---|---|---|
| ObjectNav | 主动 STOP，到目标类别任一成功视点的最短测地距离 `<1 m` | HM3D ObjectNav 为成功视点 `<0.1 m` |
| GOAT | subtask STOP，到当前目标成功视点的最短测地距离 `<1 m` | GOAT 为成功视点 `<0.25 m`；旧 NavCraft 开发运行曾用 3 m |
| 指令导航 | STOP，到终点的最短测地距离 `<3 m`，另统计有序途经约束 | VLN-CE 同为 3 m，但本轮为自建 HM3D 指令，不是官方 MP3D VLN-CE |

来源：[ObjectNav 配置](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/habitat/task/objectnav.yaml)、[距离与成功测量](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/nav/nav.py)、[VLN-CE 配置](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/vlnce_task.yaml)、[GOAT 配置](https://github.com/Ram81/goat-bench/blob/74c41d19d4a4c3608d1575b512087b5a529aee0e/config/tasks/goat_stretch_hm3d.yaml#L45-L55)及[测量](https://github.com/Ram81/goat-bench/blob/74c41d19d4a4c3608d1575b512087b5a529aee0e/goat_bench/measurements/nav.py#L266-L294)。

成功视点是预采样的合格观察位置，**视点距离不是物体中心距离**。将视点容差改为 1 m 会扩大成功区域，不能称为原始 benchmark 的复现。当前机器人为 0.88 m 高／0.18 m 半径，HM3D v2 Challenge 的 Stretch 为 1.41 m／0.17 m；输入、相机、动作和 NavMesh 也须统一才能比较。

GOAT 主指标采用成功子目标／全部预定子目标，并报告整链成功率；未尝试目标计失败。官方按 episode 内成功比例再取均值，还过滤超过 55 词的描述；本轮保留完整链，聚合与任务集合不同。[论文中“物体 1 m”](https://arxiv.org/html/2404.06609v1#S3)不能替代代码中的视点判据。

已核查的相关实现也不统一：

| 方法 | 已确认条件 | 使用边界 |
|---|---|---|
| [BSC-Nav](https://github.com/Heathcliff-saku/BSC-Nav/blob/ad4c544f7d1feb980ffad33d4eefa8ad2e39fad0/env.py#L552) | 默认参数覆盖 YAML 后为 `VIEW_POINTS@1m + STOP` | RGB-D、持续位姿与局部规划器；是发布代码默认值核查 |
| [OmniNav 快系统](https://github.com/amap-cvlab/OmniNav/blob/e8b485953c6555e841a3283c3e3a4f7d8d0f24c3/infer_ovon/ovon/configs/ovon_pano.yaml#L32) | `VIEW_POINTS@1m + STOP`，连续 waypoint 执行 | [慢快系统自定义计分](https://github.com/amap-cvlab/OmniNav/blob/e8b485953c6555e841a3283c3e3a4f7d8d0f24c3/infer_ovon_slowfast/run_nav_ovon_omni.py#L301)无额外 STOP 门槛，不能混用版本 |
| NavHarness／HarnessVLN／SpaceVLN | NavHarness 主表半径未确认；后两者论文写 1 m | 未确认距离参照时，不推定为成功视点 1 m；论文来源见 [Report](Report.md#evidence) |

对照需同时记录距离参照、阈值、STOP、输入与低层执行器；仅统一一个数字不足以建立可比实验。

## 现有结果

所有失败、服务中断与未尝试目标保留在选定集合分母；不同尝试不取最优结果合并。以下样本和版本不同，不能据此作严格模型排名。

| 记录 | ObjectNav | 自建指令 | GOAT 子目标／整链 | 性质 |
|---|---:|---:|---:|---|
| Sol 历史开发轨迹，离线统一视点 1 m | 3/6（50.0%） | — | 16/27（59.3%）／0/3 | 非新运行，原运行有源码调整或主动终止 |
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

### 接口重试与证据范围

Qwen 首次小规模四并发存在 120 秒超时；改为 300 秒后，两条重试仍因 4096 token 输出上限中断。8192 token／low reasoning 重跑中，床目标 298 动作后耗尽请求预算，指令 94 动作后 STOP 但未成功；GOAT 前三个目标成功后为切换设置主动中止，3/7 仅是部分进度。延长超时不能替代导航能力改进。

请求条件与结果分别保存在[首次四并发](../cache/visual-harness-v1/qwen-four-concurrent-20261012.json)、[超时重试](../cache/visual-harness-v1/qwen-timeout-retries-20261012.json)、[8192 重跑](../cache/visual-harness-v1/qwen-8192-low-20261012.json)。八组 Qwen 原目录仍保留，[归档表](../cache/visual-harness-v1/runs-archive-latest.json)记录路径；更早三组 smoke 原轨迹已删除，仅有[清理清单](../cache/visual-harness-v1/runs-cleanup-20261012.json)等汇总，不能严格复评。不要混淆归档与删除。

[多图与工具续接探针](../cache/visual-harness-v1/probes/summary.json)中 Qwen、Sol 通过；DeepSeek 首次完整请求超时、后续 low reasoning 探针通过。它们只证明接口条件，不是导航成功率。

<a id="harness-diagnostics"></a>

## 已定位问题与解释边界

| 问题 | 修正／结论 | 不能据此推断什么 |
|---|---|---|
| NavMesh 与官方视点约有 0.1 m 高度错位 | ObjectNav visual 显式设 `navmesh_cell_height: 0.05`；Qa 误差小于 1 微米，验收目标集合中位数小于 5 微米 | 不是改成功阈值或删目标；边缘／台阶残差仍保留，旧配置与旧结果不变 |
| 0.25 m 步长与 0.1 m 容差是否矛盾 | 几何探针执行前进后可落入成功视点区域，因为有多个合格视点 | 私有几何探针不是模型导航成绩，也不证明所有目标容易到达 |
| 摘要漏掉近期 RGB，把餐桌记成目标床 | 摘要改为读取完整闭合上下文，压缩后保留最近五组交互，禁止猜测升级为事实 | 后续 STOP 也见到近期 RGB，不能断言旧摘要是唯一原因；尚无性能收益对照 |
| 未调用图片工具被摘要写成“没看地图” | 区分自动提供图片与显式查询，检查真实输入 | 不能推断模型内部是否关注地图 |
| 跨目标沿用旧箭头、门口反复碰撞、识别目标后过早 STOP | 指南要求更新估计、核查开口和机身净空，再确认接近情况 | 指南修改不等于问题已被解决；未加入私有距离纠偏或坐标代走 |

几何证据：[开发场景诊断](../cache/visual-harness-v1/geometry-diagnosis/report.json)、[固定验收视点核验](../cache/visual-harness-v1/geometry-diagnosis/acceptance-05.json)。修正没有改写共享 NavMesh、目标坐标或模型输入；后来的 1 m 协议调整是独立变化。

Sol 旧指令开发集为 6/6，但运行中源码变化，只作诊断；后续冻结 smoke 为 0/1、332 动作、途经约束 1/3、终点距离 10.53 m，版本 `1bdef93ff4f92dfd` 前后指纹一致。错误是 `Model request time budget exhausted`，不能仅凭 `service_error` 分类归因服务故障。样本与版本不同，亦不能归因某项改动导致退化。

原 GOAT 第三链的 `worker_exit_code=-15` 来自主动停止，不是模拟器自发崩溃；当时已有六个失败，即便剩余全成功也至多 21/27，低于当轮 80% 目标。已有成功和未尝试目标继续按原协议保留。

## 后续运行与验收

固定集合为 15 个开发起点／任务链、60 个验收起点／任务链，验收 GOAT 含 149 个预定子目标。**尚未通过三任务族各 20 条、两轮均达到 80% 的冻结验收。**

运行前后核验代码、清单、配置及地图资源指纹；有变化则标为 invalidated。服务、输出截断、动作预算与真实导航失败分开统计，成本包含失败请求；无响应请求不能凭空估算 token。新条件使用独立 run，保留完整分母和重试历史。启动及生成报告的命令见 [Architecture](Architecture.md)。
