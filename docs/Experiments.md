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

GOAT 主指标采用成功子目标／全部预定子目标，并报告整链成功率；未尝试目标计失败。官方按 episode 内成功比例再取均值，还过滤超过 55 词的描述；本轮保留完整链，聚合与任务集合不同。[论文中“物体 1 m”](https://arxiv.org/html/2404.06609v1#S3)不能替代代码中的视点判据。

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

## 正式比较要求

一组对照固定场景、任务、模型条件、输入资料、动作与评分协议；不同条件分别报告。服务中断、输出截断、预算耗尽与真实导航失败分开解释，成本包含失败尝试。保留完整分母和重复尝试记录，不挑选最好一次作为总体成绩。
