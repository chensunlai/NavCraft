# 四类短任务

首批每类 4 条，用于固定状态下的 Harness 回归。构造不调用模型；Bench 只恢复冻结样例、执行和评分。

| 类型 | 公开输入与要求 | 私有验收 |
|---|---|---|
| `ground` | 当前观察、标注目标、墙图和空记忆；记录实例位置、描述和证据 | 实例范围、实际选中像素、证据有效性；描述仅报告是否填写，语义需视觉复核 |
| `recognize` | 旧观察及一条已有记忆、另一真实视角；复用已有实例或登记不同实例 | 两条同实例、两条同类不同实例；拒重复登记、错误合并和误删 |
| `revisit` | 旧物体记忆、真实离开后的当前观察；回到该实例旁并提交 | 起点目标完全在当前水平视野外，参考路线真实执行并独立重放；不声称目标被遮挡 |
| `recover` | 真实前进受阻状态、原反馈和带 P1 的地图；调整走法到达 P1 | 初始必须 `blocked`；侧移或退离路线真实重放；没有门标注时不声称测了穿门 |

前两类复用 `data/generated/spatial-pilot-20261026` 的观察、语义掩码和实例范围，重新写出当前格式快照。旧图与当前图来自不同真实视点；原目标图、旧模型历史和答案不进入输入。后两类通过当前 Habitat 与 Harness 工具采集，仅使用物理 GPU 0 或 1。

每条 `cases/<id>/` 包含：

- `task.json`：普通请求、图像引用、预算、来源和快照入口。
- `snapshot/`：当前帧、公开工作空间、独立恢复所需的私有环境与投影数据。
- `private/label.json`：评分标签；运动题另有真实参考和独立重放记录。
- `construction.json`：构造依据及未完成的检查。

模型只获得快照中的公开工作空间和请求。参考轨迹是程序辅助核验，不是教师示范。每次执行复制到独立运行目录，不修改冻结输入。图文检查页包含私有标签，仅供检查者阅读。

```bash
python -m src.dataset.short_tasks \
  --source data/generated/spatial-pilot-20261026 \
  --output data/generated/short-tasks-<日期> --gpu 0

python -m src.bench.short_tasks \
  --task data/generated/short-tasks-<日期>/cases/ground-01 \
  --output runs/short-ground-01 \
  --config config/models/qwen27.yaml --config config/harness/entity_geometry.yaml --gpu 0
```

构造支持 `--phase memory/motion/check`。`status.json`、`manifest.json`、`index.html` 随每条产出更新；终端使用 tqdm。候选没有真实走通则写入 `rejections.jsonl`，不足配额报告 `partial`。这里只冻结少量工程样例，不将共享实例的派生题视为独立场景，不报告模型成功率。

重复模型测试使用 `scripts/run_short_tasks.sh`，组合现有模型与工具配置：27B medium 每题 4 次、Sol high 每题 2 次，共 96 次，总并发 8。每次独立进程、会话与工作空间，结果按 `模型/repeat-N/任务ID/` 保存；`summary.json` 单列明确正确、明确错误、待人工核验和运行异常，不把未核验结果当作成功。
