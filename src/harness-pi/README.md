# Pi navigation Harness

独立于旧 `src/harness/` 的运行入口。使用本地维护的完整 **Pi Coding Agent v1.2.0** SDK；上游来源、许可证与构建适配见 [pi/UPSTREAM.md](pi/UPSTREAM.md)。其余 Pi 包固定为 npm `1.2.0`。源码修改后重新构建，再启动任务。

## 启动

从仓库根目录运行；Node 版本由 `.nvmrc` 固定为 `24.21.0`。

```bash
npm ci --ignore-scripts
npm run build
npm run test:pi
npm run harness -- doctor
npm run harness -- doctor --online
npm run harness -- run --case query
npm run harness -- run --case local
npm run harness -- run --case objnav
```

`doctor` 默认离线，检查 SDK 解析、bwrap、Python 库和资料路径；`--online` 会实际执行 Coding 模型测试并消耗请求额度。其余案例为 `instruction`、`goat`、`mapping`、`compaction`；`smoke` 顺序运行全部七项。`run --case CASE --max-requests N` 只调整单案例上限，不能突破全局额度。

默认配置在 `config/harness-pi/default.yaml`，案例在 `config/harness-pi/smoke.yaml`。通过 `--config PATH` 选择独立配置；每个研究批次使用新的预算文件，保留旧文件作为记录。实现验收批次的全局上限是 **200 次 HTTP 尝试**，失败与摘要同样计数，自动重试与预热关闭。

模型凭据从仓库根 `.env.harness-test` 的 `NAVCRAFT_TEST_BASE_URL`、`NAVCRAFT_TEST_MODEL`、`NAVCRAFT_TEST_API_KEY` 读取。当前服务实际接受 `gpt-6.1-sol`，原指定标识 `gpt6.1sol` 返回 404，因此配置显式使用服务公布的名称；路由为 Responses，high reasoning、8192 输出 token、300 秒请求超时。64k 是保守运行预算，未据此验证服务容量；价格未配置，SDK 显示的零成本不能作为真实费用。

## 运行结构

`cli.ts` / `run.ts` 创建单个 Pi 会话。`runtime.ts` 接入指定服务、串行工具执行与压缩；`tools.ts` 通过常驻 `python/bridge.py` 连接 Habitat。`python/memory.py` 负责可信初始化及公开记忆检查/绘图，`python/verify.py` 只用于私有核验。普通操作不需要重启模拟器。

| 工具 | 模型可执行的操作 |
|---|---|
| Pi 原生 `read/write/edit/bash` | 读资料、编辑记忆、编写并执行 Python、重读结果 |
| `view_images` | 打开已有图片，裁剪或拼图；新派生图使用新引用 |
| `render_map` | 绘制墙层、当前图元及模型位置估计；支持区域/高亮/网格/图层 |
| `nav_observe` | 新单帧或相对机体六视图全景；全景后恢复原相机状态 |
| `nav_step` | 串行执行 1–6 个原子动作，碰撞/目标切换/结束时中断 |
| `nav_finish` | 核对当前目标 ID 后显式提交；文字回答不触发 STOP |

QA 使用固定资料库，回答后评分；局部 IA 在实际穿过指定开口并到达后结束；完整导航需显式提交。GOAT smoke 是原链前两个描述目标的独立副本，不改原 manifest。

## 工作空间与记忆

每次运行创建 `runs/harness-pi/<run-id>/workspace/`。Coding 工具运行于 bwrap，无网络和服务密钥，不开放宿主项目/数据目录；Python 使用配置中的 vln 环境。隔离失败报错，不回退宿主执行。

```text
workspace/
  scene/                    # 只读墙图、初始化资料、证据、格式说明和检查脚本
  observations/             # 只读真实 RGB、帧索引、公开动作反馈
  memory/maps/              # 唯一当前非墙图元
  memory/entities/          # 唯一当前实体卡片
  memory/state.json         # 初始化提示及模型可编辑的位置估计
  scratch/                  # 脚本、派生图片与临时结果
private/                    # 模型不可访问：评分、真实状态、Pi session、审计记录
```

地图使用原图归一化坐标；区域 polygon、出入口 segment、物体 box/组合形状，未知位置可以是 `null`。卡片通过稳定实体 ID 与图片引用关联。预建资料只在初始化复制；删除当前实体后不会恢复旧记录。`scene/index.md` 说明当前格式。

模型可通过 `bash` 检查一致性，例如：

```bash
printf '%s' '{"workspace":"."}' | python -B scene/tools/memory.py check
```

允许分次编辑；检查失败不会禁止观察或移动，但绘图拒绝格式错误，不用旧图冒充更新成功。检查器核验格式与引用，空间正确性由私有几何/真实事件另行检查。地图工具不会自动定位。

## 上下文与产物

正常运行增量追加历史，不滚动删图。使用 Pi 原生切分、摘要合并和会话保存，导航要求覆盖普通及 split-turn 摘要；摘要也走统一预算入口。压缩后仅补充已有公开目标、最新观察与必要图片，不新生成位置判断或改写记忆。文字摘要不能保留所有未表达的视觉细节。

每个运行目录包含 `config.json`、`result.json`、公开图片/记忆，以及 `private/trace.jsonl`、文件版本 blobs、Pi session 和私有环境结果。trace 保存实际模型输入输出、HTTP 协议/用量、工具回执、图片与文件变化。服务未返回的指标保持缺失。私有 trace 含真值及大量图像数据，应与公开训练输入分开使用。

Python 回归：

```bash
/home2/chen/.local/anaconda3/envs/vln/bin/python -m pytest -q
```

当前转换与真实验收集中于 Qa 场景；少量 smoke 仅验证闭环，不是成功率或跨场景结果。测试结果、开发失败和剩余限制见 [实施记录](../../docs/Agents/HarnessImplementation.md)。
