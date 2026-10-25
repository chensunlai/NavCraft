# NavCraft 本地实现与取证

先检查当前文件再复用；下面是已验证过的环境线索，不是任意新场景都可直接运行的一键流水线。

## 当前资源

- 当前项目位置以会话 cwd 为准。已验证位置为 `/data/oldhome/chen/Project/NavCraft`；旧记录可能使用 `/home2/chen/Project/NavCraft`，先检查实际路径。
- 已验证 Habitat 环境：`/data/oldhome/chen/.local/anaconda3/envs/vln/bin/python`。旧 `/home2/chen/...` 路径可能失效；先检查现有环境。采集时选择空闲 GPU，不硬编码某张卡永远可用。
- 输入：`data/nas/scene_datasets/`。历史素材可能经过 data 和 NAS 符号链接。
- 示例完整归档：`data/scene_maps/hm3d/val/00829-QaLdnwvtxbs/`。重点看修正后的 `raw/layout_review.json`、`raw/validation.json`，不要复用修正前的灰区／门位置。
- 文本与房间代表图参考：`runs/runs_0_astra_xhigh_rgbd_mapping/workspace/mapping/`；该 run 仅属于 4ok3usBNeis。不能把其中的房间、帧号、几何或统计套用到其他场景。

## 当前代码边界与历史工具

2026-10-12 已将本轮场景图构建代码从 NavCraft 主代码库移出，最终采用 Astra＋本 Skill。实验记录见 `docs/SceneMap.md`。**不要默认导入 `src.dataset.scene_map`、`floor_plan`、`schematic_map`、`scene_map_evidence`、`archive_scene_maps` 或 `wall_filters`，这些模块不再是当前项目入口。**

历史工具与配置备份在 `cache/scene-map-code-archive-20261012/`，路径相对于该备份根目录：

| 历史文件 | 可参考的内容 | 限制 |
|---|---|---|
| `src/dataset/schematic_map.py` | 从独立 JSON 绘制墙、分区、box 与整图 | 可将必要绘图逻辑复制到本场景工作目录使用；检查多层布局及字段，不自动恢复到项目 src |
| `assets/floor_plans/<scene>/` | 初稿图层数据 | 最新数据优先取最终归档 `raw/`，不能用初稿覆盖已接受修正 |
| `src/dataset/scene_map_evidence.py` | 离线视角采样、按实例挑选 RGB | 依赖固定配置和旧 inventory/cache，不是任意场景通用采集器 |
| `src/dataset/archive_scene_maps.py` | 文本、关联和五项目录组织 | 绑定旧三场景及 xhigh 对应表，不直接运行 main；仅参考所需组织逻辑 |
| `src/dataset/scene_map.py`、`wall_filters.py`、`floor_plan.py` | 投影、过滤和分层实验 | 含跨模块／相邻 AstraNav 依赖，不能当独立通用工具；原始投影只作证据 |
| `config/scene_map/three_floor_plans.json` | 历史输入和取证位置 | 不是支持场景白名单，也不是新场景模板真值 |

新场景建立独立工作目录；二维双墙图绘制、门洞裁切、墙体阶段校验和散列刷新已由本 Skill 的 [墙体工具](wall-tools.md) 提供，优先复用。Habitat 取景、原始网格测量与路径查询先复用本地已验证接口，仅为缺少的能力编写适配；按本 Skill 的归档格式交付。历史备份缺失不应成为必须恢复旧流水线的理由。现有 `src/environment` 仅在接口合适时复用，不能为构图重新侵入导航主代码。

重建已有归档优先读取最新 `raw/`；查清持久数据与缓存的权威来源，避免旧输入覆盖已接受修正。

## 离线视图

HM3D `.basis.glb` 可同时配 `.semantic.glb`、`.semantic.txt` 与 `.basis.navmesh`。曾验证可用的临时 stage 配置字段：`render_asset`、`semantic_asset`、`semantic_descriptor_filename`、`nav_asset`；scan 为 Z-up 时可设置 `up: [0,0,1]`、`front: [0,1,0]`，但其他格式必须重新确认方向。

加载后先看 RGB，并检查 semantic_scene 是否非空、semantic sensor 是否全零；只有文件存在不代表语义正确加载。查询对象 ID 与语义描述映射，不能盲用数组下标。

在同层、正确房间内选相机，记录 Habitat XYZ、朝向或旋转、hfov、分辨率和来源。拍房间全景与对象近景；对每个入口从两侧拍摄。主动用自由相机检查扫描缺口时标明该视角不是可行走位置。记录 snap 前后位置，防止被吸附到另一层或墙外。

需要测量时，将深度、内参与相机位姿投回同一坐标。原扫描 `(X,Y,Z-up)` 与地图 `(Habitat X,-Z)` 通常一致，但应以实际资产验证。全高度投影会把门楣压到门洞；实例大 AABB 会把空的 L 形内角填满，均不能直接当最终轮廓。

房间图可优先复用已检查过的观测，无证据区域才补采。语义像素数可筛掉遮挡图，但不能证明标签正确。图像联系表用实际渲染帧制作；不要用生成式图片代替场景证据。

## 持久化与重复运行

最终主图由同一套修改后的源数据生成。先备份受影响文件，再有范围地更新本场景；检查归档脚本是否会把手工修正覆盖回旧缓存。持续保存修改记录和证据，使下一次请求可以从最终版本继续。

脚本与渲染器有依赖缺失时，报告具体缺口并尝试已有可用路线；不能以缺库为由虚构图片或把未执行检查标为通过。此 skill 不要求调用任何收费模型，也不预设特定视觉教师。
