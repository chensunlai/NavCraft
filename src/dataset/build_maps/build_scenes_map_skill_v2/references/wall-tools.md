# 可复用墙体工具

构建与修复墙体时优先复用以下脚本，不为每个场景重写绘图、门洞裁切、归档校验和散列计算。脚本位于本 Skill 的 `scripts/`，无固定场景 ID、坐标或楼层名。

需要 Python 3.10+、Shapely、Matplotlib、Pillow。仅这些操作不需要 Habitat、GPU 或网络。先使用现有可用环境，不直接假定某个旧绝对路径仍有效。

## 1. 双墙图层绘制

```bash
python <skill-dir>/scripts/render_wall_layers.py \
  --scene <scene-dir>/raw/scene.json \
  --walls <work-dir>/authored_walls.json \
  --real <work-dir>/authored_wall_real.json \
  --stairs <work-dir>/stairs.json \
  --output <work-dir>/rendered
```

输入约定：

- `scene.json`：`scene_id`、`levels`；每层含 `id, floor_height, bounds: [[xmin,xmax],[ymin,ymax]]`。坐标使用 Skill 的 Habitat 米制约定。
- 两套施工源线：`scene_id, segments, openings`；墙段采用 [墙体字段](wall-completion.md)。最终 wall 的段为 `status: resolved`，wall_real 每段为 `origin: observed_schematic`。不可传入已经裁切过的派生 JSON。
- 门洞含 `id, level, segment: [[x1,y1],[x2,y2]], status`。可用 `wall_ids` 精确指定需要裁切的施工线。仅裁切同层、与门洞平行的墙面，避免把附近垂直返墙或护栏抹掉。默认门洞两侧裁切半宽为 0.18 米，可用 `--portal-half-width` 或每门 `cut_half_width_m` 指定；这是绘图容差，不是实测墙厚。
- 楼梯文件含 `scene_id, stairs`，每项包含 `id, from_level, to_level, from_height, to_height, polygon, treads, ascent_points, lower_entry, upper_exit, evidence_frames`。`treads` 为线段数组；上行折线从低层指向高层。可用 `label_positions: {楼层ID: [x,y]}` 调整标签。没有楼梯的场景显式保存空 `stairs`；不要求为了工具虚构楼梯。不传 `--stairs` 时从 wall 输入的 `stairs` 读取。
- 可选 `structural_voids` 包含 `level, polygon`；`drawing_notes` 包含 `level, position, text`。护栏必须在 segments 内，不能依赖辅助线显示。

输出为两套门洞已裁切的 `walls.json` / `wall_real.json`、PNG/SVG，以及 `render_report.json`。派生段保留 `source_wall`、证据和估计来源；楼梯目标层与箭头自动按数据绘制。两层面板共用尺度。输出目录不能覆盖输入文件，脚本会拒绝这种情况。

`render_report.json` 记录输入与输出散列，`visual_review_performed` 固定为 false。生成图片后仍须实际打开整图与局部。脚本不决定墙线、补全方案、楼梯方向是否符合真实场景，也不自动去除重复墙或声称完成验收。

## 2. 墙体阶段只读校验

```bash
python <skill-dir>/scripts/validate_wall_stage.py <scene-dir>
```

输出 JSON 到标准输出；成功退出 0，错误退出 1，不写归档。检查五项顶层、双墙图层、墙段来源、重复线、门洞阻挡、两套门洞一致性、楼梯几何与引用、证据文件、文档链接、别名一致性、关键文件散列及已有检查记录是否过期。

- `scene.stage` 应为 `walls_only`，`scene.files` 可覆盖默认文件名。无需 rooms/regions/objects。已有后续阶段文件只产生提示，不自动删改。
- `scene.files.wall` 与 `scene.files.walls` 均可指向最终墙文件。额外提供 `wall.json` 与 `walls.json` 时两者必须一致。
- 要求 `wall_real` 由最终 wall 中保留的观测几何组成；允许不同分段，检查几何覆盖，不依赖段数相同。
- `stairs.json` 必须存在，无楼梯时明确空数组。检查器不依据“多层”自动虚构跨层连接。
- 检查记录使用 `inputs_sha256` 或 `checked_inputs_sha256`，键相对 `raw/`。历史工作文件名不同，可在 `layout_review.json.build_working_input_aliases` 中显式映射，例如 `{"walls.json":"authored_walls.json"}`。禁止靠改散列伪装重新运行检查。
- 机器结构通过与人工视觉通过分别记录；未提供路径报告会提示，脚本不自行运行模拟器，也不把缺失报告当路径成功。发现问题时修源数据、重绘、重跑对应检查，而不是放宽容差。

将输出保存为 `raw/validation.json` 时，先运行命令到工作目录报告，再按成功／失败真实保存；不要写“已目视复查”来替代实际读图。已有人工结论可另外保留在 `review_final.json`。

完整地图仍使用 `validate_archive.py`；两者检查范围不同，不能用墙体检查器替代完整地图的家具／分区检查。

## 3. 刷新归档校验和

```bash
python <skill-dir>/scripts/refresh_manifest.py <scene-dir>
python <skill-dir>/scripts/refresh_manifest.py <scene-dir> --write
```

第一条只预览变化；第二条原子替换 manifest 的校验和。其余字段（版本、状态、计数、检查结论）全部保留，调用者先按实际修改更新它们。此工具不复制证据、不归档缓存、不备份几何、不标记完成，也不修复过期检查记录。

需要 `raw/manifest.json` 已存在，校验和基准为场景根目录。计算所有归档文件的散列，仅排除当前 `raw/manifest.json`、`raw/validation.json`、`raw/standard_archive_validation.json`；旧版本中的同名文件照常校验。拒绝指向归档外文件的符号链接。刷新后重跑阶段校验，再保存校验结果。

## 顺序与职责

复用工具的流程为：读取最新归档并备份改动文件 → 取证并编辑施工源线／楼梯 → 生成双图层 → 人工读图并修正 → 重跑门口、同层与楼梯路径检查 → 复制本次产物及证据入归档 → 同步说明、版本、计数与复查记录 → 刷新校验和 → 阶段校验。

几何解读、取景位置、缺损补全、楼梯高度与真实连接必须结合场景决定。RGB-D 采集和 NavMesh 查询仍使用已验证的场景工具，先检查坐标与资产接口；不要直接把某场景缓存脚本当通用工具复制到 Skill。自动脚本也不替代逐场景验收。

修改工具后，可运行不依赖 GPU 的回归测试：

```bash
python -m unittest discover -s <skill-dir>/scripts/tests -v
```
