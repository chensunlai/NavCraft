# 归档格式与关联约定

## 路径与独立性

例如素材目录 `data/nas/scene_datasets/hm3d/val/00829-QaLdnwvtxbs/` 对应 `data/scene_maps/hm3d/val/00829-QaLdnwvtxbs/`。不能只取末尾短 ID，也不能因物理路径经过 NAS 符号链接而改变逻辑相对路径。

场景顶层只留 `floor_plan.png`、`scene_description.md`、`evidence/`、`evidence_rooms.jpg`、`raw/`。

`raw/` 推荐沿用 NavCraft 当前格式：

| 文件 | 内容 |
|---|---|
| `scene.json` | scene_id、米制单位、坐标约定、楼层高度与显示范围、图层文件名 |
| `walls.json` | 最终 wall：segments、openings；实体墙／围栏／场景边界／缺失补全及来源 |
| `wall_real.json` | 估计前由扫描与 RGB 支持的墙体及围栏，不含补全段 |
| `stairs.json` | 墙体阶段的独立楼梯：范围、连接层、平台、踏步、上下端和方向 |
| `regions.json` | 房间／功能区与独立楼梯多边形、unknown_areas、stair_connections |
| `objects.json` | objects，每个物体的几何 primitives、房间与帧关联 |
| `rooms.json` | 房间几何、中文名称、function、description、object_ids、doors、evidence_frames |
| `doors.json` | 门／通路、连接区域、开放／关闭／未核验状态、证据、局部路径核验 |
| `evidence.json` | frames，每帧的 ID、相对 RGB 路径、source、位姿／相机参数（已知时） |
| `layout_review.json` | 多轮检查的问题、证据、修改、复检结果及未决项 |
| `manifest.json` | 数据集相对子路径、版本、来源、计数、校验和及其路径基准 |
| `validation.json` | 最终版本机器检查与人工视觉检查的独立结论 |
| 图层／楼层子目录 | walls、regions、objects 的 PNG/SVG，floor_plan.svg，逐楼层输出 |
| `provenance/` | 必要旧版本、局部核验材料；不要复制整个模拟器数据集 |

墙体阶段设置 `scene.stage: walls_only`，不生成尚未请求的 `regions.json`、`rooms.json`、`objects.json`。`evidence_rooms.jpg` 在此阶段作为结构与楼梯证据联系表，不暗示房间阶段已经完成。两套墙体各自提供 JSON 与可浏览图层，具体字段和验收见 [墙体补全](wall-completion.md)。

可保留辅助 `raw/index.html`、`raw/README.md`。总入口放 `data/scene_maps/index.json` / `index.html`，不要在场景顶层增加第六项。

## 最小数据形状

保持既有字段兼容；下面的 ID 和数值仅说明格式，不可当作新场景的内容：

```json
{
  "scene_id": "SCENE",
  "objects": [{
    "id": "O01", "level": "L1", "room": "R01", "label": "Sofa",
    "primitives": [{"type": "rectangle", "center": [2.1, 3.0], "size": [2.0, 0.8], "angle_deg": 0}],
    "evidence_frames": ["render_room1_sofa_a"],
    "basis": "Describe the views and measured support actually used"
  }]
}
```

墙段采用 `id, level, start, end, status, basis`；开口采用 `id, level, kind, segment, connects, status, evidence_frames`。分区采用 `id, level, kind, polygon`。不同楼层可使用相同楼梯 ID，实际唯一键为 `(level, id)`。

普通物体不能任意重叠。台盆在台面内等真实包含关系用 `supported_by: "counter_id"` 显式关联，并在文本和 RGB 中可解释；该字段不允许豁免任意相交。高处电视、吊柜、桌下椅子等需要高度／附着关系，并保持图面可辨，不把高度上的投影叠加伪装成无碰撞地面占用。

## 帧与路径

- 帧 ID 示例：`xhigh_000791`、`render_L1_R08_north`。不同 run 的 791 必须是不同 ID。
- `raw/evidence.json` 中 `rgb` 使用 `../evidence/frames/<id>.jpg`，相对于该 JSON。
- 其他 JSON 文件间引用相对于其所在目录；正文引用 JSON 为 `raw/objects.json`，正文引用图为 `evidence/frames/...`。
- `manifest.json` 推荐 `checksum_path_base: ".."`，校验和键相对于场景顶层；生成 checksum 时排除 manifest 自身和会再修改的 validation 文件。
- 原始来源路径可留作溯源，但实际打开证据的字段必须指向归档内文件。不凭空补相机参数；缺失时写未知。

## 检查记录

每轮记录：检查对象、疑点、帧 ID／几何依据、采取修改、修改后结果。区分未检查、证据缺失、已选定补全、已确认开闭等状态；最终 wall 的 `resolved` 只表示几何已裁决，`origin` 仍区分观测和估计。未知不能伪装成已确认不存在。

自动检查、人工读图、NavMesh 局部查询、实际移动验证分别记录。脚本只做第一类和几何关系检查；没有运行的验证不能从旧模板复制为 passed。
