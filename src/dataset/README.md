# 数据与地图边界

`episodes/` 将官方任务转换为 `EpisodeSpec`；`build_maps/` 离线构建场景归档；`maps.py` 离线发布墙图并加载发布结果。Bench 只选择已发布地图、执行任务和评分，运行时不补墙、改标注或转换坐标。

## 文件与坐标

| 数据 | 路径／内容 | 坐标约定 |
|---|---|---|
| 原始场景 | `data/nas/scene_datasets/<scene_id>`（由环境配置指定根目录） | Habitat 世界 `[X,Y,Z]` 米，Y 向上 |
| 构建归档 | `data/scene_maps/<dataset>/<split?>/<scene>/raw/` | 平面 `[X,-Z]` 米，`scene.json` 声明坐标与单位 |
| 发布地图 | `data/scene_map_datasets/<dataset>/<split?>/<scene>/map.json` | 平面 `[X,Z]` 米；逐楼层保存共享的 `origin/size_m` 和 `walls/wall_real` 图层 |
| 运行公开地图 | `workspace/scene/map.json` | 仅 `origin/size_m/floor_height/walls`，无源文件路径、语义标签或核验材料 |
| 空间记忆 | Harness 工作空间 | 归一化 `[u,v]`，右／下递增；地图显示裁剪不改变坐标 |

`scene_id` 是相对场景根目录的 POSIX 路径，例如 `hm3d/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.basis.glb`。官方数据里的前缀在 episode 读取时统一一次，环境用 `common.scenes.resolve_scene` 定位资产；不在各运行器重复拼接。HSSD 的资产路径不包含 split，发布时由构建队列生成 `index.json`，明确关联到地图目录。

所有世界／UV／方向换算使用 `common.coordinates`。身体朝向 0° 面向世界 `-Z`，右转为正；`map_pose` 接受 Habitat `[x,y,z,w]` 四元数。

## 离线发布

```bash
python -m src.dataset.maps publish \
  --source-root data/scene_map_datasets \
  --output-root data/scene_map_datasets \
  --queue data/samples/scene_build_queue.json \
  --queue data/samples/scene_build_queue_nonhssd_100.json \
  --report cache/map-publish.json
```

可从原归档根目录发布到目标目录，也可对已经复制的墙图目录原位生成小型 `map.json`；均不修改 `scene.json`、`walls.json` 或证据。不同的源字段只在此离线边界解析。坐标、单位未知、未完成归档、缺失墙体或非法几何会明确报错；没有 `wall_real` 时不以 `walls` 冒充。源文件哈希留在发布元数据及运行私有溯源中，不进入公开地图。

运行时 `load_wall_map` 要求起点高度在一个且仅一个楼层的 ±0.5 m 内，并在画布内。楼梯中间起点、重复楼层或未发布地图显式报告，不自动改楼层或替换样本。源资料明确标为非楼层的楼梯端点不发布为独立楼层。地图结构通过不等于墙体已经完成视觉复核。

验证：`python -m pytest tests/test_coordinates.py tests/test_dataset_maps.py tests/test_build_maps.py`。
