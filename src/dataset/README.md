# 场景与任务数据

最终构建结果只有一份，直接保存到 `data/scene_maps/<dataset>/<split?>/<scene>/`。场景目录存 `scene.json`、`walls.json`、`wall_real.json`、房间/物体标注、证据及检查文件；`raw/` 单独保存原始工作区（脚本、采样中间产物、备份）。缺失结果保持缺失，不补空文件；不另建发布副本或数据目录中的中间 `map.json`。

## 从原始文件到运行

`episodes/` 读取官方任务为 `EpisodeSpec`；`build_maps/` 构建原始场景表征；`maps.py` 直接读取这些场景文件并选择运行所需的楼层和墙图层。Bench 不补墙、修改标注或自动切换图层。

| 内容 | 位置与坐标 |
|---|---|
| 仿真资产 | `data/nas/scene_datasets/<scene_id>`；Habitat `[X,Y,Z]` 米，Y 向上 |
| 原始场景表征 | `data/scene_maps/.../scene.json` 及其他文件；平面 `[X,-Z]` 米，字段显式声明坐标与单位 |
| 本次运行的公开地图 | `<run>/workspace/scene/map.json`；墙线 `[X,Z]` 米，只含 `origin/size_m/floor_height/walls` |
| 本次运行的私有来源 | `<run>/private/map_source.json`；原始文件哈希、楼层与图层 |
| 模型空间记忆 | 工作空间的 `memory/entities.json`；全局 `[u,v]`，向右/下增大 |

原始标注不被改写。读取时统一完成必要的坐标计算，再将公开几何快照放进当次运行；模型不会因此看到房间标签、目标位置、私有描述或证据索引。此快照属于实验记录，不是第二套场景数据集。后续修改原始文件只影响新运行，已有运行保留自己的地图和来源哈希。

公开世界/UV/方位换算使用 `common.coordinates`。朝向 0° 对应世界 -Z，右转为正；地图裁剪和旋转不改变存储坐标。

`maps._source_point` 是构图 `[X,-Z]` 到运行 `[X,Z]` 的唯一换算入口，墙端点与楼层范围共用它。指定图层中的护栏、外围和补全边界随墙段一起保留，不按 `kind` 静默过滤。楼层选择与两图层共享画布仍在 Dataset 完成；Harness 不再次翻转 Z。

## 场景定位与校验

`scene_id` 是相对仿真资产根目录的 POSIX 路径，如 `hm3d/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.basis.glb`。官方数据的已知前缀在读取时规范化一次。HM3D/MP3D 的表征目录与资产目录对应；HSSD 用资产文件名中的场景 ID 查找唯一 split 目录，找不到或跨 split 重复时明确报错，不另存资产索引副本。

`load_wall_map(directory, start_position, layer)` 直接读取根目录的原始文件，返回公开地图和私有来源。`prepare_map(...)` 将这两部分保存到运行目录；开发测试也可以显式传入已校准的单楼层地图文件。默认 `layer=walls`，可选 `wall_real`。

加载要求：坐标与米制单位明确；墙段有效且楼层引用存在；起点高度在唯一楼层的 ±0.5 m 内并位于画布中；指定图层有实际几何。标为非楼层的区域不充当导航楼层。缺图、楼层歧义、非法坐标、读取期间源文件变化或明确标为未完成的状态均报错，不用另一个目录或旧副本兜底。两种墙图层共享画布校准，切换图层不会改变实体 UV 的含义。

这项读取只验证结构、坐标和接入条件，不替代 RGB 检查，也不表明所有房间、物体标注都已验证正确。后续任务构造应直接使用同一场景目录里的原始标注和证据，不能再创建一份墙体副本作为中转。

```bash
python -m pytest tests/test_coordinates.py tests/test_dataset_maps.py tests/test_build_maps.py
```
