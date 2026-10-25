使用本次显式附加的 build-scene-map skill，只处理下面的一个场景。构建范围：{stage}。
指定 skill 文件：{skill}/SKILL.md。请读取这个路径及同目录引用，不使用其他目录的同名 skill。

场景记录（路径相对于仓库 {repo}）：
{scene}

本次工作目录：{workdir}
最终场景归档：{destination}
{workspace_note}
可用 Habitat Python：{python}
渲染仅允许物理 GPU {gpu}；CUDA_VISIBLE_DEVICES 已设为 {gpu}，渲染器使用逻辑设备 0。禁止使用物理 GPU 2、3，不得停止或修改任何模型服务或其他任务进程。Skill 中的旧机器路径、选卡建议以这里为准。

执行要求：
1. 先读取 SKILL.md 和引用文档。核对场景资源、实际坐标与楼层，不根据推荐高度直接忽略其他楼层。HSSD 必须加载完整 scene_instance 和 scene_dataset_config，不能只渲染空的 stage。
2. 有已有归档先读取和检查，将受影响文件备份到本次工作目录后再改。原始场景资产只读。只写本场景归档和本次工作目录，不修改项目源码、skill、总清单、其他场景或共享索引；需要脚本放在工作目录。
3. 真实渲染并查看 RGB；逐区域、开口和楼梯取证，完成初检、修改及最终复检。不得使用生成式图片充当证据，也不能把几何校验通过声称为视觉验收或代理实际走通。
4. 按 skill 交付独立可读的五项顶层归档；原始证据复制进去。完整地图同时保存房间、主要物体与证据；仅墙体阶段不提前生成房间/物体。
5. 使用 skill 对应校验器，保存实际检查与修改记录。无法确认的问题写明限制，不伪造通过。只处理这个场景，不自行开始下一场景。
6. 不读取任何凭据文件，不打印环境变量或密钥。任务无需其他模型、网络服务、训练或导航基准重跑。不要创建额外 Codex 子进程或委派其他 agent。

最终按给定 JSON schema 回答：status 为 reviewed/partial/blocked，summary 简述完成工作，limitations 列出尚存限制，review_evidence 给出归档内实际使用的复查记录或图像相对路径。reviewed 只表示你已完成本轮视觉复查，不能取代后续独立验收。
