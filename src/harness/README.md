# Python navigation Harness

Python 主导任务、环境、空间记忆、工具和实验记录；[TS Core](../agent-core/README.md) 保留 Pi Agent 执行能力。详细架构见 [Architecture](../../docs/Architecture.md)。

在已安装 Habitat、PyYAML、Pillow 的 Python 环境中，从仓库根目录运行：

```bash
npm ci --ignore-scripts
npm run build
python -m src.harness doctor
python -m src.harness run --case local
python -m src.harness smoke
```

`doctor` 默认离线；`doctor --online` 运行 Coding 测试。快速 smoke 顺序运行 Coding、局部 IA、压缩续接；`smoke --suite full` 增加查询 QA、ObjectNav、指令、GOAT 和局部构图。

```bash
python -m src.harness --config config/harness-pi/default.yaml run --case query
python -m src.harness run --case objnav --max-requests 32 --seconds 1800
```

沿用现有 `config/harness-pi` 配置。模型服务凭据由 Python 从本地私有配置读取，显式传入 Core；不写入公开 workspace。默认 high reasoning、8192 输出 token、300 秒请求超时。64k 是运行上下文预算，未据此验证服务容量。

## 代码入口

- `__main__.py`：doctor、单案例、快速／完整 smoke。
- `runner.py`：初始化与运行，连接 Core、任务和评分，保存结果。
- `session.py`：直接持有 Habitat；观察、运动、提交、公开过滤与关闭在同一线程执行。
- `tools/`：五个扩展工具的名称、参数和执行函数。
- `memory.py`：文件化地图、卡片、图片的初始化／检查／绘制。
- `core_client.py`：持续接收 Core 事件，在主执行线程处理工具回调。
- `trace.py`：实际输入输出、工具反馈和文件变化的统一记录。
- `worker.py`：episode 超时监督和后代进程回收。

查询任务只读取固定资料；局部 IA 按真实事件结束；完整导航需要 `nav_finish`，GOAT 切换目标后重新决策。模型自主决定工具顺序，程序不自动定位或规划。

工作空间为 `scene/observations/memory/scratch`，前两者只读、后两者可写。私有评分、Core 会话、请求账本和 audit 放在相邻 `private/`，模型不可访问。每次运行有独立请求账本，本轮无全局请求总限额；单 episode 限额包含摘要及失败尝试。

旧 Python Harness、旧数据生成和旧评测保留在 `src/legacy/harness`，旧入口维持原行为。新应用不依赖旧 Agent 循环。
