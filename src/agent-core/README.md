# Pi Agent Core

独立 TypeScript 运行层，使用 Pi Fork 中的 Coding Agent **v1.2.0**。完整 Pi 仓库作为 `pi/` Git 子模块保存，NavCraft 固定其提交版本；Fork 的 `navcraft` 分支维护本地修改。上游版本、许可证和源码修改见 [pi/UPSTREAM.md](pi/UPSTREAM.md)。Python 导航应用位于 [../harness](../harness/README.md)。

## 构建与运行

初次克隆可用 `git clone --recurse-submodules https://github.com/chensunlai/NavCraft.git`；已有仓库先初始化子模块。以下安装、构建与测试命令均在 **NavCraft 根目录**执行：

```bash
git submodule update --init --recursive
npm ci --ignore-scripts
npm run build
npm run test:pi
node src/agent-core/dist/server.js
```

SDK 通过 npm workspace 解析到本目录中的 `pi/packages/coding-agent/dist/index.js`。只链接 `packages/coding-agent`，其他 Pi 包仍使用 npm 发布版本。源码修改后重新构建；不要在 Pi 子模块根目录执行全仓安装／构建。依赖与生成产物不进入版本控制。

Pi 修改先提交到 `chensunlai/pi-navcraft` 的 `navcraft` 分支，再在 NavCraft 提交子模块指针。子模块更新默认检出固定提交；开发前在子模块内切换到 `navcraft` 分支。其 `upstream` 指向官方 `earendil-works/pi`，用于手动合并上游更新。

## 边界

- `session.ts`：模型协议、Pi 会话、原生工具、实际请求和事件。
- `server.ts`：双向 JSONL、外部工具注册与回调、工具批次的停止控制。
- `sandbox.ts`：原生 `read/write/edit/bash`，按调用方提供的读写目录隔离命令与文件访问。
- `budget.ts`：发送前持久计数；单 episode 限额包含普通推理、失败请求和摘要。
- `compaction.ts`：复用 Pi 压缩，保存摘要调用并恢复调用方提供的公开内容。
- `trace.ts`：通用文件版本和脱敏事件；导航应用使用流式记录，由 Python 写唯一 audit 文件。

Core 不加载 Habitat、导航任务、地图格式或 Python 工具参数表。模型凭据以显式 `auth` 传入可信运行层，不交给 Coding 命令。默认关闭用户配置和资源的隐式发现。

## 通信

stdin/stdout 每行一个 JSON，日志走 stderr。命令携带唯一 `id`、`op` 和 `args`，响应为 `{id,ok,data}` 或错误。`prompt` 立即确认，执行期间仍能接收工具回执。

| 命令 | 内容 |
|---|---|
| `init` | 显式模型配置、system prompt、公开 workspace、权限、外部工具 JSON Schema、请求限制、可选 checkpoint |
| `prompt` | 文字、图片相对路径及时间限制 |
| `tool_reply` | 调用 ID、文字／图片路径、错误状态、公开 checkpoint、通用停止标志 |
| `configure` | 调整运行中的 compaction 配置 |
| `set_checkpoint` | 更新调用方保存的公开内容，不触发模型推理 |
| `abort` / `close` | 异常取消／释放会话 |

Core 输出 `tool_request`、带序号的 `event` 和 `settled`。外部工具参数与执行代码由调用方定义；同名注册和原生工具重名被拒绝。图片必须通过同一公开路径权限检查。

`stop_remaining_tools` 为同批剩余调用产生未执行回执；`end_run` 在回执配对完成后结束。正常任务完成不使用 abort。外部执行结果不明时不自动重发工具。

checkpoint 只有 `content/images/metadata/revision`；Core 不解释定位与导航状态。平时增量追加模型历史，不每轮重新组合输入。导航摘要提示直接维护在 [Pi compaction 源码](pi/packages/coding-agent/src/core/compaction/compaction.ts)，外围不注入提示。

服务实际返回的用量、缓存和延迟写入事件；缺失项保留缺失。价格未配置，SDK 的零成本不能解释为真实免费。
