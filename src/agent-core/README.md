# Pi Agent Core

Python owns the navigation application. This process owns one Pi session, the model/tool loop, native Coding tools, request accounting, and context projection. Navigation tool definitions and execution come from Python over JSONL.

From the repository root:

```bash
npm ci --ignore-scripts
npm run build
node src/agent-core/dist/server.js
```

The npm workspace builds the local `pi/packages/coding-agent` SDK. Other Pi packages use pinned npm dependencies; the full upstream monorepo is not needed to run this application.

## Boundary and protocol

Commands use `{id, op, args}` and receive `{id, ok, data}`. Logs go to stderr. `prompt` acknowledges immediately while tool callbacks and events continue on stdout.

| Command | Purpose |
| --- | --- |
| `init` | Model/auth, workspace, Python executable, tool definitions, permission policy, limits, optional saved session/history |
| `prompt` | User text, optional image paths, effective time limit |
| `tool_reply` | Match `call_id`; return text/image references and host stop controls |
| `configure` | Update the cumulative request ceiling or compaction settings |
| `export_state` | Export paired history and the native Pi session path at an idle boundary |
| `abort`, `close` | Cancel the current prompt or release the session |

The Core emits `tool_request`, `event`, and `settled`. Tools execute serially. `end_run` completes remaining call/result pairs before returning, and leaves the session available for another prompt. Duplicate command IDs do not repeat work. A tool whose execution is uncertain is never automatically retried.

The application mounts `scene`, `observations`, and `memory` read-only, and `scratch` writable. Coding commands use bwrap without network or service credentials. A public `scene/navlib.py` is importable through the sandbox's `PYTHONPATH`.

## Context

Defaults: 32,768 tokens and a 24,576-token compaction threshold. History is appended normally; there is no per-turn rolling image-pruning policy. Transport omits receipts paired with failed/aborted assistant messages, while saved sessions and audit records retain the original content.

Compaction retains two complete rounds and calls Pi's native summarizer once for the older history. The navigation summary prompt lives in the local Pi source. It aims for at most 1,000 visible tokens and accepts up to 2,000, preserving the requested reasoning effort and an 8,192-token total output allowance. `python -m src.harness.tokenizer` counts that text with the fixed local tokenizer on CPU. Only summaries above 2,000 tokens get one shortening call; invalid summaries cancel compaction and preserve history. After failure, automatic threshold compaction waits for two more completed turns; manual and context-overflow compaction bypass that cooldown. This limits repeated summary attempts but does not guarantee that new observations fit within the remaining context.

Compaction adds no separate facts, task, or image snapshot. The model retrieves the current request and its reference images through the Python `task_read` tool when needed. Its receipt enters the ordinary tool history; later requests append to that history until the next pruning or compaction boundary. The Core owns no duplicate task state.

## Records and limits

Actual request inputs, model responses, tool receipts, and file snapshots are emitted as events. Repeated messages and image bytes use content hashes under `private/blobs`; Python can write the single audit stream by setting `tracePersist: false`. Pi additionally keeps its native JSONL session.

`requests.json` counts every HTTP attempt. `used` includes failed attempts; `charged` excludes recoverable service errors. `maxRequests` applies to charged requests and includes native summaries. Service outages retry the unchanged model request with cancellable backoff and do not consume the effective prompt deadline. Invalid requests and tool execution are not retried.

`time_budget` means the effective prompt time was exhausted. `tool_timeout` and transport failures remain infrastructure errors. Python controls navigation request state; the Core keeps the sole model history. The native continuation support remains in the Pi fork, but the application currently uses repeated `prompt` calls.
