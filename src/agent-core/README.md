# Pi Agent Core

Python owns the navigation application. This process owns one Pi session, the model/tool loop, native Coding tools, request accounting, and context projection. Navigation tool definitions and execution come from Python over JSONL.

From the repository root:

```bash
npm ci --ignore-scripts
npm run build
npm run test:pi
node src/agent-core/dist/server.js
```

The npm workspace builds the local `pi/packages/coding-agent` SDK. Other Pi packages use pinned npm dependencies; the full upstream monorepo is not needed to run this application.

## Boundary and protocol

Commands use `{id, op, args}` and receive `{id, ok, data}`. Logs go to stderr. `prompt` acknowledges immediately while tool callbacks and events continue on stdout.

| Command | Purpose |
| --- | --- |
| `init` | Model/auth, workspace, Python executable, tool definitions, permission policy, limits, optional saved session/history |
| `prompt` | User text, optional image paths, effective time limit |
| `tool_reply` | Match `call_id`; return text/image references, optional public checkpoint and host stop controls |
| `set_checkpoint` | Replace already-public facts used after compaction |
| `configure` | Update the cumulative request ceiling or compaction settings |
| `abort`, `close` | Cancel the current prompt or release the session |

The Core emits `tool_request`, `event`, and `settled`. Tools execute serially. `end_run` completes remaining call/result pairs before returning, and leaves the session available for another prompt. Duplicate command IDs do not repeat work. A tool whose execution is uncertain is never automatically retried.

The application mounts `scene`, `observations`, and `memory` read-only, and `scratch` writable. Coding commands use bwrap without network or service credentials. A public `scene/navlib.py` is importable through the sandbox's `PYTHONPATH`.

## Context

Defaults: 32,768 tokens, approximately 24,000-token compaction threshold. Before each model request, images outside the latest three complete assistant/tool rounds become reference text, while the latest current image remains. Saved sessions and image files retain their original content.

Compaction retains two complete rounds and calls Pi's native summarizer once for the older history. The navigation summary prompt lives in the local Pi source. It preserves the requested reasoning effort and an 8,192-token total output allowance; only the visible summary is limited to 300 tokens. `python -m src.harness.tokenizer` counts that text with the fixed local tokenizer on CPU. An overlong summary gets at most one native shortening call; invalid summaries cancel compaction rather than replacing history with partial text.

Python supplies the factual head through `checkpoint`; the Core never invents pose, entities, or success. After compression, its request projection contains that head, Pi's summary, and the retained rounds. A current frame and checkpoint content retain their recorded times.

## Records and limits

Actual request inputs, model responses, tool receipts, and file snapshots are emitted as events. Repeated messages and image bytes use content hashes under `private/blobs`; Python can write the single audit stream by setting `tracePersist: false`. Pi additionally keeps its native JSONL session.

`requests.json` counts every HTTP attempt. `used` includes failed attempts; `charged` excludes recoverable service errors. `maxRequests` applies to charged requests and includes native summaries. Service outages retry the unchanged model request with cancellable backoff and do not consume the effective prompt deadline. Invalid requests and tool execution are not retried.
