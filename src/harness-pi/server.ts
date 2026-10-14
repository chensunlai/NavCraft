import { createInterface } from "node:readline";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Readable, Writable } from "node:stream";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { runtime, promptBounded, type RuntimeOptions } from "./runtime.js";
import { publicPath } from "./sandbox.js";

export interface PublicCheckpoint {
  content: string | TextContent[];
  images?: string[];
  metadata?: unknown;
  revision: string | number;
}
export interface RemoteToolDefinition {
  name: string;
  description: string;
  parameters: ToolDefinition["parameters"];
}
export interface RemoteToolResult {
  content: TextContent[];
  images?: string[];
  details?: unknown;
  isError?: boolean;
}
export interface ToolReply {
  call_id: string;
  result: RemoteToolResult;
  checkpoint?: PublicCheckpoint;
  control?: { stop_remaining_tools?: boolean; end_run?: boolean };
}

interface Request { id: string; op: string; args?: Record<string, any>; }
type ActiveRuntime = Awaited<ReturnType<typeof runtime>>;
interface PendingTool {
  accept: (result: any) => void;
  reject: (error: Error) => void;
  cleanup: () => void;
  replying: boolean;
}

async function images(workspace: string, refs: string[] = []): Promise<ImageContent[]> {
  return Promise.all(refs.map(async (ref) => {
    const path = await publicPath(workspace, ref);
    return { type: "image" as const, data: (await readFile(path)).toString("base64"), mimeType: /\.jpe?g$/i.test(path) ? "image/jpeg" : /\.webp$/i.test(path) ? "image/webp" : "image/png" };
  }));
}

/** Python owns the application. This process owns only a Pi session and callbacks. */
export function serve(input: Readable = process.stdin, output: Writable = process.stdout, createRuntime = runtime) {
  let active: ActiveRuntime | undefined;
  let workspace = "";
  let checkpoint: PublicCheckpoint | undefined;
  let running: Promise<void> | undefined;
  let initializing = false;
  let closed = false;
  let endRun = false;
  let stopRemainingTools = false;
  let failureReason: string | undefined;
  let toolTimeoutSeconds = 120;
  let seq = 0;
  const pending = new Map<string, PendingTool>();
  const responses = new Map<string, { request: string; promise: Promise<unknown> }>();
  const reader = createInterface({ input, crlfDelay: Infinity });
  const send = (message: unknown) => { if (!output.destroyed) output.write(JSON.stringify(message) + "\n"); };
  const emit = (event: { type: string; data: unknown; timestamp?: string }) => send({ type: "event", seq: ++seq, event });

  async function shutdown() {
    if (closed) return;
    closed = true;
    endRun = true; stopRemainingTools = true; failureReason = "closed";
    for (const [id, call] of pending) {
      call.cleanup(); call.reject(new Error("Core transport closed; external execution may have occurred")); pending.delete(id);
    }
    await active?.session.abort();
    await running?.catch(() => {});
    active?.trace.snapshot("server_close");
    active?.session.dispose();
    reader.close();
  }

  const register = (definitions: RemoteToolDefinition[]): ToolDefinition<any>[] => {
    const names = new Set(["read", "write", "edit", "bash"]);
    return definitions.map((definition) => {
      if (!definition.name || names.has(definition.name)) throw new Error(`Duplicate or reserved tool: ${definition.name}`);
      if (!definition.parameters || (definition.parameters as Record<string, unknown>).type !== "object") throw new Error(`Tool ${definition.name} requires an object JSON schema`);
      names.add(definition.name);
      return {
        name: definition.name, label: definition.name,
        description: definition.description, parameters: definition.parameters,
        async execute(call_id, args, signal) {
          if (pending.has(call_id)) throw new Error("Duplicate pending tool call ID");
          return new Promise((accept, reject) => {
            const remove = () => {
              clearTimeout(timer); signal?.removeEventListener("abort", aborted); pending.delete(call_id);
            };
            const aborted = () => { remove(); reject(new Error("External tool interrupted; execution may have occurred")); };
            const timer = setTimeout(() => {
              remove(); endRun = true; stopRemainingTools = true; failureReason = "tool_timeout";
              reject(new Error("External tool timed out; execution may have occurred. It will not be retried."));
            }, toolTimeoutSeconds * 1000);
            pending.set(call_id, { accept, reject, cleanup: remove, replying: false });
            signal?.addEventListener("abort", aborted, { once: true });
            if (signal?.aborted) { aborted(); return; }
            send({ type: "tool_request", call_id, tool_call_id: call_id, name: definition.name, args });
          });
        },
      };
    });
  };

  async function dispatch(request: Request): Promise<unknown> {
    const args = request.args ?? {};
    if (request.op === "init") {
      if (active || initializing || closed) throw new Error("Core is already initialized or closed");
      initializing = true;
      try {
        workspace = args.workspace;
        checkpoint = args.checkpoint;
        toolTimeoutSeconds = args.toolTimeoutSeconds ?? 120;
        const definitions = register(args.tools ?? []);
        active = await createRuntime({ ...args, tools: definitions, onEvent: emit } as RuntimeOptions);
        const previousBefore = active.session.agent.beforeToolCall;
        active.session.agent.beforeToolCall = async (context, signal) => {
          if (stopRemainingTools || endRun) return { block: true, reason: "Not executed: the host stopped the remaining calls in this batch." };
          return previousBefore?.(context, signal);
        };
        const previousFinish = active.session.agent.finishTurn;
        active.session.agent.finishTurn = async (context, signal) => {
          const decision = await previousFinish?.(context, signal);
          return endRun ? { action: "end" } : decision || undefined;
        };
        active.session.subscribe((event) => { if (event.type === "turn_start") stopRemainingTools = false; });
        return { initialized: true, tools: active.session.getActiveToolNames(), session_id: active.session.sessionManager.getSessionId() };
      } finally { initializing = false; }
    }
    if (!active && request.op !== "close") throw new Error("Initialize the Core first");
    if (request.op === "tool_reply") {
      const reply = args as ToolReply;
      const call = pending.get(reply.call_id);
      if (!call || call.replying) throw new Error("Tool reply does not match an outstanding call");
      call.replying = true;
      try {
        if (!reply.result || !Array.isArray(reply.result.content) || reply.result.content.some((block) => block.type !== "text" || typeof block.text !== "string")) {
          throw new Error("Tool result content must contain text blocks; provide image paths in images");
        }
        const visual = await images(workspace, reply.result.images);
        if (reply.checkpoint !== undefined) checkpoint = reply.checkpoint;
        if (reply.control?.stop_remaining_tools) stopRemainingTools = true;
        if (reply.control?.end_run) { endRun = true; stopRemainingTools = true; }
        call.cleanup();
        call.accept({ content: [...reply.result.content, ...visual], details: reply.result.details, isError: reply.result.isError });
        return { accepted: true, call_id: reply.call_id };
      } catch (error) {
        call.cleanup(); call.reject(error instanceof Error ? error : new Error(String(error))); throw error;
      }
    }
    if (request.op === "set_checkpoint") { checkpoint = args.checkpoint; return { updated: true, revision: checkpoint?.revision }; }
    if (request.op === "configure") {
      if (args.compaction) active!.settingsManager.applyOverrides({ compaction: args.compaction });
      return { configured: true };
    }
    if (request.op === "prompt") {
      if (running || closed) throw new Error("A prompt is already running or the Core is closed");
      const text = args.text ?? args.message;
      if (typeof text !== "string") throw new Error("prompt requires args.text");
      const visual = await images(workspace, args.images);
      endRun = false; stopRemainingTools = false; failureReason = undefined;
      const current = active!;
      // Return immediately. stdin continues receiving tool replies and cancellation.
      running = (async () => {
        let error: string | undefined;
        try { await promptBounded(current.session, text, visual, args.seconds ?? 600); }
        catch (cause) { error = String(cause); }
        const last = current.session.messages.at(-1) as any;
        const reason = failureReason ?? (current.budget.exhausted ? "request_budget" : endRun ? "host_end" : last?.stopReason === "aborted" ? "aborted" : last?.stopReason === "error" ? "model_error" : error ? "runtime_error" : "model_finished");
        current.trace.snapshot("prompt_settled");
        send({ type: "settled", prompt_id: request.id, reason, answer: current.session.getLastAssistantText(), last, requests: current.budget.used, global_requests: current.budget.totalUsed, ...(error ? { error } : {}) });
      })().finally(() => { running = undefined; });
      return { accepted: true };
    }
    if (request.op === "abort") {
      endRun = true; stopRemainingTools = true; failureReason = "aborted";
      await active!.session.abort(); return { aborted: true };
    }
    if (request.op === "close") { await shutdown(); return { closed: true }; }
    throw new Error(`Unknown Core operation: ${request.op}`);
  }

  reader.on("line", (line) => {
    void (async () => {
      let request: Request;
      try {
        request = JSON.parse(line);
        if (typeof request.id !== "string" || typeof request.op !== "string") throw new Error("Commands require string id and op");
      } catch (error) { send({ id: null, ok: false, error: String(error) }); return; }
      const serialized = JSON.stringify(request);
      const existing = responses.get(request.id);
      if (existing && existing.request !== serialized) { send({ id: request.id, ok: false, error: "Command ID reused with different arguments" }); return; }
      const promise = existing?.promise ?? dispatch(request).then((data) => ({ id: request.id, ok: true, data }), (error) => ({ id: request.id, ok: false, error: String(error) }));
      responses.set(request.id, { request: serialized, promise });
      send(await promise);
    })();
  });
  reader.on("close", () => { if (!closed) void shutdown(); });
  return { close: shutdown, get checkpoint() { return checkpoint; } };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.stdout.on("error", () => process.exit(1));
  serve();
}
