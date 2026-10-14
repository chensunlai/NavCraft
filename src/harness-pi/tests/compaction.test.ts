import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { AgentMessage, StreamFn } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, type AssistantMessage, type Model, type TranscriptContext, type Usage } from "@earendil-works/pi-ai";
import { compact, findCutPoint, type ExtensionAPI, type ExtensionContext, type SessionBeforeCompactEvent, type SessionEntry, type SessionCompactEvent } from "@earendil-works/pi-coding-agent";
import { createNavigationCompaction, type NavigationPublicState } from "../compaction.js";

const usage: Usage = { input: 20, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 30, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const model: Model<"openai-responses"> = { id: "fixture", name: "fixture", api: "openai-responses", provider: "fixture", baseUrl: "http://fixture.invalid", reasoning: true, input: ["text", "image"], contextWindow: 65536, maxTokens: 8192, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
function assistant(content: AssistantMessage["content"] = [{ type: "text", text: "fixture checkpoint" }], stopReason: AssistantMessage["stopReason"] = "stop"): AssistantMessage {
  return { role: "assistant", content, stopReason, usage, model: model.id, api: model.api, provider: model.provider, timestamp: Date.now(), ...(stopReason === "error" ? { errorMessage: "fixture transport terminated" } : {}) };
}
function preparation(split = false): SessionBeforeCompactEvent["preparation"] {
  return {
    firstKeptEntryId: "kept-tool-call", tokensBefore: 43000, isSplitTurn: split,
    messagesToSummarize: [{ role: "user", content: [{ type: "text", text: "Navigate to the door; unsure whether this is D03" }], timestamp: 1 }],
    turnPrefixMessages: split ? [{ role: "user", content: [{ type: "text", text: "Continue the current door approach" }], timestamp: 2 }] : [],
    fileOps: { read: new Set(["scene/room.md"]), written: new Set(["memory/state.json"]), edited: new Set() },
    settings: { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 },
  };
}

function summaryPrompt(context: TranscriptContext): string {
  return context.messages.filter((message) => message.role === "user").map((message) => {
    if (typeof message.content === "string") return message.content;
    return message.content.filter((block) => block.type === "text").map((block) => block.text).join("\n");
  }).join("\n");
}

function checkCoreNavigationPrompt(context: TranscriptContext): string {
  const text = summaryPrompt(context);
  assert.equal(text.split("Navigation checkpoint requirements:").length - 1, 1);
  assert.match(text, /last observation in summarized segment/);
  assert.match(text, /not the current state of the whole run/);
  assert.match(text, /Later retained messages and public checkpoints take precedence/);
  // Core adds instructions to its existing prompt; no extra user message is needed.
  assert.equal(context.messages.filter((message) => message.role === "user").length, 1);
  return text;
}

async function directCore(prep: SessionBeforeCompactEvent["preparation"], instructions?: string) {
  const calls: { context: TranscriptContext; options: any }[] = [];
  const stream: StreamFn = (_model, context, options) => {
    calls.push({ context, options });
    const result = createAssistantMessageEventStream(); result.end(assistant()); return result;
  };
  const result = await compact(prep, model, "fixture-key", undefined, instructions, undefined, "high", stream, undefined, { enabled: false, maxRetries: 0, baseDelayMs: 0 }, undefined, "fixture-core-session");
  return { result, calls };
}

test("direct Pi core ordinary compaction adds navigation requirements without an outer extension", async () => {
  const prep = preparation();
  const { result, calls } = await directCore(prep);
  assert.equal(calls.length, 1);
  const text = checkCoreNavigationPrompt(calls[0].context);
  assert.match(text, /<conversation>/);
  assert.match(text, /Use this EXACT format/);
  assert.match(text, /## Goal/);
  assert.match(text, /## Constraints & Preferences/);
  assert.equal(result.firstKeptEntryId, prep.firstKeptEntryId);
  assert.equal(result.tokensBefore, prep.tokensBefore);
  assert.deepEqual(result.usage, usage);
  assert.deepEqual(result.details, { readFiles: ["scene/room.md"], modifiedFiles: ["memory/state.json"] });
  assert.equal(prep.previousSummary, undefined);
  assert.deepEqual([...prep.fileOps.read], ["scene/room.md"]);
  assert.deepEqual([...prep.fileOps.written], ["memory/state.json"]);
  assert.equal(calls[0].options.reasoning, "high");
});

test("direct Pi core updated summary retains the previous summary and update format", async () => {
  const prep = preparation(); prep.previousSummary = "prior checkpoint: pose estimate at frame_000003 is uncertain";
  const { result, calls } = await directCore(prep);
  assert.equal(calls.length, 1);
  const text = checkCoreNavigationPrompt(calls[0].context);
  assert.match(text, /<previous-summary>/);
  assert.ok(text.includes(prep.previousSummary));
  assert.match(text, /Update the existing structured summary/);
  assert.match(text, /PRESERVE exact file paths/);
  assert.equal(result.firstKeptEntryId, prep.firstKeptEntryId);
  assert.deepEqual(result.usage, usage);
});

test("direct Pi core split history and turn prefix both use the patched common prompt builder", async () => {
  const prep = preparation(true);
  const { result, calls } = await directCore(prep);
  assert.equal(calls.length, 2);
  const [history, prefix] = calls.map((call) => checkCoreNavigationPrompt(call.context));
  assert.match(history, /Use this EXACT format/);
  assert.match(prefix, /## Original Request/);
  assert.match(prefix, /## Progress So Far/);
  assert.match(prefix, /Only summarize information explicitly present above/);
  assert.match(result.summary, /Turn Context \(split turn\)/);
  assert.equal(result.firstKeptEntryId, prep.firstKeptEntryId);
  assert.equal(result.usage?.totalTokens, 60);
  assert.equal(result.usage?.input, 40);
  assert.deepEqual(result.details, { readFiles: ["scene/room.md"], modifiedFiles: ["memory/state.json"] });
});

test("direct Pi core prefix-only compaction preserves prior history with a single request", async () => {
  const prep = preparation(true); prep.messagesToSummarize = []; prep.previousSummary = "unchanged prior checkpoint";
  const { result, calls } = await directCore(prep);
  assert.equal(calls.length, 1);
  const text = checkCoreNavigationPrompt(calls[0].context);
  assert.match(text, /## Original Request/);
  assert.ok(result.summary.startsWith(prep.previousSummary));
  assert.equal(result.firstKeptEntryId, prep.firstKeptEntryId);
  assert.deepEqual(result.usage, usage);
});

test("direct Pi core retains caller focus alongside exactly one navigation requirement block", async () => {
  const prep = preparation();
  const focus = "Keep the unresolved cabinet identity and evidence frame_000047 verbatim.";
  const { calls, result } = await directCore(prep, focus);
  const text = checkCoreNavigationPrompt(calls[0].context);
  assert.ok(text.includes(`Additional focus: ${focus}`));
  assert.equal(result.firstKeptEntryId, prep.firstKeptEntryId);
  assert.equal(calls[0].options.sessionId, "fixture-core-session");
});

async function fixture(options: { response?: AssistantMessage; messages?: AgentMessage[]; state?: NavigationPublicState; workspace?: string } = {}) {
  const calls: { context: TranscriptContext; options: any }[] = [];
  const handlers = new Map<string, (event: any, ctx: ExtensionContext) => Promise<any>>();
  const sent: { message: any; options: any }[] = [];
  const traced: { type: string; data: any }[] = [];
  const stream: StreamFn = (_model, context, options_) => {
    calls.push({ context, options: options_ });
    const eventStream = createAssistantMessageEventStream();
    eventStream.end(options.response ?? assistant());
    return eventStream;
  };
  const pi = {
    on: (name: string, handler: (event: any, ctx: ExtensionContext) => Promise<any>) => { handlers.set(name, handler); },
    sendMessage: (message: any, options_: any) => { sent.push({ message, options: options_ }); },
  } as unknown as ExtensionAPI;
  const ctx = {
    cwd: options.workspace ?? "/tmp",
    model, thinkingLevel: "high",
    modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "fixture-key" }) },
    sessionManager: { getSessionId: () => "fixture-session", buildSessionProjection: () => ({ messages: options.messages ?? [] }) },
  } as unknown as ExtensionContext;
  await createNavigationCompaction(stream, () => options.state ?? { task: { goal_id: 1, text: "door" }, latestObservation: { id: "frame_7", step: 4 } }, { emit: (type, data) => traced.push({ type, data }) })(pi);
  return {
    calls, sent, traced,
    before: (prep = preparation()) => handlers.get("session_before_compact")!({ type: "session_before_compact", preparation: prep, branchEntries: [], reason: "manual", willRetry: false, signal: new AbortController().signal }, ctx),
    after: () => handlers.get("session_compact")!({ type: "session_compact", fromExtension: true, compactionEntry: { id: "compacted" } } as SessionCompactEvent, ctx),
    context: (messages: AgentMessage[]) => handlers.get("context")!({ type: "context", messages }, ctx),
  };
}

test("ordinary native summary keeps cut point, file tracking and provider usage", async () => {
  const f = await fixture();
  const result = await f.before();
  assert.equal(f.calls.length, 1);
  assert.equal(result.compaction.firstKeptEntryId, "kept-tool-call");
  assert.equal(result.compaction.tokensBefore, 43000);
  assert.deepEqual(result.compaction.usage, usage);
  assert.deepEqual(result.compaction.details.readFiles, ["scene/room.md"]);
  assert.deepEqual(result.compaction.details.modifiedFiles, ["memory/state.json"]);
  checkCoreNavigationPrompt(f.calls[0].context);
  assert.equal(f.calls[0].options.maxRetries, 0);
  assert.equal(f.calls[0].options.reasoning, "high");
});

test("history and split-turn summary BOTH receive navigation instructions and shared stream", async () => {
  const f = await fixture();
  const result = await f.before(preparation(true));
  assert.equal(f.calls.length, 2);
  for (const call of f.calls) {
    checkCoreNavigationPrompt(call.context);
    assert.equal(call.options.maxRetries, 0);
    assert.equal(call.options.reasoning, "high");
  }
  assert.match(result.compaction.summary, /Turn Context \(split turn\)/);
  assert.equal(result.compaction.usage.totalTokens, 60);
  assert.equal(result.compaction.firstKeptEntryId, "kept-tool-call");
});

test("Pi cut points retain tool-call/result pairing even for a large trailing tool result", () => {
  const entries: SessionEntry[] = [
    { type: "message", id: "old-user", parentId: null, timestamp: "2026-10-14", message: { role: "user", content: "old context ".repeat(1000), timestamp: 1 } },
    { type: "message", id: "kept-tool-call", parentId: "old-user", timestamp: "2026-10-14", message: assistant([{ type: "toolCall", id: "call_1", name: "nav_observe", arguments: { mode: "single" } }], "toolUse") },
    { type: "message", id: "paired-result", parentId: "kept-tool-call", timestamp: "2026-10-14", message: { role: "toolResult", toolCallId: "call_1", toolName: "nav_observe", content: [{ type: "text", text: "fresh frame ".repeat(1500) }], isError: false, timestamp: 3 } },
  ];
  const cut = findCutPoint(entries, 0, entries.length, 100);
  assert.equal(entries[cut.firstKeptEntryIndex].id, "kept-tool-call");
  const kept = entries.slice(cut.firstKeptEntryIndex);
  assert.equal(kept[0].type, "message");
  assert.equal((kept[0] as any).message.content[0].id, (kept[1] as any).message.toolCallId);
});

for (const reason of ["error", "length", "aborted", "deferred", "pending"] as const) {
  test(`summary ${reason} explicitly cancels without a default stream fallback`, async () => {
    const f = await fixture({ response: assistant([{ type: "text", text: "incomplete" }], reason) });
    assert.deepEqual(await f.before(), { cancel: true });
    assert.equal(f.calls.length, 1);
    assert.ok(f.traced.some((entry) => entry.type === "navigation_compaction_cancelled"));
    assert.equal(f.sent.length, 0);
  });
}

test("summary tool calls and empty summaries are rejected", async () => {
  for (const response of [assistant([{ type: "toolCall", id: "new", name: "nav_step", arguments: { actions: ["move_forward"] } }], "toolUse"), assistant([])]) {
    const f = await fixture({ response });
    assert.deepEqual(await f.before(), { cancel: true });
  }
});

test("missing public records and actual images are restored without actions or memory writes", async () => {
  const dir = await mkdtemp(join(tmpdir(), "navcraft-compact-"));
  try {
    await mkdir(join(dir, "observations")); await mkdir(join(dir, "memory"));
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64");
    await writeFile(join(dir, "observations", "frame_7.png"), png);
    await writeFile(join(dir, "memory", "state.json"), '{"pose_estimate":null}');
    const state = { task: { goal_id: 2, text: "bed", image: "observations/frame_7.png" }, latestObservation: { id: "frame_7", path: "observations/frame_7.png", step: 4 } };
    const f = await fixture({ state, workspace: dir, messages: [{ role: "compactionSummary", summary: "Old context", tokensBefore: 4000, timestamp: 5 }] });
    await f.after();
    assert.equal(f.sent.length, 1);
    assert.equal(f.sent[0].options.triggerTurn, false);
    assert.deepEqual(f.sent[0].message.content.filter((block: any) => block.type === "image").map((block: any) => block.data), [png.toString("base64")]);
    const checkpoint = JSON.parse(f.sent[0].message.content[0].text);
    assert.deepEqual(checkpoint.task, state.task);
    assert.deepEqual(checkpoint.latestObservation, state.latestObservation);
    assert.equal(checkpoint.pose_estimate, undefined);
    assert.equal(await readFile(join(dir, "memory", "state.json"), "utf8"), '{"pose_estimate":null}');
    assert.equal(f.calls.length, 0);
    const nextInput = await f.context([]);
    assert.equal(nextInput.messages.length, 1);
    assert.deepEqual(nextInput.messages[0].content, f.sent[0].message.content);
    assert.equal(await f.context(nextInput.messages), undefined);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("retained records and images are not duplicated", async () => {
  const state = { task: { goal_id: 1, text: "door" }, latestObservation: { id: "frame_7", step: 4 } };
  const f = await fixture({ state, messages: [{ role: "user", content: JSON.stringify(state), timestamp: 6 }] });
  await f.after();
  assert.equal(f.sent.length, 0);
  assert.ok(f.traced.some((entry) => entry.type === "navigation_checkpoint_retained"));
});

test("post-compaction image restoration rejects external paths and keeps public input explicit", async () => {
  const dir = await mkdtemp(join(tmpdir(), "navcraft-compact-path-"));
  try {
    const state = { task: { goal_id: 1 }, latestObservation: { id: "frame_8", path: "../private.png" } };
    const f = await fixture({ state, workspace: dir });
    await f.after();
    assert.equal(f.sent[0].message.content.filter((block: any) => block.type === "image").length, 0);
    assert.ok(f.traced.some((entry) => entry.type === "navigation_checkpoint_image_error"));
    assert.equal(f.calls.length, 0);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("restored maps keep their original pose frame and versions rather than acquiring current metadata", async () => {
  const dir = await mkdtemp(join(tmpdir(), "navcraft-compact-map-"));
  try {
    await mkdir(join(dir, "scratch")); await mkdir(join(dir, "memory"));
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64");
    await writeFile(join(dir, "scratch", "initial-map.png"), png);
    const current = '{"pose_estimate":{"at_frame":"frame_000047","position":[0.647,0.355]}}';
    await writeFile(join(dir, "memory", "state.json"), current);
    const mapMetadata = {
      map_id: "map_L1", pose_at_frame: "frame_000000",
      versions: { "memory/state.json": "old-state-hash", "memory/maps/L1.entities.json": "old-entities-hash" },
    };
    const state = { task: { goal_id: 1 }, latestObservation: { id: "frame_000102", step: 84 }, map_ref: "scratch/initial-map.png", map_metadata: mapMetadata };
    const before = JSON.stringify(state);
    const f = await fixture({ state, workspace: dir });
    await f.after();
    const checkpoint = JSON.parse(f.sent[0].message.content[0].text);
    assert.equal(checkpoint.map_ref, state.map_ref);
    assert.deepEqual(checkpoint.map_metadata, mapMetadata);
    assert.equal(checkpoint.map_metadata.pose_at_frame, "frame_000000");
    assert.equal(checkpoint.latestObservation.id, "frame_000102");
    assert.match(checkpoint.note, /pose_at_frame may be old and is not live localization/);
    assert.equal(JSON.stringify(state), before);
    assert.equal(await readFile(join(dir, "memory", "state.json"), "utf8"), current);
    assert.equal(f.calls.length, 0);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
