import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { createInterface } from "node:readline";
import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import { runtime, type RuntimeOptions } from "../session.js";
import { serve } from "../server.js";

function message(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"]): AssistantMessage {
  return { role: "assistant", content, stopReason, model: "fixture", provider: "navcraft", api: "openai-responses", timestamp: Date.now(), usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}

async function setup(outputs: AssistantMessage[], realTransport = false, appendCheckpoint = false) {
  const dir = await mkdtemp(join(tmpdir(), "navcraft-core-server-"));
  const workspace = join(dir, "workspace");
  for (const folder of ["scene", "observations", "memory", "scratch"]) await mkdir(join(workspace, folder), { recursive: true });
  const input = new PassThrough(); const output = new PassThrough();
  const records: any[] = [];
  const waiting: { test: (value: any) => boolean; accept: (value: any) => void }[] = [];
  const lines = createInterface({ input: output });
  lines.on("line", (line) => {
    const value = JSON.parse(line); records.push(value);
    for (let i = waiting.length - 1; i >= 0; i--) if (waiting[i].test(value)) { const found = waiting.splice(i, 1)[0]; found.accept(value); }
  });
  let produced = 0;
  const inputs: any[] = [];
  const createRuntime = async (options: RuntimeOptions) => {
    const result = await runtime(options);
    if (appendCheckpoint) result.session.subscribe(event => {
      if (event.type === 'message_end' && event.message.role === 'assistant') {
        void result.session.sendCustomMessage({ customType: 'public_checkpoint',
          content: 'Public facts restored', display: false }, { triggerTurn: false });
      }
    });
    if (!realTransport) result.session.agent.streamFunction = (_model, context) => {
      inputs.push(context);
      const stream = createAssistantMessageEventStream();
      stream.end(outputs[produced++] ?? message([{ type: "text", text: "done" }], "stop"));
      return stream;
    };
    return result;
  };
  const server = serve(input, output, createRuntime);
  const send = (id: string, op: string, args: any = {}) => input.write(JSON.stringify({ id, op, args }) + "\n");
  const wait = (predicate: (value: any) => boolean) => {
    const found = records.find(predicate); if (found) return Promise.resolve(found);
    return new Promise<any>((accept, reject) => {
      const timer = setTimeout(() => reject(new Error("fixture Core response timed out")), 5000);
      waiting.push({ test: predicate, accept: (value) => { clearTimeout(timer); accept(value); } });
    });
  };
  const initialize = async (extra: any = {}) => {
    send("init", "init", { workspace, directory: dir, budgetFile: join(dir, "budget.json"), maxRequests: 4, totalLimit: null, auth: { baseUrl: "http://fixture.invalid", model: "fixture", apiKey: "fixture-only-secret" }, systemPrompt: "Fixture autonomous tool agent", coding: false, compact: false, tools: [{ name: "inspect_local", description: "Inspect a local region", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"], additionalProperties: false } }], ...extra });
    const result = await wait((value) => value.id === "init"); assert.equal(result.ok, true, result.error);
  };
  return { dir, workspace, records, inputs, send, wait, initialize, server, produced: () => produced,
    close: async () => { await server.close(); lines.close(); input.end(); output.end(); await rm(dir, { recursive: true, force: true }); } };
}

test('non-thinking students explicitly disable the service chat-template thinking mode', async () => {
  const originalFetch = globalThis.fetch;
  const bodies: any[] = [];
  globalThis.fetch = async (_input, init) => {
    bodies.push(JSON.parse(String(init?.body)));
    const chunks = [
      { id: 'fixture', choices: [{ index: 0, delta: { role: 'assistant', content: 'done' }, finish_reason: null }] },
      { id: 'fixture', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
    ];
    return new Response(chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n',
      { headers: { 'content-type': 'text/event-stream' } });
  };
  const f = await setup([], true);
  try {
    await f.initialize({ api: 'openai-completions', enableThinking: false, thinkingLevel: 'off' });
    f.send('nothink', 'prompt', { text: 'Answer directly' });
    const result = await f.wait(record => record.type === 'settled');
    assert.equal(result.reason, 'model_finished');
    assert.equal(bodies.length, 1);
    assert.deepEqual(bodies[0].chat_template_kwargs, { enable_thinking: false });
    assert.equal(bodies[0].reasoning_effort, undefined);
  } finally { await f.close(); globalThis.fetch = originalFetch; }
});

test("explicit state history imports paired calls and public images into a fresh session", async () => {
  const f = await setup([message([{ type: 'text', text: 'continued' }], 'stop')]);
  try {
    await writeFile(join(f.workspace,'observations','old.png'), Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB5cAAAAASUVORK5CYII=', 'base64'));
    await f.initialize({ initialMessages: [
      {role:'user',content:'Earlier observation',timestamp:0},
      message([{type:'toolCall',id:'prefix',name:'inspect_local',arguments:{label:'old'}}],'toolUse'),
      {role:'toolResult',toolCallId:'prefix',toolName:'inspect_local',content:[{type:'text',text:'real old receipt'},{type:'image',path:'observations/old.png'}],isError:false,timestamp:1},
    ] });
    f.send('state-prompt','prompt',{text:'Now answer'});
    await f.wait(record=>record.type==='settled');
    assert.equal(f.inputs[0].messages[0].content,'Earlier observation');
    const receipt=f.inputs[0].messages.find((m:any)=>m.role==='toolResult');
    assert.equal(receipt.toolCallId,'prefix');
    assert.equal(receipt.content[1].type,'image');
    assert.ok(receipt.content[1].data);
    assert.equal(f.produced(),1);
  } finally { await f.close(); }
});

test("Responses explicitly sends non-strict Python tool schemas without requiring optional alternatives", async () => {
  const originalFetch = globalThis.fetch;
  const payloads: any[] = [];
  globalThis.fetch = async (_input, init) => {
    payloads.push(JSON.parse(String(init?.body)));
    return new Response('data: ' + JSON.stringify({ type: 'response.completed', response: {
      id: 'resp_fixture', status: 'completed', output: [],
      usage: { input_tokens: 1, output_tokens: 0, total_tokens: 1 },
    } }) + '\n\n', { headers: { 'content-type': 'text/event-stream' } });
  };
  const f = await setup([], true);
  const parameters = { type: 'object', properties: {
    target: { type: 'string', enum: ['state'] },
    ids: { type: 'array', items: { type: 'string' } },
    filter: { type: 'object', properties: { located: { type: 'boolean' } }, additionalProperties: false },
  }, required: [], additionalProperties: false };
  try {
    await f.initialize({ tools: [{ name: 'memory_read', description: 'Choose target, IDs or filter', parameters }] });
    f.send('prompt', 'prompt', { text: 'Read the current state' });
    await f.wait(record => record.type === 'settled');
    assert.equal(payloads.length, 1);
    const tool = payloads[0].tools.find((item: any) => item.name === 'memory_read');
    assert.equal(tool.strict, false);
    assert.deepEqual(tool.parameters, parameters);
  } finally { await f.close(); globalThis.fetch = originalFetch; }
});

test("generic tools callback to Python while prompt is pending, then preserve skipped-call pairing", async () => {
  const f = await setup([
    message([{ type: "toolCall", id: "call_1", name: "inspect_local", arguments: { label: "first" } }, { type: "toolCall", id: "call_2", name: "inspect_local", arguments: { label: "second" } }], "toolUse"),
    message([{ type: "text", text: "continued with the new public target" }], "stop"),
  ]);
  try {
    await f.initialize();
    f.send("prompt", "prompt", { text: "Inspect the regions" });
    assert.equal((await f.wait((record) => record.id === "prompt")).data.accepted, true);
    const call = await f.wait((record) => record.type === "tool_request");
    assert.equal(call.call_id, "call_1"); assert.equal(call.name, "inspect_local");
    assert.deepEqual(call.args, { label: "first" });
    f.send("reply", "tool_reply", { call_id: call.call_id, result: { content: [{ type: "text", text: "new public goal" }] }, checkpoint: { content: "new public goal", revision: 1 }, control: { stop_remaining_tools: true } });
    assert.equal((await f.wait((record) => record.id === "reply")).ok, true);
    const settled = await f.wait((record) => record.type === "settled");
    assert.equal(settled.answer, "continued with the new public target");
    assert.equal(f.records.filter((record) => record.type === "tool_request").length, 1);
    assert.equal(f.produced(), 2);
    const paired = f.inputs[1].messages.filter((item: any) => item.role === "toolResult");
    assert.deepEqual(paired.map((item: any) => item.toolCallId), ["call_1", "call_2"]);
    assert.match(paired[1].content[0].text, /Not executed/);
    assert.equal(f.server.checkpoint?.revision, 1);
    assert.ok(f.records.some((record) => record.type === "event" && record.event.type === "turn_start"));
    const seq = f.records.filter((record) => record.type === "event").map((record) => record.seq);
    assert.deepEqual(seq, seq.map((_value, index) => index + 1));
  } finally { await f.close(); }
});

test("a truncated answer is reported as output_limit rather than completed", async () => {
  const f = await setup([message([{ type: "text", text: "unfinished answer" }], "length")]);
  try {
    await f.initialize();
    f.send("prompt", "prompt", { text: "Answer briefly" });
    const settled = await f.wait((record) => record.type === "settled");
    assert.equal(settled.reason, "output_limit");
    assert.equal(settled.answer, "unfinished answer");
  } finally { await f.close(); }
});

test('a deferred checkpoint cannot conceal a truncated assistant response', async () => {
  const f = await setup([message([], 'length')], false, true);
  try {
    await f.initialize();
    f.send('prompt', 'prompt', { text: 'Continue after compression' });
    const result = await f.wait(record => record.type === 'settled');
    assert.ok(f.records.some(record => record.event?.type === 'message_end'
      && record.event.data.message.role === 'custom'));
    assert.equal(result.reason, 'output_limit');
    assert.equal(result.last.stopReason, 'length');
    assert.equal(result.answer, '');
  } finally { await f.close(); }
});

test("successive task prompts retain one session and host-ended tool history", async () => {
  const f = await setup([
    message([{ type: "toolCall", id: "round1_finish", name: "inspect_local", arguments: { label: "round1" } }], "toolUse"),
    message([{ type: "text", text: "second task uses the previous result" }], "stop"),
  ]);
  try {
    await f.initialize();
    const sessionId = f.records.find(record => record.id === "init").data.session_id;
    f.send("first", "prompt", { text: "first current task" });
    const call = await f.wait(record => record.type === "tool_request");
    f.send("finish", "tool_reply", { call_id: call.call_id,
      result: { content: [{ type: "text", text: "round1 completed at its actual position" }] },
      control: { end_run: true } });
    await f.wait(record => record.type === "settled" && record.prompt_id === "first");
    f.send("budget2", "configure", { maxRequests: 8 });
    assert.equal((await f.wait(record => record.id === "budget2")).ok, true);
    assert.ok(f.records.some(record => record.event?.type === "request_limit_configured"
      && record.event.data.maxRequests === 8));
    f.send("second", "prompt", { text: "second current task; same position and memories" });
    const second = await f.wait(record => record.type === "settled" && record.prompt_id === "second");
    assert.equal(second.reason, "model_finished");
    assert.equal(f.records.filter(record => record.event?.type === "runtime_started").length, 1);
    assert.ok(sessionId);
    const messages = f.inputs[1].messages;
    assert.ok(messages.some((m: any) => m.role === "toolResult" && m.toolCallId === "round1_finish"));
    assert.match(JSON.stringify(messages), /first current task/);
    f.send("invalid_budget", "configure", { maxRequests: -1 });
    assert.equal((await f.wait(record => record.id === "invalid_budget")).ok, false);
  } finally { await f.close(); }
});

test("a host-ended prompt cannot return the preceding prompt's answer", async () => {
  const f = await setup([
    message([{ type: "text", text: "answer from the first prompt" }], "stop"),
    message([{ type: "toolCall", id: "second_finish", name: "inspect_local", arguments: { label: "second" } }], "toolUse"),
    message([{ type: "text", text: "third prompt continues the same conversation" }], "stop"),
  ]);
  try {
    await f.initialize();
    f.send("first", "prompt", { text: "Answer the first question" });
    const first = await f.wait(record => record.type === "settled" && record.prompt_id === "first");
    assert.equal(first.answer, "answer from the first prompt");
    f.send("second", "prompt", { text: "Perform a local event" });
    const call = await f.wait(record => record.type === "tool_request" && record.prompt_id === "second");
    f.send("reply", "tool_reply", { call_id: call.call_id,
      result: { content: [{ type: "text", text: "the local event completed" }] }, control: { end_run: true } });
    const second = await f.wait(record => record.type === "settled" && record.prompt_id === "second");
    assert.equal(second.reason, "host_end");
    assert.equal(second.answer, "");
    assert.equal(second.last.role, "toolResult");
    assert.equal(second.message_start, first.message_end);
    assert.equal(second.session_entry_start, first.session_entry_end);
    assert.equal(second.prompt_requests, second.requests - second.request_start);
    f.send("third", "prompt", { text: "Discuss that event" });
    const third = await f.wait(record => record.type === "settled" && record.prompt_id === "third");
    assert.equal(third.answer, "third prompt continues the same conversation");
    assert.equal(third.message_start, second.message_end);
    assert.equal(f.records.filter(record => record.event?.type === "runtime_started").length, 1);
    assert.match(JSON.stringify(f.inputs[2].messages), /answer from the first prompt/);
    assert.ok(f.inputs[2].messages.some((m: any) => m.role === "toolResult" && m.toolCallId === "second_finish"));
    const starts = f.records.filter(record => record.event?.type === "prompt_started");
    const ends = f.records.filter(record => record.event?.type === "prompt_settled");
    assert.deepEqual(starts.map(record => record.prompt_id), ["first", "second", "third"]);
    assert.deepEqual(ends.map(record => record.prompt_id), ["first", "second", "third"]);
  } finally { await f.close(); }
});

test("an empty aborted response is scoped to this prompt instead of the previous answer", async () => {
  const f = await setup([
    message([{ type: "text", text: "previous answer must not be reused" }], "stop"),
    message([], "aborted"),
    message([{ type: "text", text: "a later prompt still runs" }], "stop"),
  ]);
  try {
    await f.initialize();
    f.send("first", "prompt", { text: "First question" });
    await f.wait(record => record.type === "settled" && record.prompt_id === "first");
    f.send("aborted", "prompt", { text: "Cancelled question" });
    const aborted = await f.wait(record => record.type === "settled" && record.prompt_id === "aborted");
    assert.equal(aborted.reason, "aborted");
    assert.equal(aborted.answer, "");
    assert.equal(aborted.last.stopReason, "aborted");
    f.send("third", "prompt", { text: "Continue the discussion" });
    const third = await f.wait(record => record.type === "settled" && record.prompt_id === "third");
    assert.equal(third.answer, "a later prompt still runs");
  } finally { await f.close(); }
});

test("host completion stops at the tool boundary, without another model request", async () => {
  const f = await setup([message([{ type: "toolCall", id: "event", name: "inspect_local", arguments: { label: "event" } }, { type: "toolCall", id: "extra", name: "inspect_local", arguments: { label: "extra" } }], "toolUse")]);
  try {
    await f.initialize(); f.send("prompt", "prompt", { text: "Do one event" });
    const call = await f.wait((record) => record.type === "tool_request");
    f.send("reply", "tool_reply", { call_id: call.call_id, result: { content: [{ type: "text", text: "local event ended" }] }, control: { end_run: true } });
    const settled = await f.wait((record) => record.type === "settled");
    assert.equal(settled.reason, "host_end"); assert.equal(f.produced(), 1);
    const results = f.records.filter((record) => record.type === "event" && record.event.type === "message_end" && record.event.data.message.role === "toolResult").map((record) => record.event.data.message);
    assert.deepEqual(results.map((result) => result.toolCallId), ["event", "extra"]);
    assert.match(results[1].content[0].text, /Not executed/);
    const ledger = JSON.parse(await readFile(join(f.dir, "budget.json"), "utf8"));
    assert.equal(ledger.limit, null); assert.deepEqual(ledger.attempts, []);
  } finally { await f.close(); }
});

test("duplicate commands do not execute tools twice and an invalid reply fails explicitly", async () => {
  const f = await setup([message([{ type: "toolCall", id: "inspect", name: "inspect_local", arguments: { label: "x" } }], "toolUse")]);
  try {
    await f.initialize(); f.send("prompt", "prompt", { text: "Inspect" });
    await f.wait((record) => record.type === "tool_request");
    f.send("bad", "tool_reply", { call_id: "unknown", result: { content: [] } });
    assert.equal((await f.wait((record) => record.id === "bad")).ok, false);
    const reply = { call_id: "inspect", result: { content: [{ type: "text", text: "done" }] }, control: { end_run: true } };
    f.send("reply", "tool_reply", reply); f.send("reply", "tool_reply", reply);
    await f.wait((record) => record.type === "settled");
    assert.equal(f.records.filter((record) => record.type === "tool_request").length, 1);
    f.send("reply", "tool_reply", { ...reply, call_id: "changed" });
    const bad = await f.wait((record) => record.id === "reply" && !record.ok);
    assert.match(bad.error, /reused/);
  } finally { await f.close(); }
});

test("busy prompts are rejected and abort pairs the whole batch before a subsequent prompt", async () => {
  const f = await setup([
    message([{ type: "toolCall", id: "slow", name: "inspect_local", arguments: { label: "slow" } },
      { type: "toolCall", id: "skipped", name: "inspect_local", arguments: { label: "skipped" } }], "toolUse"),
    message([{ type: "text", text: "continued after cancellation" }], "stop"),
  ]);
  try {
    await f.initialize(); f.send("prompt", "prompt", { text: "Inspect slowly" });
    await f.wait((record) => record.type === "tool_request");
    f.send("busy", "prompt", { text: "Do not queue this question" });
    const busy = await f.wait(record => record.id === "busy");
    assert.equal(busy.ok, false); assert.match(busy.error, /already running/);
    f.send("abort", "abort");
    assert.equal((await f.wait((record) => record.id === "abort")).ok, true);
    const aborted = await f.wait((record) => record.type === "settled");
    assert.equal(aborted.reason, "aborted"); assert.equal(aborted.answer, "");
    assert.equal(f.produced(), 1);
    assert.equal(f.records.filter((record) => record.type === "tool_request").length, 1);
    const paired = f.records.filter(record => record.event?.type === "message_end" && record.event.data.message.role === "toolResult");
    assert.deepEqual(paired.map(record => record.event.data.message.toolCallId), ["slow", "skipped"]);
    assert.ok(paired.every(record => record.event.data.message.isError));
    f.send("next", "prompt", { text: "Continue after cancellation" });
    const next = await f.wait(record => record.type === "settled" && record.prompt_id === "next");
    assert.equal(next.reason, "model_finished"); assert.equal(next.answer, "continued after cancellation");
    assert.equal(f.produced(), 2);
    assert.ok(!JSON.stringify(f.inputs[1].messages).includes("Do not queue this question"));
  } finally { await f.close(); }
});

test("Core accepts explicit auth, configurable compaction and stream-only recording without exposing credentials", async () => {
  const f = await setup([message([{ type: "text", text: "done" }], "stop")]);
  try {
    await f.initialize({ tracePersist: false, checkpoint: { content: "public state", revision: "r1" } });
    f.send("configure", "configure", { compaction: { enabled: true, reserveTokens: 65000, keepRecentTokens: 1200 } });
    const configured = await f.wait((record) => record.id === "configure");
    assert.equal(configured.ok, true);
    assert.deepEqual(configured.data.compaction, { enabled: true, reserveTokens: 65000, keepRecentTokens: 1200 });
    assert.ok(f.records.some((record) => record.type === "event" && record.event.type === "runtime_configured"
      && record.event.data.compaction.reserveTokens === 65000));
    assert.equal(f.server.checkpoint?.revision, "r1");
    const saved = await readFile(join(f.dir, "config.json"), "utf8");
    assert.ok(!saved.includes("fixture-only-secret"));
    assert.equal(JSON.parse(saved).auth, "[PRIVATE]");
    const records = JSON.stringify(f.records);
    assert.ok(!records.includes("fixture-only-secret"));
    await assert.rejects(readFile(join(f.dir, "private", "trace.jsonl")), /ENOENT/);
  } finally { await f.close(); }
});

test("prompt timeout stops at the outstanding callback without making another model request", async () => {
  const f = await setup([message([{ type: "toolCall", id: "timeout", name: "inspect_local", arguments: { label: "slow" } }], "toolUse")]);
  try {
    await f.initialize(); f.send("prompt", "prompt", { text: "Inspect", seconds: 0.03 });
    const settled = await f.wait((record) => record.type === "settled");
    assert.equal(settled.reason, "timeout");
    assert.equal(f.produced(), 1);
    assert.equal(f.records.filter((record) => record.type === "tool_request").length, 1);
  } finally { await f.close(); }
});

test("a Python tool error remains an error in the actual Pi message and tool event", async () => {
  const f = await setup([message([{ type: "toolCall", id: "failed", name: "inspect_local", arguments: { label: "failure" } }], "toolUse"), message([{ type: "text", text: "The tool failed; no observation was obtained" }], "stop")]);
  try {
    await f.initialize(); f.send("prompt", "prompt", { text: "Inspect" });
    await f.wait((record) => record.type === "tool_request");
    f.send("reply", "tool_reply", { call_id: "failed", result: { content: [{ type: "text", text: "Python operation failed" }], isError: true } });
    await f.wait((record) => record.type === "settled");
    const result = f.inputs[1].messages.find((item: any) => item.role === "toolResult");
    assert.equal(result.isError, true);
    assert.equal(result.content[0].text, "Python operation failed");
    const event = f.records.find((record) => record.type === "event" && record.event.type === "tool_execution_end");
    assert.equal(event.event.data.isError, true);
  } finally { await f.close(); }
});

test("tool image paths follow the same generic workspace policy as Coding tools", async () => {
  const f = await setup([message([{ type: "toolCall", id: "image", name: "inspect_local", arguments: { label: "image" } }], "toolUse")]);
  try {
    await mkdir(join(f.workspace, "assets")); await mkdir(join(f.workspace, "out"));
    const bytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64");
    await writeFile(join(f.workspace, "assets", "evidence.png"), bytes);
    await f.initialize({ executionPolicy: { readOnly: ["assets"], readWrite: ["out"] } });
    f.send("prompt", "prompt", { text: "Inspect" }); await f.wait((record) => record.type === "tool_request");
    f.send("reply", "tool_reply", { call_id: "image", result: { content: [{ type: "text", text: "saved evidence" }], images: ["assets/evidence.png"] }, control: { end_run: true } });
    await f.wait((record) => record.type === "settled");
    const returned = f.records.find((record) => record.type === "event" && record.event.type === "message_end" && record.event.data.message.role === "toolResult").event.data.message;
    assert.equal(returned.content[1].text, "IMAGE REFERENCE | assets/evidence.png");
    assert.equal(returned.content[2].type, "image");
    const image = returned.content[2].data;
    assert.equal(image.encoding, "base64");
    assert.deepEqual(await readFile(join(f.dir, "private", "blobs", image.$trace_blob)), bytes);
    // Auditing uses references; Pi's actual session still receives the image bytes.
    const sessions = join(f.dir, "private", "pi", "sessions");
    const saved = await readFile(join(sessions, (await readdir(sessions))[0]), "utf8");
    const actual = saved.trim().split("\n").map(line => JSON.parse(line))
      .find(entry => entry.message?.role === "toolResult").message;
    assert.equal(actual.content[2].data, bytes.toString("base64"));
    f.send("outside", "prompt", { text: "Invalid asset", images: ["memory/not-public.png"] });
    const denied = await f.wait((record) => record.id === "outside");
    assert.equal(denied.ok, false); assert.match(denied.error, /readable/);
  } finally { await f.close(); }
});

test("reasoning-only stop is not a completed answer, and a subsequent prompt can use tools", async () => {
  const f = await setup([
    message([{ type: "thinking", thinking: "Continue observing." }], "stop"),
    message([{ type: "toolCall", id: "after_empty", name: "inspect_local", arguments: { label: "resume" } }], "toolUse"),
  ]);
  try {
    await f.initialize();
    f.send("first_empty", "prompt", { text: "Inspect" });
    const first = await f.wait(record => record.type === "settled" && record.prompt_id === "first_empty");
    assert.equal(first.reason, "no_action_output"); assert.equal(first.answer, "");
    f.send("continue_empty", "prompt", { text: "Continue the same task" });
    const call = await f.wait(record => record.type === "tool_request");
    assert.equal(call.call_id, "after_empty");
    f.send("reply_empty", "tool_reply", { call_id: call.call_id,
      result: { content: [{ type: "text", text: "new evidence" }] }, control: { end_run: true } });
    const second = await f.wait(record => record.type === "settled" && record.prompt_id === "continue_empty");
    assert.equal(second.reason, "host_end");
    assert.ok(f.inputs[1].messages.some((m: any) => m.role === "assistant" && m.content[0]?.type === "thinking"));
  } finally { await f.close(); }
});

test("current and target captions directly precede their own images in actual model input", async () => {
  const f = await setup([
    message([{ type: "toolCall", id: "compare", name: "inspect_local", arguments: { label: "images" } }], "toolUse"),
    message([{ type: "text", text: "The current view differs from the target." }], "stop"),
  ]);
  try {
    const bytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64");
    for (const name of ["current.png", "target.png"]) await writeFile(join(f.workspace, "observations", name), bytes);
    await f.initialize(); f.send("compare_prompt", "prompt", { text: "Compare" });
    await f.wait(record => record.type === "tool_request");
    f.send("compare_reply", "tool_reply", { call_id: "compare", result: {
      content: [{ type: "text", text: "goal changed" }], images: ["observations/current.png", "observations/target.png"],
      image_labels: ["CURRENT OBSERVATION | frame_1", "TARGET REFERENCE | goal_2"],
    } });
    await f.wait(record => record.type === "settled");
    const content = (f.inputs[1].messages.find((m: any) => m.role === "toolResult") as any).content;
    assert.deepEqual(content.map((b: any) => b.type), ["text", "text", "image", "text", "image"]);
    assert.equal(content[1].text, "CURRENT OBSERVATION | frame_1");
    assert.equal(content[3].text, "TARGET REFERENCE | goal_2");
  } finally { await f.close(); }
});

test("a late Python receipt after prompt timeout is archived without converting timeout to error", async () => {
  const f = await setup([message([{ type: "toolCall", id: "late", name: "inspect_local", arguments: { label: "slow" } }], "toolUse")]);
  try {
    await f.initialize();
    f.send("prompt", "prompt", { text: "Inspect", seconds: 0.03 });
    await f.wait((record) => record.type === "tool_request");
    assert.equal((await f.wait((record) => record.type === "settled")).reason, "timeout");
    f.send("late_reply", "tool_reply", { call_id: "late", result: { content: [{ type: "text", text: "Actual execution completed before callback delivery" }] }, checkpoint: { content: "do not apply late checkpoint" }, control: { end_run: true } });
    const ack = await f.wait((record) => record.id === "late_reply");
    assert.equal(ack.ok, true);
    assert.equal(ack.data.accepted, false);
    assert.equal(ack.data.reason, "timeout");
    assert.equal(f.produced(), 1);
    assert.ok(f.records.some((record) => record.type === "event" && record.event.type === "late_tool_reply"));
    assert.notEqual(f.server.checkpoint?.content, "do not apply late checkpoint");
    f.send("unknown", "tool_reply", { call_id: "never_requested", result: { content: [] } });
    assert.equal((await f.wait((record) => record.id === "unknown")).ok, false);
    f.send("next", "prompt", { text: "Continue discussion" });
    assert.equal((await f.wait((record) => record.type === "settled" && record.prompt_id === "next")).reason, "model_finished");
  } finally { await f.close(); }
});

test('service outage time does not expire the Core prompt deadline', async () => {
  const f = await setup([message([{ type: 'toolCall', id: 'after_outage', name: 'inspect_local', arguments: { label: 'resume' } }], 'toolUse')]);
  try {
    await f.initialize();
    const started = Date.now();
    await writeFile(join(f.dir, 'service-retry.json'), JSON.stringify({ completed_ms: 0, started_ms: started }));
    f.send('prompt', 'prompt', { text: 'continue', seconds: .05 });
    const call = await f.wait(r => r.type === 'tool_request');
    await new Promise(resolve => setTimeout(resolve, 140));
    assert.equal(f.records.some(r => r.type === 'settled'), false);
    await writeFile(join(f.dir, 'service-retry.json'), JSON.stringify({ completed_ms: Date.now() - started, started_ms: null }));
    f.send('reply', 'tool_reply', { call_id: call.call_id, result: { content: [{ type: 'text', text: 'done' }] }, control: { end_run: true } });
    const settled = await f.wait(r => r.type === 'settled');
    assert.equal(settled.reason, 'host_end');
  } finally { await f.close(); }
});

test('failed assistant error remains visible after closing unexecuted tool receipts', async () => {
  const failed = message([{ type: 'toolCall', id: 'not_executed', name: 'inspect_local', arguments: { label: 'partial' } }], 'error');
  failed.errorMessage = 'Upstream response stream was interrupted';
  const f = await setup([failed]);
  try {
    await f.initialize();
    f.send('prompt', 'prompt', { text: 'continue' });
    const settled = await f.wait(r => r.type === 'settled');
    assert.equal(settled.reason, 'model_error');
    assert.equal(settled.last.errorMessage, failed.errorMessage);
    assert.equal(settled.last.stopReason, 'error');
    assert.equal(f.records.filter(r => r.type === 'tool_request').length, 0);
    assert.ok(f.records.some(r => r.type === 'event' && r.event.type === 'tool_call_cancelled'));
  } finally { await f.close(); }
});
