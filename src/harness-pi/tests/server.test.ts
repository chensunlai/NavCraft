import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { createInterface } from "node:readline";
import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import { runtime, type RuntimeOptions } from "../runtime.js";
import { serve } from "../server.js";

function message(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"]): AssistantMessage {
  return { role: "assistant", content, stopReason, model: "fixture", provider: "navcraft", api: "openai-responses", timestamp: Date.now(), usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}

async function setup(outputs: AssistantMessage[]) {
  const dir = await mkdtemp(join(tmpdir(), "navcraft-core-server-"));
  const workspace = join(dir, "workspace");
  for (const folder of ["scene", "observations", "memory", "scratch"]) await mkdir(join(workspace, folder), { recursive: true });
  const credentialFile = join(dir, "credentials");
  await writeFile(credentialFile, "NAVCRAFT_TEST_BASE_URL=http://fixture.invalid\nNAVCRAFT_TEST_MODEL=fixture\nNAVCRAFT_TEST_API_KEY=fixture-only-secret\n");
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
    result.session.agent.streamFunction = (_model, context) => {
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
    send("init", "init", { workspace, directory: dir, budgetFile: join(dir, "budget.json"), maxRequests: 4, totalLimit: null, credentialFile, systemPrompt: "Fixture autonomous tool agent", coding: false, compact: false, tools: [{ name: "inspect_local", description: "Inspect a local region", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"], additionalProperties: false } }], ...extra });
    const result = await wait((value) => value.id === "init"); assert.equal(result.ok, true, result.error);
  };
  return { dir, workspace, records, inputs, send, wait, initialize, server, produced: () => produced,
    close: async () => { await server.close(); lines.close(); input.end(); output.end(); await rm(dir, { recursive: true, force: true }); } };
}

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

test("abort is accepted during an outstanding callback, with no tool retry", async () => {
  const f = await setup([message([{ type: "toolCall", id: "slow", name: "inspect_local", arguments: { label: "slow" } }], "toolUse")]);
  try {
    await f.initialize(); f.send("prompt", "prompt", { text: "Inspect slowly" });
    await f.wait((record) => record.type === "tool_request");
    f.send("abort", "abort");
    assert.equal((await f.wait((record) => record.id === "abort")).ok, true);
    await f.wait((record) => record.type === "settled");
    assert.equal(f.produced(), 1);
    assert.equal(f.records.filter((record) => record.type === "tool_request").length, 1);
  } finally { await f.close(); }
});
