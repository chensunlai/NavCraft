import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, mkdirSync, readFileSync, existsSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Trace } from "../trace.js";
import { createHash } from "node:crypto";

test("stream-only trace redacts secrets and keeps replay blobs for arbitrary writable paths", () => {
  const dir = mkdtempSync(join(tmpdir(), "navcraft-stream-trace-"));
  try {
    const workspace = join(dir, "workspace"); mkdirSync(join(workspace, "results"), { recursive: true });
    writeFileSync(join(workspace, "results", "state.txt"), "first");
    const events: any[] = [];
    const trace = new Trace(join(dir, "private"), workspace, ["test-secret"], (event) => events.push(event), false, ["results"]);
    trace.emit("model_input", { value: "contains test-secret" });
    const versions = trace.snapshot("initial");
    assert.equal(events[0].data.value, "contains [REDACTED]");
    assert.ok(events.some((event) => event.type === "workspace_changes"));
    assert.equal(existsSync(join(dir, "private", "trace.jsonl")), false);
    assert.equal(readFileSync(join(dir, "private", "blobs", versions["results/state.txt"]), "utf8"), "first");
    assert.equal(readdirSync(join(dir, "private", "blobs")).length, 1);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("growing contexts reuse messages and images across model and HTTP events without changing the inputs", () => {
  const dir = mkdtempSync(join(tmpdir(), "navcraft-compact-trace-"));
  try {
    const image = Buffer.alloc(64 * 1024, 7);
    const base64 = image.toString("base64");
    const messages: any[] = [];
    const inputs: any[] = [];
    const trace = new Trace(dir, dir, ["fixture-secret"]);
    let oldBytes = 0;
    for (let i = 0; i < 30; i++) {
      messages.push({ role: "user", content: [{ type: "text", text: `frame ${i} fixture-secret` },
        { type: "image", data: base64, mimeType: "image/png" }] });
      inputs.push({ role: "user", content: [{ type: "input_text", text: `frame ${i} fixture-secret` },
        { type: "input_image", image_url: `data:image/png;base64,${base64}` }] });
      const model = { context: { messages }, model: "fixture" };
      const http = { number: i + 1, body: { model: "fixture", reasoning: { effort: "medium" }, input: inputs } };
      oldBytes += Buffer.byteLength(JSON.stringify(model)) + Buffer.byteLength(JSON.stringify(http));
      trace.emit("model_input", model);
      trace.emit("http_request", http);
    }
    const raw = readFileSync(join(dir, "trace.jsonl"), "utf8");
    const events = raw.trim().split("\n").map(line => JSON.parse(line));
    assert.equal(events.length, 60);
    assert.equal(events[0].data.context.messages[0].$trace_blob,
      events[58].data.context.messages[0].$trace_blob);
    const first = JSON.parse(readFileSync(join(dir, "blobs", events[0].data.context.messages[0].$trace_blob), "utf8"));
    assert.equal(first.content[0].text, "frame 0 [REDACTED]");
    const imageHash = createHash("sha256").update(image).digest("hex");
    assert.equal(first.content[1].data.$trace_blob, imageHash);
    const http = JSON.parse(readFileSync(join(dir, "blobs", events[1].data.body.input[0].$trace_blob), "utf8"));
    assert.deepEqual(http.content[1].image_url, { $trace_blob: imageHash, encoding: "data-url", prefix: "data:image/png;base64," });
    assert.deepEqual(readFileSync(join(dir, "blobs", imageHash)), image);
    const blobs = readdirSync(join(dir, "blobs"));
    assert.equal(blobs.length, 61); // 30 Pi messages, 30 HTTP items, one shared image.
    const bytes = Buffer.byteLength(raw) + blobs.reduce((sum, name) => sum + readFileSync(join(dir, "blobs", name)).length, 0);
    assert.ok(bytes < oldBytes / 100, `Trace grew to ${bytes} bytes from ${oldBytes}`);
    assert.ok(!raw.includes(base64) && !raw.includes("fixture-secret"));
    assert.equal(messages[0].content[0].text, "frame 0 fixture-secret");
    assert.equal(messages[0].content[1].data, base64);
    assert.equal(inputs[0].content[1].image_url, `data:image/png;base64,${base64}`);
    assert.equal(events[59].data.body.reasoning.effort, "medium");
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("tool images are referenced before transport even when Node does not persist its own trace", () => {
  const dir = mkdtempSync(join(tmpdir(), "navcraft-image-event-"));
  try {
    const events: any[] = [];
    const trace = new Trace(dir, dir, [], event => events.push(event), false);
    const data = Buffer.from("image fixture").toString("base64");
    trace.emit("tool_execution_end", { toolName: "view_images", result: { content: [{ type: "image", data }] } });
    assert.equal(existsSync(join(dir, "trace.jsonl")), false);
    const image = events[0].data.result.content[0].data;
    assert.equal(readFileSync(join(dir, "blobs", image.$trace_blob)).toString("base64"), data);
    assert.equal(image.encoding, "base64");
    assert.ok(!JSON.stringify(events).includes(data));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
