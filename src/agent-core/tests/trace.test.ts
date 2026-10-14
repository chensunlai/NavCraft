import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, mkdirSync, readFileSync, existsSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Trace } from "../trace.js";

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
