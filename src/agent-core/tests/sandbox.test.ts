import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import type { ExtensionToolContext, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { createCodingTools, publicPath, sandboxRun } from "../sandbox.js";

const python = "/home2/chen/.local/anaconda3/envs/vln/bin/python";
let base: string;
let workspace: string;
let tools: Map<string, ToolDefinition>;

before(async () => {
  base = await mkdtemp(join(tmpdir(), "navcraft-sandbox-"));
  workspace = join(base, "workspace");
  for (const dir of ["scene", "observations", "memory", "scratch"]) {
    await mkdir(join(workspace, dir), { recursive: true });
  }
  await writeFile(join(base, "private-answer.txt"), "hidden-answer");
  await writeFile(join(workspace, "scene", "room.md"), "room_01: bed opposite television\n");
  await writeFile(join(workspace, "observations", "feedback.json"), '{"collision":false}\n');
  tools = new Map(createCodingTools(workspace, python).map((tool) => [tool.name, tool]));
});

after(async () => { await rm(base, { recursive: true, force: true }); });

function invoke(name: string, args: Record<string, unknown>) {
  return tools.get(name)!.execute("test", args, undefined, undefined, { cwd: workspace } as ExtensionToolContext);
}

test("sandbox Python can use vln libraries and write only the two writable trees", async () => {
  const result = await sandboxRun(workspace, python, "python -c 'import pathlib, numpy; pathlib.Path(\"scratch/calc.txt\").write_text(str(numpy.sum([2,3]))); print(numpy.sum([2,3]))'");
  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(result.stdout.trim(), "5");
  assert.equal(await readFile(join(workspace, "scratch", "calc.txt"), "utf8"), "5");
  const blocked = await sandboxRun(workspace, python, "printf corrupted > scene/room.md");
  assert.notEqual(blocked.exitCode, 0);
  assert.equal(await readFile(join(workspace, "scene", "room.md"), "utf8"), "room_01: bed opposite television\n");
  assert.notEqual((await sandboxRun(workspace, python, "printf corrupted > observations/feedback.json")).exitCode, 0);
  assert.notEqual((await sandboxRun(workspace, python, "printf new > outside.txt")).exitCode, 0);
});

test("host files, inherited secrets and network are unavailable", async () => {
  process.env.NAVCRAFT_SANDBOX_SENTINEL = "must-never-inherit";
  try {
    const result = await sandboxRun(workspace, python, `python -c 'import os,pathlib,socket; assert "NAVCRAFT_SANDBOX_SENTINEL" not in os.environ; assert not pathlib.Path(${JSON.stringify(join(base, "private-answer.txt"))}).exists(); assert not pathlib.Path("/home2/chen/Project/NavCraft").exists(); s=socket.socket(); print(s.connect_ex(("1.1.1.1",53)))'`);
    assert.equal(result.exitCode, 0, result.stderr);
    assert.notEqual(Number(result.stdout.trim()), 0);
  } finally { delete process.env.NAVCRAFT_SANDBOX_SENTINEL; }
});

test("native read cannot probe outside paths or follow escaping and dangling links", async () => {
  await symlink(join(base, "private-answer.txt"), join(workspace, "scratch", "external"));
  await symlink(join(base, "missing", "private"), join(workspace, "scratch", "dangling"));
  await assert.rejects(invoke("read", { path: "../private-answer.txt" }), /workspace/);
  await assert.rejects(invoke("read", { path: join(base, "private-answer.txt") }), /workspace/);
  await assert.rejects(invoke("read", { path: "scratch/external" }), /Symlink/);
  await assert.rejects(invoke("read", { path: "scratch/dangling" }), /Dangling/);
  await assert.rejects(invoke("write", { path: "scratch/dangling/child", content: "x" }), /Dangling/);
  const bypass = await sandboxRun(workspace, python, "cat scratch/external");
  assert.notEqual(bypass.exitCode, 0);
  const good = await invoke("read", { path: "scene/room.md" });
  assert.equal(good.content[0].type, "text");
  assert.match((good.content[0] as { text: string }).text, /bed opposite television/);
});

test("native write and exact edit preserve their behavior under the same boundary", async () => {
  await invoke("write", { path: "memory/nested/state.txt", content: "heading: east\nroom: R03\n" });
  const edited = await invoke("edit", { path: "memory/nested/state.txt", edits: [{ oldText: "heading: east", newText: "heading: north" }] });
  assert.match((edited.details as { diff: string }).diff, /north/);
  assert.equal(await readFile(join(workspace, "memory/nested/state.txt"), "utf8"), "heading: north\nroom: R03\n");
  await assert.rejects(invoke("edit", { path: "memory/nested/state.txt", edits: [{ oldText: "absent", newText: "new" }] }), /exact|Could not|find/i);
  await assert.rejects(invoke("write", { path: "scene/room.md", content: "override" }), /writable/);
  await assert.rejects(invoke("edit", { path: "observations/feedback.json", edits: [{ oldText: "false", newText: "true" }] }), /writable/);
  await symlink(join(workspace, "scene"), join(workspace, "memory", "readonly-link"));
  await assert.rejects(invoke("write", { path: "memory/readonly-link/new.md", content: "override" }), /writable/);
});

test("native read returns actual image content", async () => {
  const generated = await sandboxRun(workspace, python, "python -c 'from PIL import Image; Image.new(\"RGB\",(16,16),(30,90,140)).save(\"scratch/evidence.png\")'");
  assert.equal(generated.exitCode, 0, generated.stderr);
  const result = await invoke("read", { path: "scratch/evidence.png" });
  const image = result.content.find((block) => block.type === "image");
  assert.ok(image && image.type === "image");
  assert.equal(image.mimeType, "image/png");
  assert.ok(Buffer.from(image.data, "base64").length > 0);
});

test("full native bash output is published into scratch and can be continued with read", async () => {
  const result = await invoke("bash", { command: "python -c 'for i in range(6000): print(str(i)+\":\"+\"x\"*200)'" });
  const full = (result.details as { fullOutputPath: string }).fullOutputPath;
  assert.ok(full.startsWith(join(workspace, "scratch", "tool-output")));
  assert.match((result.content[0] as { text: string }).text, /Full output:/);
  const structured = result.structuredContent as { truncated: boolean; full_output_path: string };
  assert.equal(structured.truncated, true);
  assert.equal(structured.full_output_path, full);
  const continued = await invoke("read", { path: full, offset: 5999, limit: 2 });
  assert.match((continued.content[0] as { text: string }).text, /5998:/);
  assert.match((continued.content[0] as { text: string }).text, /5999:/);
});

test("tools queue competing mutations and do not retain a failed queue", async () => {
  const writing = invoke("write", { path: "memory/queue.txt", content: "first" });
  const editing = invoke("edit", { path: "memory/queue.txt", edits: [{ oldText: "first", newText: "second" }] });
  await Promise.all([writing, editing]);
  assert.equal(await readFile(join(workspace, "memory/queue.txt"), "utf8"), "second");
  await assert.rejects(invoke("write", { path: "scene/fail", content: "fail" }));
  await invoke("write", { path: "memory/after-failure.txt", content: "ok" });
});

test("timeout and abort terminate the sandbox without host fallback", async () => {
  await assert.rejects(sandboxRun(workspace, python, "sleep 30", { timeout: 0.05 }), /timeout/);
  const controller = new AbortController();
  const pending = sandboxRun(workspace, python, "sleep 30", { signal: controller.signal });
  setTimeout(() => controller.abort(), 50);
  await assert.rejects(pending, /aborted/);
  await assert.rejects(sandboxRun(workspace, python, "true", { timeout: -1 }), /Invalid timeout/);
});

test("publicPath accepts new files only beneath a valid in-workspace ancestor", async () => {
  assert.equal(await publicPath(workspace, "memory/future/new.md", true), join(workspace, "memory", "future", "new.md"));
  await assert.rejects(publicPath(workspace, "memory/../../private-answer.txt"), /workspace/);
  await assert.rejects(publicPath(workspace, "scene/future/new.md", true), /writable/);
});

test("explicit generic policy mounts only declared paths and supports arbitrary writable names", async () => {
  for (const folder of ["assets", "state", "work", "hidden"]) await mkdir(join(workspace, folder));
  await writeFile(join(workspace, "assets", "input.txt"), "public input");
  await writeFile(join(workspace, "hidden", "answer.txt"), "not declared");
  const policy = { readOnly: ["assets"], readWrite: ["state", "work"] };
  const result = await sandboxRun(workspace, python, "python -c 'import pathlib; assert not pathlib.Path(\"hidden/answer.txt\").exists(); pathlib.Path(\"state/result.txt\").write_text(pathlib.Path(\"assets/input.txt\").read_text())'", { policy });
  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(await readFile(join(workspace, "state", "result.txt"), "utf8"), "public input");
  assert.notEqual((await sandboxRun(workspace, python, "echo invalid > assets/input.txt", { policy })).exitCode, 0);
  const generic = new Map(createCodingTools(workspace, python, policy).map((tool) => [tool.name, tool]));
  const context = { cwd: workspace } as ExtensionToolContext;
  await assert.rejects(generic.get("read")!.execute("hidden", { path: "hidden/answer.txt" }, undefined, undefined, context), /readable/);
  const written = await generic.get("write")!.execute("write", { path: "work/script.py", content: "print(4)" }, undefined, undefined, context);
  assert.ok(written);
  await assert.rejects(sandboxRun(workspace, python, "true", { policy: { readOnly: ["assets"], readWrite: ["assets/subdir"] } }), /overlap/);
});
