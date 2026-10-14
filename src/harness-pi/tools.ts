import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { appendFileSync, readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { Type } from 'typebox';
import type { ToolDefinition } from '@earendil-works/pi-coding-agent';
import { imageContent } from './runtime.js';
import { publicPath } from './sandbox.js';

export async function pythonRequest(python: string, script: string, command: string, input: Record<string, unknown>, signal?: AbortSignal) {
  return new Promise<any>((accept, reject) => {
    const child = spawn(python, ['-B', script, command], { stdio: ['pipe', 'pipe', 'pipe'], signal });
    let out = '', err = ''; child.stdout.on('data', d => { out += d; }); child.stderr.on('data', d => { err += d; });
    child.on('error', reject); child.on('close', code => {
      try { const result = JSON.parse(out); if (code !== 0 || result.ok === false) reject(new Error(result.error ?? JSON.stringify(result.errors) ?? err)); else accept(result); }
      catch (error) { reject(new Error(`Python ${command}: ${String(error)} ${err.slice(-2000)}`)); }
    });
    child.stdin.end(JSON.stringify(input) + '\n');
  });
}
export class EnvironmentBridge {
  private child: ChildProcessWithoutNullStreams;
  private pending = new Map<string, { accept: (x: any) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  constructor(python: string, repo: string, readonly directory: string) {
    this.child = spawn(python, ['-B', '-u', join(repo, 'src/harness-pi/python/bridge.py')], { cwd: repo, stdio: ['pipe', 'pipe', 'pipe'], env: { PATH: process.env.PATH, LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH, __EGL_VENDOR_LIBRARY_FILENAMES: '/usr/share/glvnd/egl_vendor.d/10_nvidia.json', PYTHONDONTWRITEBYTECODE: '1' } });
    this.child.stderr.on('data', data => appendFileSync(join(directory, 'environment.log'), data));
    createInterface({ input: this.child.stdout }).on('line', line => {
      let value: any; try { value = JSON.parse(line); } catch { this.fail(new Error('Environment protocol corrupted')); return; }
      const waiting = this.pending.get(value.id); if (!waiting) return;
      clearTimeout(waiting.timer); this.pending.delete(value.id);
      if (value.ok) waiting.accept(value.data); else waiting.reject(new Error(value.error));
    });
    this.child.on('error', error => this.fail(error));
    this.child.on('exit', code => this.fail(new Error(`Environment exited: ${code}`)));
  }
  private fail(error: Error) { for (const x of this.pending.values()) { clearTimeout(x.timer); x.reject(error); } this.pending.clear(); }
  request(id: string, op: string, args: Record<string, unknown> = {}) {
    return new Promise<any>((accept, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('Environment request timed out; execution may have occurred')); this.child.kill('SIGKILL'); }, op === 'init' ? 120000 : 60000);
      this.pending.set(id, { accept, reject, timer });
      this.child.stdin.write(JSON.stringify({ id, op, args }) + '\n');
    });
  }
  async close() { try { await this.request('close-session', 'close'); } finally { this.child.stdin.end(); this.child.kill(); } }
}
export interface ToolState { done: boolean; goalChanged: boolean; current?: any; latestMap?: string; }
export function navigationTools(repo: string, workspace: string, python: string, bridge: EnvironmentBridge | undefined, state: ToolState, allowFinish = true): ToolDefinition<any>[] {
  const memoryScript = join(repo, 'src/harness-pi/python/memory.py');
  async function result(data: any, images: string[] = []) {
    for (const ref of images) await publicPath(workspace, ref);
    return { content: [{ type: 'text' as const, text: JSON.stringify(data) }, ...imageContent(workspace, images)], details: { images } };
  }
  const definitions: ToolDefinition<any>[] = [
    { name: 'view_images', label: 'View images', description: 'Read existing workspace images or stable image IDs; optional crops use normalized original-image coordinates. Does not observe the environment.', parameters: Type.Object({ refs: Type.Array(Type.String(), { minItems: 1, maxItems: 10 }), crops: Type.Optional(Type.Array(Type.Union([Type.Tuple([Type.Number(), Type.Number(), Type.Number(), Type.Number()]), Type.Null()]))), layout: Type.Optional(Type.Union([Type.Literal('separate'), Type.Literal('grid')])) }), async execute(id, args: any, signal) {
      const r = await pythonRequest(python, memoryScript, 'view', { ...args, workspace }, signal); return result(r.data, r.images);
    } },
    { name: 'render_map', label: 'Render map', description: 'Render the public wall layer and CURRENT memory geometry/state. Never locates you or plans routes. Region coordinates refer to the full map. Invalid records return errors.', parameters: Type.Object({ map_id: Type.String(), region: Type.Optional(Type.Tuple([Type.Number(), Type.Number(), Type.Number(), Type.Number()])), highlight_ids: Type.Optional(Type.Array(Type.String())), grid: Type.Optional(Type.Boolean()), layers: Type.Optional(Type.Array(Type.String())) }), async execute(id, args: any, signal) {
      const r = await pythonRequest(python, memoryScript, 'render', { ...args, workspace }, signal); state.latestMap = r.images?.[0]; return result(r.data, r.images);
    } },
  ];
  if (bridge) {
    const act = async (id: string, op: string, args: any) => {
      if (state.done) throw new Error('Task already ended');
      if (state.goalChanged) throw new Error('Target changed during this turn; read the new goal and decide again next turn');
      const data = await bridge.request(id, op, args); state.current = { ...state.current, ...data };
      if (op !== 'observe') { delete state.current.mode; delete state.current.views; }
      else if (args.mode === 'single') state.current.views = data.views;
      state.done = !!(data.terminated || data.truncated); state.goalChanged = !!data.goal_changed;
      if (op === 'observe' && args.mode === 'panorama') {
        const montage = await pythonRequest(python, memoryScript, 'view', { workspace, refs: data.images, layout: 'grid' });
        return result({ ...data, images: montage.images, panorama_views: data.views }, montage.images);
      }
      return result(data, data.images ?? []);
    };
    definitions.push(
      { name: 'nav_observe', label: 'Observe', description: 'Capture NEW RGB at the current position. panorama gives six horizontal views relative to body heading and restores body/camera state; it does not turn the body.', parameters: Type.Object({ mode: Type.Union([Type.Literal('single'), Type.Literal('panorama')]) }), execute: (id, args) => act(id, 'observe', args) },
      { name: 'nav_step', label: 'Move', description: 'Execute 1–6 allowed atomic actions in sequence. Collision, goal change or task completion interrupts the batch. Returns actual feedback, last RGB and references to intermediate frames; no true pose.', parameters: Type.Object({ actions: Type.Array(Type.String(), { minItems: 1, maxItems: 6 }) }), execute: (id, args) => act(id, 'step', args) },
    );
    if (allowFinish) definitions.push({ name: 'nav_finish', label: 'Finish goal', description: 'Submit completion of the CURRENT goal by its ID. It may end the episode or switch goals, without certifying success.', parameters: Type.Object({ goal_id: Type.Integer() }), execute: (id, args) => act(id, 'finish', args) });
  }
  return definitions.map(x => ({ ...x, executionMode: 'sequential' }));
}
