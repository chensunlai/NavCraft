import { mkdirSync, readFileSync, writeFileSync, copyFileSync, existsSync, rmSync, readdirSync } from 'node:fs';
import { resolve, join, basename } from 'node:path';
import { parse } from 'yaml';
import { runtime, DEFAULT_PYTHON, imageContent, promptBounded } from './runtime.js';
import { EnvironmentBridge, navigationTools, pythonRequest, type ToolState } from './tools.js';

export function runFinishReason(state: ToolState, budgetExhausted: boolean, lastMessage: any, error?: string): string {
  if (state.done) return state.current?.reason ?? 'task_ended';
  if (budgetExhausted) return 'request_budget';
  if (lastMessage?.stopReason === 'aborted' || /\b(?:aborted|aborterror)\b/i.test(lastMessage?.errorMessage ?? error ?? '')) return 'timeout';
  if (lastMessage?.stopReason === 'length') return 'output_limit';
  if (lastMessage?.stopReason === 'error') return 'model_error';
  if (error) return 'runtime_error';
  return 'model_finished';
}

export async function runCase(repo: string, config: any, name: string, overrides: { maxRequests?: number; seconds?: number } = {}) {
  const startedAt = Date.now();
  const cases = parse(readFileSync(join(repo, 'config/harness-pi/smoke.yaml'), 'utf8')).cases;
  const job = cases[name]; if (!job) throw new Error(`Unknown case ${name}`);
  const remainingSeconds = () => Math.max(0.001, (overrides.seconds ?? job.seconds) - (Date.now() - startedAt) / 1000);
  const dir = resolve(repo, config.output_dir, new Date().toISOString().replace(/[:.]/g, '-') + '-' + name);
  const workspace = join(dir, 'workspace'); const privateDir = join(dir, 'private');
  for (const folder of ['scene', 'observations', 'memory', 'scratch', '../private']) mkdirSync(join(workspace, folder), { recursive: true });
  const python = config.python ?? DEFAULT_PYTHON; const script = join(repo, 'src/harness-pi/python/memory.py');
  let bridge: EnvironmentBridge | undefined; let initial: any; let prompt: string; let initialImages: string[] = [];
  const state: ToolState = { done: false, goalChanged: false };
  let r: Awaited<ReturnType<typeof runtime>> | undefined;
  try {
  if (name === 'query') {
    const sample = resolve(repo, job.sample); const task = JSON.parse(readFileSync(join(sample, 'task.json'), 'utf8'));
    mkdirSync(join(workspace, 'scene/sample'), { recursive: true });
    const resources: Record<string, string> = {};
    for (const [id, ref] of Object.entries(task.resources) as [string, string][]) {
      const target = ref.startsWith('observations/') ? ref : 'scene/sample/' + ref;
      mkdirSync(resolve(workspace, target, '..'), { recursive: true }); copyFileSync(join(sample, ref), join(workspace, target)); resources[id] = target;
    }
    writeFileSync(join(workspace, 'scene/resources.json'), JSON.stringify(resources, null, 2));
    writeFileSync(join(workspace, 'observations/index.jsonl'), JSON.stringify({ id: 'initial_observation', path: resources.initial_observation, step: 0, source: 'observation', view: null, derived_from: [] }) + '\n');
    initialImages = task.initial_images.map((ref: string) => ref.startsWith('observations/') ? ref : 'scene/sample/' + ref);
    prompt = task.question + '\nResources: scene/resources.json. Only these fixed documents/images are available; no fresh observations. Final answer must be JSON {"choice":"<opening ID>","anchor":"initial_observation"}.';
    initial = { task: { goal_id: 0, modality: 'qa', instruction: task.question }, observation: { id: 'initial_observation', path: resources.initial_observation, step: 0 } };
  } else {
    const partial = name === 'mapping';
    await pythonRequest(python, script, 'prepare', { scene_dir: resolve(repo, config.scene_archive), workspace, mode: partial ? 'partial' : 'prebuilt', ...(partial ? { allowed_ids: ['R01', 'R03'] } : {}) });
    if (partial) {
      // Mapping evidence comes only from this run; do not reveal the withheld portal through prior cards/images.
      rmSync(join(workspace, 'scene/evidence'), { recursive: true, force: true }); mkdirSync(join(workspace, 'scene/evidence')); writeFileSync(join(workspace, 'scene/evidence/index.jsonl'), '');
      for (const folder of ['memory/entities', 'scene/initial/entities']) for (const card of readdirSync(join(workspace, folder))) {
        const file = join(workspace, folder, card); const text = readFileSync(file, 'utf8').replace(/^- 证据：.*$/m, '- 证据：无').replace(/^[-] 描述：.*$/m, '- 描述：区域范围由初始图提供，具体布局与出入口需要现场观察。'); writeFileSync(file, text);
      }
    }
    mkdirSync(join(workspace, 'scene/tools')); copyFileSync(script, join(workspace, 'scene/tools/memory.py'));
    bridge = new EnvironmentBridge(python, repo, privateDir);
    initial = await bridge.request('init-session', 'init', { repo, workspace, private_dir: join(privateDir, 'environment'), env_config: job.env_config ?? 'config/environment/objectnav_visual.yaml', task_config: job.task_config, episode_id: job.episode_id, atomic_sample: job.sample, goal_limit: job.goal_limit, scene_archive: config.scene_archive, purpose: job.purpose ?? 'navigation', instruction_override: name === 'mapping' ? '局部空间记忆构建：根据真实观察建立一个出入口记录，取得新视角后修订，并在后续判断中使用。' : name === 'compaction' ? '局部空间记忆与压缩续接检查；保持真实观察引用和待核实判断。' : undefined });
    const savedState = JSON.parse(readFileSync(join(workspace, 'memory/state.json'), 'utf8'));
    savedState.goal_id = initial.task?.goal_id; savedState.pose_estimate = { ...initial.initial_hint, status: 'initial_hint' };
    writeFileSync(join(workspace, 'memory/state.json'), JSON.stringify(savedState, null, 2));
    const map = await pythonRequest(python, script, 'render', { workspace, map_id: initial.initial_hint.map_id });
    state.latestMap = map.images[0]; state.latestMapMetadata = map.data; state.current = initial;
    initialImages = [...initial.images, ...map.images];
    if (name === 'local') {
      mkdirSync(join(workspace, 'scene/task'), { recursive: true });
      copyFileSync(resolve(repo, job.sample, 'task_map.png'), join(workspace, 'scene/task/task_map.png'));
      initialImages.push('scene/task/task_map.png');
    }
    if (name === 'mapping') initial.task.instruction = '局部空间记忆构建：根据真实观察建立一个出入口记录，取得新视角后修订，并在后续判断中使用。';
    if (name === 'compaction') initial.task.instruction = '局部空间记忆与压缩续接检查；保持真实观察引用和待核实判断。';
    prompt = JSON.stringify(initial) + '\nMemory entry: scene/index.md. Attached images include the initial RGB and map with the initialization arrow.';
    if (name === 'local') prompt += '\nThe original local task map scene/task/task_map.png shows the requested T point; it is attached along with the general scene map.';
    if (partial) prompt += '\nLocal memory-building test: use actual observations to identify one doorway between the supplied regions. Create a new portal record and card with real evidence; geometry may initially be null. Check and render it, obtain a new viewpoint, revise the SAME record using the new evidence, then read/render the updated record and use it to answer how to approach this doorway from your latest view. Keep within 24 atomic actions and 4 metres. Do not claim metric precision or invent observations. Finish with a concise report of record paths, evidence and remaining uncertainty.';
    if (partial) prompt += '\n最终需以原地图归一化坐标给出 geometry:{"segment":[[x1,y1],[x2,y2]]} 表示门洞的大致位置和跨度，这是可修订估计而非精确测量；null 可作中间态，最终仍为 null 必须报告未完成，并须取得新视角后修订同一个 ID 再重绘。';
    else if (name === 'compaction') prompt += '\nShort memory-check task: inspect the current view and nearby map/card evidence. Save a tentative observation-to-map judgment with real frame references and uncertainty to memory/notes.md, read it back, then give a short final answer. Keep movements to at most two actions; no goal submission.';
  }
  const systemPrompt = name === 'query' ? 'Answer a local spatial question by autonomously reading the fixed public resource bank. Images may be opened via read or view_images. Do not access resources outside the workspace. Preserve the requested answer anchor.' : readFileSync(join(repo, 'src/harness-pi/navigation_prompt.md'), 'utf8');
  r = await runtime({ workspace, directory: dir, budgetFile: resolve(repo, config.budget_file), maxRequests: overrides.maxRequests ?? job.requests, credentialFile: resolve(repo, config.credentials), api: config.api, modelId: config.model, python, systemPrompt, tools: navigationTools(repo, workspace, python, bridge, state, !!job.task_config), compact: true, publicState: () => ({ task: state.current?.task ?? initial.task, latestObservation: state.current?.mode === 'panorama' ? state.current.views : state.current?.observation ?? initial.observation, map_ref: state.latestMap, map_metadata: state.latestMapMetadata, goal_image_ref: state.current?.task ? state.current.task.image : initial.task?.image }) });
  r.session.subscribe(event => { if (event.type === 'turn_start') state.goalChanged = false; });
  const originalFinishTurn = r.session.agent.finishTurn;
  r.session.agent.finishTurn = async (turn, signal) => {
    if (state.done) return { action: 'end' };
    const decision = await originalFinishTurn?.(turn, signal);
    return decision || undefined;
  };
  let error: string | undefined; let finishReason = 'model_finished';
  try {
    await promptBounded(r.session, prompt, imageContent(workspace, initialImages), remainingSeconds());
    if (name === 'compaction' && !state.done && r.budget.used < job.requests) {
      const before = r.session.messages; r.trace.emit('compaction_test_before', { messages: before });
      r.settingsManager.applyOverrides({ compaction: { enabled: true, reserveTokens: 65000, keepRecentTokens: 1200 } });
      let compacted = false;
      const unsubscribe = r.session.subscribe(event => { if (event.type === 'compaction_end' && event.result) { compacted = true; r!.settingsManager.applyOverrides({ compaction: { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 } }); } });
      await promptBounded(r.session, 'Continue the same memory-check task after the budget-triggered compression: reopen your saved note and its evidence image, call nav_observe(single), and distinguish any old location estimate from the new observation before giving a brief answer. Do not claim the tentative doorway is confirmed without evidence.', [], remainingSeconds());
      unsubscribe(); r.trace.emit('compaction_test_after', { compacted, messages: r.session.messages });
      writeFileSync(join(privateDir, 'compaction_check.json'), JSON.stringify({ compacted, before_messages: before.length, after_messages: r.session.messages.length }, null, 2));
    }
  }
  catch (e) { error = String(e); }
  finally {
    const lastMessage: any = r.session.messages.at(-1);
    finishReason = runFinishReason(state, r.budget.exhausted, lastMessage, error);
    if (bridge) {
      try {
        const final = await bridge.request('finalize-session', 'finalize', { reason: finishReason });
        state.current = { ...state.current, ...final };
        state.done = !!(final.terminated || final.truncated);
      } catch (e) { error ??= String(e); }
      await bridge.close().catch(e => { error ??= String(e); });
      bridge = undefined;
    }
    r.trace.snapshot('final');
    const answer = r.session.getLastAssistantText(); const last: any = r.session.messages.at(-1);
    const result: any = { case: name, sample: job.sample, episode_id: job.episode_id, goal_limit: job.goal_limit, model: config.model, api: config.api, requests: r.budget.used, global_requests: r.budget.totalUsed, answer, error, finish_reason: finishReason, last_stop_reason: last?.stopReason, last_error: last?.errorMessage, terminated: !!state.current?.terminated, truncated: !!state.current?.truncated, software_only: false };
    const envResult = join(privateDir, 'environment/result.json'); if (existsSync(envResult)) result.environment = JSON.parse(readFileSync(envResult, 'utf8'));
    if (name === 'query') {
      result.verification = await pythonRequest(python, join(repo, 'src/harness-pi/python/verify.py'), 'qa', { mode: 'qa', workspace, sample_dir: resolve(repo, job.sample), answer }).catch(e => ({ error: String(e) }));
    }
    if (name === 'mapping') {
      result.verification = await pythonRequest(python, join(repo, 'src/harness-pi/python/verify.py'), 'mapping', { mode: 'mapping', workspace, scene_dir: resolve(repo, config.scene_archive), trace_dir: privateDir, environment_private_dir: join(privateDir, 'environment') }).catch(e => ({ error: String(e) }));
    }
    if (name === 'compaction' && existsSync(join(privateDir, 'compaction_check.json'))) result.compaction = JSON.parse(readFileSync(join(privateDir, 'compaction_check.json'), 'utf8'));
    writeFileSync(join(dir, 'result.json'), JSON.stringify(result, null, 2)); r.trace.emit('run_finished', result); r.session.dispose();
    console.log(JSON.stringify({ directory: dir, case: name, requests: result.requests, global_requests: result.global_requests, error: result.error ?? result.last_error, terminated: result.terminated, environment: result.environment?.metrics ?? result.environment, verification: result.verification }, null, 2));
  }
  return dir;
  } catch (error) {
    // Keep a reviewable result even when initialization failed before a session existed.
    if (bridge) await bridge.request('finalize-initialization-failure', 'finalize', { reason: 'initialization_error' }).catch(() => {});
    if (!existsSync(join(dir, 'result.json'))) {
      const partial: any = { case: name, sample: job.sample, episode_id: job.episode_id, goal_limit: job.goal_limit,
        model: config.model, api: config.api, error: String(error), finish_reason: r ? 'runtime_error' : 'initialization_error',
        terminated: !!state.current?.terminated, truncated: true, software_only: false };
      if (r) { partial.requests = r.budget.used; partial.global_requests = r.budget.totalUsed; r.trace.snapshot('failed'); }
      const envResult = join(privateDir, 'environment/result.json');
      if (existsSync(envResult)) partial.environment = JSON.parse(readFileSync(envResult, 'utf8'));
      writeFileSync(join(dir, 'result.json'), JSON.stringify(partial, null, 2));
    }
    throw error;
  } finally {
    if (bridge) await bridge.close().catch(() => {});
    r?.session.dispose();
  }
}
