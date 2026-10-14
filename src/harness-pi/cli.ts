import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { parse } from 'yaml';
import { runtime, DEFAULT_PYTHON, imageContent, promptBounded } from './runtime.js';
import { sandboxRun } from './sandbox.js';
import { runCase } from './run.js';

const repo = resolve('.');
const args = process.argv.slice(2);
function option(name: string, fallback?: string) { const i = args.indexOf('--' + name); return i < 0 ? fallback : args[i + 1]; }
const config = parse(readFileSync(resolve(option('config', 'config/harness-pi/default.yaml')!), 'utf8'));
export async function codingSmoke() {
  const dir = resolve(config.output_dir, new Date().toISOString().replace(/[:.]/g, '-') + '-coding');
  const workspace = join(dir, 'workspace');
  for (const name of ['scene', 'observations', 'memory', 'scratch']) mkdirSync(join(workspace, name), { recursive: true });
  writeFileSync(join(workspace, 'scene', 'instructions.md'), 'Fixture: make a Python script that reads numbers [3,5,7], writes their sum into memory/result.json, and creates scratch/result.png showing the sum. Read the generated image.\n');
  execFileSync(config.python ?? DEFAULT_PYTHON, ['-B', '-c', "from PIL import Image; import sys; p=sys.argv[1]; Image.new('RGB',(160,120),'red').save(p+'/red.png'); Image.new('RGB',(160,120),'blue').save(p+'/blue.png')", join(workspace, 'scene')]);
  const r = await runtime({ workspace, directory: dir, budgetFile: resolve(config.budget_file), maxRequests: 12, credentialFile: resolve(config.credentials), api: config.api, modelId: config.model, python: config.python, systemPrompt: 'You are an autonomous agent operating in a public workspace. Use native coding tools to solve the user task. scene and observations are read-only; memory and scratch are writable. Commands run in a restricted environment with Python, NumPy and Pillow. Finish with a concise factual report. Do not access other paths.' });
  try {
    await promptBounded(r.session, 'Read scene/instructions.md, create and execute the requested Python script, inspect the saved JSON and read the generated image. Also identify the two attached color images. Continue tool calls until verified.', imageContent(workspace, ['scene/red.png', 'scene/blue.png']), 600);
    const result = { case: 'coding', api: config.api, requests: r.budget.used, global_requests: r.budget.totalUsed, answer: r.session.getLastAssistantText(), written: existsSync(join(workspace, 'memory/result.json')), generated_image: existsSync(join(workspace, 'scratch/result.png')), last: r.session.messages.at(-1) };
    writeFileSync(join(dir, 'result.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify({ directory: dir, ...result }, null, 2));
  } finally { r.trace.snapshot('final'); r.session.dispose(); }
}
async function main() {
  if (args[0] === 'doctor' && !args.includes('--online')) {
    const dir = resolve('cache/harness-pi/doctor/workspace');
    for (const name of ['scene', 'observations', 'memory', 'scratch']) mkdirSync(join(dir, name), { recursive: true });
    const result = await sandboxRun(dir, config.python ?? DEFAULT_PYTHON, "python -B -c 'import numpy,PIL; print(numpy.__version__,PIL.__version__)'");
    console.log(JSON.stringify({ node: process.version, sdk: import.meta.resolve('@earendil-works/pi-coding-agent'), python: config.python, sandbox: result, credentials_present: existsSync(resolve(config.credentials)), scene_present: existsSync(resolve(config.scene_archive)) }, null, 2)); return;
  }
  if (args[0] === 'coding' || (args[0] === 'doctor' && args.includes('--online'))) { await codingSmoke(); return; }
  if (args[0] === 'run') { await runCase(repo, config, option('case', 'query')!, { maxRequests: option('max-requests') ? Number(option('max-requests')) : undefined }); return; }
  if (args[0] === 'smoke') { for (const name of ['query','local','objnav','instruction','goat','mapping','compaction']) await runCase(repo, config, name); return; }
  throw new Error('Usage: npm run harness -- doctor [--online] | coding | run --case CASE | smoke');
}
main().catch(error => { console.error(String(error)); process.exitCode = 1; });
