import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { navigationTools, EnvironmentBridge, type ToolState } from '../tools.js';
import { DEFAULT_PYTHON } from '../runtime.js';

test('movement supersedes panorama as latest evidence and a goal switch blocks the rest of its turn', async () => {
  const ws = mkdtempSync(join(tmpdir(), 'navcraft-tools-'));
  for (const d of ['scene','observations','memory','scratch']) mkdirSync(join(ws,d));
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=', 'base64');
  writeFileSync(join(ws,'observations/front.png'),png); writeFileSync(join(ws,'observations/new.png'),png);
  writeFileSync(join(ws,'observations/index.jsonl'), [{id:'front',path:'observations/front.png',source:'observation',step:0,view:{right_degrees:0,pitch_degrees:0},derived_from:[]},{id:'new',path:'observations/new.png',source:'observation',step:1,view:{right_degrees:0,pitch_degrees:0},derived_from:[]}].map(x=>JSON.stringify(x)).join('\n')+'\n');
  const state: ToolState = { done:false, goalChanged:false, current:{task:{goal_id:0}} };
  const calls:string[]=[];
  const bridge = { request: async (_id:string, op:string) => {
    calls.push(op);
    if(op==='observe')return {mode:'panorama',views:[{id:'front',path:'observations/front.png',step:0}],images:['observations/front.png']};
    return {observation:{id:'new',path:'observations/new.png',step:1},images:['observations/new.png'],task:{goal_id:1},goal_changed:true};
  }} as unknown as EnvironmentBridge;
  try {
    const tools=navigationTools(resolve('.'),ws,DEFAULT_PYTHON,bridge,state);
    const observe=tools.find(t=>t.name==='nav_observe')!; const step=tools.find(t=>t.name==='nav_step')!;
    await observe.execute('observe1',{mode:'panorama'},undefined,undefined,{} as any);
    assert.equal(state.current.mode,'panorama'); assert.equal(state.current.task.goal_id,0);
    await step.execute('step1',{actions:['move_forward']},undefined,undefined,{} as any);
    assert.equal(state.current.observation.id,'new'); assert.equal(state.current.mode,undefined); assert.equal(state.current.views,undefined);
    await assert.rejects(step.execute('step2',{actions:['move_forward']},undefined,undefined,{} as any),/Target changed/);
    assert.deepEqual(calls,['observe','step']);
  } finally {rmSync(ws,{recursive:true,force:true});}
});

test('bridge close waits for child cleanup after acknowledgment and remains safe to repeat', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'navcraft-bridge-close-'));
  const repo = join(directory, 'repo'); const scripts = join(repo, 'src/harness-pi/python');
  mkdirSync(scripts, { recursive: true });
  const saved = join(directory, 'cleanup.json');
  writeFileSync(join(scripts, 'bridge.py'), `import json,sys,time\nfrom pathlib import Path\nfor line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'id':r['id'],'ok':True,'data':{'closed':True}}),flush=True)\n if r['op']=='close':\n  time.sleep(0.1)\n  Path(${JSON.stringify(saved)}).write_text('{"complete":true}')\n  break\n`);
  const bridge = new EnvironmentBridge(DEFAULT_PYTHON, repo, join(directory, 'private'));
  try {
    await bridge.close();
    assert.deepEqual(JSON.parse(readFileSync(saved, 'utf8')), { complete: true });
    await bridge.close();
    await assert.rejects(bridge.request('after-exit', 'observe'), /already exited/);
  } finally { await bridge.close(); rmSync(directory, { recursive: true, force: true }); }
});
