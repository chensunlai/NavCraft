import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import { retryService, serviceUnavailable } from '../service-retry.js';
import { RequestBudget } from '../budget.js';
import { Trace } from '../trace.js';

const message = (error?: string): any => ({ role: 'assistant', content: error
  ? [{ type: 'toolCall', id: 'partial', name: 'nav_step', arguments: {} }]
  : [{ type: 'text', text: 'ready' }], api: 'openai-responses', provider: 'fixture', model: 'fixture',
  stopReason: error ? 'error' : 'stop', errorMessage: error, timestamp: Date.now(),
  usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });

test('service errors retry without limit, with identical context and one charged request', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-retry-')), old = globalThis.fetch;
  globalThis.fetch = async () => new Response('{}');
  const budget = new RequestBudget(join(dir, 'requests.json'), 1, new Trace(dir, dir));
  const context: any = { messages: [{ role: 'user', content: 'same input' }] };
  let sent = 0; const waits: number[] = [];
  try {
    const stream = retryService(async (_model, received) => {
      assert.equal(received, context);
      await budget.fetch('http://fixture.invalid');
      sent++;
      const result = message(sent <= 7 ? (sent % 2 ? 'navcraft API error (503): unavailable'
        : 'Upstream response stream was interrupted') : undefined);
      const events = createAssistantMessageEventStream();
      events.push({ type: 'start', partial: result });
      if (result.stopReason === 'error') events.push({ type: 'error', reason: 'error', error: result });
      else events.push({ type: 'done', reason: 'stop', message: result });
      events.end(result); return events;
    }, budget, async (ms) => { waits.push(ms); assert.equal(budget.charged, 0); });
    const result = await stream({} as any, context);
    const seen = [];
    for await (const event of result) seen.push(event);
    assert.equal((await result.result()).stopReason, 'stop');
    assert.equal(sent, 8); assert.equal(budget.used, 8); assert.equal(budget.charged, 1);
    assert.deepEqual(waits, [10000, 20000, 40000, 60000, 60000, 60000, 60000]);
    assert.deepEqual(seen.map(e => e.type), ['start', 'done']);
    assert.equal(JSON.stringify(seen).includes('nav_step'), false);
    const reopened = new RequestBudget(join(dir, 'requests.json'), 2, new Trace(dir, dir));
    assert.equal(reopened.charged, 1);
  } finally { globalThis.fetch = old; rmSync(dir, { recursive: true, force: true }); }
});

test('request budget exhaustion is never retried as a fetch outage', async () => {
  const dir=mkdtempSync(join(tmpdir(),'navcraft-budget-not-outage-'));
  try {
    const budget=new RequestBudget(join(dir,'requests.json'),1,new Trace(dir,dir));
    let sent=0;
    const stream=retryService(()=>{
      sent++;budget.exhausted=true;
      const result=message('fetch failed: NAVCRAFT_REQUEST_BUDGET_EXHAUSTED');
      const events=createAssistantMessageEventStream();
      events.push({type:'error',reason:'error',error:result});events.end(result);return events;
    },budget,async()=>assert.fail('An exhausted local request quota must not wait/retry'));
    const response=await stream({} as any,{messages:[]} as any);
    assert.equal((await response.result()).stopReason,'error');
    assert.equal(sent,1);
  } finally {rmSync(dir,{recursive:true,force:true});}
});

test('backoff is cancellable and invalid requests are not retried', async () => {
  assert.equal(serviceUnavailable('gateway_concurrency_limit: Concurrency limit exceeded for account, please retry later'), true);
  assert.equal(serviceUnavailable('The operation was aborted due to timeout'), true);
  assert.equal(serviceUnavailable('The operation was aborted by the user'), false);
  assert.equal(serviceUnavailable('navcraft API error (400): unavailable'), false);
  assert.equal(serviceUnavailable('navcraft API error (401): unauthorized'), false);
  assert.equal(serviceUnavailable('navcraft API error (502): upstream error'), true);
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-retry-cancel-'));
  try {
    const budget = new RequestBudget(join(dir, 'requests.json'), 1, new Trace(dir, dir));
    const controller = new AbortController();
    let sent = 0;
    const stream = retryService(() => {
      sent++; const result = message('navcraft API error (503): unavailable');
      const events = createAssistantMessageEventStream();
      events.push({ type: 'error', reason: 'error', error: result }); events.end(result); return events;
    }, budget, async (_ms, signal) => { controller.abort(); signal!.throwIfAborted(); });
    const result = await stream({} as any, { messages: [] } as any, { signal: controller.signal });
    assert.equal((await result.result()).stopReason, 'aborted'); assert.equal(sent, 1);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('SSE account limits, timeouts and server overload retry the unchanged decision without charging its wait', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-retry-account-')), old = globalThis.fetch;
  globalThis.fetch = async () => new Response('{}');
  const budget = new RequestBudget(join(dir, 'requests.json'), 1, new Trace(dir, dir));
  let sent = 0;
  try {
    const stream = retryService(async () => {
      await budget.fetch('http://fixture.invalid');
      const errors = ['gateway_concurrency_limit: Concurrency limit exceeded for account, please retry later',
        'The operation was aborted due to timeout',
        'server_error: Our servers are currently overloaded. Please try again later.'];
      const result = message(errors[sent++]);
      const events = createAssistantMessageEventStream();
      events.push(result.stopReason === 'error' ? { type: 'error', reason: 'error', error: result }
        : { type: 'done', reason: 'stop', message: result });
      events.end(result); return events;
    }, budget, async () => { assert.equal(budget.charged, 0); });
    const result = await stream({} as any, { messages: [] } as any);
    const seen = [];
    for await (const event of result) seen.push(event.type);
    assert.equal((await result.result()).stopReason, 'stop');
    assert.equal(sent, 4); assert.equal(budget.charged, 1);
    assert.deepEqual(seen, ['done']);
    assert.equal(budget.state().attempts.filter(x => x.service_error).length, 3);
  } finally { globalThis.fetch = old; rmSync(dir, { recursive: true, force: true }); }
});
