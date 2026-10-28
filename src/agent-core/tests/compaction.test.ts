import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { AgentMessage, StreamFn } from '@earendil-works/pi-agent-core';
import { createAssistantMessageEventStream, type AssistantMessage, type Model, type SimpleStreamOptions } from '@earendil-works/pi-ai';
import { compact, SessionManager } from '@earendil-works/pi-coding-agent';
import { completeRoundStarts, keepRecentRounds } from '../compaction.js';
import { runtime } from '../session.js';

function assistant(text: string, stopReason: AssistantMessage['stopReason'] = 'stop'): AssistantMessage {
  return { role: 'assistant', content: [{ type: 'text', text }], api: 'openai-responses',
    provider: 'fixture', model: 'fixture', stopReason, timestamp: 1,
    usage: { input: 10, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 30,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}

function round(index: number, current = false): AgentMessage[] {
  return [
    { ...assistant(''), content: [
      { type: 'toolCall', id: `read-${index}`, name: 'read', arguments: {} },
      { type: 'toolCall', id: `observe-${index}`, name: 'observe', arguments: {} },
    ], stopReason: 'toolUse' },
    { role: 'toolResult', toolCallId: `read-${index}`, toolName: 'read',
      content: [{ type: 'text', text: 'read receipt' }], isError: false, timestamp: index },
    { role: 'toolResult', toolCallId: `observe-${index}`, toolName: 'observe',
      content: [{ type: 'text', text: `[${current ? 'CURRENT' : 'HISTORY'} f_${index}]` },
        { type: 'image', data: `image-${index}`, mimeType: 'image/png' }], isError: false, timestamp: index },
  ];
}

test('incomplete tool batches are not counted as completed rounds', () => {
  const messages = [...round(0), ...round(1), ...round(2), ...round(3).slice(0, 2)];
  assert.equal(completeRoundStarts(messages).length, 3);
});

test('compaction starts before an entire assistant batch and retains two rounds', () => {
  const manager = SessionManager.inMemory('/tmp');
  manager.appendMessage({ role: 'user', content: 'Find the chair', timestamp: 0 });
  for (const message of [round(0), round(1), round(2), round(3)].flat()) {
    if (message.role === 'assistant' || message.role === 'toolResult') manager.appendMessage(message);
  }
  const projection = manager.buildSessionProjection();
  const prepared = keepRecentRounds(projection.entries, {
    firstKeptEntryId: 'ignored', messagesToSummarize: [], turnPrefixMessages: [], isSplitTurn: true,
    tokensBefore: 24001, settings: { enabled: true, reserveTokens: 8768, keepRecentTokens: 6000 },
    fileOps: { read: new Set(), edited: new Set(), written: new Set() },
  });
  assert.ok(prepared);
  assert.equal(prepared.isSplitTurn, false);
  assert.equal(prepared.messagesToSummarize.length, 7);
  const first = projection.entries.find(entry => entry.sourceEntry.id === prepared.firstKeptEntryId)!;
  assert.equal(first.messages[0].role, 'assistant');
  const retained = projection.entries.slice(projection.entries.indexOf(first)).flatMap(entry => entry.messages);
  assert.equal(completeRoundStarts(retained).length, 2);
  assert.deepEqual(prepared.turnPrefixMessages, []);
  manager.appendMessage(round(4)[0] as AssistantMessage);
  assert.equal(keepRecentRounds(manager.buildSessionProjection().entries, prepared), undefined,
    'an unresolved tool batch must finish before compaction');
});

const model: Model<'openai-responses'> = { id: 'fixture', name: 'Fixture', api: 'openai-responses',
  provider: 'fixture', baseUrl: 'http://fixture.invalid', reasoning: true, contextWindow: 32768,
  maxTokens: 8192, input: ['text', 'image'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };

async function summarize(outputs: AssistantMessage[]) {
  const calls: { context: unknown; options: SimpleStreamOptions | undefined }[] = [];
  const stream: StreamFn = (_model, context, options) => {
    calls.push({ context, options });
    const result = createAssistantMessageEventStream();
    result.end(outputs.shift() ?? assistant('unexpected'));
    return result;
  };
  const result = await compact({ firstKeptEntryId: 'kept', messagesToSummarize: round(0),
    turnPrefixMessages: round(1), isSplitTurn: true, tokensBefore: 24001,
    previousSummary: 'Old uncertain plan', settings: { enabled: true, reserveTokens: 8768, keepRecentTokens: 6000 },
    fileOps: { read: new Set(['memory/entities.json']), edited: new Set(), written: new Set() } },
    model, 'fixture', undefined, undefined, undefined, 'xhigh', stream,
    undefined, { enabled: false, maxRetries: 0, baseDelayMs: 0 }, undefined, 'fixture-session',
    { maxVisibleTokens: 300, countTokens: async text => text.length });
  return { result, calls };
}

test('native short summary preserves xhigh and total output allowance, shortens once, and checks final text', async () => {
  const { result, calls } = await summarize([assistant('x'.repeat(350)), assistant('Door still uncertain. Inspect its far side.')]);
  assert.equal(calls.length, 2);
  assert.ok(calls.every(call => call.options?.maxTokens === 8192 && call.options.reasoning === 'xhigh'));
  assert.match(JSON.stringify(calls[0].context), /task_read/);
  assert.doesNotMatch(JSON.stringify(calls[0].context), /host separately supplies/);
  assert.match(JSON.stringify(calls[1].context), /draft-summary/);
  assert.doesNotMatch(result.summary, /read-files|Turn Context|memory\/entities/);
  assert.equal(result.usage?.totalTokens, 60);
  assert.equal(result.summary, 'Door still uncertain. Inspect its far side.');
});

test('two overlong summaries fail without truncation or unbounded retries', async () => {
  await assert.rejects(summarize([assistant('x'.repeat(301)), assistant('y'.repeat(301))]), /visible token limit/);
});

test('reasoning-only, truncated and tool-producing summaries cannot replace history', async () => {
  await assert.rejects(summarize([{ ...assistant(''), content: [{ type: 'thinking', thinking: 'reasoning only' }] }]), /visible text/);
  await assert.rejects(summarize([assistant('partial answer', 'length')]), /token cap/);
  await assert.rejects(summarize([{ ...assistant(''), content: [{ type: 'toolCall', id: 'call', name: 'move', arguments: {} }] }]), /call a tool/);
});

test('tool-loop compaction adds no facts; task_read restores task text and image as ordinary history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navcraft-tool-loop-compaction-'));
  const workspace = join(directory, 'workspace');
  await mkdir(workspace);
  const original = globalThis.fetch;
  const payloads: Record<string, any>[] = [];
  const instruction = 'Find the pictured chair beside the window using the left entrance.';
  const target = 'UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v89WAAAAA==';
  const summary = 'The doorway remains uncertain. Inspect its other side.';
  let decisions = 0, reads = 0, summaries = 0;
  globalThis.fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body));
    payloads.push(body);
    const isSummary = JSON.stringify(body).includes('<conversation>');
    if (isSummary) summaries++;
    else decisions++;
    const name = decisions === 1 ? 'observe' : decisions === 2 ? 'task_read' : undefined;
    const item = !isSummary && name
      ? { type: 'function_call', id: `fc_${decisions}`, call_id: `call_${decisions}`,
          name, arguments: '{}' }
      : { type: 'message', id: `msg_${payloads.length}`, role: 'assistant', status: 'completed',
          content: [{ type: 'output_text', text: isSummary ? summary : 'The task and reference are available.', annotations: [] }] };
    const inputTokens = !isSummary && decisions === 1 ? 25000 : 100;
    const events = [
      { type: 'response.output_item.done', output_index: 0, item },
      { type: 'response.completed', response: { id: `resp_${payloads.length}`, status: 'completed', output: [item],
        usage: { input_tokens: inputTokens, output_tokens: 10, total_tokens: inputTokens + 10 } } },
    ];
    return new Response(events.map(event => `data: ${JSON.stringify(event)}\n\n`).join(''),
      { headers: { 'content-type': 'text/event-stream' } });
  };
  let active: Awaited<ReturnType<typeof runtime>> | undefined;
  try {
    active = await runtime({ workspace, directory, budgetFile: join(directory, 'requests.json'), maxRequests: 5,
      auth: { baseUrl: 'http://fixture.invalid', model: 'fixture', apiKey: 'fixture-secret' },
      api: 'openai-responses', coding: false, compact: true, thinkingLevel: 'xhigh',
      compaction: { reserveTokens: 8768, keepRecentTokens: 100 },
      python: process.env.NAVCRAFT_TEST_PYTHON ?? '/home/chen/.local/anaconda3/envs/vln/bin/python',
      systemPrompt: 'Navigate using tools. Read the current task with task_read when needed.',
      initialMessages: [{ role: 'user', content: [{ type: 'text', text: instruction },
        { type: 'image', data: target, mimeType: 'image/webp' }], timestamp: 0 },
        ...[round(0), round(1), round(2), round(3)].flat().filter((message): message is AssistantMessage | Extract<AgentMessage, { role: 'toolResult' }> => message.role === 'assistant' || message.role === 'toolResult')],
      tools: [
        { name: 'observe', label: 'Observe', description: 'Read the current view',
          parameters: { type: 'object', properties: {} },
          execute: async () => ({ content: [{ type: 'text', text: 'Current doorway view.' }], details: {} }) },
        { name: 'task_read', label: 'Task', description: 'Read the current task and reference image',
          parameters: { type: 'object', properties: {} },
          execute: async () => { reads++; return { content: [{ type: 'text', text: instruction },
            { type: 'text', text: '[TARGET f_reference]' },
            { type: 'image', data: target, mimeType: 'image/webp' }], details: {} }; } },
      ],
    });
    await active.session.continueRun();
    assert.equal(decisions, 3);
    assert.equal(summaries, 1);
    assert.equal(reads, 1);
    assert.equal(active.budget.used, 4, 'three decisions and one summary share the request budget');
    assert.equal(payloads[1].reasoning.effort, 'xhigh');
    assert.equal(payloads[1].max_output_tokens, 8192);
    const beforeRead = payloads[2].input as Record<string, any>[];
    const afterRead = payloads[3].input as typeof beforeRead;
    const before = JSON.stringify(beforeRead);
    assert.match(before, /The doorway remains uncertain/);
    assert.doesNotMatch(before, /public_checkpoint|PUBLIC SNAPSHOT/);
    assert.ok(!before.includes(instruction), 'compaction does not inject the task');
    assert.ok(!before.includes(target), 'compaction does not inject the target image');
    assert.match(before, /observe-3/);
    assert.doesNotMatch(before, /observe-0|observe-1|observe-2/);
    assert.ok(beforeRead.some(item => item.type === 'function_call_output' && item.call_id === 'call_1'));
    assert.deepEqual(afterRead.slice(0, beforeRead.length), beforeRead,
      'the task reply appends to an unchanged request prefix');
    const taskCall = afterRead.find(item => item.type === 'function_call' && item.name === 'task_read');
    const taskReceipt = afterRead.find(item => item.type === 'function_call_output' && item.call_id === taskCall?.call_id);
    assert.ok(taskCall && taskReceipt, 'the task has a paired ordinary tool receipt');
    assert.ok(JSON.stringify(taskReceipt).includes(instruction));
    assert.ok(JSON.stringify(taskReceipt).includes(target));
    assert.ok(!active.session.sessionManager.getEntries().some(entry => entry.type === 'custom_message'),
      'compaction adds no hidden custom message');
  } finally {
    active?.session.dispose();
    globalThis.fetch = original;
    await rm(directory, { recursive: true, force: true });
  }
});
