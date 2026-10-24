import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { AgentMessage, StreamFn } from '@earendil-works/pi-agent-core';
import { createAssistantMessageEventStream, type AssistantMessage, type Model, type SimpleStreamOptions } from '@earendil-works/pi-ai';
import { compact, SessionManager } from '@earendil-works/pi-coding-agent';
import { completeRoundStarts, keepRecentRounds, pruneImages } from '../compaction.js';
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
      { type: 'toolCall', id: `observe-${index}`, name: 'nav_observe', arguments: {} },
    ], stopReason: 'toolUse' },
    { role: 'toolResult', toolCallId: `read-${index}`, toolName: 'read',
      content: [{ type: 'text', text: 'read receipt' }], isError: false, timestamp: index },
    { role: 'toolResult', toolCallId: `observe-${index}`, toolName: 'nav_observe',
      content: [{ type: 'text', text: `[${current ? 'CURRENT' : 'HISTORY'} f_${index}]` },
        { type: 'image', data: `image-${index}`, mimeType: 'image/png' }], isError: false, timestamp: index },
  ];
}

test('image projection retains three complete multi-tool rounds and the latest current image', () => {
  const messages = [round(0, true), round(1), round(2), round(3), round(4)].flat();
  const before = JSON.stringify(messages);
  const projected = pruneImages(messages);
  const images = projected.flatMap(message => 'content' in message && Array.isArray(message.content)
    ? message.content.filter(block => block.type === 'image').map(block => block.data) : []);
  assert.deepEqual(images, ['image-0', 'image-2', 'image-3', 'image-4']);
  assert.equal(JSON.stringify(messages), before);
  assert.match(JSON.stringify(projected), /Removed from context/);
  assert.equal(completeRoundStarts(messages).length, 5);
});

test('a leading checkpoint retains its latest image over an older panorama after three further rounds', () => {
  const checkpoint: AgentMessage = { role: 'custom', customType: 'public_checkpoint', display: false,
    timestamp: 10, details: { revision: 14 }, content: [
      { type: 'text', text: 'Public facts: frame=f_latest' },
      { type: 'text', text: '[CURRENT f_latest]' },
      { type: 'image', data: 'latest-image', mimeType: 'image/png' },
    ] };
  const panorama = round(1, true);
  const receipt = panorama[2] as Extract<AgentMessage, { role: 'toolResult' }>;
  receipt.content[0] = { type: 'text', text: '[CURRENT p_older]' };
  const messages = [checkpoint, ...round(0), ...panorama, ...round(2), ...round(3), ...round(4)];
  const before = JSON.stringify(messages);
  const projected = pruneImages(messages);
  const images = projected.flatMap(message => 'content' in message && Array.isArray(message.content)
    ? message.content.filter(block => block.type === 'image').map(block => block.data) : []);
  assert.deepEqual(images, ['latest-image', 'image-2', 'image-3', 'image-4']);
  assert.equal(completeRoundStarts(projected).length, 5);
  assert.equal(JSON.stringify(messages), before);
});

test('incomplete tool batches are kept and never counted as a completed round', () => {
  const messages = [...round(0), ...round(1), ...round(2), ...round(3).slice(0, 2)];
  assert.equal(completeRoundStarts(messages).length, 3);
  assert.equal(pruneImages(messages).length, messages.length);
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
  assert.match(JSON.stringify(calls[1].context), /draft-summary/);
  assert.doesNotMatch(result.summary, /read-files|Turn Context|memory\/entities/);
  assert.equal(result.usage?.totalTokens, 60);
  assert.equal(result.summary, 'Door still uncertain. Inspect its far side.');
});

test('two overlong summaries fail without truncation or unbounded retries', async () => {
  await assert.rejects(summarize([assistant('x'.repeat(301)), assistant('y'.repeat(301))]), /visible token limit/);
});

test('reasoning-only, truncated and tool-producing summaries cannot become checkpoints', async () => {
  await assert.rejects(summarize([{ ...assistant(''), content: [{ type: 'thinking', thinking: 'reasoning only' }] }]), /visible text/);
  await assert.rejects(summarize([assistant('partial answer', 'length')]), /token cap/);
  await assert.rejects(summarize([{ ...assistant(''), content: [{ type: 'toolCall', id: 'call', name: 'nav_step', arguments: {} }] }]), /call a tool/);
});

test('actual Pi compaction persists the short summary and restores public facts before continuing', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navcraft-lite-compaction-'));
  const workspace = join(directory, 'workspace');
  for (const folder of ['scene', 'observations', 'memory', 'scratch']) await mkdir(join(workspace, folder), { recursive: true });
  await writeFile(join(workspace, 'observations', 'current.png'), Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB5cAAAAASUVORK5CYII=', 'base64'));
  await writeFile(join(workspace, 'observations', 'reference.webp'), Buffer.from('UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v89WAAAAA==', 'base64'));
  const original = globalThis.fetch;
  const payloads: Record<string, unknown>[] = [];
  const checkpoint = { content: 'PUBLIC FACTS current_request=Find chair; frame=f_latest', revision: 4,
    images: ['observations/current.png', 'observations/reference.webp'],
    image_labels: ['[CURRENT f_latest]', '[TARGET f_reference]'] };
  globalThis.fetch = async (_input, init) => {
    payloads.push(JSON.parse(String(init?.body)));
    const chunks = [
      { id: 'fixture', choices: [{ index: 0, delta: { role: 'assistant', content: payloads.length === 1 ? 'Door identity uncertain; inspect its other side.' : 'I will inspect the door.' }, finish_reason: null }] },
      { id: 'fixture', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
    ];
    return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n',
      { headers: { 'content-type': 'text/event-stream' } });
  };
  let active: Awaited<ReturnType<typeof runtime>> | undefined;
  try {
    active = await runtime({ workspace, directory, budgetFile: join(directory, 'requests.json'), maxRequests: 3,
      auth: { baseUrl: 'http://fixture.invalid', model: 'fixture', apiKey: 'fixture-secret' },
      api: 'openai-completions', coding: false, compact: false, thinkingLevel: 'xhigh',
      compaction: { reserveTokens: 8768, keepRecentTokens: 100 },
      python: process.env.NAVCRAFT_TEST_PYTHON ?? '/home/chen/.local/anaconda3/envs/vln/bin/python',
      systemPrompt: 'Navigation fixture', executionPolicy: { readOnly: ['scene', 'observations', 'memory'], readWrite: ['scratch'] },
      tools: [{ name: 'nav_observe', label: 'Observe', description: 'Fixture observation',
        parameters: { type: 'object', properties: {} }, execute: async () => ({ content: [{ type: 'text', text: 'fixture' }], details: {} }) }],
      initialMessages: [{ role: 'user', content: 'OLD DISCARDED HISTORY '.repeat(1000), timestamp: 0 },
        ...[round(0), round(1), round(2), round(3)].flat().filter((message): message is AssistantMessage | Extract<AgentMessage, { role: 'toolResult' }> => message.role === 'assistant' || message.role === 'toolResult')],
      checkpoint: () => checkpoint,
    });
    const result = await active.session.compact();
    assert.equal(result.summary, 'Door identity uncertain; inspect its other side.');
    assert.equal(active.budget.used, 1);
    checkpoint.content = 'PUBLIC FACTS current_request=Find chair; frame=f_after_moving';
    checkpoint.image_labels = ['[CURRENT f_after_moving]', '[TARGET f_reference]'];
    checkpoint.revision = 5;
    await active.session.prompt('Continue');
    assert.equal(active.budget.used, 2);
    assert.equal(payloads[0].reasoning_effort, 'xhigh');
    assert.equal(payloads[0].max_completion_tokens, 8192);
    const input = JSON.stringify(payloads[1]);
    assert.match(input, /PUBLIC FACTS/);
    assert.match(input, /CURRENT f_after_moving/);
    assert.match(input, /data:image\/webp;base64/);
    assert.doesNotMatch(input, /f_latest/);
    assert.match(input, /Door identity uncertain/);
    assert.doesNotMatch(input, /OLD DISCARDED HISTORY/);
    assert.doesNotMatch(input, /observe-0|observe-1/);
    assert.match(input, /observe-2/);
    assert.match(input, /observe-3/);
  } finally {
    active?.session.dispose();
    globalThis.fetch = original;
    await rm(directory, { recursive: true, force: true });
  }
});
