import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
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
  await assert.rejects(summarize([{ ...assistant(''), content: [{ type: 'toolCall', id: 'call', name: 'move', arguments: {} }] }]), /call a tool/);
});

test('actual model requests append to a frozen checkpoint until the next compaction', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navcraft-lite-compaction-'));
  const workspace = join(directory, 'workspace');
  for (const folder of ['scene', 'observations', 'memory', 'scratch']) await mkdir(join(workspace, folder), { recursive: true });
  await writeFile(join(workspace, 'observations', 'current.png'), Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB5cAAAAASUVORK5CYII=', 'base64'));
  await writeFile(join(workspace, 'observations', 'reference.webp'), Buffer.from('UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v89WAAAAA==', 'base64'));
  const original = globalThis.fetch;
  const payloads: Record<string, unknown>[] = [];
  const instruction = 'Find the pictured chair beside the window. Use the left entrance, not the doorway behind the sofa.';
  const systemPrompt = 'Navigation fixture: maintain evidence-backed memory.\nA submission acknowledgement is not a success score.';
  const notesPath = 'scratch/route-notes.md';
  const checkpoint = { content: `PUBLIC SNAPSHOT step=12; current_request=${instruction}; current_question=Which doorway matches the picture?; frame=f_latest; workspace_files=${notesPath}`, revision: 4,
    images: ['observations/current.png', 'observations/reference.webp'],
    image_labels: ['[CURRENT f_latest]', '[TARGET f_reference]'] };
  globalThis.fetch = async (_input, init) => {
    payloads.push(JSON.parse(String(init?.body)));
    const chunks = [
      { id: 'fixture', choices: [{ index: 0, delta: { role: 'assistant', content: JSON.stringify(payloads.at(-1)).includes('<conversation>') ? 'Door identity uncertain; inspect its other side.' : 'I will inspect the door.' }, finish_reason: null }] },
      { id: 'fixture', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
    ];
    return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n',
      { headers: { 'content-type': 'text/event-stream' } });
  };
  let active: Awaited<ReturnType<typeof runtime>> | undefined;
  try {
    const options: Parameters<typeof runtime>[0] = { workspace, directory, budgetFile: join(directory, 'requests.json'), maxRequests: 8,
      auth: { baseUrl: 'http://fixture.invalid', model: 'fixture', apiKey: 'fixture-secret' },
      api: 'openai-completions', coding: false, compact: false, thinkingLevel: 'xhigh',
      compaction: { reserveTokens: 8768, keepRecentTokens: 100 },
      python: process.env.NAVCRAFT_TEST_PYTHON ?? '/home/chen/.local/anaconda3/envs/vln/bin/python',
      systemPrompt, executionPolicy: { readOnly: ['scene', 'observations', 'memory'], readWrite: ['scratch'] },
      tools: [{ name: 'observe', label: 'Observe', description: 'Fixture observation',
        parameters: { type: 'object', properties: {} }, execute: async () => ({ content: [{ type: 'text', text: 'fixture' }], details: {} }) }],
      initialMessages: [{ role: 'user', content: 'OLD DISCARDED HISTORY '.repeat(1000), timestamp: 0 },
        ...[round(0), round(1), round(2), round(3)].flat().filter((message): message is AssistantMessage | Extract<AgentMessage, { role: 'toolResult' }> => message.role === 'assistant' || message.role === 'toolResult')],
      checkpoint: () => checkpoint,
    };
    active = await runtime(options);
    const result = await active.session.compact();
    assert.equal(result.summary, 'Door identity uncertain; inspect its other side.');
    assert.equal(active.budget.used, 1);
    await active.session.continueRun();
    assert.equal(active.budget.used, 2);
    assert.equal(payloads[0].reasoning_effort, 'xhigh');
    assert.equal(payloads[0].max_completion_tokens, 8192);
    const input = JSON.stringify(payloads[1]);
    const firstMessages = payloads[1].messages as { role: string; content: unknown }[];
    assert.ok(input.includes(JSON.stringify(systemPrompt).slice(1, -1)));
    assert.ok(input.includes(instruction));
    assert.ok(input.includes(notesPath));
    assert.match(input, /Which doorway matches the picture/);
    assert.match(input, /PUBLIC SNAPSHOT step=12/);
    assert.match(input, /CURRENT f_latest/);
    assert.match(input, /data:image\/webp;base64/);
    assert.match(input, /Door identity uncertain/);
    assert.doesNotMatch(input, /OLD DISCARDED HISTORY|observe-0|observe-1/);
    assert.match(input, /observe-2/);
    assert.match(input, /observe-3/);
    assert.ok(input.indexOf('observe-3') < input.indexOf('PUBLIC SNAPSHOT'));
    assert.ok(!firstMessages.some(message => message.role === 'user' && message.content === 'Continue'));

    // Public state changes alone must not rewrite an earlier snapshot or image.
    checkpoint.content = 'PUBLIC SNAPSHOT step=20; current_request=Find a table; frame=f_after_moving';
    checkpoint.images = ['observations/current.png'];
    checkpoint.image_labels = ['[CURRENT f_after_moving]'];
    checkpoint.revision = 5;
    const sessionFile = active.session.sessionManager.getSessionFile();
    assert.ok(sessionFile);
    active.session.dispose();
    active = await runtime({ ...options, initialMessages: undefined, sessionFile });
    for (let index = 0; index < 4; index++) {
      await active.session.prompt(index ? `Keep discussing the table: ${index}` : 'Next goal: find a table');
      const messages = payloads[index + 2].messages as typeof firstMessages;
      assert.deepEqual(messages.slice(0, firstMessages.length), firstMessages);
      assert.match(JSON.stringify(messages), /CURRENT f_latest/);
      assert.doesNotMatch(JSON.stringify(messages), /CURRENT f_after_moving|Removed from context/);
    }

    // The old target is released at the next boundary, not on each new prompt.
    await active.session.compact();
    await active.session.continueRun();
    assert.equal(active.budget.used, 8);
    const next = JSON.stringify(payloads.at(-1));
    assert.match(next, /PUBLIC SNAPSHOT step=20|Find a table|CURRENT f_after_moving/);
    assert.doesNotMatch(next, /CURRENT f_latest|TARGET f_reference|data:image\/webp;base64/);
    assert.ok(!next.includes(instruction));
    const checkpoints = active.session.sessionManager.getEntries().filter(entry =>
      entry.type === 'custom_message' && entry.customType === 'public_checkpoint');
    assert.equal(checkpoints.length, 2, 'original checkpoints remain in the persisted audit trail');
    checkpoint.images = ['observations/missing.png'];
    const beforeFailure = active.session.sessionManager.getEntries();
    await assert.rejects(active.session.compact(), /Compaction cancelled/);
    assert.deepEqual(active.session.sessionManager.getEntries(), beforeFailure);
    assert.equal(active.budget.used, 8, 'invalid public state cannot discard history or request a summary');
  } finally {
    active?.session.dispose();
    globalThis.fetch = original;
    await rm(directory, { recursive: true, force: true });
  }
});

test('automatic compaction appends its snapshot before the following request', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navcraft-auto-compaction-'));
  const workspace = join(directory, 'workspace');
  await mkdir(workspace);
  const original = globalThis.fetch;
  const payloads: Record<string, unknown>[] = [];
  globalThis.fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body));
    payloads.push(body);
    const summary = JSON.stringify(body).includes('<conversation>');
    const chunks = [
      { id: 'fixture', choices: [{ index: 0, delta: { role: 'assistant', content: summary ? 'Review the doorway.' : 'Observed the doorway.' }, finish_reason: null }] },
      { id: 'fixture', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      { id: 'fixture', choices: [], usage: { prompt_tokens: payloads.length === 1 ? 25000 : 100, completion_tokens: 10, total_tokens: payloads.length === 1 ? 25010 : 110 } },
    ];
    return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n',
      { headers: { 'content-type': 'text/event-stream' } });
  };
  let active: Awaited<ReturnType<typeof runtime>> | undefined;
  try {
    active = await runtime({ workspace, directory, budgetFile: join(directory, 'requests.json'), maxRequests: 4,
      auth: { baseUrl: 'http://fixture.invalid', model: 'fixture', apiKey: 'fixture-secret' },
      api: 'openai-completions', coding: false, compact: true,
      compaction: { reserveTokens: 8768, keepRecentTokens: 100 },
      python: process.env.NAVCRAFT_TEST_PYTHON ?? '/home/chen/.local/anaconda3/envs/vln/bin/python',
      systemPrompt: 'Fixture agent.',
      initialMessages: [{ role: 'user', content: 'Inspect the doorway.', timestamp: 0 },
        ...[round(0), round(1), round(2), round(3)].flat().filter((message): message is AssistantMessage | Extract<AgentMessage, { role: 'toolResult' }> => message.role === 'assistant' || message.role === 'toolResult')],
      checkpoint: () => ({ content: 'PUBLIC SNAPSHOT step=4; frame=f_0004', revision: 4 }),
    });
    await active.session.continueRun();
    assert.equal(active.budget.used, 2, 'one normal request and one threshold summary');
    await active.session.prompt('Continue checking the doorway.');
    assert.equal(active.budget.used, 3);
    const input = JSON.stringify(payloads[2]);
    assert.match(input, /PUBLIC SNAPSHOT step=4/);
    assert.match(input, /Review the doorway/);
    assert.doesNotMatch(input, /observe-0|observe-1|observe-2/);
    assert.ok(input.indexOf('Observed the doorway') < input.indexOf('PUBLIC SNAPSHOT'));
    assert.ok(input.indexOf('PUBLIC SNAPSHOT') < input.indexOf('Continue checking the doorway'));
  } finally {
    active?.session.dispose();
    globalThis.fetch = original;
    await rm(directory, { recursive: true, force: true });
  }
});
