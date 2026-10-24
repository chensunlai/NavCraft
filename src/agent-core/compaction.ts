import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import type { AgentMessage, StreamFn } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';
import { compact, type ExtensionFactory, type ProjectedSessionEntry } from '@earendil-works/pi-coding-agent';
import { publicPath, type ExecutionPolicy } from './sandbox.js';
import { labelledImages } from './image-content.js';

export interface PublicCheckpoint {
  content: string | TextContent[];
  images?: string[];
  image_labels?: string[];
  metadata?: unknown;
  revision: string | number;
}
export interface CompactionTrace { emit(type: string, data?: unknown): void; }
type CompactionPreparation = Parameters<typeof compact>[0];

/** A round is one assistant decision and all its paired tool results. */
export function completeRoundStarts(messages: AgentMessage[]): number[] {
  const starts: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (message.role !== 'assistant') continue;
    const pending = new Set(message.content.filter(b => b.type === 'toolCall').map(b => b.id));
    for (let j = i + 1; j < messages.length && pending.size; j++) {
      const receipt = messages[j];
      if (receipt.role === 'assistant' || receipt.role === 'user') break;
      if (receipt.role === 'toolResult') pending.delete(receipt.toolCallId);
    }
    if (!pending.size) starts.push(i);
  }
  return starts;
}

/** Change the request projection only; full images remain in the saved session. */
export function pruneImages(messages: AgentMessage[]): AgentMessage[] {
  const starts = completeRoundStarts(messages);
  const cutoff = starts.length > 3 ? starts[starts.length - 3] : 0;
  let checkpointCurrent = '';
  let latestCurrent = '';
  let latestImage = '';
  for (let m = 0; m < messages.length; m++) {
    const message = messages[m];
    if (!('content' in message) || !Array.isArray(message.content)) continue;
    for (let i = 0; i < message.content.length; i++) {
      const block = message.content[i];
      if (block.type !== 'image') continue;
      latestImage = `${m}:${i}`;
      const label = message.content[i - 1];
      if (label?.type === 'text' && /\[CURRENT\b/.test(label.text)) {
        latestCurrent = latestImage;
        // The refreshed factual head precedes history but contains the newest observation.
        if (message.role === 'custom' && message.customType === 'public_checkpoint') checkpointCurrent = latestImage;
      }
    }
  }
  const keep = checkpointCurrent || latestCurrent || latestImage;
  return messages.map((message, index) => {
    if (index >= cutoff || !('content' in message) || !Array.isArray(message.content)) return message;
    const content = message.content;
    return { ...message, content: content.map((block, i) => {
      if (block.type !== 'image' || `${index}:${i}` === keep) return block;
      const previous = content[i - 1];
      const label = previous?.type === 'text' ? previous.text.match(/\[(?:CURRENT|HISTORY|MAP|TARGET)[^\]]*\]/)?.[0] : undefined;
      return { type: 'text' as const, text: `${label ?? 'Earlier image'} 已移出上下文，可用 view_images 重新打开。` };
    }) } as AgentMessage;
  });
}

/** Move the native cut point to the start of the last two complete tool rounds. */
export function keepRecentRounds(entries: ProjectedSessionEntry[], preparation: CompactionPreparation): CompactionPreparation | undefined {
  const projected = entries.flatMap(entry => entry.messages.filter(message => message.role !== 'system' && entry.sourceEntry.type !== 'compaction').map(message => ({ message, id: entry.sourceEntry.id })));
  const messages = projected.map(entry => entry.message);
  const rounds = completeRoundStarts(messages);
  if (rounds.length <= 2) return undefined;
  const cut = rounds[rounds.length - 2];
  const older = messages.slice(0, cut).filter(message => !(message.role === 'custom' && message.customType === 'public_checkpoint'));
  if (!older.length) return undefined;
  return { ...preparation, firstKeptEntryId: projected[cut].id,
    messagesToSummarize: pruneImages(older), turnPrefixMessages: [], isSplitTurn: false };
}

/** Count visible text with the application's fixed local tokenizer, on CPU only. */
export function countSummaryTokens(python: string, text: string, signal?: AbortSignal): Promise<number> {
  return new Promise((accept, reject) => {
    const child = spawn(python, ['-m', 'src.harness.tokenizer'], { cwd: process.cwd(),
      env: { ...process.env, CUDA_VISIBLE_DEVICES: '', TOKENIZERS_PARALLELISM: 'false' },
      stdio: ['pipe', 'pipe', 'pipe'], signal });
    let stdout = '', stderr = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 60000);
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.stdin.on('error', () => {});
    child.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(`Summary tokenizer failed: ${stderr.slice(-1000)}`)); return; }
      try {
        const result = JSON.parse(stdout);
        if (!Number.isInteger(result.count) || result.count < 0) throw new Error('Invalid tokenizer count');
        accept(result.count);
      } catch (error) { reject(error); }
    });
    child.stdin.end(text);
  });
}

async function factualHead(workspace: string, state: PublicCheckpoint, policy?: ExecutionPolicy): Promise<AgentMessage> {
  const content: (TextContent | ImageContent)[] = typeof state.content === 'string' ? [{ type: 'text', text: state.content }] : [...state.content];
  const images = await Promise.all((state.images ?? []).map(async ref => {
    const path = await publicPath(workspace, ref, false, policy);
    return { type: 'image' as const, data: (await readFile(path)).toString('base64'),
      mimeType: /\.jpe?g$/i.test(path) ? 'image/jpeg' : 'image/png' };
  }));
  content.push(...labelledImages(images, state.images ?? [], state.image_labels));
  return { role: 'custom', customType: 'public_checkpoint', content,
    timestamp: Date.now(), display: false, details: { revision: state.revision } };
}

/** Pi owns history and native summaries; Python supplies an already-public factual head. */
export function createCheckpointCompaction(sharedStream: StreamFn,
  checkpoint: () => PublicCheckpoint | undefined | Promise<PublicCheckpoint | undefined>,
  trace: CompactionTrace, policy?: ExecutionPolicy, python?: string): ExtensionFactory {
  return pi => {
    let facts: AgentMessage | undefined;
    pi.on('session_before_compact', async (event, ctx) => {
      trace.emit('checkpoint_compaction_start', { reason: event.reason, tokensBefore: event.preparation.tokensBefore });
      try {
        if (!ctx.model || !python) throw new Error('Compaction requires model and local tokenizer Python');
        const preparation = keepRecentRounds(ctx.sessionManager.buildSessionProjection().entries, event.preparation);
        if (!preparation) { trace.emit('checkpoint_compaction_skipped', { reason: 'fewer_than_three_complete_rounds' }); return { cancel: true }; }
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
        if (!auth.ok) throw new Error(auth.error);
        const headers = auth.headers ? Object.fromEntries(Object.entries(auth.headers).filter((entry): entry is [string, string] => typeof entry[1] === 'string')) : undefined;
        const stream: StreamFn = async (model, context, options) => {
          trace.emit('checkpoint_summary_input', { model: model.id, context });
          const result = await sharedStream(model, context, { ...options, maxTokens: 8192, maxRetries: 0 });
          const original = result.result.bind(result);
          result.result = async () => {
            const message = await original();
            trace.emit('checkpoint_summary_output', message);
            return message;
          };
          return result;
        };
        const result = await compact(preparation, auth.baseUrl ? { ...ctx.model, baseUrl: auth.baseUrl } : ctx.model,
          auth.apiKey, headers, event.customInstructions, event.signal, ctx.thinkingLevel, stream,
          auth.env, { enabled: false, maxRetries: 0, baseDelayMs: 0 }, undefined, ctx.sessionManager.getSessionId(),
          { maxVisibleTokens: 300, countTokens: async text => {
            const count = await countSummaryTokens(python, text, event.signal);
            trace.emit('summary_length', { visible_tokens: count, limit: 300 });
            return count;
          } });
        event.signal.throwIfAborted();
        trace.emit('checkpoint_compaction_result', result);
        return { compaction: result };
      } catch (error) {
        trace.emit('checkpoint_compaction_cancelled', { error: String(error) });
        return { cancel: true };
      }
    });
    pi.on('session_compact', async (event, ctx) => {
      const state = await checkpoint();
      if (!state) return;
      facts = await factualHead(ctx.cwd, state, policy);
      if (facts.role !== 'custom') throw new Error('Invalid factual head');
      pi.sendMessage({ customType: facts.customType, content: facts.content, display: false, details: facts.details }, { triggerTurn: false });
      trace.emit('checkpoint_restored', { revision: state.revision, images: state.images ?? [], compactionId: event.compactionEntry.id });
    });
    pi.on('context', async (event, ctx) => {
      // Reload the persisted factual head when reopening an existing Pi session.
      facts ??= ctx.sessionManager.buildSessionProjection().messages.findLast(message => message.role === 'custom' && message.customType === 'public_checkpoint');
      if (facts) {
        const state = await checkpoint();
        if (state) facts = await factualHead(ctx.cwd, state, policy);
      }
      const messages = event.messages.filter(message => !(message.role === 'custom' && message.customType === 'public_checkpoint'));
      return { messages: pruneImages(facts ? [facts, ...messages] : messages) };
    });
  };
}
