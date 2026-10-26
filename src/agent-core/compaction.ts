import { spawn } from 'node:child_process';
import type { AgentMessage, StreamFn } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';
import { compact, type ExtensionFactory, type ProjectedSessionEntry } from '@earendil-works/pi-coding-agent';
import type { ExecutionPolicy } from './sandbox.js';
import { labelledImages, loadImages } from './image-content.js';

export interface PublicCheckpoint {
  content: string | TextContent[];
  images?: string[];
  image_labels?: string[];
  revision: string | number;
}
export interface CompactionTrace { emit(type: string, data?: unknown): void; }
type CompactionPreparation = Parameters<typeof compact>[0];
type PublicSnapshot = Extract<AgentMessage, { role: 'custom' }> & { details: { revision: string | number } };

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

/** Move the native cut point to the start of the last two complete tool rounds. */
export function keepRecentRounds(entries: ProjectedSessionEntry[], preparation: CompactionPreparation): CompactionPreparation | undefined {
  const projected = entries.flatMap(entry => entry.messages.filter(message => message.role !== 'system' && entry.sourceEntry.type !== 'compaction').map(message => ({ message, id: entry.sourceEntry.id })));
  const messages = projected.map(entry => entry.message);
  const rounds = completeRoundStarts(messages);
  if (rounds.length <= 2 || rounds.length !== messages.filter(message => message.role === 'assistant').length) return undefined;
  const cut = rounds[rounds.length - 2];
  const older = messages.slice(0, cut).filter(message => !(message.role === 'custom' && message.customType === 'public_checkpoint'));
  if (!older.length) return undefined;
  return { ...preparation, firstKeptEntryId: projected[cut].id,
    messagesToSummarize: older, turnPrefixMessages: [], isSplitTurn: false };
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

async function publicSnapshot(workspace: string, state: PublicCheckpoint, policy?: ExecutionPolicy): Promise<PublicSnapshot> {
  const content: (TextContent | ImageContent)[] = [
    { type: 'text', text: 'Public state snapshot at this compaction boundary. Later messages supersede it. Memory and workspace notes are agent records, not verified facts.' },
    ...(typeof state.content === 'string' ? [{ type: 'text' as const, text: state.content }] : state.content),
  ];
  const images = await loadImages(workspace, state.images, policy);
  content.push(...labelledImages(images, state.images ?? [], state.image_labels));
  return { role: 'custom', customType: 'public_checkpoint', content,
    timestamp: Date.now(), display: false, details: { revision: state.revision } };
}

/** Pi owns history and summaries; Python supplies public state only at compaction boundaries. */
export function createCheckpointCompaction(sharedStream: StreamFn,
  checkpoint: () => PublicCheckpoint | undefined | Promise<PublicCheckpoint | undefined>,
  trace: CompactionTrace, policy?: ExecutionPolicy, python?: string): ExtensionFactory {
  return pi => {
    let preparedSnapshot: PublicSnapshot | undefined;
    pi.on('session_before_compact', async (event, ctx) => {
      preparedSnapshot = undefined;
      trace.emit('checkpoint_compaction_start', { reason: event.reason, tokensBefore: event.preparation.tokensBefore });
      try {
        if (!ctx.model || !python) throw new Error('Compaction requires model and local tokenizer Python');
        const preparation = keepRecentRounds(ctx.sessionManager.buildSessionProjection().entries, event.preparation);
        if (!preparation) { trace.emit('checkpoint_compaction_skipped', { reason: 'fewer_than_three_complete_rounds' }); return { cancel: true }; }
        // Validate and freeze public content before changing history.
        const state = await checkpoint();
        const snapshot = state ? await publicSnapshot(ctx.cwd, state, policy) : undefined;
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
        preparedSnapshot = snapshot;
        trace.emit('checkpoint_compaction_result', result);
        return { compaction: result };
      } catch (error) {
        trace.emit('checkpoint_compaction_cancelled', { error: String(error) });
        return { cancel: true };
      }
    });
    pi.on('session_compact', event => {
      const snapshot = preparedSnapshot;
      preparedSnapshot = undefined;
      if (!snapshot) return;
      // Append after the retained rounds: this state belongs to the boundary,
      // not to an earlier point in their history.
      pi.sendMessage({ customType: snapshot.customType, content: snapshot.content,
        display: false, details: snapshot.details }, { triggerTurn: false });
      trace.emit('checkpoint_restored', { revision: snapshot.details.revision, compactionId: event.compactionEntry.id });
    });
    pi.on('context', event => {
      // A new compaction snapshot supersedes older snapshots. Keep its original
      // position and bytes; ordinary turns never refresh snapshots or prune images.
      const latest = event.messages.findLastIndex(message => message.role === 'custom' && message.customType === 'public_checkpoint');
      if (latest < 0) return;
      return { messages: event.messages.filter((message, index) =>
        message.role !== 'custom' || message.customType !== 'public_checkpoint' || index === latest) };
    });
  };
}
