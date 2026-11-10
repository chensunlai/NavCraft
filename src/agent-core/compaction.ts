import { spawn } from 'node:child_process';
import type { AgentMessage, StreamFn } from '@earendil-works/pi-agent-core';
import { compact, type ExtensionFactory, type ProjectedSessionEntry } from '@earendil-works/pi-coding-agent';

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

/** Move the native cut point to the start of the last two complete tool rounds. */
export function keepRecentRounds(entries: ProjectedSessionEntry[], preparation: CompactionPreparation): CompactionPreparation | undefined {
  const projected = entries.flatMap(entry => entry.messages.filter(message => message.role !== 'system' && entry.sourceEntry.type !== 'compaction').map(message => ({ message, id: entry.sourceEntry.id })));
  const messages = projected.map(entry => entry.message);
  const rounds = completeRoundStarts(messages);
  if (rounds.length <= 2 || rounds.length !== messages.filter(message => message.role === 'assistant').length) return undefined;
  const cut = rounds[rounds.length - 2];
  const older = messages.slice(0, cut);
  if (!older.length) return undefined;
  return { ...preparation, firstKeptEntryId: projected[cut].id,
    messagesToSummarize: older, turnPrefixMessages: [], isSplitTurn: false };
}

/** Count visible text with the application's fixed local tokenizer, on CPU only. */
export function countSummaryTokens(python: string, text: string, signal?: AbortSignal): Promise<number> {
  return new Promise((accept, reject) => {
    const child = spawn(python, ['-m', 'src.harness.runtime.tokenizer'], { cwd: process.cwd(),
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

/** Keep Pi's short summary and complete recent rounds; task state is read through tools. */
export function createNavigationCompaction(sharedStream: StreamFn, trace: CompactionTrace,
  python?: string): ExtensionFactory {
  return pi => {
    let completedTurns = 0, retryAfterTurn = 0;
    pi.on('turn_end', async () => { completedTurns++; });
    pi.on('session_before_compact', async (event, ctx) => {
      if (event.reason === 'threshold' && completedTurns < retryAfterTurn) {
        trace.emit('summary_compaction_skipped', { reason: 'failure_cooldown', remaining_turns: retryAfterTurn - completedTurns });
        return { cancel: true };
      }
      trace.emit('summary_compaction_start', { reason: event.reason, tokensBefore: event.preparation.tokensBefore });
      try {
        if (!ctx.model || !python) throw new Error('Compaction requires model and local tokenizer Python');
        const preparation = keepRecentRounds(ctx.sessionManager.buildSessionProjection().entries, event.preparation);
        if (!preparation) { trace.emit('summary_compaction_skipped', { reason: 'fewer_than_three_complete_rounds' }); return { cancel: true }; }
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
        if (!auth.ok) throw new Error(auth.error);
        const headers = auth.headers ? Object.fromEntries(Object.entries(auth.headers).filter((entry): entry is [string, string] => typeof entry[1] === 'string')) : undefined;
        const stream: StreamFn = async (model, context, options) => {
          trace.emit('summary_input', { model: model.id, context });
          const result = await sharedStream(model, context, { ...options, maxTokens: 8192, maxRetries: 0 });
          const original = result.result.bind(result);
          result.result = async () => {
            const message = await original();
            trace.emit('summary_output', message);
            return message;
          };
          return result;
        };
        const result = await compact(preparation, auth.baseUrl ? { ...ctx.model, baseUrl: auth.baseUrl } : ctx.model,
          auth.apiKey, headers, event.customInstructions, event.signal, ctx.thinkingLevel, stream,
          auth.env, { enabled: false, maxRetries: 0, baseDelayMs: 0 }, undefined, ctx.sessionManager.getSessionId(),
          { targetVisibleTokens: 1000, maxVisibleTokens: 2000, countTokens: async text => {
            const count = await countSummaryTokens(python, text, event.signal);
            trace.emit('summary_length', { visible_tokens: count, target: 1000, limit: 2000 });
            return count;
          } });
        event.signal.throwIfAborted();
        retryAfterTurn = 0;
        trace.emit('summary_compaction_result', result);
        return { compaction: result };
      } catch (error) {
        retryAfterTurn = completedTurns + 2;
        trace.emit('summary_compaction_cancelled', { error: String(error) });
        return { cancel: true };
      }
    });
  };
}
