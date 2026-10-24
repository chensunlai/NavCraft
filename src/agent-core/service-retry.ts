import { setTimeout as sleep } from 'node:timers/promises';
import { createAssistantMessageEventStream, type AssistantMessageEvent, type AssistantMessage } from '@earendil-works/pi-ai';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import type { RequestBudget } from './budget.js';

export function serviceUnavailable(message: string): boolean {
  // Some gateways return account limits inside an HTTP 200 SSE response.
  if (/gateway_concurrency_limit|concurrency limit exceeded for account/i.test(message)) return true;
  const code = message.match(/(?:API error\s*\(|\bHTTP\s+)(\d{3})/i)?.[1];
  if (code) return [408, 429, 500, 502, 503, 504, 529].includes(Number(code));
  return /(?:service|upstream|servers?).*?(?:temporarily unavailable|unavailable|overloaded)|(?:upstream response )?stream (?:was interrupted|interrupted|ended unexpectedly)|fetch failed|connection (?:error|reset)|request timed out|operation was aborted due to timeout/i.test(message);
}

// Buffer one bounded model response: a failed stream may already contain partial
// tool calls. Only a completed attempt may enter Pi history or execute tools.
export function retryService(stream: StreamFn, budget: RequestBudget,
  wait: (ms: number, signal?: AbortSignal) => Promise<unknown> = (ms, signal) => sleep(ms, undefined, { signal })): StreamFn {
  return (model, context, options) => {
    const output = createAssistantMessageEventStream();
    void (async () => {
      let retries = 0;
      let last: AssistantMessage | undefined;
      try {
        while (true) {
          const started = Date.now(), from = budget.totalUsed;
          const events: AssistantMessageEvent[] = [];
          const attempt = await stream(model, context, options);
          for await (const event of attempt) {
            if (event.type === 'start') budget.endServiceWait(started);
            events.push(event);
          }
          last = await attempt.result();
          if (budget.exhausted || last.stopReason !== 'error' || !serviceUnavailable(last.errorMessage ?? '') || options?.signal?.aborted) {
            budget.endServiceWait(started);
            for (const event of events) output.push(event);
            output.end(last);
            return;
          }
          budget.serviceFailure(from, started);
          const delay = Math.min(60000, 10000 * 2 ** Math.min(retries++, 3));
          budget.trace.emit('service_retry', { attempt: retries, delay_ms: delay, error: last,
            requests: budget.used, budget_requests: budget.charged });
          await wait(delay, options?.signal);
        }
      } catch (error) {
        const aborted = Boolean(options?.signal?.aborted);
        const failure: AssistantMessage = { ...(last ?? {
          role: 'assistant', content: [], api: model.api, provider: model.provider, model: model.id,
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, timestamp: Date.now(),
        }), content: [], stopReason: aborted ? 'aborted' : 'error', errorMessage: String(error) };
        output.push({ type: 'error', reason: failure.stopReason as 'error' | 'aborted', error: failure });
        output.end(failure);
      } finally { budget.endServiceWait(); }
    })();
    return output;
  };
}
