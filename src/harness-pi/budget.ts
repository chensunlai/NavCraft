import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname } from 'node:path';
import type { Trace } from './trace.js';

export class RequestBudget {
  readonly start: number;
  constructor(readonly file: string, readonly caseLimit: number, readonly trace: Trace, readonly totalLimit = 200) {
    mkdirSync(dirname(file), { recursive: true });
    if (!existsSync(file)) writeFileSync(file, JSON.stringify({ limit: totalLimit, attempts: [] }, null, 2));
    const state = this.state();
    if (state.limit !== totalLimit) throw new Error('Budget file limit does not match configured global limit');
    this.start = state.attempts.length;
  }
  state(): { limit: number; attempts: { number: number; time: string; url: string; run: string }[] } {
    return JSON.parse(readFileSync(this.file, 'utf8'));
  }
  get used() { return this.state().attempts.length - this.start; }
  get totalUsed() { return this.state().attempts.length; }
  fetch: typeof globalThis.fetch = async (input, init) => {
    const state = this.state();
    if (state.attempts.length >= state.limit || this.used >= this.caseLimit) throw new Error('NAVCRAFT_REQUEST_BUDGET_EXHAUSTED');
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const attempt = { number: state.attempts.length + 1, time: new Date().toISOString(), url, run: this.trace.directory };
    state.attempts.push(attempt); writeFileSync(this.file, JSON.stringify(state, null, 2));
    this.trace.emit('http_request', { ...attempt, body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined });
    const started = Date.now();
    try {
      const response = await globalThis.fetch(input, { ...init, signal: init?.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(300000)]) : AbortSignal.timeout(300000) });
      this.trace.emit('http_response', { number: attempt.number, status: response.status, latency_ms: Date.now() - started });
      return response;
    } catch (error) { this.trace.emit('http_error', { number: attempt.number, error: String(error), latency_ms: Date.now() - started }); throw error; }
  };
}
