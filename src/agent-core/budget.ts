import { readFileSync, writeFileSync, mkdirSync, existsSync, openSync, closeSync, unlinkSync, renameSync } from 'node:fs';
import { dirname } from 'node:path';
import type { Trace } from './trace.js';

export class RequestBudget {
  exhausted = false;
  constructor(readonly file: string, public caseLimit: number, readonly trace: Trace, readonly totalLimit: number | null = null) {
    mkdirSync(dirname(file), { recursive: true });
    if (!existsSync(file)) writeFileSync(file, JSON.stringify({ limit: totalLimit, attempts: [] }, null, 2));
    const state = this.state();
    if (state.limit !== totalLimit) throw new Error('Budget file limit does not match configured global limit');
  }
  state(): { limit: number | null; attempts: { number: number; time: string; url: string; run: string; service_error?: boolean }[] } {
    return JSON.parse(readFileSync(this.file, 'utf8'));
  }
  get used() { return this.state().attempts.filter(x => x.run === this.trace.directory).length; }
  get totalUsed() { return this.state().attempts.length; }
  get charged() { return this.state().attempts.filter(x => x.run === this.trace.directory && !x.service_error).length; }
  private save(value: unknown, file = this.file) {
    writeFileSync(file + '.tmp', JSON.stringify(value, null, 2)); renameSync(file + '.tmp', file);
  }
  private get waitFile() { return this.file.replace(/[^/]+$/, 'service-retry.json'); }
  private waitState(): { completed_ms: number; started_ms: number | null } {
    return existsSync(this.waitFile) ? JSON.parse(readFileSync(this.waitFile, 'utf8')) : { completed_ms: 0, started_ms: null };
  }
  get serviceWaitMs() {
    const state = this.waitState();
    return state.completed_ms + (state.started_ms === null ? 0 : Math.max(0, Date.now() - state.started_ms));
  }
  serviceFailure(from: number, started: number) {
    const state = this.state();
    for (const attempt of state.attempts.slice(from)) attempt.service_error = true;
    this.save(state);
    const wait = this.waitState();
    if (wait.started_ms === null) wait.started_ms = started;
    this.save(wait, this.waitFile);
  }
  endServiceWait(at = Date.now()) {
    const wait = this.waitState();
    if (wait.started_ms === null) return;
    wait.completed_ms += Math.max(0, at - wait.started_ms); wait.started_ms = null;
    this.save(wait, this.waitFile);
  }
  fetch: typeof globalThis.fetch = async (input, init) => {
    const lock = this.file + '.lock';
    let fd: number;
    try { fd = openSync(lock, 'wx'); } catch { throw new Error('Request budget is locked by another writer; no request was sent'); }
    let state: ReturnType<RequestBudget['state']>;
    try {
      state = this.state();
      if ((state.limit !== null && state.attempts.filter(x => !x.service_error).length >= state.limit) || this.charged >= this.caseLimit) {
        this.exhausted = true;
        this.trace.emit('request_budget_exhausted', { totalUsed: state.attempts.length, caseUsed: this.used, caseLimit: this.caseLimit, totalLimit: state.limit });
        throw new Error('NAVCRAFT_REQUEST_BUDGET_EXHAUSTED');
      }
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const attempt = { number: state.attempts.length + 1, time: new Date().toISOString(), url, run: this.trace.directory };
    state.attempts.push(attempt); this.save(state);
    } finally { closeSync(fd); unlinkSync(lock); }
    const attempt = state.attempts.at(-1)!;
    const url = attempt.url;
    this.trace.emit('http_request', { ...attempt,
      body_bytes: typeof init?.body === 'string' ? Buffer.byteLength(init.body) : undefined,
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined });
    const started = Date.now();
    try {
      const response = await globalThis.fetch(input, { ...init, signal: init?.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(300000)]) : AbortSignal.timeout(300000) });
      this.trace.emit('http_response', { number: attempt.number, status: response.status, latency_ms: Date.now() - started });
      return response;
    } catch (error) { this.trace.emit('http_error', { number: attempt.number, error: String(error), latency_ms: Date.now() - started }); throw error; }
  };
}
