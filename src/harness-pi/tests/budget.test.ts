import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Trace } from '../trace.js';
import { RequestBudget } from '../budget.js';
test('HTTP attempts include failed requests and restart-safe global/case limits', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-budget-')); const old = globalThis.fetch;
  let sent = 0; globalThis.fetch = async () => { sent++; throw new Error('fixture failure'); };
  try {
    const trace = new Trace(join(dir, 'private'), dir);
    const b = new RequestBudget(join(dir, 'budget.json'), 1, trace, 2);
    await assert.rejects(b.fetch('http://invalid.local'), /fixture/);
    await assert.rejects(b.fetch('http://invalid.local'), /BUDGET/);
    const restarted = new RequestBudget(join(dir, 'budget.json'), 5, trace, 2);
    await assert.rejects(restarted.fetch('http://invalid.local'), /fixture/);
    await assert.rejects(restarted.fetch('http://invalid.local'), /BUDGET/);
    assert.equal(sent, 2); assert.equal(restarted.totalUsed, 2);
  } finally { globalThis.fetch = old; rmSync(dir, { recursive: true, force: true }); }
});
