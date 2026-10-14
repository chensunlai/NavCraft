import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, unlinkSync } from 'node:fs';
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
    assert.equal(b.exhausted, false);
    await assert.rejects(b.fetch('http://invalid.local'), /BUDGET/);
    assert.equal(b.exhausted, true);
    const restarted = new RequestBudget(join(dir, 'budget.json'), 5, trace, 2);
    await assert.rejects(restarted.fetch('http://invalid.local'), /fixture/);
    await assert.rejects(restarted.fetch('http://invalid.local'), /BUDGET/);
    assert.equal(sent, 2); assert.equal(restarted.totalUsed, 2);
  } finally { globalThis.fetch = old; rmSync(dir, { recursive: true, force: true }); }
});


test('separate runs share the global ceiling and an occupied ledger lock fails closed', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-budget-lock-')); const old = globalThis.fetch;
  let sent = 0; globalThis.fetch = async () => { sent++; return new Response('{}'); };
  try {
    const file = join(dir, 'budget.json');
    const a = new RequestBudget(file, 2, new Trace(join(dir, 'a'), dir), 2);
    const b = new RequestBudget(file, 2, new Trace(join(dir, 'b'), dir), 2);
    writeFileSync(file + '.lock', 'fixture');
    await assert.rejects(a.fetch('http://invalid.local'), /locked/);
    assert.equal(sent, 0); unlinkSync(file + '.lock');
    await a.fetch('http://invalid.local'); await b.fetch('http://invalid.local');
    assert.equal(a.used, 1); assert.equal(b.used, 1);
    assert.equal(a.exhausted, false); assert.equal(b.exhausted, false);
    await assert.rejects(a.fetch('http://invalid.local'), /BUDGET/);
    assert.equal(sent, 2);
  } finally { globalThis.fetch = old; rmSync(dir, { recursive: true, force: true }); }
});

test('an unlimited global ledger persists attempts while each run has its own hard ceiling', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-budget-unlimited-')); const old = globalThis.fetch;
  let sent = 0; globalThis.fetch = async () => { sent++; return new Response('{}'); };
  try {
    const file = join(dir, 'budget.json');
    const first = new RequestBudget(file, 1, new Trace(join(dir, 'first'), dir), null);
    await first.fetch('http://invalid.local');
    await assert.rejects(first.fetch('http://invalid.local'), /BUDGET/);
    const second = new RequestBudget(file, 2, new Trace(join(dir, 'second'), dir), null);
    await second.fetch('http://invalid.local'); await second.fetch('http://invalid.local');
    assert.equal(second.state().limit, null);
    assert.equal(second.totalUsed, 3); assert.equal(second.used, 2); assert.equal(sent, 3);
    assert.throws(() => new RequestBudget(file, 10, new Trace(join(dir, 'wrong'), dir), 200), /limit does not match/);
  } finally { globalThis.fetch = old; rmSync(dir, { recursive: true, force: true }); }
});
