import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, unlinkSync, symlinkSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { Trace } from '../trace.js';
test('audit retains earlier bytes, edits, deletions and symlink metadata without following links', () => {
  const dir = mkdtempSync(join(tmpdir(), 'navcraft-audit-')); const workspace = join(dir, 'workspace');
  mkdirSync(join(workspace, 'memory'), { recursive: true }); const trace = new Trace(join(dir, 'private'), workspace, ['private-fixture-key']);
  try {
    const file = join(workspace, 'memory/state.json'); writeFileSync(file, '{"step":0}');
    const initial = trace.snapshot('initial'); writeFileSync(file, '{"step":1}'); trace.snapshot('after-edit');
    symlinkSync('/unavailable/private', join(workspace, 'memory/link')); trace.snapshot('link');
    unlinkSync(file); trace.snapshot('delete'); trace.emit('fixture', { key: 'private-fixture-key' });
    assert.equal(readFileSync(join(dir, 'private/blobs', initial['memory/state.json']), 'utf8'), '{"step":0}');
    const events = readFileSync(join(dir, 'private/trace.jsonl'), 'utf8');
    assert(!events.includes('private-fixture-key'));
    const lines = events.trim().split('\n').map(x => JSON.parse(x));
    assert(lines.some(x => x.data.changes?.some((c: any) => c.path === 'memory/link' && c.after)));
    assert(lines.some(x => x.data.changes?.some((c: any) => c.path === 'memory/state.json' && c.before && !c.after)));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
