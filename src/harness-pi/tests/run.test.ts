import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runCase, runFinishReason } from '../run.js';
import { DEFAULT_PYTHON } from '../runtime.js';

test('a final successful request at the exact limit is not budget exhaustion', () => {
  const state = { done: false, goalChanged: false };
  assert.equal(runFinishReason(state, false, { stopReason: 'stop' }), 'model_finished');
  assert.equal(runFinishReason(state, true, { stopReason: 'error' }), 'request_budget');
  assert.equal(runFinishReason(state, false, { stopReason: 'length' }), 'output_limit');
  assert.equal(runFinishReason(state, false, { stopReason: 'error', errorMessage: 'This operation was aborted' }), 'timeout');
  assert.equal(runFinishReason(state, false, { stopReason: 'error', errorMessage: 'Provider unavailable' }), 'model_error');
  assert.equal(runFinishReason(state, false, undefined, 'Invalid image reference'), 'runtime_error');
  assert.equal(runFinishReason({ ...state, done: true, current: { reason: 'local_event_complete' } }, true, undefined), 'local_event_complete');
});

test('initialization failure leaves a result without contacting any model', async () => {
  const repo = mkdtempSync(join(tmpdir(), 'navcraft-run-failure-'));
  mkdirSync(join(repo, 'config/harness-pi'), { recursive: true });
  writeFileSync(join(repo, 'config/harness-pi/smoke.yaml'), 'cases:\n  local:\n    sample: missing-sample\n    requests: 2\n    seconds: 10\n');
  try {
    await assert.rejects(runCase(repo, { output_dir: 'runs', python: DEFAULT_PYTHON,
      scene_archive: 'absent-scene', model: 'fixture', api: 'openai-responses' }, 'local'), /Python prepare/);
    const directory = join(repo, 'runs', readdirSync(join(repo, 'runs'))[0]);
    const result = JSON.parse(readFileSync(join(directory, 'result.json'), 'utf8'));
    assert.equal(result.finish_reason, 'initialization_error');
    assert.equal(result.terminated, false);
    assert.match(result.error, /Python prepare/);
    assert.equal(result.requests, undefined);
  } finally { rmSync(repo, { recursive: true, force: true }); }
});
