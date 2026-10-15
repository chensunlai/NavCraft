import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager, shouldCompact, type ResourceLoader } from '@earendil-works/pi-coding-agent';
import { runtime } from '../session.js';

test('SDK resolves local workspace and creates an isolated empty session', async () => {
  const entry = await realpath(new URL(import.meta.resolve('@earendil-works/pi-coding-agent')).pathname);
  assert.match(entry, /src\/agent-core\/pi\/packages\/coding-agent\/dist\/index\.js$/);
  const cwd = await mkdtemp(join(tmpdir(), 'navcraft-lifecycle-'));
  const resourceLoader: ResourceLoader = {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }), getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }), getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => 'Navigation test.', getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [], getAppendSystemPromptSources: () => [], extendResources: () => {}, reload: async () => {},
  };
  const modelRuntime = await ModelRuntime.create({ authPath: join(cwd, 'auth.json'), modelsPath: null, allowModelNetwork: false });
  modelRuntime.registerProvider('navcraft-test', { baseUrl: 'http://invalid.local/v1', api: 'openai-responses', apiKey: 'local-fixture', models: [{ id: 'fixture', name: 'fixture', reasoning: true, input: ['text', 'image'], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 65536, maxTokens: 8192 }] });
  const model = modelRuntime.getModel('navcraft-test', 'fixture')!;
  const { session } = await createAgentSession({ cwd, agentDir: cwd, modelRuntime, model, tools: [], resourceLoader, sessionManager: SessionManager.inMemory(cwd), settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }) });
  assert.deepEqual(session.getActiveToolNames(), []);
  session.dispose();
  await rm(cwd, { recursive: true, force: true });
});

test('runtime applies the configured window and Pi compacts at the intended threshold', async () => {
  const cwd = await mkdtemp(join(tmpdir(), 'navcraft-context-'));
  for (const configuration of [
    { options: {}, window: 128000, threshold: 100000, recent: 20000 },
    { options: { contextWindow: 96000, compaction: { reserveTokens: 26000, keepRecentTokens: 12000 } }, window: 96000, threshold: 70000, recent: 12000 },
  ]) {
    const active = await runtime({ workspace: cwd, directory: cwd, budgetFile: join(cwd, 'requests.json'), maxRequests: 2,
      auth: { baseUrl: 'http://fixture.invalid', model: 'fixture', apiKey: 'local-fixture' },
      systemPrompt: 'Context test.', coding: false, compact: true, ...configuration.options });
    try {
      const window = active.modelRuntime.getModel('navcraft', 'fixture')!.contextWindow;
      const settings = active.settingsManager.getCompactionSettings();
      assert.equal(window, configuration.window);
      assert.equal(settings.keepRecentTokens, configuration.recent);
      assert.equal(shouldCompact(configuration.threshold, window, settings), false);
      assert.equal(shouldCompact(configuration.threshold + 1, window, settings), true);
      assert.equal(active.budget.used, 0);
    } finally { active.session.dispose(); }
  }
  await rm(cwd, { recursive: true, force: true });
});
