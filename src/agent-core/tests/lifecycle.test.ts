import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager, type ResourceLoader } from '@earendil-works/pi-coding-agent';

test('SDK resolves local workspace and creates an isolated empty session', async () => {
  const entry = await realpath(new URL(import.meta.resolve('@earendil-works/pi-coding-agent')).pathname);
  assert.match(entry, /src\/agent-core\/pi\/coding-agent\/dist\/index\.js$/);
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
