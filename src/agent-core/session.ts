import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createAgentSession, createExtensionRuntime, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager, type ResourceLoader, type ToolDefinition, type ExtensionFactory, type AgentSession } from '@earendil-works/pi-coding-agent';
import type { ImageContent } from '@earendil-works/pi-ai';
import { createCodingTools, type ExecutionPolicy } from './sandbox.js';
import { Trace } from './trace.js';
import { RequestBudget } from './budget.js';
import { createCheckpointCompaction, type PublicCheckpoint } from './compaction.js';
import type { StreamFn } from '@earendil-works/pi-agent-core';

export const DEFAULT_PYTHON = '/home2/chen/.local/anaconda3/envs/vln/bin/python';
export function explicitLoader(systemPrompt: string): ResourceLoader {
  return {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }), getPrompts: () => ({ prompts: [], diagnostics: [] }), getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }), getSystemPrompt: () => systemPrompt, getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [], getAppendSystemPromptSources: () => [], extendResources: () => {}, reload: async () => {},
  };
}
export interface RuntimeOptions {
  workspace: string; directory: string; budgetFile: string; maxRequests: number;
  auth: { baseUrl: string; model: string; apiKey: string };
  modelId?: string; api?: 'openai-responses' | 'openai-completions'; python?: string; systemPrompt: string; tools?: ToolDefinition[];
  coding?: boolean; timeoutSeconds?: number; extensions?: ExtensionFactory[]; compact?: boolean;
  checkpoint?: () => PublicCheckpoint | undefined | Promise<PublicCheckpoint | undefined>;
  executionPolicy?: ExecutionPolicy; tracePersist?: boolean;
  totalLimit?: number | null; onEvent?: (event: { timestamp: string; type: string; data: unknown }) => void;
}
export async function runtime(options: RuntimeOptions) {
  const auth = options.auth ? { ...options.auth } : undefined;
  if (!auth?.baseUrl || !auth.model || !auth.apiKey) throw new Error('Missing model credentials');
  const requestedModel = auth.model;
  if (options.modelId) auth.model = options.modelId;
  const trace = new Trace(join(options.directory, 'private'), options.workspace, [auth.apiKey], options.onEvent, options.tracePersist !== false, options.executionPolicy?.readWrite);
  const budget = new RequestBudget(options.budgetFile, options.maxRequests, trace, options.totalLimit);
  const agentDir = join(options.directory, 'private', 'pi'); mkdirSync(agentDir, { recursive: true });
  const modelRuntime = await ModelRuntime.create({ authPath: join(agentDir, 'auth.json'), modelsPath: null, allowModelNetwork: false });
  const api = options.api ?? 'openai-responses';
  modelRuntime.registerProvider('navcraft', { baseUrl: auth.baseUrl.endsWith('/v1') ? auth.baseUrl : auth.baseUrl + '/v1', api, apiKey: auth.apiKey, models: [{ id: auth.model, name: auth.model, reasoning: true, input: ['text', 'image'], contextWindow: 65536, maxTokens: 8192, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }] });
  await modelRuntime.setRuntimeApiKey('navcraft', auth.apiKey);
  const tools = [...(options.coding === false ? [] : createCodingTools(options.workspace, options.python ?? DEFAULT_PYTHON, options.executionPolicy)), ...(options.tools ?? [])];
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: options.compact ?? false, reserveTokens: 16384, keepRecentTokens: 20000 }, retry: { enabled: false }, cacheWarming: 'off' });
  let controlledStream!: StreamFn;
  const factories = [...(options.extensions ?? []), ...(options.checkpoint ? [createCheckpointCompaction((...args) => controlledStream(...args), options.checkpoint, trace, options.executionPolicy)] : [])];
  let resourceLoader: ResourceLoader = explicitLoader(options.systemPrompt);
  if (factories.length) {
    const loader = new DefaultResourceLoader({ cwd: options.workspace, agentDir, settingsManager, extensionFactories: factories, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: options.systemPrompt });
    await loader.reload();
    if (loader.getExtensions().errors.length) throw new Error(JSON.stringify(loader.getExtensions().errors));
    resourceLoader = loader;
  }
  const { session } = await createAgentSession({ cwd: options.workspace, agentDir, model: modelRuntime.getModel('navcraft', auth.model)!, modelRuntime, thinkingLevel: 'high', tools: tools.map(t => t.name), customTools: tools, resourceLoader, settingsManager, sessionManager: SessionManager.create(options.workspace, join(agentDir, 'sessions')) });
  session.agent.toolExecution = 'sequential';
  const originalStream = session.agent.streamFunction;
  session.agent.streamFunction = (model, context, streamOptions) => {
    trace.emit('model_input', { model: model.id, api: model.api, reasoning: streamOptions?.reasoning, context, files: trace.snapshot('before_model') });
    return originalStream(model, context, { ...streamOptions, maxRetries: 0, maxTokens: 8192, fetch: budget.fetch, timeoutMs: 300000, transport: 'sse' });
  };
  controlledStream = session.agent.streamFunction;
  await session.bindExtensions({});
  session.subscribe(event => {
    if (event.type === 'turn_start' || event.type === 'message_end' || event.type === 'tool_execution_start' || event.type === 'tool_execution_end' || event.type.includes('compaction') || event.type === 'agent_settled') trace.emit(event.type, event);
    if (event.type === 'tool_execution_end') trace.snapshot(`after_tool:${event.toolName}`);
  });
  trace.snapshot('initial');
  trace.emit('runtime_started', { pi: '1.2.0', requestedModel, model: auth.model, api, baseUrl: auth.baseUrl, thinking: 'high', maxTokens: 8192, contextWindow: 65536, tools: tools.map(t => t.name) });
  writeFileSync(join(options.directory, 'config.json'), JSON.stringify({ ...options, auth: options.auth ? '[PRIVATE]' : undefined, tools: tools.map(t => t.name), extensions: undefined, baseUrl: auth.baseUrl, model: auth.model }, null, 2));
  return { session, trace, budget, settingsManager, modelRuntime };
}
export function imageContent(workspace: string, refs: string[]): ImageContent[] {
  return refs.map(ref => ({ type: 'image', data: readFileSync(join(workspace, ref)).toString('base64'), mimeType: ref.endsWith('.jpg') || ref.endsWith('.jpeg') ? 'image/jpeg' : 'image/png' }));
}
export async function promptBounded(session: AgentSession, text: string, images: ImageContent[], seconds: number) {
  const timer = setTimeout(() => { void session.abort(); }, seconds * 1000);
  try { await session.prompt(text, { images }); }
  finally { clearTimeout(timer); }
}
