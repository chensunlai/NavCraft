import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createAgentSession, createExtensionRuntime, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager, type ResourceLoader, type ToolDefinition, type ExtensionFactory } from '@earendil-works/pi-coding-agent';
import type { Context, Message } from '@earendil-works/pi-ai';
import { createCodingTools, type ExecutionPolicy } from './sandbox.js';
import { Trace } from './trace.js';
import { RequestBudget } from './budget.js';
import { retryService } from './service-retry.js';
import { createCheckpointCompaction, type PublicCheckpoint } from './compaction.js';
import { createImageTransport } from './image-transport.js';
import type { StreamFn, ThinkingLevel } from '@earendil-works/pi-agent-core';

// Pi omits failed assistant turns during transport conversion. Their paired
// receipts must also be omitted; leave the saved session and audit untouched.
export function transportContext<T extends Context>(context: T): T {
  const discarded = new Set<string>();
  const messages = context.messages.filter(message => {
    if (message.role === 'assistant') {
      const failed = message.stopReason === 'error' || message.stopReason === 'aborted';
      for (const block of message.content) {
        if (block.type === 'toolCall') {
          if (failed) discarded.add(block.id);
          else discarded.delete(block.id);
        }
      }
    }
    return message.role !== 'toolResult' || !discarded.has(message.toolCallId);
  });
  return { ...context, messages };
}

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
  thinkingLevel?: ThinkingLevel;
  enableThinking?: boolean;
  sessionFile?: string;
  initialMessages?: Message[];
  coding?: boolean; extensions?: ExtensionFactory[]; compact?: boolean;
  contextWindow?: number; compaction?: { reserveTokens: number; keepRecentTokens: number };
  checkpoint?: () => PublicCheckpoint | undefined | Promise<PublicCheckpoint | undefined>;
  executionPolicy?: ExecutionPolicy; tracePersist?: boolean;
  totalLimit?: number | null; onEvent?: (event: { timestamp: string; type: string; data: unknown }) => void;
}
export async function runtime(options: RuntimeOptions) {
  const auth = options.auth ? { ...options.auth } : undefined;
  if (!auth?.baseUrl || !auth.model || !auth.apiKey) throw new Error('Missing model credentials');
  const requestedModel = auth.model;
  if (options.modelId) auth.model = options.modelId;
  if (options.coding !== false && !options.python) throw new Error('Coding tools require an explicit Python executable');
  const contextWindow = options.contextWindow ?? 32768;
  const compaction = { enabled: options.compact ?? true, reserveTokens: contextWindow - 24000, keepRecentTokens: 6000, ...options.compaction };
  if (![contextWindow, compaction.reserveTokens, compaction.keepRecentTokens].every(Number.isInteger)
      || compaction.reserveTokens < 8192 || compaction.keepRecentTokens <= 0
      || compaction.keepRecentTokens >= contextWindow - compaction.reserveTokens) {
    throw new Error('Invalid context window or compaction settings');
  }
  const trace = new Trace(join(options.directory, 'private'), options.workspace, [auth.apiKey], options.onEvent, options.tracePersist !== false,
    options.executionPolicy ? [...new Set(['memory', ...options.executionPolicy.readWrite])] : undefined);
  const budget = new RequestBudget(options.budgetFile, options.maxRequests, trace, options.totalLimit);
  const agentDir = join(options.directory, 'private', 'pi'); mkdirSync(agentDir, { recursive: true });
  const modelRuntime = await ModelRuntime.create({ authPath: join(agentDir, 'auth.json'), modelsPath: null, allowModelNetwork: false });
  const api = options.api ?? 'openai-responses';
  const thinkingLevel = options.thinkingLevel ?? 'high';
  const thinkingLevelMap = thinkingLevel === 'xhigh' || thinkingLevel === 'max' ? { [thinkingLevel]: thinkingLevel } : undefined;
  // Pi otherwise omits strict entirely for custom Responses providers, letting
  // the service turn optional Python tool arguments into required ones.
  modelRuntime.registerProvider('navcraft', { baseUrl: auth.baseUrl.endsWith('/v1') ? auth.baseUrl : auth.baseUrl + '/v1', api, apiKey: auth.apiKey, models: [{ id: auth.model, name: auth.model, reasoning: true, thinkingLevelMap, input: ['text', 'image'], contextWindow, maxTokens: 8192, ...(api === 'openai-responses' ? { compat: { supportsStrictMode: true } } : {}), cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }] });
  await modelRuntime.setRuntimeApiKey('navcraft', auth.apiKey);
  const tools = [...(options.coding === false ? [] : createCodingTools(options.workspace, options.python!, options.executionPolicy)), ...(options.tools ?? [])];
  const settingsManager = SettingsManager.inMemory({ compaction, retry: { enabled: false }, cacheWarming: 'off' });
  let controlledStream!: StreamFn;
  const factories = [...(options.extensions ?? []), createCheckpointCompaction((...args) => controlledStream(...args),
    options.checkpoint ?? (() => undefined), trace, options.executionPolicy, options.python)];
  let resourceLoader: ResourceLoader = explicitLoader(options.systemPrompt);
  if (factories.length) {
    const loader = new DefaultResourceLoader({ cwd: options.workspace, agentDir, settingsManager, extensionFactories: factories, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: options.systemPrompt });
    await loader.reload();
    if (loader.getExtensions().errors.length) throw new Error(JSON.stringify(loader.getExtensions().errors));
    resourceLoader = loader;
  }
  const sessionManager = options.sessionFile ? SessionManager.open(options.sessionFile, join(agentDir, 'sessions'), options.workspace)
    : SessionManager.create(options.workspace, join(agentDir, 'sessions'));
  if (options.initialMessages?.length) {
    if (options.sessionFile) throw new Error('State history and sessionFile are mutually exclusive');
    for (const message of options.initialMessages) sessionManager.appendMessage(message);
    trace.emit('history_imported', { count: options.initialMessages.length });
  }
  const { session } = await createAgentSession({ cwd: options.workspace, agentDir, model: modelRuntime.getModel('navcraft', auth.model)!, modelRuntime, thinkingLevel, tools: tools.map(t => t.name), customTools: tools, resourceLoader, settingsManager, sessionManager });
  session.agent.toolExecution = 'sequential';
  const originalStream = retryService(session.agent.streamFunction, budget);
  const encodeImages = createImageTransport((type, data) => trace.emit(type, data));
  session.agent.streamFunction = (model, context, streamOptions) => {
    context = encodeImages(transportContext(context));
    trace.emit('model_input', { model: model.id, api: model.api, reasoning: streamOptions?.reasoning, context, files: trace.snapshot('before_model') });
    return originalStream(model, context, { ...streamOptions, maxRetries: 0, maxTokens: streamOptions?.maxTokens ?? 8192, fetch: budget.fetch, timeoutMs: 300000, transport: 'sse',
      ...(options.enableThinking === undefined ? {} : { onPayload: async (payload: unknown) => {
        const previous = await streamOptions?.onPayload?.(payload, model);
        const body = { ...(previous ?? payload) as Record<string, unknown> };
        if (options.enableThinking === false) delete body.reasoning_effort;
        return { ...body, chat_template_kwargs: {
          ...(body.chat_template_kwargs as Record<string, unknown> ?? {}), enable_thinking: options.enableThinking,
        } };
      } }),
    });
  };
  controlledStream = session.agent.streamFunction;
  await session.bindExtensions({});
  session.subscribe(event => {
    if (event.type === 'turn_start' || event.type === 'message_end' || event.type === 'tool_execution_start' || event.type === 'tool_execution_end' || event.type.includes('compaction') || event.type === 'agent_settled') trace.emit(event.type, event);
    if (event.type === 'tool_execution_end') trace.snapshot(`after_tool:${event.toolName}`);
  });
  trace.snapshot('initial');
  trace.emit('runtime_started', { pi: '1.2.0', requestedModel, model: auth.model, api, baseUrl: auth.baseUrl, thinking: session.thinkingLevel, maxTokens: 8192, contextWindow, compaction, tools: tools.map(t => t.name) });
  writeFileSync(join(options.directory, 'config.json'), JSON.stringify({ ...options, initialMessages: undefined, auth: options.auth ? '[PRIVATE]' : undefined, tools: tools.map(t => t.name), extensions: undefined, baseUrl: auth.baseUrl, model: auth.model }, null, 2));
  return { session, trace, budget, settingsManager, modelRuntime };
}
