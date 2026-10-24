import { appendFileSync, mkdirSync, readFileSync, readdirSync, lstatSync, writeFileSync, existsSync, readlinkSync } from 'node:fs';
import { join, relative } from 'node:path';
import { createHash } from 'node:crypto';

export class Trace {
  private previous = new Map<string, string>();
  private secrets: string[];
  constructor(readonly directory: string, readonly workspace: string, secrets: string[] = [], private onEvent?: (event: { timestamp: string; type: string; data: unknown }) => void, private persist = true, private trackedPaths = ['memory', 'scratch']) {
    this.secrets = secrets.filter(Boolean);
    mkdirSync(directory, { recursive: true });
    mkdirSync(join(directory, 'blobs'), { recursive: true });
  }
  private blob(bytes: Buffer, encoding: string, prefix?: string) {
    const hash = createHash('sha256').update(bytes).digest('hex');
    const path = join(this.directory, 'blobs', hash);
    if (!existsSync(path)) writeFileSync(path, bytes);
    return { $trace_blob: hash, encoding, ...(prefix === undefined ? {} : { prefix }) };
  }
  private archive(value: any, messages = false): any {
    if (typeof value === 'string') {
      const image = /^(data:image\/[^;,]+;base64,)([A-Za-z0-9+/]*={0,2})$/.exec(value);
      if (image) return this.blob(Buffer.from(image[2], 'base64'), 'data-url', image[1]);
      for (const secret of this.secrets) value = value.split(secret).join('[REDACTED]');
      return value;
    }
    if (Array.isArray(value)) return value.map(item => this.archive(item, messages));
    if (!value || typeof value !== 'object') return value;
    const result: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
      if (value.type === 'image' && key === 'data' && typeof child === 'string'
          && /^[A-Za-z0-9+/]*={0,2}$/.test(child)) {
        result[key] = this.blob(Buffer.from(child, 'base64'), 'base64');
      } else if (messages && (key === 'messages' || key === 'input') && Array.isArray(child)) {
        result[key] = child.map(message => message?.$trace_blob ? message
          : this.blob(Buffer.from(JSON.stringify(this.archive(message))), 'json'));
      } else {
        result[key] = this.archive(child, messages);
      }
    }
    return result;
  }
  emit(type: string, data: unknown = {}) {
    // Compact only the audit copy. Pi's messages and the outgoing request stay untouched.
    const archived = this.archive(data, ['model_input', 'http_request', 'checkpoint_summary_input'].includes(type));
    const line = JSON.stringify({ timestamp: new Date().toISOString(), type, data: archived });
    if (this.persist) appendFileSync(join(this.directory, 'trace.jsonl'), line + '\n');
    this.onEvent?.(JSON.parse(line));
  }
  snapshot(label: string) {
    const next = new Map<string, string>();
    const changes: { path: string; before?: string; after?: string }[] = [];
    const walk = (directory: string) => {
      if (!existsSync(directory)) return;
      for (const item of readdirSync(directory)) {
        const path = join(directory, item); const stat = lstatSync(path);
        if (stat.isDirectory() && !stat.isSymbolicLink()) { walk(path); continue; }
        if (!stat.isFile() && !stat.isSymbolicLink()) continue;
        const bytes = stat.isSymbolicLink() ? Buffer.from(JSON.stringify({ type: 'symlink', target: readlinkSync(path) })) : readFileSync(path); const hash = createHash('sha256').update(bytes).digest('hex');
        const ref = relative(this.workspace, path); next.set(ref, hash);
        if (!existsSync(join(this.directory, 'blobs', hash))) writeFileSync(join(this.directory, 'blobs', hash), bytes);
        if (this.previous.get(ref) !== hash) changes.push({ path: ref, before: this.previous.get(ref), after: hash });
      }
    };
    for (const name of this.trackedPaths) walk(join(this.workspace, name));
    for (const [path, hash] of this.previous) if (!next.has(path)) changes.push({ path, before: hash });
    this.previous = next;
    if (changes.length) this.emit('workspace_changes', { label, changes });
    return Object.fromEntries(next);
  }
}
