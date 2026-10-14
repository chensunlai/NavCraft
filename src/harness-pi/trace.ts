import { appendFileSync, mkdirSync, readFileSync, readdirSync, lstatSync, writeFileSync, existsSync, readlinkSync } from 'node:fs';
import { join, relative } from 'node:path';
import { createHash } from 'node:crypto';

export class Trace {
  private previous = new Map<string, string>();
  private secrets: string[];
  constructor(readonly directory: string, readonly workspace: string, secrets: string[] = []) {
    this.secrets = secrets.filter(Boolean);
    mkdirSync(directory, { recursive: true });
    mkdirSync(join(directory, 'blobs'), { recursive: true });
  }
  emit(type: string, data: unknown = {}) {
    let line = JSON.stringify({ timestamp: new Date().toISOString(), type, data });
    for (const secret of this.secrets) line = line.split(secret).join('[REDACTED]');
    appendFileSync(join(this.directory, 'trace.jsonl'), line + '\n');
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
    for (const name of ['memory', 'scratch']) walk(join(this.workspace, name));
    for (const [path, hash] of this.previous) if (!next.has(path)) changes.push({ path, before: hash });
    this.previous = next;
    if (changes.length) this.emit('workspace_changes', { label, changes });
    return Object.fromEntries(next);
  }
}
