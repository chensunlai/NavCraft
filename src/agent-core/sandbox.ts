import { spawn } from "node:child_process";
import { constants } from "node:os";
import { lstat, readFile, realpath, unlink } from "node:fs/promises";
import { existsSync, realpathSync } from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createBashToolDefinition,
  createEditToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
  defineTool,
  type ExtensionToolContext,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";

export interface SandboxRunOptions {
  /** Seconds, matching Pi's native bash tool. */
  timeout?: number;
  signal?: AbortSignal;
  onData?: (data: Buffer) => void;
  policy?: ExecutionPolicy;
}

export interface ExecutionPolicy {
  /** Public workspace paths, relative to its root. */
  readOnly: string[];
  readWrite: string[];
}

export interface SandboxResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function inside(parent: string, child: string): boolean {
  const rel = relative(parent, child);
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

/** The kernel file view is shared by every Coding operation, not just bash. */
function bwrapArgs(workspace: string, python: string, command: string, policy?: ExecutionPolicy): string[] {
  const root = realpathSync(workspace);
  const runtime = realpathSync(dirname(dirname(python)));
  if (policy) {
    for (const readonly of policy.readOnly) for (const writable of policy.readWrite) {
      const a = resolve(root, readonly), b = resolve(root, writable);
      if (inside(a, b) || inside(b, a)) throw new Error("Read-only and writable workspace paths must not overlap");
    }
  }
  const args = ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv"];
  for (const mount of ["/usr", "/bin", "/lib", "/lib64"]) {
    if (existsSync(mount)) args.push("--ro-bind", mount, mount);
  }
  args.push("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp");
  for (const mount of ["/etc/ld.so.cache", "/etc/fonts"]) {
    if (existsSync(mount)) args.push("--ro-bind", mount, mount);
  }
  if (!inside("/usr", runtime)) args.push("--ro-bind", runtime, runtime);
  if (policy) {
    args.push("--tmpfs", root);
    for (const readable of policy.readOnly) {
      const path = resolve(root, readable);
      if (!inside(root, path) || realpathSync(path) !== path) throw new Error(`Invalid read-only workspace path: ${readable}`);
      args.push("--ro-bind", path, path);
    }
  } else args.push("--ro-bind", root, root);
  for (const writable of policy?.readWrite ?? ["memory", "scratch"]) {
    const path = resolve(root, writable);
    if (!inside(root, path) || path === root || !existsSync(path) || realpathSync(path) !== path) {
      throw new Error(`Sandbox requires a real writable directory: ${path}`);
    }
    args.push("--bind", path, path);
  }
  if (policy) args.push("--remount-ro", root);
  args.push(
    "--setenv", "PATH", `${runtime}/bin:/usr/bin:/bin`,
    "--setenv", "HOME", "/tmp",
    "--setenv", "TMPDIR", "/tmp",
    "--setenv", "LANG", "C.UTF-8",
    "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
    "--setenv", "PYTHONPATH", join(root, "scene"),
    "--chdir", root,
    "--", "/bin/bash", "--noprofile", "--norc", "-c", command,
  );
  return args;
}

async function run(
  workspace: string,
  python: string,
  command: string,
  options: SandboxRunOptions = {},
  input?: string,
): Promise<SandboxResult> {
  if (options.signal?.aborted) throw new Error("aborted");
  if (options.timeout !== undefined && (!Number.isFinite(options.timeout) || options.timeout <= 0)) {
    throw new Error("Invalid timeout: must be a finite number of seconds");
  }
  const args = bwrapArgs(workspace, python, command, options.policy);
  return new Promise((accept, reject) => {
    // The launcher receives no model service credentials either.
    const child = spawn("/usr/bin/bwrap", args, {
      env: { PATH: "/usr/bin:/bin" },
      detached: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let timedOut = false;
    let settled = false;
    let timeout: NodeJS.Timeout | undefined;
    const kill = () => {
      if (!child.pid) return;
      try { process.kill(-child.pid, "SIGKILL"); } catch { /* exited */ }
    };
    const cleanup = () => {
      if (timeout) clearTimeout(timeout);
      options.signal?.removeEventListener("abort", kill);
    };
    child.once("error", (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(`Sandbox execution failed: ${error.message}`));
    });
    child.stdout.on("data", (data: Buffer) => {
      stdout.push(data);
      options.onData?.(data);
    });
    child.stderr.on("data", (data: Buffer) => {
      stderr.push(data);
      options.onData?.(data);
    });
    child.stdin.on("error", () => {});
    child.stdin.end(input);
    if (options.timeout !== undefined) {
      timeout = setTimeout(() => { timedOut = true; kill(); }, options.timeout * 1000);
    }
    options.signal?.addEventListener("abort", kill, { once: true });
    if (options.signal?.aborted) kill();
    child.once("close", (code, signal) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (options.signal?.aborted) return reject(new Error("aborted"));
      if (timedOut) return reject(new Error(`timeout:${options.timeout}`));
      const exitCode = code ?? (signal ? 128 + (constants.signals[signal] ?? 0) : 1);
      accept({ stdout: Buffer.concat(stdout).toString("utf8"), stderr: Buffer.concat(stderr).toString("utf8"), exitCode });
    });
  });
}

export function sandboxRun(
  workspace: string,
  python: string,
  command: string,
  options: SandboxRunOptions = {},
): Promise<SandboxResult> {
  return run(workspace, python, command, options);
}

// Native tools otherwise probe the host path before invoking their operations.
// Validate arguments first; actual file contents are accessed only inside bwrap.
async function safePath(workspace: string, path: string, writable: boolean, policy?: ExecutionPolicy): Promise<string> {
  if (typeof path !== "string" || path.includes("\0") || path.startsWith("~")) {
    throw new Error("Path must be inside the public workspace");
  }
  const root = await realpath(workspace);
  let normalized = path.replace(/[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g, " ");
  if (normalized.startsWith("@")) normalized = normalized.slice(1);
  if (normalized.startsWith("file://")) normalized = fileURLToPath(normalized);
  const logicalRoot = resolve(workspace);
  const absolute = resolve(logicalRoot, normalized);
  if (!inside(logicalRoot, absolute) && !inside(root, absolute)) {
    throw new Error("Path must be inside the public workspace");
  }
  const mapped = inside(logicalRoot, absolute) ? resolve(root, relative(logicalRoot, absolute)) : absolute;
  let existing = mapped;
  while (true) {
    try {
      // lstat notices dangling links, which realpath alone cannot distinguish.
      await lstat(existing);
      const canonical = await realpath(existing);
      if (!inside(root, canonical)) throw new Error("Symlink escapes the public workspace");
      const effective = resolve(canonical, relative(existing, mapped));
      const permitted = writable ? policy?.readWrite ?? ["memory", "scratch"] : policy ? [...policy.readOnly, ...policy.readWrite] : undefined;
      if (permitted && !permitted.some((dir) => inside(resolve(root, dir), effective))) {
        throw new Error(writable ? "Path is outside the writable workspace paths" : "Path is outside the readable workspace paths");
      }
      return mapped;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      // A dangling symlink cannot be accepted as a not-yet-created directory.
      try {
        if ((await lstat(existing)).isSymbolicLink()) throw new Error("Dangling symlink is not an accessible path");
      } catch (linkError) {
        if ((linkError as NodeJS.ErrnoException).code !== "ENOENT") throw linkError;
      }
      if (existing === root) throw error;
      existing = dirname(existing);
    }
  }
}

export function publicPath(workspace: string, path: string, writable = false, policy?: ExecutionPolicy): Promise<string> {
  return safePath(workspace, path, writable, policy);
}

const FILE_OPERATION = `
import base64, json, os, pathlib, sys
r=json.load(sys.stdin)
p=pathlib.Path(r['path'])
op=r['op']
if op == 'read':
    print(base64.b64encode(p.read_bytes()).decode())
elif op == 'write':
    p.write_bytes(base64.b64decode(r['content']))
elif op == 'mkdir':
    p.mkdir(parents=True, exist_ok=True)
elif op == 'access':
    if not os.access(p, os.R_OK | (os.W_OK if r.get('write') else 0)):
        raise PermissionError(str(p))
elif op == 'mime':
    with p.open('rb') as f: data=f.read(16)
    mime = 'image/png' if data.startswith(b'\\x89PNG\\r\\n\\x1a\\n') else 'image/jpeg' if data.startswith(b'\\xff\\xd8\\xff') else 'image/gif' if data.startswith((b'GIF87a', b'GIF89a')) else 'image/webp' if data.startswith(b'RIFF') and data[8:12] == b'WEBP' else 'image/bmp' if data.startswith(b'BM') else None
    print(json.dumps(mime))
`;

/** Original Pi schemas, text/image processing, exact edits and output truncation. */
export function createCodingTools(workspace: string, python: string, policy?: ExecutionPolicy): ToolDefinition[] {
  python = realpathSync(python); // The sandbox mounts the canonical runtime, not host-only symlink aliases.
  const root = realpathSync(workspace);
  const fileOperation = async (op: string, path: string, extra: Record<string, unknown> = {}, writable = false) => {
    const allowed = await safePath(root, path, writable, policy);
    const result = await run(root, python, `${quote(python)} -c ${quote(FILE_OPERATION)}`, { policy }, JSON.stringify({ op, path: allowed, ...extra }));
    if (result.exitCode !== 0) throw new Error(result.stderr.trim() || `Sandbox ${op} failed (${result.exitCode})`);
    return result.stdout.trim();
  };
  const readFileOp = async (path: string) => Buffer.from(await fileOperation("read", path), "base64");
  const writeFileOp = async (path: string, content: string) => {
    await fileOperation("write", path, { content: Buffer.from(content).toString("base64") }, true);
  };
  const tools: ToolDefinition<any, any, any>[] = [
    createReadToolDefinition(root, {
      operations: {
        readFile: readFileOp,
        access: async (path) => { await fileOperation("access", path); },
        detectImageMimeType: async (path) => JSON.parse(await fileOperation("mime", path)),
      },
    }),
    createWriteToolDefinition(root, {
      operations: {
        writeFile: writeFileOp,
        mkdir: async (path) => { await fileOperation("mkdir", path, {}, true); },
      },
    }),
    createEditToolDefinition(root, {
      operations: {
        readFile: readFileOp,
        writeFile: writeFileOp,
        access: async (path) => { await fileOperation("access", path, { write: true }, true); },
      },
    }),
    createBashToolDefinition(root, {
      exposeSessionEnvironment: false,
      operations: {
        exec: async (command, _cwd, options) => {
          const result = await sandboxRun(root, python, command, { ...options, policy });
          return { exitCode: result.exitCode };
        },
      },
    }),
  ];

  async function publishOutput(source: string): Promise<string> {
    if (!/^pi-bash-[0-9a-f]{16}\.log$/.test(basename(source))) {
      throw new Error("Unexpected native output file");
    }
    const target = join(root, policy?.readWrite.at(-1) ?? "scratch", "tool-output", basename(source));
    await fileOperation("mkdir", dirname(target), {}, true);
    const data = await readFile(source);
    await fileOperation("write", target, { content: data.toString("base64") }, true);
    await unlink(source);
    return target;
  }

  const rewrite = (value: unknown, source: string, target: string): unknown => {
    if (typeof value === "string") return value.replaceAll(source, target);
    if (Array.isArray(value)) return value.map((item) => rewrite(item, source, target));
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, rewrite(item, source, target)]));
    }
    return value;
  };

  let pending = Promise.resolve();
  return tools.map((tool) => defineTool({
    ...tool,
    async execute(id, args, signal, onUpdate, ctx) {
      const previous = pending;
      let release!: () => void;
      pending = new Promise<void>((accept) => { release = accept; });
      await previous;
      try {
        const inputArgs = args as Record<string, any>;
        const input = tool.name === "bash" ? inputArgs : {
          ...inputArgs,
          path: await safePath(workspace, inputArgs.path, tool.name !== "read", policy),
        };
        const guardedUpdate = onUpdate ? (partial: Parameters<NonNullable<typeof onUpdate>>[0]) => {
          // Streaming tails contain no reference; publish the closed full file only.
          if (tool.name === "bash" && partial.details) {
            onUpdate({ ...partial, details: { ...partial.details, fullOutputPath: undefined } });
          } else onUpdate(partial);
        } : undefined;
        const context = ctx ? { ...ctx, cwd: root } : { cwd: root } as ExtensionToolContext;
        const result = await tool.execute(id, input, signal, guardedUpdate, context);
        if (tool.name === "bash") {
          const source = (result.details as { fullOutputPath?: string } | undefined)?.fullOutputPath;
          if (source) return rewrite(result, source, await publishOutput(source)) as typeof result;
        }
        return result;
      } catch (error) {
        if (tool.name === "bash" && error instanceof Error) {
          const source = error.message.match(/\/[^\s\]]*pi-bash-[0-9a-f]{16}\.log/)?.[0];
          if (source) throw new Error(error.message.replaceAll(source, await publishOutput(source)), { cause: error });
        }
        throw error;
      } finally { release(); }
    },
  }));
}
