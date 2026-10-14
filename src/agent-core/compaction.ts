import { readFile } from "node:fs/promises";
import type { AgentMessage, StreamFn } from "@earendil-works/pi-agent-core";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { compact, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { publicPath, type ExecutionPolicy } from "./sandbox.js";

export interface PublicCheckpoint {
  content: string | TextContent[];
  images?: string[];
  metadata?: unknown;
  revision: string | number;
}

export interface CompactionTrace {
  emit(type: string, data?: unknown): void;
}

function contextText(messages: AgentMessage[]): string {
  return messages.map((message) => {
    if ("summary" in message && typeof message.summary === "string") return message.summary;
    if (!("content" in message)) return "";
    if (typeof message.content === "string") return message.content;
    if (!Array.isArray(message.content)) return "";
    return message.content.filter((item) => item.type === "text").map((item) => (item as TextContent).text).join("\n");
  }).join("\n");
}

function hasImage(messages: AgentMessage[], image: ImageContent): boolean {
  return messages.some((message) => "content" in message && Array.isArray(message.content) && message.content.some((block) => block.type === "image" && (block as ImageContent).data === image.data));
}

/** Budgeted transport, audit and public-state restoration; prompts live in local Pi core. */
export function createCheckpointCompaction(
  sharedStreamFn: StreamFn,
  getCheckpoint: () => PublicCheckpoint | undefined | Promise<PublicCheckpoint | undefined>,
  trace: CompactionTrace,
  policy?: ExecutionPolicy,
): ExtensionFactory {
  return (pi) => {
    let restored: AgentMessage | undefined;
    pi.on("session_before_compact", async (event, ctx) => {
      restored = undefined;
      trace.emit("checkpoint_compaction_start", {
        reason: event.reason,
        firstKeptEntryId: event.preparation.firstKeptEntryId,
        tokensBefore: event.preparation.tokensBefore,
        isSplitTurn: event.preparation.isSplitTurn,
      });
      try {
        event.signal.throwIfAborted();
        if (!ctx.model) throw new Error("No model selected for navigation compaction");
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
        if (!auth.ok) throw new Error(auth.error);
        const stream: StreamFn = async (model, context, options) => {
          trace.emit("checkpoint_summary_input", { model: model.id, context, maxRetries: 0 });
          const resultStream = await sharedStreamFn(model, context, { ...options, maxRetries: 0 });
          const originalResult = resultStream.result.bind(resultStream);
          resultStream.result = async () => {
            const response = await originalResult();
            trace.emit("checkpoint_summary_output", response);
            if (["aborted", "deferred", "pending"].includes(response.stopReason)) {
              throw new Error(`Navigation summary did not complete: ${response.stopReason}`);
            }
            if (response.stopReason === "stop" && !response.content.some((block) => block.type === "text" && block.text.trim())) {
              throw new Error("Navigation summary returned empty text");
            }
            return response;
          };
          return resultStream;
        };
        const model = auth.baseUrl ? { ...ctx.model, baseUrl: auth.baseUrl } : ctx.model;
        const headers = auth.headers ? Object.fromEntries(Object.entries(auth.headers).filter((entry): entry is [string, string] => typeof entry[1] === "string")) : undefined;
        const result = await compact(
          event.preparation,
          model,
          auth.apiKey,
          headers,
          event.customInstructions,
          event.signal,
          ctx.thinkingLevel ?? "high",
          stream,
          auth.env,
          { enabled: false, maxRetries: 0, baseDelayMs: 0 },
          undefined,
          ctx.sessionManager.getSessionId(),
        );
        event.signal.throwIfAborted();
        trace.emit("checkpoint_compaction_result", result);
        return { compaction: result };
      } catch (error) {
        // Pi catches hook exceptions and would otherwise run the default generator.
        // Returning cancel explicitly prevents a hidden, unbudgeted fallback.
        trace.emit("checkpoint_compaction_cancelled", {
          error: error instanceof Error ? error.message : String(error),
          aborted: event.signal.aborted,
        });
        return { cancel: true };
      }
    });

    pi.on("session_compact", async (event, ctx) => {
      const state = await getCheckpoint();
      if (!state) return;
      const messages = ctx.sessionManager.buildSessionProjection().messages;
      const retainedText = contextText(messages);
      const content: TextContent[] = typeof state.content === "string" ? [{ type: "text", text: state.content }] : state.content;
      const missingContent = content.some((block) => !retainedText.includes(block.text));
      const marker = JSON.stringify({ source: "saved_public_checkpoint", revision: state.revision, metadata: state.metadata,
        note: "Saved records and images supplied by the host; this did not acquire a new observation. Image metadata retains its original time and version. Prefer later tool observations where they differ." });
      const refs = [...new Set(state.images ?? [])];
      const images: ImageContent[] = [];
      const addedRefs: string[] = [];
      for (const ref of refs) {
        try {
          const path = await publicPath(ctx.cwd, ref, false, policy);
          const data = (await readFile(path)).toString("base64");
          const mimeType = /\.jpe?g$/i.test(path) ? "image/jpeg" : /\.webp$/i.test(path) ? "image/webp" : "image/png";
          const image: ImageContent = { type: "image", data, mimeType };
          if (!hasImage(messages, image)) { images.push(image); addedRefs.push(ref); }
        } catch (error) {
          trace.emit("checkpoint_image_error", { ref, error: error instanceof Error ? error.message : String(error) });
        }
      }
      const missingMetadata = state.metadata !== undefined && !retainedText.includes(JSON.stringify(state.metadata));
      if (!missingContent && !missingMetadata && !images.length) {
        trace.emit("checkpoint_retained", { compactionId: event.compactionEntry.id, revision: state.revision });
        return;
      }
      const checkpoint = { revision: state.revision, metadata: state.metadata, images: addedRefs };

      restored = {
        role: "custom", customType: "public_checkpoint",
        content: [{ type: "text", text: marker }, ...content, ...images],
        display: false, timestamp: Date.now(),
        details: { compactionId: event.compactionEntry.id, refs: addedRefs },
      };
      pi.sendMessage({
        customType: "public_checkpoint",
        content: restored.content,
        display: false,
        details: { compactionId: event.compactionEntry.id, refs: addedRefs },
      }, { triggerTurn: false });
      trace.emit("checkpoint_restored", checkpoint);
    });

    // During an active run Pi defers non-triggering custom messages until the
    // turn boundary. An overflow retry may need these same public inputs sooner.
    pi.on("context", async (event) => {
      if (!restored || restored.role !== "custom") return;
      const marker = (restored.content as (TextContent | ImageContent)[])[0] as TextContent;
      if (contextText(event.messages).includes(marker.text) && (restored.content as (TextContent | ImageContent)[]).filter((block): block is ImageContent => block.type === "image").every((image) => hasImage(event.messages, image))) return;
      trace.emit("checkpoint_request_attached", { details: restored.details });
      return { messages: [...event.messages, restored] };
    });
  };
}
