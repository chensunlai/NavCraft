import { readFile } from "node:fs/promises";
import type { AgentMessage, StreamFn } from "@earendil-works/pi-agent-core";
import { normalizeContext, type ImageContent, type TextContent } from "@earendil-works/pi-ai";
import { compact, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { publicPath } from "./sandbox.js";

export interface NavigationPublicState {
  task: unknown;
  /** Public observation record(s), with path/image_ref/images identifying saved RGB. */
  latestObservation: unknown;
  map_ref?: string;
  /** Metadata saved when this image was rendered, not the current memory state. */
  map_metadata?: unknown;
  goal_image_ref?: string;
}

export interface CompactionTrace {
  emit(type: string, data?: unknown): void;
}

export const NAVIGATION_SUMMARY_INSTRUCTIONS = `Navigation checkpoint requirements:
This summary covers only the supplied older messages. "Latest" means the last observation within this summarized segment, not the current state of the whole run.
Label it "last observation in summarized segment" with its frame/step. Later retained messages and public checkpoints take precedence for current goals, observations and state; do not relabel historical data as globally current.
Preserve the current goal and goal ID, its description/image references, and which goals have already been submitted.
Preserve the latest actual observation's frame ID, action step, image references and view/camera information.
Preserve exact map, memory and evidence file references so they can be reopened.
Keep position/heading estimates explicitly labelled as estimates and bound to their supporting observation time; preserve uncertainty and conflicts.
Keep actual executed actions, collisions, unfinished local intentions and questions requiring more evidence.
Distinguish observed facts from model hypotheses and old plans from the current goal.
Summarize only supplied history. Do not solve navigation, invent a new location, infer private scores, generate evidence, or write memory files.
The image contents may no longer be present: retain their references rather than claiming a text summary preserves all visual detail.`;

function contextText(messages: AgentMessage[]): string {
  return messages.map((message) => {
    if ("summary" in message && typeof message.summary === "string") return message.summary;
    if (!("content" in message)) return "";
    if (typeof message.content === "string") return message.content;
    if (!Array.isArray(message.content)) return "";
    return message.content.filter((item) => item.type === "text").map((item) => (item as TextContent).text).join("\n");
  }).join("\n");
}

function imageRefs(observation: unknown): string[] {
  if (!observation || typeof observation !== "object") return [];
  if (Array.isArray(observation)) return observation.flatMap(imageRefs);
  const record = observation as Record<string, unknown>;
  const refs: string[] = [];
  for (const key of ["path", "image_ref", "rgb_ref", "rgb_path"]) {
    if (typeof record[key] === "string") refs.push(record[key]);
  }
  for (const key of ["images", "image_refs"]) {
    if (Array.isArray(record[key])) refs.push(...record[key].filter((item): item is string => typeof item === "string"));
  }
  for (const key of ["observation", "observations", "views"]) {
    if (record[key] && typeof record[key] === "object") refs.push(...imageRefs(record[key]));
  }
  return refs;
}

function hasImage(messages: AgentMessage[], image: ImageContent): boolean {
  return messages.some((message) => "content" in message && Array.isArray(message.content) && message.content.some((block) => block.type === "image" && (block as ImageContent).data === image.data));
}

/** Keep Pi's prepared cut point, summary merge, file tracking, usage and persistence. */
export function createNavigationCompaction(
  sharedStreamFn: StreamFn,
  getPublicState: () => NavigationPublicState | Promise<NavigationPublicState>,
  trace: CompactionTrace,
): ExtensionFactory {
  return (pi) => {
    let restored: AgentMessage | undefined;
    pi.on("session_before_compact", async (event, ctx) => {
      restored = undefined;
      trace.emit("navigation_compaction_start", {
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
          // Both the history and split-turn generator pass through this function.
          const augmented = normalizeContext({ messages: [
            ...context.messages,
            { role: "user", content: [{ type: "text", text: NAVIGATION_SUMMARY_INSTRUCTIONS }], timestamp: Date.now() },
          ] });
          trace.emit("navigation_summary_input", { model: model.id, context: augmented, maxRetries: 0 });
          const resultStream = await sharedStreamFn(model, augmented, { ...options, maxRetries: 0 });
          const originalResult = resultStream.result.bind(resultStream);
          resultStream.result = async () => {
            const response = await originalResult();
            trace.emit("navigation_summary_output", response);
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
        trace.emit("navigation_compaction_result", result);
        return { compaction: result };
      } catch (error) {
        // Pi catches hook exceptions and would otherwise run the default generator.
        // Returning cancel explicitly prevents a hidden, unbudgeted fallback.
        trace.emit("navigation_compaction_cancelled", {
          error: error instanceof Error ? error.message : String(error),
          aborted: event.signal.aborted,
        });
        return { cancel: true };
      }
    });

    pi.on("session_compact", async (event, ctx) => {
      const state = await getPublicState();
      const messages = ctx.sessionManager.buildSessionProjection().messages;
      const retainedText = contextText(messages);
      const missing: Record<string, unknown> = {};
      for (const key of ["task", "latestObservation"] as const) {
        const value = state[key];
        if (value !== undefined && !retainedText.includes(JSON.stringify(value))) missing[key] = value;
      }
      const missingMapMetadata = state.map_metadata !== undefined && !retainedText.includes(JSON.stringify(state.map_metadata));
      const missingMapRef = !!state.map_ref && !retainedText.includes(state.map_ref);
      const refs = [...new Set([
        ...imageRefs(state.latestObservation),
        ...[state.map_ref, state.goal_image_ref].filter((ref): ref is string => !!ref),
      ])];
      const images: ImageContent[] = [];
      const addedRefs: string[] = [];
      for (const ref of refs) {
        try {
          const path = await publicPath(ctx.cwd, ref);
          const data = (await readFile(path)).toString("base64");
          const mimeType = /\.jpe?g$/i.test(path) ? "image/jpeg" : /\.webp$/i.test(path) ? "image/webp" : "image/png";
          const image: ImageContent = { type: "image", data, mimeType };
          if (!hasImage(messages, image)) { images.push(image); addedRefs.push(ref); }
        } catch (error) {
          trace.emit("navigation_checkpoint_image_error", { ref, error: error instanceof Error ? error.message : String(error) });
        }
      }
      if (!Object.keys(missing).length && !images.length && !missingMapMetadata && !missingMapRef) {
        trace.emit("navigation_checkpoint_retained", { compactionId: event.compactionEntry.id });
        return;
      }
      const checkpoint = {
        source: "existing_public_state_after_compaction",
        ...missing,
        ...(state.map_ref ? { map_ref: state.map_ref } : {}),
        ...(state.map_metadata !== undefined ? { map_metadata: state.map_metadata } : {}),
        images: addedRefs,
        note: "These are existing public records and saved images. No new observation or position estimate was generated. Map metadata describes the saved rendering; its pose_at_frame may be old and is not live localization. Later retained observations take precedence over the older summary.",
      };
      restored = {
        role: "custom", customType: "navigation_checkpoint",
        content: [{ type: "text", text: JSON.stringify(checkpoint) }, ...images],
        display: false, timestamp: Date.now(),
        details: { compactionId: event.compactionEntry.id, refs: addedRefs },
      };
      pi.sendMessage({
        customType: "navigation_checkpoint",
        content: restored.content,
        display: false,
        details: { compactionId: event.compactionEntry.id, refs: addedRefs },
      }, { triggerTurn: false });
      trace.emit("navigation_checkpoint_restored", checkpoint);
    });

    // During an active run Pi defers non-triggering custom messages until the
    // turn boundary. An overflow retry may need these same public inputs sooner.
    pi.on("context", async (event) => {
      if (!restored || restored.role !== "custom") return;
      const marker = (restored.content as (TextContent | ImageContent)[])[0] as TextContent;
      if (contextText(event.messages).includes(marker.text)) return;
      trace.emit("navigation_checkpoint_request_attached", { details: restored.details });
      return { messages: [...event.messages, restored] };
    });
  };
}
