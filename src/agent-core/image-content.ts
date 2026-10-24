import { readFile } from "node:fs/promises";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { publicPath, type ExecutionPolicy } from "./sandbox.js";

/** Load only images permitted by the same public workspace policy as Coding tools. */
export function loadImages(workspace: string, refs: string[] = [], policy?: ExecutionPolicy): Promise<ImageContent[]> {
  return Promise.all(refs.map(async (ref) => {
    const path = await publicPath(workspace, ref, false, policy);
    return { type: "image" as const, data: (await readFile(path)).toString("base64"),
      mimeType: /\.jpe?g$/i.test(path) ? "image/jpeg" : /\.webp$/i.test(path) ? "image/webp" : "image/png" };
  }));
}

/** Captions are supplied by the host; the Core does not infer domain semantics. */
export function labelledImages(images: ImageContent[], refs: string[], labels?: string[]): (TextContent | ImageContent)[] {
  if (labels && (labels.length !== images.length || labels.some(label => typeof label !== "string"))) {
    throw new Error("Image labels must correspond one-to-one with image references");
  }
  return images.flatMap((image, index): (TextContent | ImageContent)[] => [
    { type: "text", text: labels?.[index] ?? `IMAGE REFERENCE | ${refs[index]}` }, image,
  ]);
}
