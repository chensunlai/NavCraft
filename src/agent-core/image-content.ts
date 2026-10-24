import type { ImageContent, TextContent } from "@earendil-works/pi-ai";

/** Captions are supplied by the host; the Core does not infer domain semantics. */
export function labelledImages(images: ImageContent[], refs: string[], labels?: string[]): (TextContent | ImageContent)[] {
  if (labels && (labels.length !== images.length || labels.some(label => typeof label !== "string"))) {
    throw new Error("Image labels must correspond one-to-one with image references");
  }
  return images.flatMap((image, index): (TextContent | ImageContent)[] => [
    { type: "text", text: labels?.[index] ?? `IMAGE REFERENCE | ${refs[index]}` }, image,
  ]);
}
