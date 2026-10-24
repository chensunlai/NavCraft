import { createHash } from 'node:crypto';
import { PhotonImage } from '@silvia-odwyer/photon-node';
import type { Context, ImageContent } from '@earendil-works/pi-ai';

/** Preserve original files/session images; encode opaque PNGs once for model transport. */
export function createImageTransport(emit: (type: string, data: unknown) => void) {
  const cache = new Map<string, ImageContent>();
  function encode(image: ImageContent): ImageContent {
    if (image.mimeType !== 'image/png' || image.data.length < 64 * 1024) return image;
    const key = createHash('sha256').update(image.data).digest('hex');
    const cached = cache.get(key);
    if (cached) return cached;
    let decoded: PhotonImage | undefined;
    let result = image;
    try {
      const bytes = Buffer.from(image.data, 'base64');
      decoded = PhotonImage.new_from_byteslice(bytes);
      const pixels = decoded.get_raw_pixels();
      let opaque = true;
      for (let i = 3; i < pixels.length; i += 4) {
        if (pixels[i] !== 255) { opaque = false; break; }
      }
      if (opaque) {
        const data = Buffer.from(decoded.get_bytes_jpeg(85)).toString('base64');
        if (data.length < image.data.length) {
          result = { ...image, data, mimeType: 'image/jpeg' };
          emit('image_transport_encoded', { source_sha256: key, width: decoded.get_width(),
            height: decoded.get_height(), quality: 85, source_base64_bytes: image.data.length,
            sent_base64_bytes: data.length, mime_type: result.mimeType });
        }
      }
    } catch (error) {
      emit('image_transport_error', { source_sha256: key, error: String(error) });
    } finally { decoded?.free(); }
    if (cache.size >= 128) cache.delete(cache.keys().next().value!);
    cache.set(key, result);
    return result;
  }
  return <T extends Context>(context: T): T => ({ ...context,
    messages: context.messages.map(message => {
      if (message.role === 'assistant' || !Array.isArray(message.content)) return message;
      return { ...message, content: message.content.map(block => block.type === 'image' ? encode(block) : block) };
    }),
  });
}
