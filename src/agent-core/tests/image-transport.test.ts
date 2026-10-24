import assert from 'node:assert/strict';
import test from 'node:test';
import { PhotonImage } from '@silvia-odwyer/photon-node';
import type { Context, ImageContent } from '@earendil-works/pi-ai';
import { createImageTransport } from '../image-transport.js';

function fixture(alpha = 255): ImageContent {
  const pixels = new Uint8Array(256 * 256 * 4);
  let seed = 123;
  for (let i = 0; i < pixels.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      pixels[i + c] = seed >>> 24;
    }
    pixels[i + 3] = alpha;
  }
  const image = new PhotonImage(pixels, 256, 256);
  try { return { type: 'image', data: Buffer.from(image.get_bytes()).toString('base64'), mimeType: 'image/png' }; }
  finally { image.free(); }
}

test('model transport compresses user and tool images without resizing or changing history', () => {
  const image = fixture();
  const events: any[] = [];
  const encode = createImageTransport((type, data) => events.push({ type, data }));
  const context: Context = { messages: [
    { role: 'user', content: [{ type: 'text', text: 'current view' }, image], timestamp: 1 },
    { role: 'toolResult', toolCallId: 'call-1', toolName: 'nav_observe',
      content: [{ type: 'text', text: 'front' }, image], isError: false, timestamp: 2 },
  ] };
  const original = JSON.stringify(context);
  const sent = encode(context);
  const result = (sent.messages[0].content as any[])[1] as ImageContent;
  assert.equal(result.mimeType, 'image/jpeg');
  assert.ok(result.data.length < image.data.length / 2);
  const decoded = PhotonImage.new_from_byteslice(Buffer.from(result.data, 'base64'));
  try { assert.deepEqual([decoded.get_width(), decoded.get_height()], [256, 256]); }
  finally { decoded.free(); }
  assert.deepEqual((sent.messages[1].content as any[])[1], result);
  assert.equal((sent.messages[1] as any).toolCallId, 'call-1');
  assert.deepEqual((sent.messages[0].content as any[])[0], { type: 'text', text: 'current view' });
  assert.equal(JSON.stringify(context), original);
  assert.equal(JSON.stringify(encode(context)), JSON.stringify(sent));
  assert.equal(events.length, 1); // Reuse the same encoded bytes, including across requests.
  assert.equal(events[0].type, 'image_transport_encoded');
});

test('transparent, small and already compressed images keep their original bytes', () => {
  const transparent = fixture(128);
  const jpeg: ImageContent = { type: 'image', data: 'jpeg-fixture', mimeType: 'image/jpeg' };
  const small: ImageContent = { type: 'image', data: 'small-fixture', mimeType: 'image/png' };
  const context: Context = { messages: [{ role: 'user', timestamp: 1, content: [transparent, jpeg, small] }] };
  const encode = createImageTransport(() => assert.fail('No image should be re-encoded'));
  assert.deepEqual(encode(context), context);
});
