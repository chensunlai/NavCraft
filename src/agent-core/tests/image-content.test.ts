import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadImages } from '../image-content.js';

test('public image loading preserves reference order and bytes with PNG, JPEG and WebP MIME types', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'navcraft-images-'));
  const refs = ['view.PNG', 'view.jpg', 'view.JPEG', 'view.WeBp'];
  const bytes = refs.map((_, index) => Buffer.from([0, 1, index, 255]));
  try {
    await Promise.all(refs.map((ref, index) => writeFile(join(workspace, ref), bytes[index])));
    const images = await loadImages(workspace, refs);
    assert.deepEqual(images.map(image => image.mimeType), ['image/png', 'image/jpeg', 'image/jpeg', 'image/webp']);
    assert.deepEqual(images.map(image => Buffer.from(image.data, 'base64')), bytes);
    assert.ok(images.every(image => image.type === 'image'));
    assert.deepEqual(await loadImages(workspace), []);
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test('public image loading rejects undeclared paths, traversal and escaping symlinks', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navcraft-image-policy-'));
  const workspace = join(directory, 'workspace');
  for (const name of ['observations', 'scratch', 'private']) await mkdir(join(workspace, name), { recursive: true });
  const policy = { readOnly: ['observations'], readWrite: ['scratch'] };
  try {
    await writeFile(join(directory, 'outside.png'), 'outside');
    await writeFile(join(workspace, 'private', 'answer.png'), 'private');
    await writeFile(join(workspace, 'observations', 'current.png'), 'current');
    await symlink(join(directory, 'outside.png'), join(workspace, 'observations', 'escape.png'));
    assert.equal((await loadImages(workspace, ['observations/current.png'], policy)).length, 1);
    await assert.rejects(loadImages(workspace, ['private/answer.png'], policy), /readable/);
    await assert.rejects(loadImages(workspace, ['../outside.png'], policy), /workspace/);
    await assert.rejects(loadImages(workspace, ['observations/escape.png'], policy), /Symlink escapes/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
