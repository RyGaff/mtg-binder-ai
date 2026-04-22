jest.mock('../../modules/card-detector/src', () => ({
  encodeImage: jest.fn(),
}));

import {
  encodeCardImage, isImageSearchReady, __resetImageEncoderReadiness,
} from '../../src/embeddings/imageEncoder';
import { encodeImage } from '../../modules/card-detector/src';

beforeEach(() => {
  jest.clearAllMocks();
  __resetImageEncoderReadiness();
});

it('returns null when native returns null (asset missing)', async () => {
  (encodeImage as jest.Mock).mockResolvedValue(null);
  const v = await encodeCardImage('file:///x.jpg');
  expect(v).toBeNull();
  expect(await isImageSearchReady()).toBe(false);
});

it('returns null when native throws', async () => {
  (encodeImage as jest.Mock).mockRejectedValue(new Error('boom'));
  const v = await encodeCardImage('file:///x.jpg');
  expect(v).toBeNull();
  expect(await isImageSearchReady()).toBe(false);
});

it('returns null when vector has wrong length', async () => {
  (encodeImage as jest.Mock).mockResolvedValue(new Float32Array(128));
  const v = await encodeCardImage('file:///x.jpg');
  expect(v).toBeNull();
  expect(await isImageSearchReady()).toBe(false);
});

it('returns the vector and flips readiness to true on success', async () => {
  const good = new Float32Array(256); good[0] = 1;
  (encodeImage as jest.Mock).mockResolvedValue(good);
  const v = await encodeCardImage('file:///x.jpg');
  expect(v).toBe(good);
  expect(await isImageSearchReady()).toBe(true);
});

it('isImageSearchReady stays true after a subsequent failure (memoized once)', async () => {
  // Documents intentional behavior: one good scan per session flips readiness
  // to true and it never flips back, even if the native call later errors.
  // If the device unloads the model mid-session, this flag lies.
  const good = new Float32Array(256); good[0] = 1;
  (encodeImage as jest.Mock).mockResolvedValueOnce(good);
  await encodeCardImage('file:///x.jpg');
  expect(await isImageSearchReady()).toBe(true);

  (encodeImage as jest.Mock).mockResolvedValueOnce(null);
  const v2 = await encodeCardImage('file:///y.jpg');
  expect(v2).toBeNull();
  expect(await isImageSearchReady()).toBe(true);  // still true — memoized
});
