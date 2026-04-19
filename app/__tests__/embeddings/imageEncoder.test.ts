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
