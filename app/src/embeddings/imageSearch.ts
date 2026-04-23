import { encodeCardImage } from './imageEncoder';
import { getImageEmbeddingMap } from './parser';

export type ImageMatch = {
  scryfallId: string;
  score: number;                           // top-1 cosine similarity
  topK: Array<{ scryfallId: string; score: number }>;
};

const TOP_K = 3;

/**
 * Encode the rectified card image and find the nearest neighbor in the
 * loaded image-embeddings database. The input MUST be a rectified
 * 400×560 card crop — feeding a raw photo yields a garbage embedding.
 * Returns null when:
 *   - The encoder asset is not bundled
 *   - The embeddings file is not loaded / doesn't exist
 *   - The loaded file is the v1 text-embeddings format (wrong file)
 *   - Any step errors
 *
 * Both query and database vectors are L2-normalized, so dot-product
 * equals cosine similarity.
 */
export async function findCardByImage(rectifiedUri: string): Promise<ImageMatch | null> {
  const query = await encodeCardImage(rectifiedUri);
  if (!query) return null;

  let index;
  try {
    index = await getImageEmbeddingMap();
  } catch {
    return null;
  }
  if (index.version !== 2) return null;
  if (index.size === 0) return null;
  // Guard: mismatched encoder / embeddings would produce NaN scores silently
  // (vec[i] undefined for i ≥ vec.length). Refuse to search rather than
  // returning garbage that passes MATCH_MIN because `NaN < x` is false.
  if (index.dim !== query.length) return null;

  // Linear scan — 35k × 256 dot products ≈ 9M MACs, ~30 ms on phone.
  const { ids, vectors, dim, size } = index;
  type Hit = { id: string; score: number };
  const topK: Hit[] = [];
  for (let r = 0; r < size; r++) {
    const offset = r * dim;
    const limit = dim - (dim % 4);
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    let i = 0;
    for (; i < limit; i += 4) {
      s0 += query[i]     * vectors[offset + i];
      s1 += query[i + 1] * vectors[offset + i + 1];
      s2 += query[i + 2] * vectors[offset + i + 2];
      s3 += query[i + 3] * vectors[offset + i + 3];
    }
    let score = s0 + s1 + s2 + s3;
    for (; i < dim; i++) score += query[i] * vectors[offset + i];

    if (topK.length < TOP_K) {
      topK.push({ id: ids[r], score });
      topK.sort((a, b) => b.score - a.score);
    } else if (score > topK[TOP_K - 1].score) {
      topK[TOP_K - 1] = { id: ids[r], score };
      topK.sort((a, b) => b.score - a.score);
    }
  }

  if (topK.length === 0) return null;
  if (__DEV__) {
    const pretty = topK
      .map((h) => `${h.id.slice(0, 8)}=${h.score.toFixed(4)}`)
      .join(' ');
    console.log(`[match] top${TOP_K} ${pretty}`);
  }
  return {
    scryfallId: topK[0].id,
    score:      topK[0].score,
    topK:       topK.map(h => ({ scryfallId: h.id, score: h.score })),
  };
}
