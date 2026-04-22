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
  if (index.byId.size === 0) return null;
  // Guard: mismatched encoder / embeddings would produce NaN scores silently
  // (vec[i] undefined for i ≥ vec.length). Refuse to search rather than
  // returning garbage that passes MATCH_MIN because `NaN < x` is false.
  if (index.dim !== query.length) return null;

  // Linear scan — 35k × 256 dot products ≈ 9M MACs, ~30 ms on phone.
  type Hit = { id: string; score: number };
  const topK: Hit[] = [];
  for (const [id, vec] of index.byId) {
    let score = 0;
    for (let i = 0; i < query.length; i++) score += query[i] * vec[i];

    if (topK.length < TOP_K) {
      topK.push({ id, score });
      topK.sort((a, b) => b.score - a.score);
    } else if (score > topK[TOP_K - 1].score) {
      topK[TOP_K - 1] = { id, score };
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
