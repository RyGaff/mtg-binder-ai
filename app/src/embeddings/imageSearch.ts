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
 * loaded image-embeddings database. Returns null when:
 *   - The encoder asset is not bundled
 *   - The embeddings file is not loaded / doesn't exist
 *   - The loaded file is the v1 text-embeddings format (wrong file)
 *   - Any step errors
 *
 * Both query and database vectors are L2-normalized, so dot-product
 * equals cosine similarity.
 */
export async function findCardByImage(uri: string): Promise<ImageMatch | null> {
  const query = await encodeCardImage(uri);
  if (!query) {
    console.log('[imageSearch] encoder returned null');
    return null;
  }

  let index;
  try {
    index = await getImageEmbeddingMap();
  } catch (err) {
    console.log('[imageSearch] parser error:', err instanceof Error ? err.message : String(err));
    return null;
  }
  console.log(`[imageSearch] index loaded version=${index.version} byId.size=${index.byId.size} dim=${index.dim}`);
  if (index.version !== 2) {
    console.log(`[imageSearch] wrong version (expected 2, got ${index.version}) — is the correct embeddings file downloaded?`);
    return null;
  }
  if (index.byId.size === 0) {
    console.log('[imageSearch] empty byId map');
    return null;
  }

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
  return {
    scryfallId: topK[0].id,
    score:      topK[0].score,
    topK:       topK.map(h => ({ scryfallId: h.id, score: h.score })),
  };
}
