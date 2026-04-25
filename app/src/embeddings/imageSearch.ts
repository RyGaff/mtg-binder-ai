import { encodeCardImage } from './imageEncoder';
import { getImageEmbeddingMap } from './parser';

export type ImageMatch = {
  scryfallId: string;
  score: number;
  topK: Array<{ scryfallId: string; score: number }>;
};

const TOP_K = 3;

export async function findCardByImage(rectifiedUri: string): Promise<ImageMatch | null> {
  const query = await encodeCardImage(rectifiedUri);
  if (!query) return null;

  let index;
  try { index = await getImageEmbeddingMap(); } catch { return null; }
  if (index.version !== 2 || index.size === 0 || index.dim !== query.length) return null;

  const { ids, vectors, dim, size } = index;
  type Hit = { id: string; score: number };
  const topK: Hit[] = [];
  const limit = dim - (dim % 4);

  for (let r = 0; r < size; r++) {
    const offset = r * dim;
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
    const pretty = topK.map((h) => `${h.id.slice(0, 8)}=${h.score.toFixed(4)}`).join(' ');
    console.log(`[match] top${TOP_K} ${pretty}`);
  }
  return {
    scryfallId: topK[0].id,
    score: topK[0].score,
    topK: topK.map((h) => ({ scryfallId: h.id, score: h.score })),
  };
}
