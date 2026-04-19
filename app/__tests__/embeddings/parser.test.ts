import { parseEmbeddingBuffer, normalize } from '../../src/embeddings/parser';

function makeBuffer(cards: Array<{ id: string; name?: string; vector: number[] }>): ArrayBuffer {
  const d = cards[0].vector.length;
  const n = cards.length;
  const recordSize = 36 + 64 + d * 4;
  const buffer = new ArrayBuffer(8 + n * recordSize);
  const view = new DataView(buffer);
  view.setUint32(0, n, true);
  view.setUint32(4, d, true);
  for (let i = 0; i < n; i++) {
    const base = 8 + i * recordSize;
    const idPadded = cards[i].id.padEnd(36, '\0');
    for (let j = 0; j < 36; j++) view.setUint8(base + j, idPadded.charCodeAt(j));
    const namePadded = (cards[i].name ?? '').padEnd(64, '\0');
    for (let j = 0; j < 64; j++) view.setUint8(base + 36 + j, namePadded.charCodeAt(j));
    for (let j = 0; j < d; j++) view.setFloat32(base + 100 + j * 4, cards[i].vector[j], true);
  }
  return buffer;
}

const ID_A = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
const ID_B = 'ffffffff-0000-1111-2222-333333333333';

describe('parseEmbeddingBuffer', () => {
  it('returns empty maps for 0 cards', () => {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setUint32(0, 0, true);
    view.setUint32(4, 3, true);
    const { byId, byName } = parseEmbeddingBuffer(buffer);
    expect(byId.size).toBe(0);
    expect(byName.size).toBe(0);
  });

  it('parses a single card into byId', () => {
    const buffer = makeBuffer([{ id: ID_A, name: 'Lightning Bolt', vector: [1, 0, 0] }]);
    const { byId } = parseEmbeddingBuffer(buffer);
    expect(byId.has(ID_A)).toBe(true);
  });

  it('parses card name into byName', () => {
    const buffer = makeBuffer([{ id: ID_A, name: 'Lightning Bolt', vector: [1, 0, 0] }]);
    const { byName } = parseEmbeddingBuffer(buffer);
    expect(byName.get('Lightning Bolt')).toBe(ID_A);
  });

  it('parses two cards into maps with two entries each', () => {
    const buffer = makeBuffer([
      { id: ID_A, name: 'Lightning Bolt', vector: [1, 0, 0] },
      { id: ID_B, name: 'Counterspell', vector: [0, 1, 0] },
    ]);
    const { byId, byName } = parseEmbeddingBuffer(buffer);
    expect(byId.size).toBe(2);
    expect(byName.size).toBe(2);
  });

  it('normalizes vectors to unit length', () => {
    const buffer = makeBuffer([{ id: ID_A, vector: [3, 4, 0] }]);
    const { byId } = parseEmbeddingBuffer(buffer);
    const v = byId.get(ID_A)!;
    const norm = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
    expect(norm).toBeCloseTo(1.0, 5);
  });

  it('returns Float32Array for each card', () => {
    const buffer = makeBuffer([{ id: ID_A, vector: [1, 2, 3] }]);
    const { byId } = parseEmbeddingBuffer(buffer);
    const v = byId.get(ID_A)!;
    expect(v).toBeInstanceOf(Float32Array);
    expect(v.length).toBe(3);
  });

  it('handles cards with no name (empty byName entry skipped)', () => {
    const buffer = makeBuffer([{ id: ID_A, name: '', vector: [1, 0, 0] }]);
    const { byName } = parseEmbeddingBuffer(buffer);
    expect(byName.size).toBe(0);
  });
});

function buildV2Buffer(records: Array<{ id: string; vec: number[] }>, modelHash = 0xDEADBEEF): ArrayBuffer {
  const dim = records[0].vec.length;
  const recSize = 36 + dim * 4;
  const buf = new ArrayBuffer(20 + records.length * recSize);
  const view = new DataView(buf);
  view.setUint32(0,  0x4D544745, true); // 'MTGE' magic
  view.setUint32(4,  2, true);           // version 2
  view.setUint32(8,  records.length, true);
  view.setUint32(12, dim, true);
  view.setUint32(16, modelHash, true);

  records.forEach((r, i) => {
    const base = 20 + i * recSize;
    for (let j = 0; j < 36; j++) {
      view.setUint8(base + j, j < r.id.length ? r.id.charCodeAt(j) : 0);
    }
    const vec = new Float32Array(buf, base + 36, dim);
    r.vec.forEach((x, k) => vec[k] = x);
  });
  return buf;
}

describe('parseEmbeddingBuffer v2 (image embeddings)', () => {
  it('parses a v2 buffer and returns image variant', () => {
    const buf = buildV2Buffer([
      { id: 'aaa-111', vec: [1, 0, 0, 0] },
      { id: 'bbb-222', vec: [0, 1, 0, 0] },
    ], 0x12345678);

    const result = parseEmbeddingBuffer(buf);
    expect(result.version).toBe(2);
    expect(result.dim).toBe(4);
    expect(result.modelHash).toBe(0x12345678);
    expect(result.byId.size).toBe(2);
    expect(Array.from(result.byId.get('aaa-111')!)).toEqual([1, 0, 0, 0]);
    expect(result.byName.size).toBe(0);
  });
});

describe('parseEmbeddingBuffer v1 (text embeddings, backward-compat)', () => {
  it('still parses the legacy no-magic format', () => {
    const dim = 2;
    const rec = 36 + 64 + dim * 4;
    const buf = new ArrayBuffer(8 + rec);
    const view = new DataView(buf);
    view.setUint32(0, 1, true);
    view.setUint32(4, dim, true);
    for (let j = 0; j < 3; j++) view.setUint8(8 + j, 'zzz'.charCodeAt(j));
    for (let j = 0; j < 8; j++) view.setUint8(8 + 36 + j, 'Testcard'.charCodeAt(j));
    const vec = new Float32Array(buf, 8 + 36 + 64, dim);
    vec[0] = 0.6; vec[1] = 0.8;

    const result = parseEmbeddingBuffer(buf);
    expect(result.version).toBe(1);
    expect(result.byId.size).toBe(1);
    expect(result.byName.get('Testcard')).toBe('zzz');
  });
});

describe('normalize', () => {
  it('returns a unit vector for a non-zero input', () => {
    const v = new Float32Array([3, 4]);
    const n = normalize(v);
    expect(n[0]).toBeCloseTo(0.6, 5);
    expect(n[1]).toBeCloseTo(0.8, 5);
  });

  it('returns zeros for a zero vector (no NaN)', () => {
    const v = new Float32Array([0, 0, 0]);
    const n = normalize(v);
    expect(n[0]).toBe(0);
    expect(n[1]).toBe(0);
    expect(n[2]).toBe(0);
  });

  it('does not modify the input vector', () => {
    const v = new Float32Array([3, 4]);
    normalize(v);
    expect(v[0]).toBe(3);
    expect(v[1]).toBe(4);
  });
});
