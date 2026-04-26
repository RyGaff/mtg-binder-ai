import { resolveDeckCards } from '../../src/utils/deckImport';

global.fetch = jest.fn();

const mockCard = (over: { id: string; name: string; set?: string; collector_number?: string }) => ({
  id: over.id,
  name: over.name,
  set: over.set ?? 'lea',
  collector_number: over.collector_number ?? '1',
  mana_cost: '',
  type_line: 'Instant',
  oracle_text: '',
  color_identity: [],
  prices: {},
  keywords: [],
});

beforeEach(() => {
  (fetch as jest.Mock).mockReset();
});

describe('resolveDeckCards', () => {
  it('returns empty result without making a request when given empty input', async () => {
    const result = await resolveDeckCards([]);
    expect(result.resolved.size).toBe(0);
    expect(result.unresolved).toEqual([]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('skips empty / whitespace-only names without sending requests', async () => {
    const result = await resolveDeckCards(['', '  ', '\t']);
    expect(result.resolved.size).toBe(0);
    expect(result.unresolved).toEqual([]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('single-chunk happy path: resolves all names with one request', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [
          mockCard({ id: 'a', name: 'Lightning Bolt' }),
          mockCard({ id: 'b', name: 'Sol Ring' }),
        ],
        not_found: [],
      }),
    });

    const result = await resolveDeckCards(['Lightning Bolt', 'Sol Ring']);
    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = (fetch as jest.Mock).mock.calls[0];
    expect(url).toBe('https://api.scryfall.com/cards/collection');
    expect(init.method).toBe('POST');
    expect(init.headers['User-Agent']).toBe('MTGBinderApp/1.0');
    expect(init.headers['Accept']).toBe('application/json');
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(init.body)).toEqual({
      identifiers: [{ name: 'Lightning Bolt' }, { name: 'Sol Ring' }],
    });

    expect(result.resolved.size).toBe(2);
    expect(result.resolved.get('lightning bolt')?.scryfall_id).toBe('a');
    expect(result.resolved.get('sol ring')?.scryfall_id).toBe('b');
    expect(result.unresolved).toEqual([]);
  });

  it('dedupes case-insensitively before requesting', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [mockCard({ id: 'a', name: 'Lightning Bolt' })],
        not_found: [],
      }),
    });

    await resolveDeckCards(['Lightning Bolt', 'lightning bolt', 'LIGHTNING BOLT']);
    const [, init] = (fetch as jest.Mock).mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      identifiers: [{ name: 'Lightning Bolt' }],
    });
  });

  it('multi-chunk path: chunks into ≤75 per request and merges results', async () => {
    const names = Array.from({ length: 160 }, (_, i) => `Card ${i + 1}`);
    const responses = [
      {
        data: names.slice(0, 75).map((n, i) => mockCard({ id: `c${i}`, name: n })),
        not_found: [],
      },
      {
        data: names.slice(75, 150).map((n, i) => mockCard({ id: `c${i + 75}`, name: n })),
        not_found: [],
      },
      {
        data: names.slice(150).map((n, i) => mockCard({ id: `c${i + 150}`, name: n })),
        not_found: [],
      },
    ];
    (fetch as jest.Mock)
      .mockResolvedValueOnce({ ok: true, json: async () => responses[0] })
      .mockResolvedValueOnce({ ok: true, json: async () => responses[1] })
      .mockResolvedValueOnce({ ok: true, json: async () => responses[2] });

    const result = await resolveDeckCards(names);
    expect(fetch).toHaveBeenCalledTimes(3);

    const sizes = (fetch as jest.Mock).mock.calls.map(([, init]) => JSON.parse(init.body).identifiers.length);
    expect(sizes).toEqual([75, 75, 10]);

    expect(result.resolved.size).toBe(160);
    expect(result.resolved.get('card 1')?.scryfall_id).toBe('c0');
    expect(result.resolved.get('card 76')?.scryfall_id).toBe('c75');
    expect(result.resolved.get('card 160')?.scryfall_id).toBe('c159');
    expect(result.unresolved).toEqual([]);
  });

  it('reports unresolved names from not_found', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [mockCard({ id: 'a', name: 'Lightning Bolt' })],
        not_found: [{ name: 'Notarealcard' }],
      }),
    });

    const result = await resolveDeckCards(['Lightning Bolt', 'Notarealcard']);
    expect(result.resolved.size).toBe(1);
    expect(result.resolved.has('lightning bolt')).toBe(true);
    expect(result.unresolved).toEqual(['notarealcard']);
  });

  it('keys resolved map under input name when Scryfall returns alternate canonical name', async () => {
    // Input: "Bruna" (a face of "Bruna, the Fading Light" meld pair).
    // Scryfall returns the canonical card name; we should index under both.
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [mockCard({ id: 'bruna', name: 'Bruna, the Fading Light' })],
        not_found: [],
      }),
    });

    const result = await resolveDeckCards(['Bruna']);
    expect(result.resolved.get('bruna')?.scryfall_id).toBe('bruna');
    expect(result.resolved.get('bruna, the fading light')?.scryfall_id).toBe('bruna');
    expect(result.unresolved).toEqual([]);
  });

  it('aborts in-flight request when signal is aborted', async () => {
    const controller = new AbortController();
    const fetchError = new DOMException('The operation was aborted.', 'AbortError');

    (fetch as jest.Mock).mockImplementationOnce((_url, init: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => reject(fetchError));
      });
    });

    const promise = resolveDeckCards(['Lightning Bolt'], controller.signal);
    controller.abort();
    await expect(promise).rejects.toThrow(/abort/i);
  });

  it('does not start subsequent chunks once aborted between chunks', async () => {
    const names = Array.from({ length: 100 }, (_, i) => `Card ${i + 1}`);
    const controller = new AbortController();

    (fetch as jest.Mock).mockImplementationOnce(async () => {
      // Abort after the first chunk completes, before the second starts.
      controller.abort();
      return {
        ok: true,
        json: async () => ({
          data: names.slice(0, 75).map((n, i) => mockCard({ id: `c${i}`, name: n })),
          not_found: [],
        }),
      };
    });

    await expect(resolveDeckCards(names, controller.signal)).rejects.toThrow();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('throws when Scryfall returns a non-ok response', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 500 });
    await expect(resolveDeckCards(['Lightning Bolt'])).rejects.toThrow('Scryfall 500');
  });
});
