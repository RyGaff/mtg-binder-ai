import {
  parseDeckSourceUrl,
  fetchMoxfieldDeck,
  fetchArchidektDeck,
  fetchDeckFromUrl,
} from '../../src/api/deckSources';

global.fetch = jest.fn();

beforeEach(() => {
  (fetch as jest.Mock).mockReset();
});

describe('parseDeckSourceUrl', () => {
  it('matches moxfield without scheme or www', () => {
    expect(parseDeckSourceUrl('moxfield.com/decks/abc')).toEqual({ source: 'moxfield', id: 'abc' });
  });
  it('matches moxfield with www', () => {
    expect(parseDeckSourceUrl('www.moxfield.com/decks/abc')).toEqual({ source: 'moxfield', id: 'abc' });
  });
  it('matches moxfield with https and trailing slash', () => {
    expect(parseDeckSourceUrl('https://moxfield.com/decks/abc/')).toEqual({ source: 'moxfield', id: 'abc' });
  });
  it('matches archidekt bare id', () => {
    expect(parseDeckSourceUrl('archidekt.com/decks/123')).toEqual({ source: 'archidekt', id: '123' });
  });
  it('matches archidekt with slug', () => {
    expect(parseDeckSourceUrl('archidekt.com/decks/123/some-slug')).toEqual({ source: 'archidekt', id: '123' });
  });
  it('matches archidekt with trailing slash', () => {
    expect(parseDeckSourceUrl('archidekt.com/decks/123/')).toEqual({ source: 'archidekt', id: '123' });
  });
  it('rejects mtggoldfish urls', () => {
    expect(parseDeckSourceUrl('https://mtggoldfish.com/deck/12345')).toBeNull();
  });
  it('rejects bare strings', () => {
    expect(parseDeckSourceUrl('not a url')).toBeNull();
    expect(parseDeckSourceUrl('')).toBeNull();
  });
});

describe('fetchMoxfieldDeck', () => {
  const fixture = {
    name: 'My Deck',
    format: 'commander',
    boards: {
      commanders: {
        cards: {
          k1: { quantity: 1, card: { name: 'Atraxa, Praetors’ Voice' } },
        },
      },
      mainboard: {
        cards: {
          k2: { quantity: 4, card: { name: 'Sol Ring' } },
        },
      },
      sideboard: {
        cards: {
          k3: { quantity: 2, card: { name: 'Pithing Needle' } },
        },
      },
      maybeboard: {
        cards: {
          k4: { quantity: 1, card: { name: 'Cyclonic Rift' } },
        },
      },
    },
  };

  it('flattens boards into ParsedLine[] and normalizes format', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => fixture });
    const out = await fetchMoxfieldDeck('abc');
    expect(fetch).toHaveBeenCalledWith(
      'https://api.moxfield.com/v3/decks/all/abc',
      expect.objectContaining({ headers: expect.objectContaining({ Accept: 'application/json' }) }),
    );
    expect(out.name).toBe('My Deck');
    expect(out.format).toBe('Commander');
    expect(out.lines).toEqual(
      expect.arrayContaining([
        { quantity: 1, name: 'Atraxa, Praetors’ Voice', board: 'commander' },
        { quantity: 4, name: 'Sol Ring', board: 'main' },
        { quantity: 2, name: 'Pithing Needle', board: 'side' },
        { quantity: 1, name: 'Cyclonic Rift', board: 'considering' },
      ]),
    );
    expect(out.lines).toHaveLength(4);
  });

  it('throws on non-200', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) });
    await expect(fetchMoxfieldDeck('missing')).rejects.toThrow(/404/);
  });
});

describe('fetchArchidektDeck', () => {
  const fixture = {
    name: 'Edgar EDH',
    format: 'Commander',
    cards: [
      { quantity: 1, categories: ['Commander'], card: { oracleCard: { name: 'Edgar Markov' } } },
      { quantity: 2, categories: ['Sideboard'], card: { oracleCard: { name: 'Bojuka Bog' } } },
      { quantity: 3, categories: [], card: { oracleCard: { name: 'Swamp' } } },
      { quantity: 1, categories: ['Maybeboard'], card: { oracleCard: { name: 'Necropotence' } } },
    ],
  };

  it('maps categories to boards', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => fixture });
    const out = await fetchArchidektDeck('123');
    expect(fetch).toHaveBeenCalledWith(
      'https://archidekt.com/api/decks/123/',
      expect.objectContaining({ headers: expect.objectContaining({ Accept: 'application/json' }) }),
    );
    expect(out.name).toBe('Edgar EDH');
    expect(out.format).toBe('Commander');
    expect(out.lines).toEqual([
      { quantity: 1, name: 'Edgar Markov', board: 'commander' },
      { quantity: 2, name: 'Bojuka Bog', board: 'side' },
      { quantity: 3, name: 'Swamp', board: 'main' },
      { quantity: 1, name: 'Necropotence', board: 'considering' },
    ]);
  });

  it('falls back to Commander on unknown format', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ name: 'X', format: 'brawl', cards: [] }),
    });
    const out = await fetchArchidektDeck('1');
    expect(out.format).toBe('Commander');
  });

  it('throws on network error', async () => {
    (fetch as jest.Mock).mockRejectedValueOnce(new TypeError('network down'));
    await expect(fetchArchidektDeck('1')).rejects.toThrow(/Network error/);
  });
});

describe('fetchDeckFromUrl', () => {
  it('dispatches to moxfield', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ name: 'M', format: 'modern', boards: {} }),
    });
    const out = await fetchDeckFromUrl('https://moxfield.com/decks/abc');
    expect((fetch as jest.Mock).mock.calls[0][0]).toBe('https://api.moxfield.com/v3/decks/all/abc');
    expect(out.format).toBe('Modern');
  });

  it('dispatches to archidekt', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ name: 'A', format: 'pauper', cards: [] }),
    });
    const out = await fetchDeckFromUrl('https://archidekt.com/decks/9999/foo-bar');
    expect((fetch as jest.Mock).mock.calls[0][0]).toBe('https://archidekt.com/api/decks/9999/');
    expect(out.format).toBe('Pauper');
  });

  it('throws on unsupported URL', async () => {
    await expect(fetchDeckFromUrl('https://mtggoldfish.com/deck/1')).rejects.toThrow(/Unsupported/);
  });
});
