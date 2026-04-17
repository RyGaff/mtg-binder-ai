import { fetchCardById, fetchCardBySetNumber, fetchCardByName, searchScryfall, fetchPrintings } from '../../src/api/scryfall';

global.fetch = jest.fn();

const mockScryfallCard = {
  id: 'abc-123',
  name: 'Lightning Bolt',
  set: 'lea',
  collector_number: '161',
  mana_cost: '{R}',
  type_line: 'Instant',
  oracle_text: 'Lightning Bolt deals 3 damage to any target.',
  color_identity: ['R'],
  image_uris: { normal: 'https://example.com/bolt.jpg' },
  prices: { usd: '1.20', usd_foil: '4.50' },
  keywords: [],
};

beforeEach(() => {
  (fetch as jest.Mock).mockResolvedValue({
    ok: true,
    json: async () => mockScryfallCard,
  });
});

describe('scryfall API', () => {
  it('fetchCardById calls correct URL', async () => {
    await fetchCardById('abc-123');
    expect(fetch).toHaveBeenCalledWith(
      'https://api.scryfall.com/cards/abc-123',
      expect.any(Object)
    );
  });

  it('fetchCardBySetNumber calls correct URL', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ data: [mockScryfallCard] }),
    });
    await fetchCardBySetNumber('lea', '161');
    expect(fetch).toHaveBeenCalledWith(
      'https://api.scryfall.com/cards/search?q=set%3Alea+cn%3A161',
      expect.any(Object)
    );
  });

  it('fetchCardByName calls fuzzy endpoint', async () => {
    await fetchCardByName('Lightning Bolt');
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/cards/named?fuzzy='),
      expect.any(Object)
    );
  });

  it('fetchCardById returns normalized CachedCard', async () => {
    const card = await fetchCardById('abc-123');
    expect(card.scryfall_id).toBe('abc-123');
    expect(card.set_code).toBe('lea');
    expect(typeof card.color_identity).toBe('string'); // JSON stringified
  });
});

describe('fetchPrintings', () => {
  const mockList = {
    data: [
      {
        id: 'print-1',
        name: 'Lightning Bolt',
        set: 'lea',
        set_name: 'Limited Edition Alpha',
        collector_number: '161',
        prices: { usd: '1200.00', usd_foil: null },
      },
      {
        id: 'print-2',
        name: 'Lightning Bolt',
        set: 'm11',
        set_name: 'Magic 2011',
        collector_number: '149',
        prices: { usd: '1.20', usd_foil: '4.50' },
      },
    ],
    has_more: false,
  };

  beforeEach(() => {
    (fetch as jest.Mock).mockReset();
  });

  it('fetchPrintings calls correct URL', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockList });
    await fetchPrintings('Lightning Bolt');
    expect(fetch).toHaveBeenCalledWith(
      'https://api.scryfall.com/cards/search?q=!%22Lightning%20Bolt%22&unique=prints&order=released&dir=desc',
      expect.any(Object)
    );
  });

  it('fetchPrintings returns mapped PrintingSummary array', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => mockList });
    const results = await fetchPrintings('Lightning Bolt');
    expect(results).toHaveLength(2);
    expect(results[0]).toEqual({
      scryfall_id: 'print-1',
      set_code: 'lea',
      set_name: 'Limited Edition Alpha',
      collector_number: '161',
      image_uri: '',
      prices: { usd: '1200.00', usd_foil: null },
    });
    expect(results[1].prices.usd_foil).toBe('4.50');
  });

  it('fetchPrintings throws on non-ok response', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(fetchPrintings('Nonexistent Card')).rejects.toThrow('Scryfall 404');
  });
});
