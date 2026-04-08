import { fetchCardById, fetchCardBySetNumber, fetchCardByName, searchScryfall } from '../../src/api/scryfall';

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
    await fetchCardBySetNumber('lea', '161');
    expect(fetch).toHaveBeenCalledWith(
      'https://api.scryfall.com/cards/lea/161',
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
