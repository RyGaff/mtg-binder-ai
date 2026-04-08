import { upsertCard, getCardById, getCardBySetNumber, searchCardsLocal } from '../../src/db/cards';

// Mock expo-sqlite
jest.mock('expo-sqlite', () => ({
  openDatabaseSync: jest.fn(() => ({
    execSync: jest.fn(),
    runSync: jest.fn(),
    getFirstSync: jest.fn(),
    getAllSync: jest.fn(() => []),
    withTransactionSync: jest.fn((fn: () => void) => fn()),
  })),
}));

const mockCard = {
  scryfall_id: 'abc-123',
  name: 'Lightning Bolt',
  set_code: 'lea',
  collector_number: '161',
  mana_cost: '{R}',
  type_line: 'Instant',
  oracle_text: 'Lightning Bolt deals 3 damage to any target.',
  color_identity: JSON.stringify(['R']),
  image_uri: 'https://example.com/bolt.jpg',
  prices: JSON.stringify({ usd: '1.20', usd_foil: '4.50' }),
  keywords: JSON.stringify([]),
  cached_at: Date.now(),
};

describe('card cache', () => {
  it('upsertCard does not throw', () => {
    expect(() => upsertCard(mockCard)).not.toThrow();
  });

  it('getCardById returns null when not found', () => {
    const result = getCardById('nonexistent');
    expect(result).toBeNull();
  });

  it('getCardBySetNumber returns null when not found', () => {
    const result = getCardBySetNumber('lea', '161');
    expect(result).toBeNull();
  });

  it('searchCardsLocal returns array', () => {
    const result = searchCardsLocal('bolt');
    expect(Array.isArray(result)).toBe(true);
  });
});
