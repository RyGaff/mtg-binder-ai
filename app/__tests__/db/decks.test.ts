import { createDeck, getDecks, addCardToDeck, getDeckCards, deleteDeck } from '../../src/db/decks';

jest.mock('expo-sqlite', () => ({
  openDatabaseSync: jest.fn(() => ({
    execSync: jest.fn(),
    runSync: jest.fn(() => ({ lastInsertRowId: 1 })),
    getFirstSync: jest.fn(() => null),
    getAllSync: jest.fn(() => []),
    withTransactionSync: jest.fn((fn: () => void) => fn()),
  })),
}));

describe('decks', () => {
  it('createDeck returns a numeric id', () => {
    const id = createDeck({ name: 'My Deck', format: 'commander' });
    expect(typeof id).toBe('number');
  });

  it('getDecks returns array', () => {
    expect(Array.isArray(getDecks())).toBe(true);
  });

  it('addCardToDeck does not throw', () => {
    expect(() => addCardToDeck({ deck_id: 1, scryfall_id: 'abc', quantity: 1, board: 'main' })).not.toThrow();
  });

  it('getDeckCards returns array', () => {
    expect(Array.isArray(getDeckCards(1))).toBe(true);
  });

  it('deleteDeck does not throw', () => {
    expect(() => deleteDeck(1)).not.toThrow();
  });
});
