import { addToCollection, getCollection, updateQuantity, removeFromCollection, getFoilCount, getTotalCardCount } from '../../src/db/collection';

jest.mock('expo-sqlite', () => ({
  openDatabaseSync: jest.fn(() => ({
    execSync: jest.fn(),
    runSync: jest.fn(),
    getFirstSync: jest.fn(() => null),
    getAllSync: jest.fn(() => []),
    withTransactionSync: jest.fn((fn: () => void) => fn()),
  })),
}));

describe('collection', () => {
  it('addToCollection does not throw', () => {
    expect(() => addToCollection({ scryfall_id: 'abc', quantity: 1, foil: false, condition: 'NM' })).not.toThrow();
  });

  it('getCollection returns array', () => {
    expect(Array.isArray(getCollection())).toBe(true);
  });

  it('updateQuantity does not throw', () => {
    expect(() => updateQuantity(1, 3)).not.toThrow();
  });

  it('removeFromCollection does not throw', () => {
    expect(() => removeFromCollection(1)).not.toThrow();
  });

  it('getFoilCount returns number', () => {
    expect(typeof getFoilCount()).toBe('number');
  });

  it('getTotalCardCount returns number', () => {
    expect(typeof getTotalCardCount()).toBe('number');
  });
});
