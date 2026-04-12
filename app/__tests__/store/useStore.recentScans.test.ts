jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
  removeItem: jest.fn(() => Promise.resolve()),
}));

import { useStore } from '../../src/store/useStore';
import type { CachedCard } from '../../src/db/cards';

function makeCard(id: string): CachedCard {
  return {
    scryfall_id: id,
    name: `Card ${id}`,
    set_code: 'lea',
    collector_number: '1',
    mana_cost: '{1}',
    type_line: 'Instant',
    oracle_text: '',
    color_identity: '[]',
    image_uri: `https://example.com/${id}.jpg`,
    prices: JSON.stringify({ usd: '1.00' }),
    keywords: '[]',
    cached_at: Date.now(),
  };
}

describe('recentScans', () => {
  beforeEach(() => {
    useStore.setState({ recentScans: [] });
  });

  it('starts empty', () => {
    expect(useStore.getState().recentScans).toEqual([]);
  });

  it('addRecentScan prepends card (newest first)', () => {
    const { addRecentScan } = useStore.getState();
    addRecentScan(makeCard('a'));
    addRecentScan(makeCard('b'));
    const scans = useStore.getState().recentScans;
    expect(scans[0].scryfall_id).toBe('b');
    expect(scans[1].scryfall_id).toBe('a');
  });

  it('caps list at 10 entries', () => {
    const { addRecentScan } = useStore.getState();
    for (let i = 0; i < 12; i++) addRecentScan(makeCard(`card-${i}`));
    const scans = useStore.getState().recentScans;
    expect(scans).toHaveLength(10);
    expect(scans[0].scryfall_id).toBe('card-11');
    expect(scans[9].scryfall_id).toBe('card-2');
  });
});
