jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
  removeItem: jest.fn(() => Promise.resolve()),
}));

import { extractEffectCategories, parseManaValue, buildSimilarQuery, buildPrintingsQueryKey } from '../../src/api/hooks';
import type { CachedCard } from '../../src/db/cards';

// Mock expo-sqlite to avoid dependency issues during testing
jest.mock('expo-sqlite', () => ({
  openDatabaseSync: jest.fn(() => ({
    execSync: jest.fn(),
    runSync: jest.fn(),
    getFirstSync: jest.fn(),
    getAllSync: jest.fn(() => []),
    withTransactionSync: jest.fn((fn: () => void) => fn()),
  })),
}));

function makeCard(overrides: Partial<CachedCard> = {}): CachedCard {
  return {
    scryfall_id: 'test-id',
    name: 'Test Card',
    set_code: 'tst',
    collector_number: '1',
    mana_cost: '{R}',
    type_line: 'Instant',
    oracle_text: '',
    color_identity: '["R"]',
    image_uri: '',
    image_uri_back: '',
    card_faces: '[]',
    all_parts: '[]',
    prices: '{}',
    keywords: '[]',
    cached_at: Date.now(),
    ...overrides,
  };
}

describe('extractEffectCategories', () => {
  it('detects damage', () => {
    expect(extractEffectCategories('Lightning Bolt deals 3 damage to any target.')).toContain('damage');
  });

  it('detects draw', () => {
    expect(extractEffectCategories('Draw a card.')).toContain('draw');
  });

  it('detects draw (plural)', () => {
    expect(extractEffectCategories('Draw two cards.')).toContain('draw');
  });

  it('detects destroy', () => {
    expect(extractEffectCategories('Destroy target creature.')).toContain('destroy');
  });

  it('detects exile', () => {
    expect(extractEffectCategories('Exile target artifact.')).toContain('exile');
  });

  it('detects counter', () => {
    expect(extractEffectCategories('Counter target spell.')).toContain('counter');
  });

  it('detects tokens', () => {
    expect(extractEffectCategories('Create a 1/1 white Soldier creature token.')).toContain('tokens');
  });

  it('detects ramp via add mana', () => {
    expect(extractEffectCategories('Add {G}{G} to your mana pool.')).toContain('ramp');
  });

  it('detects ramp via land search', () => {
    expect(extractEffectCategories('Search your library for a basic land card.')).toContain('ramp');
  });

  it('detects lifegain', () => {
    expect(extractEffectCategories('You gain 3 life.')).toContain('lifegain');
  });

  it('detects discard', () => {
    expect(extractEffectCategories('Target player discards a card.')).toContain('discard');
  });

  it('detects bounce', () => {
    expect(extractEffectCategories('Return target creature to its owner\'s hand.')).toContain('bounce');
  });

  it('detects tutor', () => {
    expect(extractEffectCategories('Search your library for a card and put it into your hand.')).toContain('tutor');
  });

  it('detects pump', () => {
    expect(extractEffectCategories('Target creature gets +2/+2 until end of turn.')).toContain('pump');
  });

  it('detects multiple categories', () => {
    const cats = extractEffectCategories('Deal 2 damage to target creature. Draw a card.');
    expect(cats).toContain('damage');
    expect(cats).toContain('draw');
  });

  it('returns empty array for unrecognized text', () => {
    expect(extractEffectCategories('Untap all lands you control.')).toEqual([]);
  });
});

describe('parseManaValue', () => {
  it('parses single colored pip', () => {
    expect(parseManaValue('{R}')).toBe(1);
  });

  it('parses generic + colored', () => {
    expect(parseManaValue('{2}{R}')).toBe(3);
  });

  it('parses multi-color', () => {
    expect(parseManaValue('{1}{W}{B}')).toBe(3);
  });

  it('treats X as 0', () => {
    expect(parseManaValue('{X}{R}')).toBe(1);
  });

  it('returns 0 for empty mana cost', () => {
    expect(parseManaValue('')).toBe(0);
  });

  it('returns 0 for zero cost', () => {
    expect(parseManaValue('{0}')).toBe(0);
  });
});

describe('buildSimilarQuery', () => {
  it('includes oracle clause for known category', () => {
    const card = makeCard({ oracle_text: 'Lightning Bolt deals 3 damage to any target.', mana_cost: '{R}', color_identity: '["R"]' });
    const query = buildSimilarQuery(card);
    expect(query).toContain('o:"deals"');
    expect(query).toContain('o:"damage"');
  });

  it('includes mana value range', () => {
    const card = makeCard({ oracle_text: 'Deal 3 damage.', mana_cost: '{2}{R}', color_identity: '["R"]' });
    const query = buildSimilarQuery(card);
    expect(query).toContain('mv>=2');
    expect(query).toContain('mv<=4');
  });

  it('clamps mv lower bound to 0', () => {
    const card = makeCard({ oracle_text: 'Deal 1 damage.', mana_cost: '{R}', color_identity: '["R"]' });
    const query = buildSimilarQuery(card);
    expect(query).toContain('mv>=0');
  });

  it('includes color identity', () => {
    const card = makeCard({ oracle_text: 'Destroy target creature.', mana_cost: '{1}{B}', color_identity: '["B"]' });
    const query = buildSimilarQuery(card);
    expect(query).toContain('c:B');
  });

  it('falls back to mv+color only when no categories match', () => {
    const card = makeCard({ oracle_text: 'Untap all lands you control.', mana_cost: '{2}{G}', color_identity: '["G"]' });
    const query = buildSimilarQuery(card);
    expect(query).toContain('mv>=');
    expect(query).toContain('c:G');
    expect(query).not.toContain('o:');
  });

  it('handles colorless cards', () => {
    const card = makeCard({ oracle_text: 'Draw a card.', mana_cost: '{2}', color_identity: '[]' });
    const query = buildSimilarQuery(card);
    expect(query).not.toContain('c:');
  });
});

describe('buildPrintingsQueryKey', () => {
  it('returns array with card name', () => {
    const card = makeCard({ name: 'Lightning Bolt' });
    expect(buildPrintingsQueryKey(card)).toEqual(['printings', 'Lightning Bolt']);
  });
});
