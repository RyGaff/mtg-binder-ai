import { buildSections } from '../../src/utils/deckSections';
import type { DeckCard } from '../../src/db/decks';

const card = (over: Partial<DeckCard>): DeckCard => ({
  deck_id: 1, scryfall_id: 'x', quantity: 1, board: 'main',
  name: 'X', set_code: 'set', collector_number: '1',
  mana_cost: '', type_line: 'Creature', oracle_text: '',
  color_identity: '[]', image_uri: '', image_uri_back: '',
  card_faces: '[]', all_parts: '[]', prices: '{}', keywords: '[]',
  layout: 'normal', cached_at: 0,
  ...over,
});

test('order: commander → main (with type sub-sections) → sideboard', () => {
  const cards = [
    card({ scryfall_id: 'a', name: 'Atraxa', board: 'commander', type_line: 'Legendary Creature — Phyrexian Angel' }),
    card({ scryfall_id: 'b', name: 'Sol Ring', board: 'main', type_line: 'Artifact', mana_cost: '{1}' }),
    card({ scryfall_id: 'c', name: 'Forest', board: 'main', type_line: 'Basic Land — Forest' }),
    card({ scryfall_id: 'd', name: 'Veil of Summer', board: 'side', type_line: 'Instant', mana_cost: '{G}' }),
  ];
  const out = buildSections(cards);
  expect(out.map((s) => s.title)).toEqual(['Commander', 'Main', 'Artifact', 'Land', 'Sideboard']);
  expect(out.map((s) => s.kind)).toEqual(['board', 'board', 'type', 'type', 'board']);
});

test('main board header has empty data; type sub-sections carry rows', () => {
  const cards = [
    card({ scryfall_id: 'a', name: 'Sol Ring', board: 'main', type_line: 'Artifact' }),
  ];
  const out = buildSections(cards);
  const main = out.find((s) => s.title === 'Main')!;
  const artifact = out.find((s) => s.title === 'Artifact')!;
  expect(main.data).toEqual([]);
  expect(artifact.data).toHaveLength(1);
});

test('counts + prices: board headers include price, type headers do not', () => {
  const cards = [
    card({ name: 'A', board: 'main', type_line: 'Creature', quantity: 2, prices: '{"usd":"3.00"}' }),
    card({ name: 'B', board: 'main', type_line: 'Creature', quantity: 1, prices: '{"usd":"1.50"}' }),
  ];
  const out = buildSections(cards);
  const main = out.find((s) => s.title === 'Main')!;
  const creature = out.find((s) => s.title === 'Creature')!;
  expect(main.count).toBe(3);
  expect(main.price).toBeCloseTo(7.5);
  expect(creature.count).toBe(3);
  expect(creature.price).toBeUndefined();
});

test('inner sort: by mana value asc, then name', () => {
  const cards = [
    card({ name: 'Z', board: 'main', type_line: 'Creature', mana_cost: '{1}' }),
    card({ name: 'A', board: 'main', type_line: 'Creature', mana_cost: '{2}' }),
    card({ name: 'B', board: 'main', type_line: 'Creature', mana_cost: '{1}' }),
  ];
  const out = buildSections(cards);
  const creature = out.find((s) => s.title === 'Creature')!;
  expect(creature.data.map((c) => c.name)).toEqual(['B', 'Z', 'A']);
});

test('omitted boards: empty commander / empty side leave their section out', () => {
  const cards = [card({ name: 'A', board: 'main', type_line: 'Creature' })];
  const out = buildSections(cards);
  expect(out.find((s) => s.title === 'Commander')).toBeUndefined();
  expect(out.find((s) => s.title === 'Sideboard')).toBeUndefined();
});
