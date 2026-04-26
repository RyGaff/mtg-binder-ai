import { manaCurve, colorPipCounts, typeCounts, avgCmc, boardPrice } from '../../src/utils/deckStats';
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

test('manaCurve buckets, lands → 0, 6+ collapse', () => {
  const cards = [
    card({ mana_cost: '', type_line: 'Land', quantity: 24 }),
    card({ mana_cost: '{1}', quantity: 1 }),
    card({ mana_cost: '{2}{U}', quantity: 3 }),
    card({ mana_cost: '{6}{B}', quantity: 1 }),
  ];
  expect(manaCurve(cards)).toEqual({ 0: 24, 1: 1, 2: 0, 3: 3, 4: 0, 5: 0, '6+': 1 });
});

test('colorPipCounts identity-weighted by quantity, colorless → C', () => {
  const cards = [
    card({ color_identity: '["U","B"]', quantity: 2 }),
    card({ color_identity: '["B"]', quantity: 1 }),
    card({ color_identity: '[]', quantity: 1 }),
  ];
  expect(colorPipCounts(cards)).toEqual({ W: 0, U: 2, B: 3, R: 0, G: 0, C: 1 });
});

test('typeCounts grouped by parseTypeLine', () => {
  const cards = [
    card({ type_line: 'Creature — Elf', quantity: 2 }),
    card({ type_line: 'Instant', quantity: 4 }),
    card({ type_line: 'Basic Land — Forest', quantity: 24 }),
  ];
  const r = typeCounts(cards);
  expect(r.Creature).toBe(2);
  expect(r.Instant).toBe(4);
  expect(r.Land).toBe(24);
});

test('avgCmc excludes lands; zero non-lands → 0', () => {
  expect(avgCmc([
    card({ mana_cost: '{2}', type_line: 'Instant', quantity: 4 }),
    card({ mana_cost: '{4}', type_line: 'Sorcery', quantity: 2 }),
    card({ mana_cost: '', type_line: 'Land', quantity: 24 }),
  ])).toBeCloseTo((2 * 4 + 4 * 2) / 6, 5);
  expect(avgCmc([card({ type_line: 'Land', quantity: 24 })])).toBe(0);
});

test('boardPrice sums usd, skips null', () => {
  expect(boardPrice([
    card({ prices: '{"usd":"1.50"}', quantity: 2 }),
    card({ prices: '{"usd":"0.10"}', quantity: 4 }),
    card({ prices: '{"usd":null}', quantity: 1 }),
    card({ prices: '{}', quantity: 1 }),
  ])).toBeCloseTo(3.4, 5);
});
