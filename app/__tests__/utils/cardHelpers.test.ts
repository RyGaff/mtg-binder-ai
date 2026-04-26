import { parseTypeLine, parseCmc, CARD_TYPE_ORDER } from '../../src/utils/cardHelpers';

test('parseTypeLine — em dash + hyphen, recognized + Other fallback', () => {
  expect(parseTypeLine('Legendary Creature — Phyrexian Angel')).toBe('Creature');
  expect(parseTypeLine('Creature - Human')).toBe('Creature');
  expect(parseTypeLine('Legendary Planeswalker — Ajani')).toBe('Planeswalker');
  expect(parseTypeLine('Basic Land — Forest')).toBe('Land');
  expect(parseTypeLine('')).toBe('Other');
  expect(parseTypeLine('Conspiracy')).toBe('Other');
});

test('parseCmc — numeric sum, hybrid = 1, X = 0, null/empty = 0', () => {
  expect(parseCmc('')).toBe(0);
  expect(parseCmc(null)).toBe(0);
  expect(parseCmc('{2}{U}{U}')).toBe(4);
  expect(parseCmc('{4}{B}')).toBe(5);
  expect(parseCmc('{G/W}{G/W}{G/W}')).toBe(3);
  expect(parseCmc('{U/P}{U/P}')).toBe(2);
  expect(parseCmc('{X}{R}{R}')).toBe(2);
});

test('CARD_TYPE_ORDER lists 9 types in canonical order', () => {
  expect(CARD_TYPE_ORDER).toEqual([
    'Creature', 'Planeswalker', 'Battle', 'Sorcery',
    'Instant', 'Artifact', 'Enchantment', 'Land', 'Other',
  ]);
});
