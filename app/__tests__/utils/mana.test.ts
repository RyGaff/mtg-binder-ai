import { parseManaCost, manaGlyph, manaTint } from '../../src/utils/mana';

test('parseManaCost', () => {
  expect(parseManaCost('')).toEqual([]);
  expect(parseManaCost(null)).toEqual([]);
  expect(parseManaCost('{2}{U}{U}')).toEqual(['2', 'U', 'U']);
  expect(parseManaCost('{G/W}{U/P}')).toEqual(['G/W', 'U/P']);
  // Multiface castable cost (split / mdfc / adventure) — emit "//" sentinel.
  expect(parseManaCost('{1}{U} // {3}{R}')).toEqual(['1', 'U', '//', '3', 'R']);
  expect(parseManaCost('{W} // {2}{U}')).toEqual(['W', '//', '2', 'U']);
});

test('manaGlyph returns 1-char for known tokens, null for unknown', () => {
  expect(manaGlyph('W')?.length).toBe(1);
  expect(manaGlyph('5')?.length).toBe(1);
  expect(manaGlyph('20')?.length).toBe(1);
  expect(manaGlyph('G/W')?.length).toBe(1);
  expect(manaGlyph('FOO')).toBeNull();
});

test('manaTint covers single colors + hybrid', () => {
  expect(manaTint('W')).toBe('#f0e6c0');
  expect(manaTint('G/W')).toBe('#6bc88a');
  expect(manaTint('5')).toBe('#a4abbb');
});

test('hybrid + phyrexian collapse to a colored glyph + tint (intentional)', () => {
  expect(manaGlyph('U/P')).toBe(manaGlyph('U'));
  expect(manaGlyph('G/W')).toBe(manaGlyph('G'));
  expect(manaTint('U/P')).toBe(manaTint('U'));
  // Monocolored hybrid {2/W}: prefer the colored half for glyph + tint.
  expect(manaGlyph('2/W')).toBe(manaGlyph('W'));
  expect(manaTint('2/W')).toBe(manaTint('W'));
  // Colorless hybrid + phyrexian-hybrid still resolve.
  expect(manaGlyph('C/W')).toBe(manaGlyph('W'));
  expect(manaGlyph('W/U/P')).toBe(manaGlyph('W'));
  expect(manaTint('W/U/P')).toBe(manaTint('W'));
});
