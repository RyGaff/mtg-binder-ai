import { parseManaCost, manaGlyph, manaTint } from '../../src/utils/mana';

test('parseManaCost', () => {
  expect(parseManaCost('')).toEqual([]);
  expect(parseManaCost(null)).toEqual([]);
  expect(parseManaCost('{2}{U}{U}')).toEqual(['2', 'U', 'U']);
  expect(parseManaCost('{G/W}{U/P}')).toEqual(['G/W', 'U/P']);
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

test('hybrid + phyrexian collapse to first-color glyph + tint (intentional)', () => {
  expect(manaGlyph('U/P')).toBe(manaGlyph('U'));
  expect(manaGlyph('G/W')).toBe(manaGlyph('G'));
  expect(manaTint('U/P')).toBe(manaTint('U'));
});
