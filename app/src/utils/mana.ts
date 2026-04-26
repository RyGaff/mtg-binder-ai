// Mana font codepoints. Sourced from andrewgioia/mana (.ms-* classes in mana.css).
// Hybrid + phyrexian render as the first color's glyph (single-glyph approximation;
// the upstream font composes hybrid pips from two layered half-glyphs).
// Single colors + variable + colorless.
const SINGLE: Record<string, number> = {
  W: 0xe600, U: 0xe601, B: 0xe602, R: 0xe603, G: 0xe604,
  C: 0xe904, S: 0xe619, X: 0xe615, Y: 0xe616, Z: 0xe617,
};
// Numeric 0..15 are sequential at 0xe605..0xe614; 16..20 jump to 0xe62a..0xe62e.
const NUMERIC: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  for (let n = 0; n <= 15; n++) m[String(n)] = 0xe605 + n;
  for (let n = 16; n <= 20; n++) m[String(n)] = 0xe62a + (n - 16);
  return m;
})();

export const MANA_TINT: Record<string, string> = {
  W: '#f0e6c0', U: '#6ba8e8', B: '#bdb6b6', R: '#e8826b', G: '#6bc88a', C: '#bbb',
};

export function manaGlyph(token: string): string | null {
  if (SINGLE[token] != null) return String.fromCharCode(SINGLE[token]);
  if (NUMERIC[token] != null) return String.fromCharCode(NUMERIC[token]);
  // Hybrid (G/W) and Phyrexian (U/P): use first color's glyph as a single-glyph approximation.
  const m = /^([WUBRG])\/([WUBRGP])$/.exec(token);
  if (m && SINGLE[m[1]] != null) return String.fromCharCode(SINGLE[m[1]]);
  return null;
}

export function manaTint(token: string): string {
  if (MANA_TINT[token]) return MANA_TINT[token];
  const m = /^([WUBRG])\/[WUBRGP]$/.exec(token);
  if (m) return MANA_TINT[m[1]];
  return '#a4abbb'; // numeric / unknown
}

/** Split a Scryfall mana_cost string like "{2}{U}{U}" into tokens ["2","U","U"]. */
export function parseManaCost(cost: string | null | undefined): string[] {
  if (!cost) return [];
  return cost.split('}').map((s) => s.replace(/^\{/, '').trim()).filter(Boolean);
}
