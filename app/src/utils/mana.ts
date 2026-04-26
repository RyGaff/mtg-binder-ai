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

// For hybrid pips ({W/U}, {2/W}, {C/W}, {U/P}, {W/U/P}, …), pick the most
// representative single symbol: prefer a WUBRG color, then any single-letter
// symbol (C/X/S/Y/Z), then a numeric. Single-glyph approximation — the upstream
// font composes hybrid pips from two layered half-glyphs, which we don't render.
function pickHybridPrimary(token: string): string | null {
  if (!token.includes('/')) return null;
  const parts = token.split('/').map(p => p.trim()).filter(Boolean);
  for (const p of parts) if (p === 'W' || p === 'U' || p === 'B' || p === 'R' || p === 'G') return p;
  for (const p of parts) if (SINGLE[p] != null) return p;
  for (const p of parts) if (NUMERIC[p] != null) return p;
  return null;
}

export function manaGlyph(token: string): string | null {
  if (SINGLE[token] != null) return String.fromCharCode(SINGLE[token]);
  if (NUMERIC[token] != null) return String.fromCharCode(NUMERIC[token]);
  const primary = pickHybridPrimary(token);
  if (primary == null) return null;
  if (SINGLE[primary] != null) return String.fromCharCode(SINGLE[primary]);
  if (NUMERIC[primary] != null) return String.fromCharCode(NUMERIC[primary]);
  return null;
}

export function manaTint(token: string): string {
  if (MANA_TINT[token]) return MANA_TINT[token];
  const primary = pickHybridPrimary(token);
  if (primary && MANA_TINT[primary]) return MANA_TINT[primary];
  return '#a4abbb'; // numeric / unknown
}

/** Sentinel token emitted between castable faces (split / mdfc / adventure). */
export const MANA_FACE_SEP = '//';

/**
 * Split a Scryfall mana_cost string into tokens.
 * - `"{2}{U}{U}"` → `["2","U","U"]`
 * - `"{1}{U} // {3}{R}"` → `["1","U","//","3","R"]` (multiface castable cost)
 */
export function parseManaCost(cost: string | null | undefined): string[] {
  if (!cost) return [];
  const sides = cost.split('//');
  const out: string[] = [];
  sides.forEach((side, i) => {
    if (i > 0) out.push(MANA_FACE_SEP);
    for (const part of side.split('}')) {
      const t = part.trim().replace(/^\{/, '');
      if (t) out.push(t);
    }
  });
  return out;
}
