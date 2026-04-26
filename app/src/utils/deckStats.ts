import type { DeckCard } from '../db/decks';
import { parseCmc, parseTypeLine, type CardType } from './cardHelpers';

type CurveBuckets = { 0: number; 1: number; 2: number; 3: number; 4: number; 5: number; '6+': number };
type ColorCounts = Record<'W' | 'U' | 'B' | 'R' | 'G' | 'C', number>;

export function manaCurve(cards: DeckCard[]): CurveBuckets {
  const out: CurveBuckets = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, '6+': 0 };
  for (const c of cards) {
    if (parseTypeLine(c.type_line ?? '') === 'Land') { out[0] += c.quantity; continue; }
    const v = parseCmc(c.mana_cost ?? '');
    if (v <= 5) out[v as 0 | 1 | 2 | 3 | 4 | 5] += c.quantity;
    else out['6+'] += c.quantity;
  }
  return out;
}

export function colorPipCounts(cards: DeckCard[]): ColorCounts {
  const out: ColorCounts = { W: 0, U: 0, B: 0, R: 0, G: 0, C: 0 };
  for (const c of cards) {
    let id: string[] = [];
    try { id = JSON.parse(c.color_identity || '[]'); } catch { /* skip */ }
    if (id.length === 0) { out.C += c.quantity; continue; }
    for (const k of id) if ('WUBRG'.includes(k)) out[k as 'W' | 'U' | 'B' | 'R' | 'G'] += c.quantity;
  }
  return out;
}

export function typeCounts(cards: DeckCard[]): Record<CardType, number> {
  const out: Record<CardType, number> = {
    Creature: 0, Planeswalker: 0, Battle: 0, Sorcery: 0,
    Instant: 0, Artifact: 0, Enchantment: 0, Land: 0, Other: 0,
  };
  for (const c of cards) out[parseTypeLine(c.type_line ?? '')] += c.quantity;
  return out;
}

export function avgCmc(cards: DeckCard[]): number {
  let n = 0, q = 0;
  for (const c of cards) {
    if (parseTypeLine(c.type_line ?? '') === 'Land') continue;
    n += parseCmc(c.mana_cost ?? '') * c.quantity;
    q += c.quantity;
  }
  return q === 0 ? 0 : n / q;
}

/** Per-card USD price from Scryfall's `prices.usd`. Returns null if missing/unparseable. */
export function cardPriceUsd(card: DeckCard): number | null {
  let p: { usd?: string | null } = {};
  try { p = JSON.parse(card.prices || '{}'); } catch { return null; }
  const usd = p.usd;
  if (usd == null) return null;
  const v = parseFloat(usd);
  return Number.isFinite(v) ? v : null;
}

/** Sum prices.usd weighted by quantity, skipping null/missing. */
export function boardPrice(cards: DeckCard[]): number {
  let total = 0;
  for (const c of cards) {
    let p: { usd?: string | null } = {};
    try { p = JSON.parse(c.prices || '{}'); } catch { /* skip */ }
    const usd = p.usd;
    if (usd == null) continue;
    const v = parseFloat(usd);
    if (Number.isFinite(v)) total += v * c.quantity;
  }
  return total;
}
