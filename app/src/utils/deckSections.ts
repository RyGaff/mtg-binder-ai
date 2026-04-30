import type { DeckCard } from '../db/decks';
import { CARD_TYPE_ORDER, parseCmc, parseTypeLine } from './cardHelpers';
import { boardPrice, cardPriceUsd } from './deckStats';

export type RowSection = {
  /** Bold board-level label OR type sub-label inside Main. */
  title: string;
  /** "board" headers carry count + price; "type" headers carry just count. */
  kind: 'board' | 'type';
  count: number;
  price?: number;
  data: DeckCard[];
};

export type SortMode = 'type' | 'name' | 'mana' | 'color' | 'price';

const sumQty = (xs: DeckCard[]) => xs.reduce((s, c) => s + c.quantity, 0);

const byManaThenName = (a: DeckCard, b: DeckCard) =>
  parseCmc(a.mana_cost ?? '') - parseCmc(b.mana_cost ?? '') || a.name.localeCompare(b.name);
const byNameOnly = (a: DeckCard, b: DeckCard) => a.name.localeCompare(b.name);

// Color identity sort key: colorless first, then mono-W/U/B/R/G in WUBRG order,
// then multicolor grouped by length and lexicographic CI string. Falls back to
// name within the same key.
function colorKey(c: DeckCard): string {
  let arr: string[] = [];
  try { arr = JSON.parse(c.color_identity || '[]') as string[]; } catch { /* skip */ }
  const wubrg = arr.filter((k) => 'WUBRG'.includes(k)).sort((a, b) => 'WUBRG'.indexOf(a) - 'WUBRG'.indexOf(b)).join('');
  // 0-prefix the length so JS string compare orders shorter first.
  return `${String(wubrg.length).padStart(2, '0')}|${wubrg}`;
}
const byColorThenName = (a: DeckCard, b: DeckCard) =>
  colorKey(a).localeCompare(colorKey(b)) || a.name.localeCompare(b.name);

// Price sort base is ascending (cheapest first). Cards with no USD price sit
// at the bottom in asc and the top in desc — treated as +Infinity since the
// `desc` direction multiplies the result by -1 (flipping +Infinity to a
// large negative would land them at the top in asc, which is wrong).
const byPriceAscThenName = (a: DeckCard, b: DeckCard) => {
  const pa = cardPriceUsd(a) ?? Number.POSITIVE_INFINITY;
  const pb = cardPriceUsd(b) ?? Number.POSITIVE_INFINITY;
  return pa - pb || a.name.localeCompare(b.name);
};

function comparatorFor(mode: SortMode): (a: DeckCard, b: DeckCard) => number {
  if (mode === 'name') return byNameOnly;
  if (mode === 'mana') return byManaThenName;
  if (mode === 'color') return byColorThenName;
  if (mode === 'price') return byPriceAscThenName;
  return byManaThenName; // 'type' falls back to mana-then-name within each type bucket
}

/**
 * Output order:
 *   Commander (board, flat list)
 *   Main (board header) → 'type' splits into Creatures / Planeswalkers / ...
 *                         non-'type' modes flatten into a single sorted list.
 *   Sideboard (board, flat list)
 *   Considering (board, flat list — "maybeboard" / cards on the bubble)
 *
 * `sortMode` controls Main's sub-grouping behavior and the in-list sort across
 * every board. `sortDir` flips comparator output ('desc' = negated). Type
 * sub-section ORDER (Creatures → Lands etc.) stays fixed regardless of dir;
 * only rows inside each section get reversed.
 */
export function buildSections(
  cards: DeckCard[],
  sortMode: SortMode = 'type',
  sortDir: 'asc' | 'desc' = 'asc',
): RowSection[] {
  const commander = cards.filter((c) => c.board === 'commander');
  const main = cards.filter((c) => c.board === 'main');
  const side = cards.filter((c) => c.board === 'side');
  const considering = cards.filter((c) => c.board === 'considering');

  const baseCmp = comparatorFor(sortMode);
  const sign = sortDir === 'desc' ? -1 : 1;
  const cmp = (a: DeckCard, b: DeckCard) => sign * baseCmp(a, b);
  const out: RowSection[] = [];

  if (commander.length > 0) {
    out.push({ title: 'Commander', kind: 'board', count: sumQty(commander), price: boardPrice(commander), data: [...commander].sort(cmp) });
  }

  if (main.length > 0) {
    // Main always splits into type sub-sections (Creatures / Lands / etc.).
    // sortMode controls the row order WITHIN each sub-section; the type-bucket
    // ORDER itself is fixed by CARD_TYPE_ORDER so the deck reads consistently
    // regardless of the chosen sort.
    out.push({ title: 'Main', kind: 'board', count: sumQty(main), price: boardPrice(main), data: [] });
    const byType: Record<string, DeckCard[]> = {};
    for (const c of main) {
      const t = parseTypeLine(c.type_line ?? '');
      (byType[t] ??= []).push(c);
    }
    for (const t of CARD_TYPE_ORDER) {
      if (!byType[t]?.length) continue;
      out.push({ title: t, kind: 'type', count: sumQty(byType[t]), data: [...byType[t]].sort(cmp) });
    }
  }

  if (side.length > 0) {
    out.push({ title: 'Sideboard', kind: 'board', count: sumQty(side), price: boardPrice(side), data: [...side].sort(cmp) });
  }

  if (considering.length > 0) {
    out.push({ title: 'Considering', kind: 'board', count: sumQty(considering), price: boardPrice(considering), data: [...considering].sort(cmp) });
  }

  return out;
}
